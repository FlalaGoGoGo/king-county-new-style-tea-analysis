#!/usr/bin/env python3
"""Fetch Google Place geometry (lat/lng) for store place_ids.

Input:
- data/raw/google_places_store_meta_king_county.csv

Output:
- data/raw/google_places_store_coords_king_county.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import time

import pandas as pd
import requests


DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-meta-input",
        default="data/raw/google_places_store_meta_king_county.csv",
        help="Store metadata with place_id.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/google_places_store_coords_king_county.csv",
        help="Output coordinate CSV.",
    )
    parser.add_argument(
        "--shop-input",
        default="data/interim/supply_shop_master.csv",
        help="Shop master table used for local coordinate fallback.",
    )
    parser.add_argument(
        "--raw-input",
        default="data/raw/king_county_food_inspections.csv",
        help="Raw inspection table with latitude/longitude used for fallback.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_MAPS_API_KEY",
        help="Environment variable for Google API key.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.08,
        help="Sleep time between requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=25.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--max-stores",
        type=int,
        default=0,
        help="Optional debug row limit (0 means all).",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Ignore output cache and refetch all rows.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def normalize_text(value: object) -> str:
    txt = str(value) if value is not None else ""
    txt = txt.upper().strip()
    txt = re.sub(r"[^A-Z0-9]+", "", txt)
    return txt


def extract_zip5(value: object) -> str:
    m = re.search(r"(\d{5})", str(value) if value is not None else "")
    return m.group(1) if m else ""


def load_store_meta(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = ["store_id", "place_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"store meta missing columns: {missing}")
    keep = ["store_id", "place_id", "shop_name_input", "matched_place_name", "matched_formatted_address"]
    for c in keep:
        if c not in df.columns:
            df[c] = ""
    out = df[keep].copy()
    out["store_id"] = out["store_id"].astype(str).str.strip()
    out["place_id"] = out["place_id"].astype(str).str.strip()
    out = out[out["store_id"] != ""].drop_duplicates(subset=["store_id"], keep="last")
    return out


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "store_id",
                "place_id",
                "latitude",
                "longitude",
                "details_status",
                "error",
                "source",
                "fetched_at",
            ]
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in ["store_id", "place_id", "latitude", "longitude", "details_status", "error", "source", "fetched_at"]:
        if c not in df.columns:
            df[c] = ""
    return df[["store_id", "place_id", "latitude", "longitude", "details_status", "error", "source", "fetched_at"]]


def load_shop_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["shop_key", "business_id", "trade_name", "street_address", "city", "zip5"]
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in ["shop_key", "business_id", "trade_name", "street_address", "city", "zip5"]:
        if c not in df.columns:
            df[c] = ""
    out = df[["shop_key", "business_id", "trade_name", "street_address", "city", "zip5"]].copy()
    out["store_id"] = out["shop_key"].astype(str).str.strip()
    out["business_id"] = out["business_id"].astype(str).str.strip()
    out["trade_name"] = out["trade_name"].astype(str).str.strip()
    out["street_address"] = out["street_address"].astype(str).str.strip()
    out["city"] = out["city"].astype(str).str.strip()
    out["zip5"] = out["zip5"].map(extract_zip5)
    out["name_norm"] = out["trade_name"].map(normalize_text)
    out = out[out["store_id"] != ""].drop_duplicates(subset=["store_id"], keep="last")
    return out


def load_raw_coord_index(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        return (
            pd.DataFrame(columns=["business_id", "latitude", "longitude"]),
            pd.DataFrame(columns=["name_norm", "zip5", "latitude", "longitude"]),
        )
    raw = pd.read_csv(path, dtype=str).fillna("")
    for c in ["business_id", "inspection_business_name", "zip_code", "latitude", "longitude"]:
        if c not in raw.columns:
            raw[c] = ""
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw[raw["latitude"].notna() & raw["longitude"].notna()].copy()
    raw["business_id"] = raw["business_id"].astype(str).str.strip()
    raw["zip5"] = raw["zip_code"].map(extract_zip5)
    raw["name_norm"] = raw["inspection_business_name"].map(normalize_text)

    by_bid = (
        raw[raw["business_id"] != ""]
        .groupby("business_id", as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
    )
    by_name_zip = (
        raw[(raw["name_norm"] != "") & (raw["zip5"] != "")]
        .groupby(["name_norm", "zip5"], as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
    )
    return by_bid, by_name_zip


def fallback_from_local(
    store_id: str,
    place_id: str,
    shop_by_store: dict[str, dict[str, object]],
    raw_by_bid: dict[str, dict[str, object]],
    raw_by_name_zip: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shop = shop_by_store.get(store_id, {})
    business_id = str(shop.get("business_id", "")).strip()
    if not business_id and store_id.startswith("BID|"):
        business_id = store_id.split("|", 1)[1].strip()

    if business_id and business_id in raw_by_bid:
        item = raw_by_bid[business_id]
        return {
            "store_id": store_id,
            "place_id": place_id,
            "latitude": to_float(item.get("latitude")),
            "longitude": to_float(item.get("longitude")),
            "details_status": "FALLBACK_RAW_BY_BUSINESS_ID",
            "error": "",
            "source": "king_county_inspection_raw",
            "fetched_at": now,
        }

    name_norm = normalize_text(shop.get("trade_name", "") or shop.get("name_norm", ""))
    zip5 = extract_zip5(shop.get("zip5", ""))
    key = (name_norm, zip5)
    if name_norm and zip5 and key in raw_by_name_zip:
        item = raw_by_name_zip[key]
        return {
            "store_id": store_id,
            "place_id": place_id,
            "latitude": to_float(item.get("latitude")),
            "longitude": to_float(item.get("longitude")),
            "details_status": "FALLBACK_RAW_BY_NAME_ZIP",
            "error": "",
            "source": "king_county_inspection_raw",
            "fetched_at": now,
        }

    return {
        "store_id": store_id,
        "place_id": place_id,
        "latitude": None,
        "longitude": None,
        "details_status": "NO_COORD_FALLBACK",
        "error": "no_google_or_raw_coordinate_match",
        "source": "king_county_inspection_raw",
        "fetched_at": now,
    }


def fetch_geometry(
    session: requests.Session,
    api_key: str,
    place_id: str,
    timeout_seconds: float,
) -> tuple[float | None, float | None, str, str]:
    params = {
        "place_id": place_id,
        "fields": "geometry/location",
        "key": api_key,
    }
    try:
        resp = session.get(DETAILS_URL, params=params, headers=UA, timeout=timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return None, None, "HTTP_ERROR", f"{type(exc).__name__}:{exc}"

    status = str(data.get("status", "")).strip() if isinstance(data, dict) else ""
    if status != "OK":
        err = str(data.get("error_message", "")).strip() if isinstance(data, dict) else ""
        return None, None, status or "UNKNOWN_STATUS", err

    result = data.get("result", {}) if isinstance(data, dict) else {}
    location = result.get("geometry", {}).get("location", {}) if isinstance(result, dict) else {}
    lat = to_float(location.get("lat"))
    lon = to_float(location.get("lng"))
    if lat is None or lon is None:
        return None, None, "NO_GEOMETRY", "geometry.location missing"
    return lat, lon, "OK", ""


def main() -> None:
    args = parse_args()
    root = project_root()
    store_meta_path = root / args.store_meta_input
    shop_path = root / args.shop_input
    raw_path = root / args.raw_input
    output_path = root / args.output

    api_key = os.environ.get(args.api_key_env, "").strip()

    stores = load_store_meta(store_meta_path)
    if args.max_stores > 0:
        stores = stores.head(args.max_stores).copy()
    shop_df = load_shop_master(shop_path)
    by_bid_df, by_name_zip_df = load_raw_coord_index(raw_path)
    shop_by_store = shop_df.set_index("store_id", drop=False).to_dict(orient="index")
    raw_by_bid = by_bid_df.set_index("business_id", drop=False).to_dict(orient="index")
    raw_by_name_zip = {
        (str(r["name_norm"]), str(r["zip5"])): r
        for r in by_name_zip_df.to_dict(orient="records")
    }

    cache = load_cache(output_path)
    cache_by_store = cache.drop_duplicates(subset=["store_id"], keep="last").set_index("store_id", drop=False).to_dict(
        orient="index"
    )

    session = requests.Session()
    rows: list[dict[str, object]] = []
    fetched = 0
    cached = 0
    ok = 0
    fallback_used = 0

    for _, r in stores.iterrows():
        store_id = str(r["store_id"]).strip()
        place_id = str(r["place_id"]).strip()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not place_id:
            item = fallback_from_local(
                store_id=store_id,
                place_id="",
                shop_by_store=shop_by_store,
                raw_by_bid=raw_by_bid,
                raw_by_name_zip=raw_by_name_zip,
            )
            if item.get("latitude") is not None and item.get("longitude") is not None:
                fallback_used += 1
            else:
                item["details_status"] = "NO_PLACE_ID"
                item["error"] = "missing_place_id"
                item["source"] = "google_place_details_geometry"
                item["fetched_at"] = now
            rows.append(item)
            continue

        existing = cache_by_store.get(store_id)
        if (
            not args.refresh_all
            and existing
            and str(existing.get("place_id", "")).strip() == place_id
            and to_float(existing.get("latitude")) is not None
            and to_float(existing.get("longitude")) is not None
        ):
            rows.append(existing)
            cached += 1
            continue

        if api_key:
            lat, lon, status, err = fetch_geometry(
                session=session,
                api_key=api_key,
                place_id=place_id,
                timeout_seconds=args.timeout_seconds,
            )
            fetched += 1
            if status == "OK":
                ok += 1
                rows.append(
                    {
                        "store_id": store_id,
                        "place_id": place_id,
                        "latitude": lat,
                        "longitude": lon,
                        "details_status": status,
                        "error": err,
                        "source": "google_place_details_geometry",
                        "fetched_at": now,
                    }
                )
                time.sleep(max(0.0, args.sleep_seconds))
                continue
            fallback_err = f"google_status={status}; {err}".strip("; ")
        else:
            fallback_err = "missing_google_api_key"

        item = fallback_from_local(
            store_id=store_id,
            place_id=place_id,
            shop_by_store=shop_by_store,
            raw_by_bid=raw_by_bid,
            raw_by_name_zip=raw_by_name_zip,
        )
        if item.get("latitude") is not None and item.get("longitude") is not None:
            fallback_used += 1
            item["error"] = fallback_err
        else:
            item["details_status"] = "NO_COORD_FALLBACK"
            item["error"] = fallback_err
            item["source"] = "google_place_details_geometry"
            item["fetched_at"] = now
        rows.append(item)
        time.sleep(max(0.0, args.sleep_seconds))

    out = pd.DataFrame(rows)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.sort_values(["store_id"]).reset_index(drop=True)

    ensure_parent(output_path)
    out.to_csv(output_path, index=False)

    print(
        "[35] Completed. "
        f"stores={len(out)}, cached={cached}, fetched={fetched}, ok={ok}, "
        f"fallback_used={fallback_used}, "
        f"coord_rows_with_lat={int(out['latitude'].notna().sum())}, "
        f"coord_rows_with_lng={int(out['longitude'].notna().sum())}, output={output_path}"
    )


if __name__ == "__main__":
    main()
