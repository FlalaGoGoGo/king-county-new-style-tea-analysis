#!/usr/bin/env python3
"""Collect official Google Places reviews and store metadata for King County tea shops.

Important notes:
- Uses official Google Places Web Service endpoints.
- Requires API key (env var by default: GOOGLE_MAPS_API_KEY).
- Place Details returns at most a small review subset (typically <= 5 per place).

Input:
- data/interim/supply_shop_master.csv

Outputs:
- data/raw/google_places_reviews_king_county.csv
- data/raw/google_places_store_meta_king_county.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time

import pandas as pd
import requests


FIND_PLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

REVIEW_COLS = [
    "zip5",
    "shop_name",
    "review_text",
    "review_date",
    "rating",
    "source_platform",
    "source_url",
    "reviewer_alias",
    "platform_review_id",
    "platform_business_id",
    "google_rating",
    "google_user_ratings_total",
]

STORE_COLS = [
    "store_id",
    "brand_id",
    "brand",
    "primary_beverage_category",
    "shop_name_input",
    "zip5_input",
    "query_text",
    "place_id",
    "matched_place_name",
    "matched_formatted_address",
    "matched_google_url",
    "website_url",
    "google_rating",
    "google_user_ratings_total",
    "google_price_level",
    "review_snippet_count",
    "extracted_review_rows",
    "find_status",
    "details_status",
    "fetch_ok",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shop-input",
        default="data/interim/supply_shop_master.csv",
        help="Shop master input.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/google_places_reviews_king_county.csv",
        help="Output review CSV.",
    )
    parser.add_argument(
        "--store-output",
        default="data/raw/google_places_store_meta_king_county.csv",
        help="Matched store metadata output.",
    )
    parser.add_argument(
        "--shop-brand-input",
        default="outputs/tables/shop_brand_category_king_county.csv",
        help="Optional store-brand mapping table used for brand->store sorting.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/interim/google_places_reviews_summary.json",
        help="Summary JSON output.",
    )
    parser.add_argument(
        "--failure-log",
        default="data/interim/google_places_reviews_failures.log",
        help="Failure log path.",
    )
    parser.add_argument(
        "--review-cache-input",
        default="",
        help="Optional existing review CSV used for cache reuse. Default = --output path.",
    )
    parser.add_argument(
        "--store-cache-input",
        default="",
        help="Optional existing store-meta CSV used for cache reuse. Default = --store-output path.",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Ignore cache and refetch all stores from API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_MAPS_API_KEY",
        help="Environment variable name for API key.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.12,
        help="Sleep between requests to reduce rate-limit issues.",
    )
    parser.add_argument(
        "--max-shops",
        type=int,
        default=0,
        help="Optional debug limit (0 = all shops).",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=15,
        help="Minimum review text length.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    s = str(text) if text is not None else ""
    s = s.replace("\n", " ").replace("\r", " ")
    return " ".join(s.split()).strip()


def to_float(v: object) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def to_int(v: object) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def norm_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper().strip())


def load_shops(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    req = ["shop_key", "trade_name", "street_address", "city", "state", "zip5"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"shop input missing columns: {miss}")

    out = df[req].copy()
    out["store_id"] = out["shop_key"].fillna("").astype(str)
    out["shop_name"] = out["trade_name"].fillna("").astype(str).str.strip()
    out["street_address"] = out["street_address"].fillna("").astype(str).str.strip()
    out["city"] = out["city"].fillna("").astype(str).str.strip()
    out["state"] = out["state"].fillna("WA").astype(str).str.strip()
    out["zip5"] = out["zip5"].fillna("").astype(str).str.extract(r"(\d{5})", expand=False).fillna("")
    out = out[["store_id", "shop_name", "street_address", "city", "state", "zip5"]].drop_duplicates(subset=["store_id"])
    return out


def build_query(row: pd.Series) -> str:
    parts = [
        str(row.get("shop_name", "")).strip(),
        str(row.get("street_address", "")).strip(),
        str(row.get("city", "")).strip(),
        str(row.get("state", "")).strip() or "WA",
        str(row.get("zip5", "")).strip(),
    ]
    return ", ".join([p for p in parts if p])


def prepare_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols]


def load_shop_brand_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["store_id", "brand_id", "brand", "primary_beverage_category"])
    raw = pd.read_csv(path, dtype=str).fillna("")
    for c in ["store_id", "brand_id", "brand", "primary_beverage_category"]:
        if c not in raw.columns:
            raw[c] = ""
    out = raw[["store_id", "brand_id", "brand", "primary_beverage_category"]].copy()
    out = out.drop_duplicates(subset=["store_id"], keep="first")
    return out


def load_store_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in STORE_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[STORE_COLS].copy()
    df["store_id"] = df["store_id"].astype(str).str.strip()
    df = df[df["store_id"] != ""].drop_duplicates(subset=["store_id"], keep="last")
    return {str(r["store_id"]): {k: r[k] for k in STORE_COLS} for _, r in df.iterrows()}


def load_review_cache(path: Path, min_text_len: int) -> dict[tuple[str, str], list[dict]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in REVIEW_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[REVIEW_COLS].copy()
    df["zip5"] = df["zip5"].astype(str).str.extract(r"(\d{5})", expand=False).fillna("")
    df["shop_name"] = df["shop_name"].fillna("").astype(str).str.strip()
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df = df[df["review_text"].str.len() >= int(min_text_len)].copy()
    out: dict[tuple[str, str], list[dict]] = {}
    for _, row in df.iterrows():
        key = (str(row["zip5"]), norm_name(str(row["shop_name"])))
        out.setdefault(key, []).append({c: row.get(c, "") for c in REVIEW_COLS})
    return out


def is_cache_row_usable(row: dict | None) -> bool:
    if not row:
        return False
    place_id = str(row.get("place_id", "")).strip()
    fetch_ok = str(row.get("fetch_ok", "")).strip().lower() in {"1", "true", "yes"}
    details_ok = str(row.get("details_status", "")).strip().upper() == "OK"
    has_google_meta = (
        str(row.get("google_rating", "")).strip() != ""
        or str(row.get("google_user_ratings_total", "")).strip() != ""
        or str(row.get("google_price_level", "")).strip() != ""
    )
    return bool(place_id) and (fetch_ok or details_ok or has_google_meta)


def overlay_store_meta(base: dict, store_id: str, row: pd.Series, query_text: str) -> dict:
    out = {k: base.get(k, "") for k in STORE_COLS}
    out["store_id"] = store_id
    out["brand_id"] = str(row.get("brand_id", "independent_other")).strip() or "independent_other"
    out["brand"] = str(row.get("brand", "Independent/Other")).strip() or "Independent/Other"
    out["primary_beverage_category"] = str(row.get("primary_beverage_category", "Unknown")).strip() or "Unknown"
    out["shop_name_input"] = str(row.get("shop_name", "")).strip()
    out["zip5_input"] = str(row.get("zip5", "")).strip()
    out["query_text"] = query_text
    return out


def main() -> None:
    args = parse_args()
    root = project_root()

    shop_path = root / args.shop_input
    out_path = root / args.output
    store_out_path = root / args.store_output
    review_cache_path = root / (args.review_cache_input or args.output)
    store_cache_path = root / (args.store_cache_input or args.store_output)
    shop_brand_path = root / args.shop_brand_input
    summary_path = root / args.summary_output
    failure_path = root / args.failure_log

    if not shop_path.exists():
        raise FileNotFoundError(f"shop input not found: {shop_path}")

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"Missing API key in env var {args.api_key_env}. "
            "Set the key before running this script."
        )

    shops = load_shops(shop_path)
    shop_brand_map = load_shop_brand_map(shop_brand_path)
    shops = shops.merge(shop_brand_map, on="store_id", how="left")
    for c, default_v in [
        ("brand_id", "independent_other"),
        ("brand", "Independent/Other"),
        ("primary_beverage_category", "Unknown"),
    ]:
        shops[c] = shops.get(c, "").fillna("").astype(str).str.strip()
        shops[c] = shops[c].where(shops[c] != "", default_v)

    if args.max_shops > 0:
        shops = shops.head(args.max_shops).copy()

    store_cache = {} if args.refresh_all else load_store_cache(store_cache_path)
    review_cache = {} if args.refresh_all else load_review_cache(review_cache_path, min_text_len=args.min_text_len)
    reused_review_keys: set[tuple[str, str]] = set()

    session = requests.Session()
    review_rows: list[dict] = []
    store_rows: list[dict] = []
    failures: list[str] = []

    find_ok = 0
    details_ok = 0
    cache_reused_store_rows = 0
    cache_reused_after_error = 0
    cache_reused_review_rows = 0

    for _, r in shops.iterrows():
        store_id = str(r["store_id"])
        shop_name = str(r["shop_name"])
        zip5 = str(r["zip5"])
        query_text = build_query(r)
        cache_key = (zip5, norm_name(shop_name))
        cache_row = store_cache.get(store_id)
        cache_usable = is_cache_row_usable(cache_row)

        store_meta = {
            "store_id": store_id,
            "brand_id": str(r.get("brand_id", "independent_other")).strip() or "independent_other",
            "brand": str(r.get("brand", "Independent/Other")).strip() or "Independent/Other",
            "primary_beverage_category": str(r.get("primary_beverage_category", "Unknown")).strip() or "Unknown",
            "shop_name_input": shop_name,
            "zip5_input": zip5,
            "query_text": query_text,
            "place_id": "",
            "matched_place_name": "",
            "matched_formatted_address": "",
            "matched_google_url": "",
            "website_url": "",
            "google_rating": None,
            "google_user_ratings_total": None,
            "google_price_level": None,
            "review_snippet_count": 0,
            "extracted_review_rows": 0,
            "find_status": "",
            "details_status": "",
            "fetch_ok": 0,
            "error": "",
        }

        if not query_text:
            err = "empty_query"
            failures.append(f"{store_id}\t{err}")
            if cache_usable:
                store_rows.append(overlay_store_meta(cache_row or {}, store_id, r, query_text))
                cache_reused_store_rows += 1
                if cache_key not in reused_review_keys:
                    cached_reviews = review_cache.get(cache_key, [])
                    review_rows.extend(cached_reviews)
                    cache_reused_review_rows += len(cached_reviews)
                    reused_review_keys.add(cache_key)
            else:
                store_meta["error"] = err
                store_rows.append(store_meta)
            continue

        if cache_usable and not args.refresh_all:
            store_rows.append(overlay_store_meta(cache_row or {}, store_id, r, query_text))
            cache_reused_store_rows += 1
            if cache_key not in reused_review_keys:
                cached_reviews = review_cache.get(cache_key, [])
                review_rows.extend(cached_reviews)
                cache_reused_review_rows += len(cached_reviews)
                reused_review_keys.add(cache_key)
            continue

        try:
            find_params = {
                "input": query_text,
                "inputtype": "textquery",
                "fields": "place_id,name,formatted_address",
                "key": api_key,
            }
            f_resp = session.get(FIND_PLACE_URL, params=find_params, headers=UA, timeout=25)
            f_resp.raise_for_status()
            f_data = f_resp.json()

            if isinstance(f_data, dict):
                store_meta["find_status"] = str(f_data.get("status", "")).strip()

            candidates = f_data.get("candidates", []) if isinstance(f_data, dict) else []
            if not candidates:
                err = "no_place_candidate"
                failures.append(f"{store_id}\t{err}")
                if cache_usable and not args.refresh_all:
                    cached = overlay_store_meta(cache_row or {}, store_id, r, query_text)
                    cached["error"] = f"{err}|cache_reused"
                    store_rows.append(cached)
                    cache_reused_after_error += 1
                    if cache_key not in reused_review_keys:
                        cached_reviews = review_cache.get(cache_key, [])
                        review_rows.extend(cached_reviews)
                        cache_reused_review_rows += len(cached_reviews)
                        reused_review_keys.add(cache_key)
                else:
                    store_meta["error"] = err
                    store_rows.append(store_meta)
                time.sleep(max(0.0, args.sleep_seconds))
                continue

            candidate = candidates[0]
            place_id = str(candidate.get("place_id", "")).strip()
            if not place_id:
                err = "missing_place_id"
                failures.append(f"{store_id}\t{err}")
                if cache_usable and not args.refresh_all:
                    cached = overlay_store_meta(cache_row or {}, store_id, r, query_text)
                    cached["error"] = f"{err}|cache_reused"
                    store_rows.append(cached)
                    cache_reused_after_error += 1
                    if cache_key not in reused_review_keys:
                        cached_reviews = review_cache.get(cache_key, [])
                        review_rows.extend(cached_reviews)
                        cache_reused_review_rows += len(cached_reviews)
                        reused_review_keys.add(cache_key)
                else:
                    store_meta["error"] = err
                    store_rows.append(store_meta)
                time.sleep(max(0.0, args.sleep_seconds))
                continue

            find_ok += 1
            store_meta["place_id"] = place_id
            store_meta["matched_place_name"] = str(candidate.get("name", "")).strip()
            store_meta["matched_formatted_address"] = str(candidate.get("formatted_address", "")).strip()

            detail_params = {
                "place_id": place_id,
                "fields": "name,formatted_address,url,website,rating,user_ratings_total,price_level,reviews",
                "key": api_key,
            }
            d_resp = session.get(DETAILS_URL, params=detail_params, headers=UA, timeout=25)
            d_resp.raise_for_status()
            d_data = d_resp.json()

            if isinstance(d_data, dict):
                store_meta["details_status"] = str(d_data.get("status", "")).strip()
            result = d_data.get("result", {}) if isinstance(d_data, dict) else {}
            details_ok += 1

            reviews = result.get("reviews", []) or []
            if not isinstance(reviews, list):
                reviews = []
            store_meta["review_snippet_count"] = len(reviews)

            source_url = str(result.get("url", "")).strip()
            if not source_url:
                source_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
            store_meta["matched_google_url"] = source_url
            store_meta["website_url"] = str(result.get("website", "")).strip()

            g_rating = to_float(result.get("rating"))
            g_total = to_int(result.get("user_ratings_total"))
            g_price = to_int(result.get("price_level"))
            store_meta["google_rating"] = g_rating
            store_meta["google_user_ratings_total"] = g_total
            store_meta["google_price_level"] = g_price

            extracted_rows = 0
            for rv in reviews:
                txt = normalize_text(rv.get("text", ""))
                if len(txt) < args.min_text_len:
                    continue

                ts = rv.get("time")
                review_date = ""
                try:
                    if ts is not None:
                        review_date = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                except Exception:
                    review_date = ""

                rating = to_float(rv.get("rating"))

                alias = str(rv.get("author_name", "")).strip()
                author_url = str(rv.get("author_url", "")).strip()
                if author_url:
                    alias = alias or author_url

                review_rows.append(
                    {
                        "zip5": zip5,
                        "shop_name": shop_name,
                        "review_text": txt,
                        "review_date": review_date,
                        "rating": rating,
                        "source_platform": "Google Places API (official)",
                        "source_url": source_url,
                        "reviewer_alias": alias,
                        "platform_review_id": f"{place_id}|{alias}|{review_date}|{rating}",
                        "platform_business_id": place_id,
                        "google_rating": g_rating,
                        "google_user_ratings_total": g_total,
                    }
                )
                extracted_rows += 1

            store_meta["extracted_review_rows"] = extracted_rows
            store_meta["fetch_ok"] = int(bool(store_meta["place_id"]) and store_meta["details_status"] == "OK")
            store_rows.append(store_meta)

        except Exception as exc:
            err = f"{type(exc).__name__}:{exc}"
            failures.append(f"{store_id}\t{err}")
            if cache_usable and not args.refresh_all:
                cached = overlay_store_meta(cache_row or {}, store_id, r, query_text)
                cached["error"] = f"{err}|cache_reused"
                store_rows.append(cached)
                cache_reused_after_error += 1
                if cache_key not in reused_review_keys:
                    cached_reviews = review_cache.get(cache_key, [])
                    review_rows.extend(cached_reviews)
                    cache_reused_review_rows += len(cached_reviews)
                    reused_review_keys.add(cache_key)
            else:
                store_meta["error"] = err
                store_rows.append(store_meta)

        time.sleep(max(0.0, args.sleep_seconds))

    review_df = pd.DataFrame(review_rows)
    if not review_df.empty:
        review_df["_key"] = review_df[["zip5", "shop_name", "review_text", "review_date", "rating"]].astype(str).agg("||".join, axis=1)
        review_df = review_df.drop_duplicates(subset=["_key"]).drop(columns=["_key"]).reset_index(drop=True)
        review_df = review_df.sort_values(["zip5", "shop_name", "review_date"], na_position="last")
    review_df = prepare_frame(review_df, REVIEW_COLS)

    store_df = pd.DataFrame(store_rows)
    if not store_df.empty:
        store_df = store_df.drop_duplicates(subset=["store_id"], keep="last")
        store_df = store_df.sort_values(
            ["brand", "shop_name_input", "zip5_input"],
            ascending=[True, True, True],
            na_position="last",
        ).reset_index(drop=True)
    store_df = prepare_frame(store_df, STORE_COLS)

    ensure_parent(out_path)
    ensure_parent(store_out_path)
    review_df.to_csv(out_path, index=False)
    store_df.to_csv(store_out_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "shop_input_count": int(shops.shape[0]),
        "find_place_success": int(find_ok),
        "details_success": int(details_ok),
        "cache_reused_store_rows": int(cache_reused_store_rows),
        "cache_reused_after_error": int(cache_reused_after_error),
        "cache_reused_review_rows": int(cache_reused_review_rows),
        "store_place_matched": int(store_df["place_id"].fillna("").astype(str).str.len().gt(0).sum()),
        "store_meta_rows": int(store_df.shape[0]),
        "output_rows": int(review_df.shape[0]),
        "zip_coverage": int(review_df["zip5"].nunique()),
        "shop_coverage": int(review_df["shop_name"].nunique()),
        "store_output": str(store_out_path),
        "note": "Google Place Details returns limited review snippets (typically <=5 per place).",
    }
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        ensure_parent(failure_path)
        failure_path.write_text("\n".join(failures), encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()

    print(
        "[25] Completed. "
        f"shops={summary['shop_input_count']}, find_ok={summary['find_place_success']}, "
        f"details_ok={summary['details_success']}, matched={summary['store_place_matched']}, "
        f"cache_reused={summary['cache_reused_store_rows']}, reused_reviews={summary['cache_reused_review_rows']}, "
        f"output_rows={summary['output_rows']}, zip_coverage={summary['zip_coverage']}, "
        f"shop_coverage={summary['shop_coverage']}, output={out_path}, store_output={store_out_path}"
    )
    if failures:
        print(f"[25] failures={len(failures)} log={failure_path}")


if __name__ == "__main__":
    main()
