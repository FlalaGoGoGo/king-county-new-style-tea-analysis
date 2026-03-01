#!/usr/bin/env python3
"""Build streamlined v7 dashboard HTML for King County new tea analysis.

Design updates in this version:
- Single clean typography system
- Left vertical control rail (filters, scenarios, KPI)
- Replace oversized Opportunity bar chart with compact opportunity matrix
- Remove advanced filter block
- Improve chart readability (no scatter text overlap)
- Add map explorer (pin/heat views) with store coordinates
- Keep robust logo fallback and review evidence interactions
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCORING_WEIGHTS = {
    "demand_potential": 0.28,
    "supply_gap": 0.25,
    "growth_momentum": 0.17,
    "text_signal": 0.10,
    "feasibility": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-master",
        default="data/reference/zip_master.csv",
        help="ZIP-level panel table.",
    )
    parser.add_argument(
        "--store-master",
        default="data/reference/store_master.csv",
        help="Store-level table.",
    )
    parser.add_argument(
        "--brand-summary",
        default="data/reference/brand_summary.csv",
        help="Brand summary table.",
    )
    parser.add_argument(
        "--license-trend",
        default="data/reference/license_trend_top10.csv",
        help="Top ZIP annual license trend table.",
    )
    parser.add_argument(
        "--review-samples",
        default="data/reference/review_samples.csv",
        help="Review samples table.",
    )
    parser.add_argument(
        "--store-logos",
        default="data/reference/store_logo_map.csv",
        help="Store logo map table.",
    )
    parser.add_argument(
        "--brand-logos",
        default="data/reference/brand_logo_map.csv",
        help="Brand logo map table.",
    )
    parser.add_argument(
        "--kpi-summary",
        default="data/reference/kpi_summary.csv",
        help="KPI summary table.",
    )
    parser.add_argument(
        "--store-coords",
        default="data/raw/google_places_store_coords_king_county.csv",
        help="Store coordinates table (lat/lng by store_id).",
    )
    parser.add_argument(
        "--wa-zip-geojson",
        default="data/external/geo/wa_zip_codes.geojson",
        help="Washington ZIP boundary GeoJSON.",
    )
    parser.add_argument(
        "--us-counties-geojson",
        default="data/external/geo/us_counties_fips.geojson",
        help="US counties GeoJSON (for King County boundary).",
    )
    parser.add_argument(
        "--output",
        default="outputs/dashboard/index.html",
        help="Output dashboard path.",
    )
    parser.add_argument(
        "--hero-logo-input",
        default="assets/king_county_logo.png",
        help="Optional hero logo image path (png/jpg/jpeg/svg/webp).",
    )
    parser.add_argument(
        "--weights-config",
        default="configs/opportunity_weights.yaml",
        help="Scoring weights config for dashboard formula note.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_csv_optional(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_scoring_weights(path: Path) -> dict[str, float]:
    defaults = DEFAULT_SCORING_WEIGHTS
    if not path.exists():
        return defaults
    weights: dict[str, float] = {}
    in_weights = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("weights:"):
            in_weights = True
            continue
        if in_weights and ":" in s and not s.startswith("formula:"):
            k, v = [x.strip() for x in s.split(":", 1)]
            try:
                weights[k] = float(v)
            except ValueError:
                continue
        if s.startswith("formula:"):
            break
    if "supply_pressure" in weights and "supply_gap" not in weights:
        weights["supply_gap"] = weights["supply_pressure"]
    kept = {k: float(v) for k, v in weights.items() if k in defaults}
    if not kept:
        return defaults
    total = sum(kept.values())
    if total <= 0:
        return defaults
    return {k: v / total for k, v in kept.items()}


def scoring_formula_note(weights: dict[str, float]) -> str:
    w = {**DEFAULT_SCORING_WEIGHTS, **weights}
    return (
        "Opportunity = "
        f"{w['demand_potential']:.2f}*Demand + "
        f"{w['supply_gap']:.2f}*Gap + "
        f"{w['growth_momentum']:.2f}*Momentum + "
        f"{w['text_signal']:.2f}*Text + "
        f"{w['feasibility']:.2f}*Feasibility (ZIP rent/home + store price proxy)"
    )


def normalize_zip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{5})", expand=False).fillna("")


def normalize_zip_token(value: Any) -> str:
    txt = str(value or "")
    m = re.search(r"(\d{5})", txt)
    return m.group(1) if m else ""


def feature_zip_code(feature: dict[str, Any]) -> str:
    props = feature.get("properties", {}) if isinstance(feature, dict) else {}
    for key in ("ZCTA5CE10", "ZCTA5CE20", "ZIPCODE", "zip", "zip5", "GEOID10", "GEOID20"):
        if key in props:
            z = normalize_zip_token(props.get(key))
            if z:
                return z
    return normalize_zip_token(feature.get("id") if isinstance(feature, dict) else "")


def build_map_layers(zip_df: pd.DataFrame, wa_zip_geo: dict[str, Any], counties_geo: dict[str, Any]) -> dict[str, Any]:
    zip_set = set(zip_df.get("zip5", pd.Series(dtype=str)).astype(str).map(normalize_zip_token).tolist())
    zip_features: list[dict[str, Any]] = []
    county_features: list[dict[str, Any]] = []

    wa_features = wa_zip_geo.get("features", []) if isinstance(wa_zip_geo, dict) else []
    for ft in wa_features:
        z = feature_zip_code(ft if isinstance(ft, dict) else {})
        if z and z in zip_set:
            zip_features.append(ft)

    county_rows = counties_geo.get("features", []) if isinstance(counties_geo, dict) else []
    for ft in county_rows:
        if not isinstance(ft, dict):
            continue
        props = ft.get("properties", {}) if isinstance(ft.get("properties", {}), dict) else {}
        fid = str(ft.get("id", "")).strip()
        state = str(props.get("STATE", "")).zfill(2)
        county = str(props.get("COUNTY", "")).zfill(3)
        if fid == "53033" or (state == "53" and county == "033"):
            county_features.append(ft)

    return {
        "zip_boundaries": {"type": "FeatureCollection", "features": zip_features},
        "king_county_boundary": {"type": "FeatureCollection", "features": county_features},
    }


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    if pd.isna(value):
        return None
    return value


def to_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out: list[dict[str, Any]] = []
    for row in df[columns].to_dict(orient="records"):
        out.append({k: clean_value(v) for k, v in row.items()})
    return out


def pick_palette_pair(seed_text: str) -> tuple[str, str]:
    palette = [
        ("#1f2937", "#f8fafc"),
        ("#1d3557", "#f1faee"),
        ("#2b2d42", "#edf2f4"),
        ("#264653", "#f1faee"),
        ("#3d405b", "#fdf0d5"),
        ("#2f3e46", "#edf6f9"),
        ("#344e41", "#f6fff8"),
        ("#274c77", "#f8f9fa"),
    ]
    digest = hashlib.md5((seed_text or "tea").encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(palette)
    return palette[idx]


def make_initials(brand: str, store: str) -> str:
    seed = brand or store or "Tea"
    words = re.findall(r"[A-Za-z0-9]+", seed)
    if not words:
        return "TE"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def make_logo_data_uri(brand: str, store: str) -> str:
    initials = make_initials(brand, store)
    bg, fg = pick_palette_pair(f"{brand}|{store}")
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='96' height='96' viewBox='0 0 96 96'>"
        f"<rect width='96' height='96' rx='20' fill='{bg}'/>"
        "<circle cx='48' cy='48' r='26' fill='rgba(255,255,255,0.1)'/>"
        f"<text x='48' y='56' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' "
        f"font-size='30' font-weight='700' fill='{fg}'>{initials}</text>"
        "</svg>"
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


IMAGE_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".avif": "image/avif",
}


def canonical_brand_key(value: Any) -> str:
    txt = str(value or "").lower()
    txt = txt.replace("&", " and ").replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "", txt)


def file_data_uri(path: Path) -> str:
    mime = IMAGE_MIME.get(path.suffix.lower())
    if not mime:
        return ""
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return f"data:{mime};base64,{encoded}"


def build_local_brand_logo_map(images_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not images_dir.exists() or not images_dir.is_dir():
        return out
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_MIME:
            continue
        stem = p.stem.lower()
        if stem.startswith("king_county_logo"):
            continue
        stem = re.sub(r"(_logo|-logo)$", "", stem)
        key = canonical_brand_key(stem.replace("_", " ").replace("-", " "))
        if not key:
            continue
        uri = file_data_uri(p)
        if not uri:
            continue
        out[key] = uri
    return out


FALLBACK_HERO_LOGO_URI = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNjAiIGhlaWdodD0iMTIwIiB2aWV3Qm94PSIwIDAgMzYwIDEyMCIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsPSJLaW5nIENvdW50eSBsb2dvIj48cmVjdCB3aWR0aD0iMzYwIiBoZWlnaHQ9IjEyMCIgcng9IjE2IiBmaWxsPSIjZmZmZmZmIi8+PGNpcmNsZSBjeD0iNTYiIGN5PSI2MCIgcj0iMzYiIGZpbGw9IiMyNDVjM2EiLz48cGF0aCBkPSJNNDIgNzNoMjh2NUg0MnpNNDQgNjhsNC0xNiA4IDYgOC02IDQgMTZ6TTUyIDUybDQtOSA0IDkiIGZpbGw9IiNmZmZmZmYiLz48dGV4dCB4PSIxMDQiIHk9IjU4IiBmb250LWZhbWlseT0iTWFucm9wZSxTZWdvZSBVSSxzYW5zLXNlcmlmIiBmb250LXNpemU9IjMwIiBmb250LXdlaWdodD0iODAwIiBmaWxsPSIjMWYzZDVhIj5LSU5HIENPVU5UWTwvdGV4dD48dGV4dCB4PSIxMDQiIHk9Ijg0IiBmb250LWZhbWlseT0iTWFucm9wZSxTZWdvZSBVSSxzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmb250LXdlaWdodD0iNzAwIiBsZXR0ZXItc3BhY2luZz0iMS4yIiBmaWxsPSIjNWM2YjgwIj5XQVNISU5HVE9OPC90ZXh0Pjwvc3ZnPg=="
)


def resolve_hero_logo_data_uri(root: Path, logo_input: str) -> str:
    path = Path(logo_input).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return FALLBACK_HERO_LOGO_URI
    uri = file_data_uri(path)
    return uri or FALLBACK_HERO_LOGO_URI


def build_payload(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    zip_df = read_csv(root / args.zip_master, dtype={"zip5": str})
    zip_df["zip5"] = normalize_zip(zip_df["zip5"])
    zip_df = zip_df.drop_duplicates(subset=["zip5"]).copy()

    store_df = read_csv(root / args.store_master, dtype={"zip5": str})
    store_df["zip5"] = normalize_zip(store_df["zip5"])
    store_df = store_df.drop_duplicates(subset=["store_id"]).copy()
    for c in ["category_tags", "auxiliary_tags", "category_assignment_method", "website_url"]:
        if c not in store_df.columns:
            store_df[c] = ""

    brand_df = read_csv(root / args.brand_summary)
    license_df = read_csv(root / args.license_trend, dtype={"zip5": str})
    license_df["zip5"] = normalize_zip(license_df["zip5"])
    review_df = read_csv(root / args.review_samples, dtype={"zip5": str})
    review_df["zip5"] = normalize_zip(review_df["zip5"])
    coords_df = read_csv_optional(root / args.store_coords, dtype=str).fillna("")
    wa_zip_geo = read_json_optional(root / args.wa_zip_geojson)
    counties_geo = read_json_optional(root / args.us_counties_geojson)

    store_logo_df = read_csv(root / args.store_logos)
    brand_logo_df = read_csv(root / args.brand_logos)
    kpi_df = read_csv(root / args.kpi_summary)
    weights = load_scoring_weights(root / args.weights_config)

    store_logo_df = store_logo_df.drop_duplicates(subset=["store_id"])[
        ["store_id", "logo_url", "logo_source"]
    ]
    brand_logo_df = brand_logo_df.drop_duplicates(subset=["brand"])[
        ["brand", "logo_url"]
    ].rename(columns={"logo_url": "brand_logo_url"})

    store_df = store_df.merge(store_logo_df, on="store_id", how="left")
    store_df = store_df.merge(brand_logo_df, on="brand", how="left")
    store_df["logo_url"] = store_df["logo_url"].fillna(store_df["brand_logo_url"])
    store_df = store_df.drop(columns=["brand_logo_url"])

    local_brand_logo_map = build_local_brand_logo_map(root / "images")
    if local_brand_logo_map:
        store_df["brand_key"] = store_df["brand"].map(canonical_brand_key)

        def pick_local_logo(brand_key: Any) -> str:
            bkey = canonical_brand_key(brand_key)
            if not bkey:
                return ""
            if bkey in local_brand_logo_map:
                return local_brand_logo_map[bkey]
            # Safe fallback: allow prefix match in both directions (brand/file), avoiding short noisy keys.
            prefix_hits: list[tuple[int, str]] = []
            for key, uri in local_brand_logo_map.items():
                if len(key) < 4:
                    continue
                if bkey.startswith(key):
                    prefix_hits.append((len(key), uri))
                    continue
                if len(bkey) >= 4 and key.startswith(bkey):
                    prefix_hits.append((len(bkey), uri))
            if prefix_hits:
                prefix_hits.sort(key=lambda x: x[0], reverse=True)
                return prefix_hits[0][1]
            return ""

        store_df["local_logo_url"] = store_df["brand_key"].map(pick_local_logo)
        use_local = store_df["local_logo_url"].astype(str).str.len() > 0
        store_df.loc[use_local, "logo_url"] = store_df.loc[use_local, "local_logo_url"]
        store_df.loc[use_local, "logo_source"] = "local_images_brand_logo"
        store_df = store_df.drop(columns=["brand_key", "local_logo_url"], errors="ignore")

    if not coords_df.empty:
        for c in ["store_id", "latitude", "longitude", "details_status", "source"]:
            if c not in coords_df.columns:
                coords_df[c] = ""
        coords_df["store_id"] = coords_df["store_id"].astype(str).str.strip()
        coords_df["latitude"] = pd.to_numeric(coords_df["latitude"], errors="coerce")
        coords_df["longitude"] = pd.to_numeric(coords_df["longitude"], errors="coerce")
        coords_df = (
            coords_df[["store_id", "latitude", "longitude", "details_status", "source"]]
            .drop_duplicates(subset=["store_id"], keep="last")
            .rename(columns={"details_status": "coord_status", "source": "coord_source"})
        )
        store_df = store_df.merge(coords_df, on="store_id", how="left")
    else:
        store_df["latitude"] = pd.NA
        store_df["longitude"] = pd.NA
        store_df["coord_status"] = ""
        store_df["coord_source"] = ""

    store_df["latitude"] = pd.to_numeric(store_df["latitude"], errors="coerce")
    store_df["longitude"] = pd.to_numeric(store_df["longitude"], errors="coerce")
    store_df["coord_status"] = store_df["coord_status"].fillna("").astype(str)
    store_df["coord_source"] = store_df["coord_source"].fillna("").astype(str)

    # Fallback for invalid/retired place_ids: fill from ZIP centroid of valid stores.
    zip_centroids = (
        store_df.dropna(subset=["latitude", "longitude"])
        .groupby("zip5", as_index=False)
        .agg(zip_latitude=("latitude", "mean"), zip_longitude=("longitude", "mean"))
    )
    store_df = store_df.merge(zip_centroids, on="zip5", how="left")
    need_fill = store_df["latitude"].isna() | store_df["longitude"].isna()
    store_df.loc[need_fill, "latitude"] = store_df.loc[need_fill, "zip_latitude"]
    store_df.loc[need_fill, "longitude"] = store_df.loc[need_fill, "zip_longitude"]
    imputed = need_fill & store_df["zip_latitude"].notna() & store_df["zip_longitude"].notna()
    store_df.loc[imputed, "coord_source"] = "zip_centroid_imputed"
    store_df.loc[imputed & store_df["coord_status"].eq(""), "coord_status"] = "IMPUTED"
    store_df = store_df.drop(columns=["zip_latitude", "zip_longitude"])
    store_df["logo_fallback"] = store_df.apply(
        lambda r: make_logo_data_uri(str(r.get("brand", "")), str(r.get("store_name", ""))),
        axis=1,
    )

    brand_df["logo_fallback"] = brand_df.apply(
        lambda r: make_logo_data_uri(str(r.get("brand", "")), str(r.get("brand", ""))),
        axis=1,
    )

    zip_cols_for_store = [
        "zip5",
        "rank",
        "opportunity_score",
        "demand_score",
        "supply_score",
        "gap_score",
        "gap_component_score",
        "momentum_score",
        "text_signal_score",
        "feasibility_score",
        "regional_cost_score",
        "regional_feasibility_score",
        "store_cost_score",
        "store_affordability_score",
        "store_cost_coverage",
        "median_gross_rent",
        "median_home_value",
        "median_google_price_level",
        "quadrant",
        "recommended_strategy",
        "action_note",
    ]
    for c in zip_cols_for_store:
        if c not in zip_df.columns:
            zip_df[c] = pd.NA
    store_df = store_df.merge(zip_df[zip_cols_for_store].copy(), on="zip5", how="left")

    review_df["review_rank"] = pd.to_numeric(review_df["review_rank"], errors="coerce").fillna(999)
    review_df = review_df.sort_values(["store_id", "review_rank", "review_date"], ascending=[True, True, False])

    kpi_map: dict[str, Any] = {}
    if {"metric", "value"}.issubset(kpi_df.columns):
        for _, row in kpi_df.iterrows():
            kpi_map[str(row["metric"])] = clean_value(row["value"])

    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scope": "King County, WA",
            "stores_total": int(store_df["store_id"].nunique()),
            "brands_total": int(store_df["brand"].nunique()),
            "zips_total": int(zip_df["zip5"].nunique()),
            "review_rows_total": int(len(review_df)),
            "stores_with_coordinates": int((store_df["latitude"].notna() & store_df["longitude"].notna()).sum()),
            "kpi": kpi_map,
            "scoring_weights": weights,
            "scoring_formula_note": scoring_formula_note(weights),
        },
        "map_layers": build_map_layers(zip_df=zip_df, wa_zip_geo=wa_zip_geo, counties_geo=counties_geo),
        "zip": to_records(
            zip_df,
            [
                "zip5",
                "rank",
                "opportunity_score",
                "demand_score",
                "supply_score",
                "gap_score",
                "gap_component_score",
                "momentum_score",
                "text_signal_score",
                "feasibility_score",
                "regional_cost_score",
                "regional_feasibility_score",
                "store_cost_score",
                "store_affordability_score",
                "store_cost_coverage",
                "median_gross_rent",
                "median_home_value",
                "median_google_price_level",
                "active_shop_count",
                "population",
                "median_income",
                "young_adult_share",
                "quadrant",
                "opportunity_tier",
                "top_topic_terms",
                "mean_sentiment",
                "positive_share",
                "negative_share",
                "avg_rating",
                "recommended_strategy",
                "action_note",
                "substitute_store_count",
                "coffee_store_count",
                "matcha_store_count",
                "vietnamese_coffee_store_count",
                "starbucks_flag_count",
            ],
        ),
        "stores": to_records(
            store_df,
            [
                "store_id",
                "store_name",
                "brand",
                "city",
                "zip5",
                "primary_beverage_category",
                "category_tags",
                "auxiliary_tags",
                "category_assignment_method",
                "hero_products",
                "positioning",
                "business_model",
                "origin_region",
                "license_start_year",
                "google_rating",
                "google_user_ratings_total",
                "google_price_level",
                "review_count",
                "source_count",
                "avg_rating",
                "date_min",
                "date_max",
                "matched_google_url",
                "website_url",
                "latitude",
                "longitude",
                "coord_source",
                "coord_status",
                "logo_url",
                "logo_source",
                "logo_fallback",
                "rank",
                "opportunity_score",
                "demand_score",
                "supply_score",
                "gap_score",
                "gap_component_score",
                "momentum_score",
                "text_signal_score",
                "feasibility_score",
                "regional_cost_score",
                "regional_feasibility_score",
                "store_cost_score",
                "store_affordability_score",
                "store_cost_coverage",
                "quadrant",
                "recommended_strategy",
                "action_note",
            ],
        ),
        "reviews": to_records(
            review_df,
            [
                "store_id",
                "store_name",
                "brand",
                "city",
                "zip5",
                "review_rank",
                "review_date",
                "rating",
                "source_platform",
                "reviewer_alias",
                "source_url",
                "review_text",
                "review_text_short",
            ],
        ),
        "brand": to_records(
            brand_df,
            [
                "brand",
                "store_count",
                "city_count",
                "zip_count",
                "primary_beverage_category",
                "business_model",
                "avg_google_rating",
                "total_google_ratings",
                "total_local_review_rows",
                "logo_fallback",
            ],
        ),
        "license": to_records(
            license_df,
            ["zip5", "year", "shops_started", "rolling3_started"],
        ),
    }
    return payload


def dashboard_template() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>King County New Tea Intelligence Dashboard</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap\" rel=\"stylesheet\">
  <script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
  <style>
    :root {
      --bg: #eef1f5;
      --shell: #f8f5ef;
      --panel: #ffffff;
      --line: #e8dfd3;
      --ink: #101828;
      --muted: #667085;
      --accent: #1f3d5a;
      --mint: #eaf6ef;
      --sand: #f7f0dc;
      --peach: #fceee6;
      --rose: #f6eaee;
      --r-shell: 18px;
      --r-panel: 12px;
      --r-card: 10px;
      --r-soft: 8px;
      --r-pill: 999px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: \"Manrope\", \"Segoe UI\", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 12%, rgba(250, 217, 180, 0.30), transparent 36%),
        radial-gradient(circle at 92% 88%, rgba(174, 214, 241, 0.28), transparent 42%),
        var(--bg);
    }
    .shell {
      width: min(1520px, 97vw);
      margin: 12px auto;
      padding: 0;
      border-radius: 0;
      border: 0;
      background: transparent;
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--r-panel);
      padding: 12px;
      min-width: 0;
    }

    .sidebar {
      position: sticky;
      top: 12px;
      max-height: calc(100vh - 24px);
      overflow: auto;
      display: grid;
      gap: 10px;
    }
    .side-block { border-top: 1px dashed #eadfce; padding-top: 8px; }
    .side-block:first-child { border-top: 0; padding-top: 0; }
    .block-title {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      color: #6b7280;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .ctl { display: grid; gap: 4px; margin-bottom: 6px; }
    .ctl label {
      font-size: 10px;
      color: #667085;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    select, input, button { font: inherit; }
    .input {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      padding: 0 10px;
      background: #fff;
      font-size: 13px;
      color: #1f2937;
    }
    .multi-select { position: relative; }
    .multi-select-btn {
      width: 100%;
      text-align: left;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      cursor: pointer;
    }
    .multi-select-btn .summary {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .multi-select-btn .caret {
      color: #64748b;
      font-size: 12px;
      line-height: 1;
      transition: transform 0.16s ease;
      flex: 0 0 auto;
    }
    .multi-select.open .multi-select-btn .caret {
      transform: rotate(180deg);
    }
    .multi-menu {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      background: #fff;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
      padding: 5px;
      max-height: 220px;
      overflow: auto;
      z-index: 40;
    }
    .multi-actions {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      border-bottom: 1px dashed #eadfce;
      padding: 2px 2px 6px 2px;
      margin-bottom: 4px;
    }
    .multi-mini-btn {
      border: 0;
      background: transparent;
      color: #1f3d5a;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      padding: 2px 4px;
      border-radius: var(--r-soft);
    }
    .multi-mini-btn:hover { background: #f3f7fc; }
    .multi-option {
      width: 100%;
      border: 0;
      background: #fff;
      border-radius: var(--r-soft);
      padding: 6px 8px;
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      color: #1f2937;
      font-size: 13px;
      text-align: left;
    }
    .multi-option:hover { background: #f8f5ef; }
    .multi-option.active {
      background: #eef5fc;
      color: #1f3d5a;
      font-weight: 700;
    }
    .multi-checkbox {
      width: 15px;
      height: 15px;
      margin: 0;
      accent-color: #1f3d5a;
      pointer-events: none;
      flex: 0 0 auto;
    }
    .hidden { display: none !important; }
    .btn {
      height: 36px;
      border-radius: var(--r-card);
      border: 1px solid #101827;
      background: #101827;
      color: #fff;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      padding: 0 12px;
    }

    .scenario-stack { display: grid; gap: 6px; }
    .scenario {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      padding: 6px 10px;
      background: #fff;
      color: #344054;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .scenario.active {
      border-color: #1f3d5a;
      box-shadow: 0 0 0 2px rgba(31,61,90,0.14);
      background: #f4f8fc;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }
    .kpi {
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      padding: 7px 8px;
      background: #edf5ff;
    }
    .kpi-l { font-size: 11px; color: #4b5563; font-weight: 600; }
    .kpi-v { margin-top: 2px; font-size: 28px; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }

    .main {
      display: grid;
      gap: 12px;
      min-width: 0;
    }
    .hero {
      background: linear-gradient(135deg, #ffffff 0%, #f6efe3 62%, #eef5fc 100%);
    }
    .hero-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
    }
    .hero-main { min-width: 0; }
    .hero-logo-wrap {
      width: 250px;
      height: 86px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      align-self: center;
    }
    .hero-logo-wrap img {
      width: auto;
      height: 100%;
      max-width: 100%;
      object-fit: contain;
      display: block;
    }
    h1 {
      margin: 0;
      font-size: clamp(30px, 3.3vw, 56px);
      line-height: 1.04;
      letter-spacing: -0.02em;
      font-weight: 700;
    }
    .subtitle {
      margin: 8px 0 0 0;
      color: var(--muted);
      font-size: 14px;
      max-width: 900px;
      line-height: 1.45;
      font-weight: 500;
      white-space: nowrap;
    }
    .team-note {
      margin: 10px 0 0 0;
      color: #7a8599;
      font-size: 13px;
      line-height: 1.35;
      font-weight: 600;
    }
    .school-note {
      margin: 2px 0 0 0;
      color: #8a95a8;
      font-size: 12px;
      line-height: 1.35;
      font-weight: 500;
    }
    .side-meta {
      border: 0;
      border-radius: 0;
      padding: 0;
      font-size: 12px;
      color: #475467;
      background: transparent;
      line-height: 1.45;
      font-weight: 600;
    }
    .refs {
      margin-top: 6px;
      padding-top: 6px;
      border-top: 1px dashed #eadfce;
    }
    .refs-title {
      font-size: 12px;
      color: #475467;
      font-weight: 600;
      margin-bottom: 3px;
    }
    .ref-item {
      font-size: 12px;
      color: #667085;
      line-height: 1.4;
      font-weight: 600;
    }
    .ref-link {
      color: #667085;
      text-decoration: none;
      font-weight: 600;
    }
    .ref-link:hover { text-decoration: underline; color: #475467; }

    .chart-grid, .insight-grid, .evidence-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      align-items: stretch;
      min-width: 0;
    }
    .chart-grid > .panel,
    .insight-grid > .panel,
    .evidence-grid > .panel {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    .card-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .card-title h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .hint { color: var(--muted); font-size: 11px; font-weight: 600; }
    .plot { width: 100%; height: 300px; }
    .formula-note {
      margin-top: 6px;
      font-size: 11px;
      color: #667085;
      line-height: 1.4;
    }

    .zip-list {
      display: grid;
      gap: 6px;
      max-height: 210px;
      overflow: auto;
    }
    .zip-item {
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      padding: 7px 9px;
      background: #fff;
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      text-align: left;
    }
    .zip-item.active {
      border-color: #1f3d5a;
      background: #f5f9fd;
      box-shadow: 0 0 0 2px rgba(31,61,90,0.14);
    }
    .zip-r { font-size: 11px; color: #6b7280; font-weight: 700; }
    .zip-z {
      font-size: 13px;
      font-weight: 700;
      display: inline-flex;
      align-items: baseline;
      gap: 5px;
    }
    .zip-city {
      color: #98a2b3;
      font-weight: 500;
      font-size: 11px;
    }
    .zip-s { font-size: 12px; font-weight: 700; color: #1f3d5a; }

    .zip-insight {
      border: 0;
      border-radius: 0;
      padding: 0;
      background: transparent;
      height: 100%;
      display: flex;
      flex-direction: column;
    }
    .zip-head { font-size: 15px; font-weight: 700; }
    .zip-meta { margin-top: 3px; font-size: 11px; color: #667085; font-weight: 600; }
    .chips { margin-top: 7px; display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      font-size: 11px;
      font-weight: 600;
      background: #f2ebdf;
      border-radius: var(--r-pill);
      padding: 3px 8px;
      color: #344054;
    }
    .bars { margin-top: 8px; display: grid; gap: 7px; }
    .bar-row { display: grid; grid-template-columns: 82px 1fr 34px; gap: 7px; align-items: center; }
    .bar-label { font-size: 11px; color: #667085; font-weight: 600; }
    .bar-track { height: 7px; border-radius: var(--r-pill); overflow: hidden; background: #efe9de; }
    .bar-fill { height: 100%; border-radius: var(--r-pill); background: #1f3d5a; }
    .bar-value { font-size: 11px; font-weight: 700; color: #344054; text-align: right; }
    .zip-formula-title,
    .zip-action-title {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px dashed #e5ddcf;
      font-size: 10.5px;
      color: #667085;
      line-height: 1.35;
      font-weight: 800;
      letter-spacing: 0.01em;
    }
    .zip-action {
      margin-top: 4px;
      border-top: 0;
      padding-top: 0;
      font-size: 11px;
      color: #4b5563;
      line-height: 1.42;
    }
    .zip-action-line {
      display: grid;
      grid-template-columns: 18px 1fr;
      column-gap: 6px;
      align-items: start;
    }
    .zip-action-line + .zip-action-line { margin-top: 4px; }
    .zip-action-idx {
      font-weight: 800;
      color: #667085;
      line-height: 1.42;
    }
    .zip-action-label {
      font-weight: 800;
      line-height: 1.42;
      color: #0f3460;
    }
    .zip-action-label.entry { color: #0f3460; }
    .zip-action-label.menu { color: #0f3460; }
    .zip-action-label.gtm { color: #0f3460; }
    .zip-action-label.ops { color: #0f3460; }
    .zip-action-label.risk { color: #0f3460; }
    .zip-action-text {
      color: #4b5563;
      line-height: 1.42;
    }
    .zip-formulas {
      margin-top: 4px;
      border-top: 0;
      padding-top: 0;
      display: grid;
      gap: 4px;
    }
    .zip-formula-line {
      display: grid;
      grid-template-columns: 18px 1fr;
      column-gap: 6px;
      align-items: start;
      font-size: 11px;
      color: #4b5563;
      line-height: 1.42;
    }
    .zip-formula-idx {
      font-weight: 800;
      color: #667085;
      line-height: 1.42;
    }
    .zip-formula-text {
      min-width: 0;
      line-height: 1.42;
    }
    .zip-formula-label {
      font-weight: 800;
      color: #0f3460;
      line-height: 1.42;
    }

    .mini-tabs {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .tab-btn {
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      padding: 5px 9px;
      background: #fff;
      font-size: 11px;
      font-weight: 600;
      color: #4b5563;
      cursor: pointer;
    }
    .tab-btn.active {
      background: #eaf1f8;
      border-color: #1f3d5a;
      color: #1f3d5a;
      font-weight: 700;
    }

    .store-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .store-header h3 { margin: 0; font-size: 18px; font-weight: 700; }
    .store-sort {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
    }
    .store-sort-btn {
      width: 28px;
      height: 28px;
      border-radius: var(--r-pill);
      border: 1px solid var(--line);
      background: #ffffff;
      color: #6b7280;
      display: inline-grid;
      place-items: center;
      padding: 0;
      line-height: 0;
      cursor: pointer;
      transition: background-color .15s ease, border-color .15s ease, color .15s ease;
    }
    .store-sort-btn svg {
      width: 15px;
      height: 15px;
      display: block;
    }
    .store-sort-btn .i-stroke { stroke: currentColor; stroke-width: 1.35; fill: none; stroke-linecap: round; stroke-linejoin: round; }
    .store-sort-btn .i-fill { fill: currentColor; }
    .store-sort-btn[data-state="off"] {
      color: #94a3b8;
      border-color: #d6dde6;
      background: #ffffff;
    }
    .store-sort-btn[data-state="asc"] {
      color: #1f3d5a;
      border-color: #1f3d5a;
      background: #eef4fb;
    }
    .store-sort-btn[data-state="desc"] {
      color: #7e1f24;
      border-color: #b4535a;
      background: #fff1f2;
    }
    .store-sort-btn:hover {
      border-color: #b7c5d7;
      background: #f8fbff;
    }
    .tiny { font-size: 11px; color: #667085; font-weight: 600; }
    .store-grid {
      flex: 1;
      min-height: 0;
      overflow: auto;
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fill, minmax(245px, 1fr));
      grid-auto-rows: 112px;
      align-content: start;
      padding-right: 2px;
    }
    .store-pager {
      margin-top: 8px;
      padding-top: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
    .store-pager-actions {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .store-page-btn {
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      background: #fff;
      color: #374151;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      cursor: pointer;
    }
    .store-page-btn:disabled {
      opacity: 0.45;
      cursor: default;
    }
    .store-page-text {
      font-size: 11px;
      color: #667085;
      font-weight: 600;
      order: -1;
    }
    .store-item {
      width: 100%;
      height: 112px;
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      background: #fff;
      padding: 8px;
      display: grid;
      grid-template-columns: 42px 1fr;
      gap: 9px;
      text-align: left;
      cursor: pointer;
      align-self: start;
      align-content: start;
      overflow: hidden;
    }
    .store-item.active {
      border-color: #1f3d5a;
      box-shadow: 0 0 0 2px rgba(31,61,90,0.14);
      background: #f7fbff;
    }
    .logo {
      width: 42px;
      height: 42px;
      border-radius: var(--r-card);
      object-fit: cover;
      object-position: center center;
      display: block;
      border: 1px solid var(--line);
      background: #f2f4f7;
    }
    .store-name { margin-top: 3px; font-size: 13px; font-weight: 700; line-height: 1.28; }
    .store-meta { margin-top: 2px; font-size: 11px; color: #6b7280; line-height: 1.35; font-weight: 600; }
    .star-row {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      flex-wrap: nowrap;
      vertical-align: middle;
    }
    .star-track {
      position: relative;
      display: inline-block;
      color: #cdd5df;
      font-size: 12px;
      letter-spacing: 0.08em;
      line-height: 1;
      white-space: nowrap;
      font-weight: 700;
    }
    .star-fill {
      position: absolute;
      top: 0;
      left: 0;
      overflow: hidden;
      color: #f59e0b;
      white-space: nowrap;
      font-size: inherit;
      letter-spacing: inherit;
      line-height: 1;
      font-weight: inherit;
    }
    .star-num {
      color: #344054;
      font-size: 11px;
      font-weight: 700;
    }
    .star-count {
      color: #667085;
      font-size: 11px;
      font-weight: 600;
    }
    .star-na {
      color: #98a2b3;
      font-size: 11px;
      font-weight: 600;
    }
    .focus-rating {
      margin-top: 5px;
      display: flex;
      align-items: center;
      gap: 6px;
      min-height: 18px;
    }
    .focus-rating .star-track { font-size: 13px; }
    .focus-rating .star-num, .focus-rating .star-count, .focus-rating .star-na {
      font-size: 11px;
    }

    .focus-head {
      display: grid;
      grid-template-columns: 56px 1fr;
      gap: 10px;
      align-items: start;
      margin-bottom: 8px;
    }
    .focus-logo {
      width: 56px;
      height: 56px;
      border-radius: var(--r-card);
      border: 1px solid var(--line);
      background: #f2f4f7;
      object-fit: cover;
      object-position: center center;
      display: block;
    }
    .focus-name-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-width: 0;
    }
    .focus-name { font-size: 15px; font-weight: 700; line-height: 1.28; }
    .focus-link-icon {
      width: 24px;
      height: 24px;
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      background: #ffffff;
      color: #1f3d5a;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      flex-shrink: 0;
      transition: background-color .15s ease, border-color .15s ease;
    }
    .focus-link-icon:hover { background: #f7fbff; border-color: #b7c5d7; }
    .focus-link-icon svg {
      width: 13px;
      height: 13px;
      stroke: currentColor;
      stroke-width: 2;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .focus-link-icon.hidden { display: none; }
    .focus-meta { margin-top: 3px; font-size: 11px; color: #6b7280; line-height: 1.38; font-weight: 600; }

    .review-box {
      border: 1px solid var(--line);
      border-radius: var(--r-card);
      background: #fffdfa;
      padding: 10px;
      min-height: 145px;
      max-height: 235px;
      overflow: auto;
    }
    .review-text {
      font-size: 12px;
      line-height: 1.55;
      color: #1f2937;
      white-space: pre-wrap;
      font-weight: 500;
    }
    .review-meta {
      margin-top: 8px;
      font-size: 11px;
      color: #6b7280;
      line-height: 1.45;
      font-weight: 600;
    }
    .review-controls {
      margin-top: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .pill-btn {
      border: 1px solid var(--line);
      border-radius: var(--r-pill);
      background: #fff;
      color: #374151;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      cursor: pointer;
    }
    .review-tone {
      display: inline-block;
      border-radius: var(--r-pill);
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .tone-positive { background: #d8f2e1; color: #166534; }
    .tone-neutral { background: #e9edf3; color: #374151; }
    .tone-negative { background: #fde2e2; color: #991b1b; }
    .kw-wrap { margin-top: 8px; font-size: 11px; color: #6b7280; line-height: 1.4; font-weight: 600; display: grid; gap: 6px; }
    .kw-row {
      display: block;
      min-width: 0;
      line-height: 1.8;
    }
    .kw-label {
      color: #64748b;
      font-weight: 800;
      min-width: 0;
      letter-spacing: 0.01em;
      text-align: left;
      line-height: 1.8;
      display: inline;
      margin-right: 2px;
    }
    .kw-values {
      display: inline;
      min-width: 0;
      line-height: 1.8;
    }
    .kw-token {
      display: inline-block;
      margin: 0 4px 4px 0;
      border-radius: var(--r-pill);
      padding: 2px 8px;
      font-weight: 700;
    }
    .kw-token-signal { background: #f2ebdf; color: #4b5563; }
    .kw-token-positive { background: #d8f2e1; color: #166534; }
    .kw-token-negative { background: #fde2e2; color: #991b1b; }
    .kw-none {
      color: #98a2b3;
      font-weight: 600;
      letter-spacing: 0.01em;
      line-height: 1.8;
      display: inline;
    }
    .kw-highlight {
      background: #fff4a8;
      border-radius: var(--r-soft);
      padding: 0 2px;
      color: #4a3f1f;
      font-weight: 700;
    }
    .kw-highlight-positive {
      background: #d8f2e1;
      border-radius: var(--r-soft);
      padding: 0 2px;
      color: #166534;
      font-weight: 700;
    }
    .kw-highlight-negative {
      background: #fde2e2;
      border-radius: var(--r-soft);
      padding: 0 2px;
      color: #991b1b;
      font-weight: 700;
    }
    .timeline {
      margin-top: 8px;
      border: 0;
      border-radius: 0;
      flex: 1;
      min-height: 0;
      overflow: auto;
      background: transparent;
      padding: 0;
      display: grid;
      gap: 6px;
    }
    .context-panel #miniChart {
      flex: 1;
      min-height: 420px;
      height: 100%;
    }
    #miniChart.logo-wall {
      min-height: 420px;
      height: 100%;
      overflow: auto;
      padding-right: 2px;
    }
    .logo-wall-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
      gap: 12px;
      align-content: start;
      padding: 4px;
    }
    .logo-node {
      display: grid;
      justify-items: center;
      align-content: start;
      gap: 6px;
    }
    .logo-circle {
      width: 64px;
      height: 64px;
      border-radius: var(--r-pill);
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
    }
    .logo-circle img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .logo-name {
      max-width: 84px;
      font-size: 11px;
      line-height: 1.2;
      color: #64748b;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .store-panel, .review-panel {
      height: 620px;
      min-height: 620px;
    }
    .map-panel {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .map-wrap {
      position: relative;
    }
    .map-plot {
      width: 100%;
      height: 430px;
      border: 0;
      border-radius: 0;
      overflow: hidden;
      background: #fff;
    }
    .map-toolbox {
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: flex;
      gap: 8px;
      z-index: 8;
      align-items: center;
    }
    .map-tool-btn {
      width: 34px;
      height: 34px;
      border: 1px solid rgba(31, 41, 55, 0.22);
      background: rgba(255, 255, 255, 0.95);
      border-radius: var(--r-card);
      color: #374151;
      font-size: 16px;
      font-weight: 700;
      line-height: 0;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 14px rgba(16, 24, 40, 0.10);
    }
    .map-tool-btn svg {
      width: 16px;
      height: 16px;
      display: block;
      stroke: currentColor;
      fill: none;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .map-tool-btn:hover {
      border-color: #7e1f24;
      color: #7e1f24;
    }
    .timeline-item {
      border: 1px solid #ece3d7;
      border-radius: var(--r-card);
      background: #fffdfa;
      padding: 7px;
      text-align: left;
      cursor: pointer;
    }
    .timeline-item.active {
      border-color: #1f3d5a;
      box-shadow: 0 0 0 2px rgba(31,61,90,0.14);
      background: #f7fbff;
    }
    .timeline-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 11px;
      color: #6b7280;
      font-weight: 600;
    }
    .timeline-snippet {
      margin-top: 4px;
      font-size: 11px;
      color: #374151;
      line-height: 1.4;
      font-weight: 500;
    }

    .empty-plot {
      height: 100%;
      display: grid;
      place-items: center;
      color: #6b7280;
      font-size: 12px;
      border: 1px dashed var(--line);
      border-radius: var(--r-card);
      background: #fcfbf9;
      font-weight: 600;
    }
    .link {
      color: #1f3d5a;
      font-weight: 700;
      text-decoration: none;
    }

    @media (max-width: 1180px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
      .chart-grid, .insight-grid, .evidence-grid { grid-template-columns: 1fr; }
      .kpi-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .chart-grid > .panel, .insight-grid > .panel, .evidence-grid > .panel { height: auto; }
      .store-panel, .review-panel { height: auto; min-height: 0; }
      .store-grid { min-height: 300px; }
      .timeline { min-height: 160px; max-height: 260px; }
      .context-panel #miniChart { min-height: 300px; height: 300px; }
      .map-plot { height: 340px; }
    }
    @media (max-width: 760px) {
      .shell { width: 98vw; margin: 8px auto; padding: 0; }
      .panel { border-radius: var(--r-panel); padding: 10px; }
      .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .store-grid { grid-template-columns: 1fr; min-height: 260px; grid-auto-rows: 104px; }
      .store-pager { gap: 6px; }
      .store-page-text { order: 0; width: 100%; }
      .store-item { height: 104px; }
      .plot { height: 250px; }
      .timeline { max-height: 220px; }
      .context-panel #miniChart { min-height: 260px; height: 260px; }
      .map-plot { height: 280px; }
      .map-toolbox { left: 10px; bottom: 10px; gap: 6px; }
      .map-tool-btn { width: 31px; height: 31px; font-size: 15px; border-radius: var(--r-card); }
      h1 { font-size: 44px; }
      .subtitle { white-space: normal; }
      .hero-top { grid-template-columns: 1fr; }
      .hero-logo-wrap { width: 120px; height: 44px; }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <aside class=\"panel sidebar\">
      <div class=\"side-block\">
        <div class=\"block-title\">Filters</div>
        <label class=\"ctl\"><label>City</label><select id=\"cityFilter\" class=\"input\"></select></label>
        <label class=\"ctl\">
          <label>ZIP Code</label>
          <div id=\"zipFilterWrap\" class=\"multi-select\">
            <button id=\"zipFilterBtn\" class=\"input multi-select-btn\" type=\"button\" aria-haspopup=\"listbox\" aria-expanded=\"false\">
              <span id=\"zipFilterText\" class=\"summary\">All ZIP Codes</span>
              <span class=\"caret\">▾</span>
            </button>
            <div id=\"zipFilterMenu\" class=\"multi-menu hidden\" role=\"listbox\" aria-multiselectable=\"true\"></div>
          </div>
        </label>
        <label class=\"ctl\">
          <label>Category</label>
          <div id=\"categoryFilterWrap\" class=\"multi-select\">
            <button id=\"categoryFilterBtn\" class=\"input multi-select-btn\" type=\"button\" aria-haspopup=\"listbox\" aria-expanded=\"false\">
              <span id=\"categoryFilterText\" class=\"summary\">All Categories</span>
              <span class=\"caret\">▾</span>
            </button>
            <div id=\"categoryFilterMenu\" class=\"multi-menu hidden\" role=\"listbox\" aria-multiselectable=\"true\"></div>
          </div>
        </label>
        <label class=\"ctl\"><label>Brand</label><select id=\"brandFilter\" class=\"input\"></select></label>
        <label class=\"ctl\"><label>Store Search</label><input id=\"storeSearch\" class=\"input\" type=\"text\" placeholder=\"Brand or store name\" /></label>
        <button id=\"resetFocusBtn\" class=\"btn\">Reset All Filters</button>
      </div>

      <div class=\"side-block\">
        <div class=\"block-title\">Scenarios</div>
        <div class=\"scenario-stack\" id=\"scenarioRow\">
          <button class=\"scenario\" data-scenario=\"top\">Scenario: Top Opportunity</button>
          <button class=\"scenario\" data-scenario=\"fruit\">Scenario: Fruit Tea Expansion</button>
          <button class=\"scenario\" data-scenario=\"gap\">Scenario: High Demand Low Supply</button>
        </div>
      </div>

      <div class=\"side-block\">
        <div class=\"block-title\">Quick KPIs</div>
        <div class=\"kpi-grid\">
          <div class=\"kpi\"><div class=\"kpi-l\">Visible Stores</div><div class=\"kpi-v\" id=\"kpiStores\">-</div></div>
          <div class=\"kpi\"><div class=\"kpi-l\">Visible Brands</div><div class=\"kpi-v\" id=\"kpiBrands\">-</div></div>
          <div class=\"kpi\"><div class=\"kpi-l\">Focused ZIP</div><div class=\"kpi-v\" id=\"kpiZip\">-</div></div>
          <div class=\"kpi\"><div class=\"kpi-l\">Avg Google Rating</div><div class=\"kpi-v\" id=\"kpiRating\">-</div></div>
        </div>
      </div>

      <div class=\"side-block\">
        <div class=\"side-meta\">
          <div><strong>Scope:</strong> <span id=\"scopeText\">King County, WA</span></div>
          <div><strong>Generated:</strong> <span id=\"generatedAt\">-</span></div>
          <div><strong>Rows:</strong> <span id=\"metaRows\">-</span></div>
          <div class=\"refs\">
            <div class=\"refs-title\">References</div>
            <div class=\"ref-item\">1. <a class=\"ref-link\" href=\"https://data.kingcounty.gov/resource/f29f-zza5.csv\" target=\"_blank\" rel=\"noopener noreferrer\">King County Open Data (Food Establishment Inspections)</a></div>
            <div class=\"ref-item\">2. <a class=\"ref-link\" href=\"https://trends.google.com/\" target=\"_blank\" rel=\"noopener noreferrer\">Google Trends</a></div>
            <div class=\"ref-item\">3. <a class=\"ref-link\" href=\"https://api.census.gov/data/2023/acs/acs5\" target=\"_blank\" rel=\"noopener noreferrer\">U.S. Census ACS 5-Year API</a></div>
            <div class=\"ref-item\">4. <a class=\"ref-link\" href=\"https://developers.google.com/maps/documentation/places/web-service/overview\" target=\"_blank\" rel=\"noopener noreferrer\">Google Maps Platform Places API</a></div>
            <div class=\"ref-item\">5. <a class=\"ref-link\" href=\"https://business.yelp.com/data/resources/open-dataset/\" target=\"_blank\" rel=\"noopener noreferrer\">Yelp Open Dataset</a></div>
          </div>
        </div>
      </div>
    </aside>

    <main class=\"main\">
      <section class=\"panel hero\">
        <div class=\"hero-top\">
          <div class=\"hero-main\">
            <h1>King County New Style Tea</h1>
            <p class=\"team-note\">Gold Cohort Group 2: Flala, Daksha, Alisha, Yosup</p>
            <p class=\"school-note\">Foster School of Business, University of Washington</p>
          </div>
          <div class=\"hero-logo-wrap\" aria-label=\"King County logo\">
            <img
              alt=\"King County logo\"
              src=\"__HERO_LOGO_URI__\"
            />
          </div>
        </div>
      </section>

      <section class=\"chart-grid\">
        <article class=\"panel\">
          <div class=\"card-title\">
            <h3>Opportunity Driver Matrix (Top 12 ZIPs)</h3>
          </div>
          <div id=\"oppHeatmap\" class=\"plot\"></div>
          <div id=\"oppFormulaNote\" class=\"formula-note\"></div>
        </article>

        <article class=\"panel\">
          <div class=\"card-title\">
            <h3>Demand vs Supply</h3>
            <span class=\"hint\">Bubble Size = Store Count</span>
          </div>
          <div id=\"gapScatter\" class=\"plot\"></div>
          <div id=\"gapMedianNote\" class=\"formula-note\"></div>
        </article>
      </section>

      <section class=\"insight-grid\">
        <article class=\"panel\">
          <div class=\"card-title\"><h3>Focused ZIP Insight</h3></div>
          <div class=\"zip-insight\">
            <div id=\"zipHead\" class=\"zip-head\">-</div>
            <div id=\"zipMeta\" class=\"zip-meta\">-</div>
            <div id=\"zipChips\" class=\"chips\"></div>
            <div id=\"zipBars\" class=\"bars\"></div>
            <div id=\"zipFormulaTitle\" class=\"zip-formula-title\">Score Formula Breakdown</div>
            <div id=\"zipFormulaLines\" class=\"zip-formulas\"></div>
            <div id=\"zipActionTitle\" class=\"zip-action-title\">Actionable ZIP Strategy</div>
            <div id=\"zipAction\" class=\"zip-action\">Select ZIP for recommendation.</div>
          </div>
        </article>

        <article class=\"panel context-panel\">
          <div class=\"card-title\"><h3>Context View</h3></div>
          <div class=\"mini-tabs\" id=\"miniTabs\">
            <button class=\"tab-btn active\" data-tab=\"brand\">Brand Mix</button>
            <button class=\"tab-btn\" data-tab=\"city\">City x Category</button>
            <button class=\"tab-btn\" data-tab=\"license\">License Trend</button>
            <button class=\"tab-btn\" data-tab=\"logos\">Brand Logos</button>
          </div>
          <div id=\"miniChart\" class=\"plot\"></div>
        </article>
      </section>

      <section class=\"panel map-panel\">
        <div class=\"card-title\"><h3>Store Geo Explorer</h3><span class=\"hint\" id=\"mapSummary\">-</span></div>
        <div class=\"map-wrap\">
          <div id=\"storeMap\" class=\"map-plot\"></div>
          <div class=\"map-toolbox\">
            <button id=\"mapLocateBtn\" class=\"map-tool-btn\" title=\"Go to my area\" aria-label=\"Go to my area\">
              <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><circle cx=\"12\" cy=\"12\" r=\"5\"></circle><line x1=\"12\" y1=\"2\" x2=\"12\" y2=\"6\"></line><line x1=\"12\" y1=\"18\" x2=\"12\" y2=\"22\"></line><line x1=\"2\" y1=\"12\" x2=\"6\" y2=\"12\"></line><line x1=\"18\" y1=\"12\" x2=\"22\" y2=\"12\"></line></svg>
            </button>
            <button id=\"mapCountyBtn\" class=\"map-tool-btn\" title=\"Reset to King County\" aria-label=\"Reset to King County\">
              <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><rect x=\"4\" y=\"4\" width=\"16\" height=\"16\" rx=\"2\"></rect><path d=\"M9 4v16M15 4v16M4 9h16M4 15h16\"></path></svg>
            </button>
            <button id=\"mapZoomInBtn\" class=\"map-tool-btn\" title=\"Zoom in\" aria-label=\"Zoom in\">
              <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><line x1=\"12\" y1=\"5\" x2=\"12\" y2=\"19\"></line><line x1=\"5\" y1=\"12\" x2=\"19\" y2=\"12\"></line></svg>
            </button>
            <button id=\"mapZoomOutBtn\" class=\"map-tool-btn\" title=\"Zoom out\" aria-label=\"Zoom out\">
              <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><line x1=\"5\" y1=\"12\" x2=\"19\" y2=\"12\"></line></svg>
            </button>
          </div>
        </div>
      </section>

      <section class=\"evidence-grid\">
        <article class=\"panel store-panel\">
          <div class=\"store-header\">
            <h3>Brand -> Store Explorer</h3>
            <div id=\"storeSortRow\" class=\"store-sort\" aria-label=\"Store sorting controls\">
              <button id=\"sortNameBtn\" class=\"store-sort-btn\" data-sort-key=\"name\" data-state=\"off\" title=\"Sort by store name\" aria-label=\"Sort by store name\"></button>
              <button id=\"sortRatingBtn\" class=\"store-sort-btn\" data-sort-key=\"rating\" data-state=\"off\" title=\"Sort by rating\" aria-label=\"Sort by rating\"></button>
              <button id=\"sortLogoBtn\" class=\"store-sort-btn\" data-sort-key=\"logo\" data-state=\"off\" title=\"Sort by logo availability\" aria-label=\"Sort by logo availability\"></button>
            </div>
          </div>
          <div id=\"storeGrid\" class=\"store-grid\"></div>
          <div class=\"store-pager\">
            <span id=\"storePageText\" class=\"store-page-text\">0 stores · page 1/1</span>
            <div class=\"store-pager-actions\">
              <button id=\"storePrevBtn\" class=\"store-page-btn\">Prev</button>
              <button id=\"storeNextBtn\" class=\"store-page-btn\">Next</button>
            </div>
          </div>
        </article>

        <article class=\"panel review-panel\">
          <div class=\"card-title\"><h3>Real Review Evidence</h3></div>

          <div class=\"focus-head\">
            <img id=\"focusStoreLogo\" class=\"focus-logo\" alt=\"store logo\" />
            <div>
              <div class=\"focus-name-row\">
                <div class=\"focus-name\" id=\"focusStoreName\">-</div>
                <a
                  id=\"focusStoreLinkIcon\"
                  class=\"focus-link-icon hidden\"
                  href=\"#\"
                  target=\"_blank\"
                  rel=\"noopener noreferrer\"
                  title=\"Open Google profile\"
                  aria-label=\"Open Google profile\"
                >
                  <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\">
                    <path d=\"M14 5h5v5\"></path>
                    <path d=\"M10 14L19 5\"></path>
                    <path d=\"M19 13v6h-14v-14h6\"></path>
                  </svg>
                </a>
              </div>
              <div class=\"focus-meta\" id=\"focusStoreMeta\">Select one store card.</div>
              <div class=\"focus-rating\" id=\"focusStoreRatingVisual\"></div>
              <div class=\"chips\" id=\"focusStoreChips\"></div>
            </div>
          </div>

          <div class=\"review-box\">
            <div id=\"reviewText\" class=\"review-text\">Select store to inspect one review at a time.</div>
            <div id=\"reviewMeta\" class=\"review-meta\"></div>
          </div>

          <div class=\"review-controls\">
            <button id=\"reviewPrevBtn\" class=\"pill-btn\">Prev</button>
            <button id=\"reviewNextBtn\" class=\"pill-btn\">Next</button>
            <span id=\"reviewPager\" class=\"tiny\">0 / 0</span>
          </div>
          <div id=\"reviewKeywords\" class=\"kw-wrap\">
            <div class=\"kw-row\"><span class=\"kw-label\">Positive Cues:</span><span class=\"kw-values\"><span class=\"kw-none\">none</span></span></div>
            <div class=\"kw-row\"><span class=\"kw-label\">Risk Cues:</span><span class=\"kw-values\"><span class=\"kw-none\">none</span></span></div>
            <div class=\"kw-row\"><span class=\"kw-label\">Product & Service Signals:</span><span class=\"kw-values\"><span class=\"kw-none\">none</span></span></div>
          </div>
          <div id=\"reviewTimeline\" class=\"timeline\"></div>
        </article>
      </section>
    </main>
  </div>

  <script>
    const DATA = __DATA_JSON__;

    const zipData = (DATA.zip || []).map((d) => ({ ...d, zip5: String(d.zip5 || "") }));
    const storeData = (DATA.stores || []).map((d) => ({ ...d, zip5: String(d.zip5 || "") }));
    const reviewData = (DATA.reviews || []).map((d) => ({ ...d, zip5: String(d.zip5 || "") }));
    const licenseData = (DATA.license || []).map((d) => ({ ...d, zip5: String(d.zip5 || "") }));
    const mapLayers = DATA.map_layers || {};

    const zipCityMap = (() => {
      const counter = {};
      for (const s of storeData) {
        const zip = String(s.zip5 || "").trim();
        const city = String(s.city || "").trim();
        if (!zip || !city) continue;
        if (!counter[zip]) counter[zip] = {};
        counter[zip][city] = (counter[zip][city] || 0) + 1;
      }
      const out = {};
      for (const [zip, byCity] of Object.entries(counter)) {
        const pick = Object.entries(byCity).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];
        out[zip] = pick ? pick[0] : "";
      }
      return out;
    })();

    const HAS_PLOTLY = typeof Plotly !== "undefined";

    const reviewsByStore = {};
    for (const r of reviewData) {
      const sid = String(r.store_id || "");
      if (!sid) continue;
      if (!reviewsByStore[sid]) reviewsByStore[sid] = [];
      reviewsByStore[sid].push(r);
    }
    Object.keys(reviewsByStore).forEach((sid) => {
      reviewsByStore[sid].sort((a, b) => {
        const bt = Date.parse(String(b.review_date || "")) || 0;
        const at = Date.parse(String(a.review_date || "")) || 0;
        if (bt !== at) return bt - at;
        return asNum(a.review_rank, 999) - asNum(b.review_rank, 999);
      });
    });

    const POSITIVE_WORDS = [
      "good", "great", "love", "best", "amazing", "friendly", "clean", "fresh",
      "delicious", "recommend", "recommended", "nice", "awesome", "fast", "perfect",
      "smooth", "tasty", "flavorful", "authentic", "cozy", "quality"
    ];
    const NEGATIVE_WORDS = [
      "bad", "worst", "slow", "rude", "dirty", "expensive", "overpriced", "wait",
      "waiting", "disappoint", "disappointed", "bland", "issue", "problem", "cold",
      "wrong", "soggy", "not worth", "watery", "inconsistent", "stale", "hard",
      "old", "burnt", "rubbery", "sour", "mushy", "unfriendly"
    ];
    const NEGATION_TERMS = [
      "not", "no", "never", "hardly", "barely", "without",
      "isn't", "wasn't", "weren't", "don't", "didn't", "can't",
      "couldn't", "won't", "cannot"
    ];
    const POSITIVE_PHRASE_OVERRIDES = [
      "worth the wait",
      "worth waiting",
      "without losing quality",
      "without losing any quality",
      "without sacrificing quality",
      "without compromising quality",
      "without compromise on quality",
      "without compromise quality"
    ];
    const NEGATION_TOKEN_SET = new Set(NEGATION_TERMS.map((x) => x.replaceAll("'", "")));

    const state = {
      city: "ALL",
      zipFilters: [],
      categories: [],
      brand: "ALL",
      zip: "ALL",
      search: "",
      storePage: 1,
      storeId: null,
      reviewIndex: 0,
      miniTab: "brand",
      mapCenterLat: null,
      mapCenterLon: null,
      mapZoom: null,
      scenario: null,
      storePageManual: false,
      storeSortKey: "",
      storeSortDir: "off",
    };

    const el = {
      cityFilter: document.getElementById("cityFilter"),
      zipFilterWrap: document.getElementById("zipFilterWrap"),
      zipFilterBtn: document.getElementById("zipFilterBtn"),
      zipFilterMenu: document.getElementById("zipFilterMenu"),
      zipFilterText: document.getElementById("zipFilterText"),
      categoryFilterWrap: document.getElementById("categoryFilterWrap"),
      categoryFilterBtn: document.getElementById("categoryFilterBtn"),
      categoryFilterMenu: document.getElementById("categoryFilterMenu"),
      categoryFilterText: document.getElementById("categoryFilterText"),
      brandFilter: document.getElementById("brandFilter"),
      storeSearch: document.getElementById("storeSearch"),
      resetFocusBtn: document.getElementById("resetFocusBtn"),
      scenarioRow: document.getElementById("scenarioRow"),
      miniTabs: document.getElementById("miniTabs"),
      mapSummary: document.getElementById("mapSummary"),
      mapLocateBtn: document.getElementById("mapLocateBtn"),
      mapCountyBtn: document.getElementById("mapCountyBtn"),
      mapZoomInBtn: document.getElementById("mapZoomInBtn"),
      mapZoomOutBtn: document.getElementById("mapZoomOutBtn"),
      generatedAt: document.getElementById("generatedAt"),
      scopeText: document.getElementById("scopeText"),
      metaRows: document.getElementById("metaRows"),
      kpiStores: document.getElementById("kpiStores"),
      kpiBrands: document.getElementById("kpiBrands"),
      kpiZip: document.getElementById("kpiZip"),
      kpiRating: document.getElementById("kpiRating"),
      zipHead: document.getElementById("zipHead"),
      zipMeta: document.getElementById("zipMeta"),
      zipChips: document.getElementById("zipChips"),
      zipBars: document.getElementById("zipBars"),
      zipFormulaLines: document.getElementById("zipFormulaLines"),
      zipAction: document.getElementById("zipAction"),
      storeGrid: document.getElementById("storeGrid"),
      storePrevBtn: document.getElementById("storePrevBtn"),
      storeNextBtn: document.getElementById("storeNextBtn"),
      storePageText: document.getElementById("storePageText"),
      storeSortRow: document.getElementById("storeSortRow"),
      sortNameBtn: document.getElementById("sortNameBtn"),
      sortRatingBtn: document.getElementById("sortRatingBtn"),
      sortLogoBtn: document.getElementById("sortLogoBtn"),
      focusStoreLogo: document.getElementById("focusStoreLogo"),
      focusStoreName: document.getElementById("focusStoreName"),
      focusStoreMeta: document.getElementById("focusStoreMeta"),
      focusStoreRatingVisual: document.getElementById("focusStoreRatingVisual"),
      focusStoreChips: document.getElementById("focusStoreChips"),
      focusStoreLinkIcon: document.getElementById("focusStoreLinkIcon"),
      reviewText: document.getElementById("reviewText"),
      reviewMeta: document.getElementById("reviewMeta"),
      reviewPager: document.getElementById("reviewPager"),
      reviewPrevBtn: document.getElementById("reviewPrevBtn"),
      reviewNextBtn: document.getElementById("reviewNextBtn"),
      reviewKeywords: document.getElementById("reviewKeywords"),
      reviewTimeline: document.getElementById("reviewTimeline"),
      oppFormulaNote: document.getElementById("oppFormulaNote"),
      gapMedianNote: document.getElementById("gapMedianNote"),
    };

    function asNum(v, fallback = 0) {
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    }
    function asNumSafe(v, fallback = 0) {
      if (v === null || v === undefined) return fallback;
      const raw = String(v).trim();
      if (!raw) return fallback;
      const n = Number(raw);
      return Number.isFinite(n) ? n : fallback;
    }
    function medianOf(values, fallback = NaN) {
      const arr = (values || [])
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v))
        .sort((a, b) => a - b);
      if (!arr.length) return fallback;
      const mid = Math.floor(arr.length / 2);
      if (arr.length % 2 === 1) return arr[mid];
      return (arr[mid - 1] + arr[mid]) / 2;
    }
    function fmtInt(v) {
      return asNum(v).toLocaleString();
    }
    function fmt1(v) {
      return asNum(v).toFixed(1);
    }
    function esc(text) {
      return String(text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }
    function escRegex(text) {
      return String(text || "").replace(new RegExp("[.*+?^${}()|[\\\\]\\\\\\\\]", "g"), "\\\\$&");
    }
    function optionList(values, selected, allLabel, labelFormatter = null) {
      const fmtLabel = (value) => {
        if (typeof labelFormatter === "function") {
          const out = labelFormatter(value);
          if (out !== null && out !== undefined && String(out).trim() !== "") return String(out);
        }
        return String(value || "");
      };
      const head = `<option value=\"ALL\"${selected === "ALL" ? " selected" : ""}>${allLabel}</option>`;
      const rest = values
        .map((v) => `<option value=\"${esc(v)}\"${selected === v ? " selected" : ""}>${esc(fmtLabel(v))}</option>`)
        .join("");
      return head + rest;
    }
    function categorySummaryText(selected, allOptions) {
      const sel = (selected || []).filter(Boolean);
      if (!sel.length) return "All Categories";
      if (sel.length === 1) return sel[0];
      if (sel.length === allOptions.length) return `All Categories (${sel.length})`;
      return `${sel.length} categories selected`;
    }
    function zipSummaryText(selected, allOptions) {
      const sel = (selected || []).filter(Boolean);
      if (!sel.length) return "All ZIP Codes";
      if (sel.length === 1) return sel[0];
      if (sel.length === allOptions.length) return `All ZIP Codes (${sel.length})`;
      return `${sel.length} ZIP codes selected`;
    }
    function allZipOptions() {
      return [...new Set(zipData.map((z) => String(z.zip5 || "").trim()).filter(Boolean))]
        .sort((a, b) => asNum(a, 0) - asNum(b, 0) || a.localeCompare(b));
    }
    function isZipMenuOpen() {
      return el.zipFilterWrap?.classList.contains("open");
    }
    function closeZipMenu() {
      el.zipFilterWrap?.classList.remove("open");
      el.zipFilterBtn?.setAttribute("aria-expanded", "false");
      el.zipFilterMenu?.classList.add("hidden");
    }
    function openZipMenu() {
      el.zipFilterWrap?.classList.add("open");
      el.zipFilterBtn?.setAttribute("aria-expanded", "true");
      el.zipFilterMenu?.classList.remove("hidden");
    }
    function toggleZipMenu(forceOpen = null) {
      const open = isZipMenuOpen();
      const next = forceOpen === null ? !open : Boolean(forceOpen);
      if (next) openZipMenu();
      else closeZipMenu();
    }
    function isCategoryMenuOpen() {
      return el.categoryFilterWrap?.classList.contains("open");
    }
    function closeCategoryMenu() {
      el.categoryFilterWrap?.classList.remove("open");
      el.categoryFilterBtn?.setAttribute("aria-expanded", "false");
      el.categoryFilterMenu?.classList.add("hidden");
    }
    function openCategoryMenu() {
      el.categoryFilterWrap?.classList.add("open");
      el.categoryFilterBtn?.setAttribute("aria-expanded", "true");
      el.categoryFilterMenu?.classList.remove("hidden");
    }
    function toggleCategoryMenu(forceOpen = null) {
      const open = isCategoryMenuOpen();
      const next = forceOpen === null ? !open : Boolean(forceOpen);
      if (next) openCategoryMenu();
      else closeCategoryMenu();
    }
    function uniqSorted(rows, key) {
      return [...new Set(rows.map((r) => String(r[key] || "").trim()).filter(Boolean))].sort((a, b) => a.localeCompare(b));
    }
    function parseCategoryTags(raw) {
      return String(raw || "")
        .split("|")
        .map((x) => x.trim())
        .filter(Boolean);
    }
    function getStoreCategories(store) {
      const tags = parseCategoryTags(store?.category_tags);
      if (tags.length) return tags;
      const primary = String(store?.primary_beverage_category || "").trim();
      return primary ? [primary] : [];
    }
    function categoryDisplayText(store) {
      const tags = getStoreCategories(store);
      if (!tags.length) return "Unknown";
      return tags.join(" | ");
    }
    function allCategoryOptions(rows) {
      const set = new Set();
      rows.forEach((r) => {
        getStoreCategories(r).forEach((c) => set.add(String(c || "").trim()));
      });
      return [...set].filter(Boolean).sort((a, b) => a.localeCompare(b));
    }
    function getZipRow(zip5) {
      return zipData.find((z) => String(z.zip5) === String(zip5 || "")) || null;
    }
    function toTitleCaseCity(text) {
      return String(text || "")
        .trim()
        .replaceAll("_", " ")
        .toLowerCase()
        .replace(/(^|\\s)([a-z])/g, (_m, p1, p2) => p1 + p2.toUpperCase());
    }
    function getZipCity(zip5) {
      return toTitleCaseCity(zipCityMap[String(zip5 || "")] || "");
    }
    function getStoreLogo(store) {
      return String(store?.logo_url || store?.logo_fallback || "");
    }
    function isFallbackLogoUrl(url, fallback) {
      const u = String(url || "").trim().toLowerCase();
      const fb = String(fallback || "").trim().toLowerCase();
      if (!u) return true;
      if (fb && u === fb) return true;
      if (u.includes("ui-avatars.com/api")) return true;
      if (u.startsWith("data:image/svg+xml;base64,phn2zy")) return true;
      return false;
    }
    function hasRealStoreLogo(store) {
      const source = String(store?.logo_source || "").trim().toLowerCase();
      const logo = String(store?.logo_url || "").trim();
      const fallback = String(store?.logo_fallback || "").trim();
      if (!logo) return false;
      if (source === "ui_avatar_fallback") return false;
      if (isFallbackLogoUrl(logo, fallback)) return false;
      return true;
    }
    function hasCuratedBrandLogo(store) {
      const source = String(store?.logo_source || "").trim().toLowerCase();
      const logo = String(store?.logo_url || "").trim();
      if (!logo) return false;
      return source === "local_images_brand_logo";
    }
    function getSortStateForKey(key) {
      if (String(state.storeSortKey || "") !== String(key || "")) return "off";
      const dir = String(state.storeSortDir || "off");
      return dir === "asc" || dir === "desc" ? dir : "off";
    }
    function sortIconSvg(key, mode) {
      const arrowUp = `<path class=\"i-stroke\" d=\"M16 16V7\"></path><path class=\"i-stroke\" d=\"M14 9l2-2 2 2\"></path>`;
      const arrowDown = `<path class=\"i-stroke\" d=\"M16 7v9\"></path><path class=\"i-stroke\" d=\"M14 14l2 2 2-2\"></path>`;
      const arrowBoth = `${arrowUp}${arrowDown}`;
      const arrow = mode === "asc" ? arrowUp : mode === "desc" ? arrowDown : arrowBoth;
      if (key === "name") {
        return `
          <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\">
            <text x=\"5.2\" y=\"8.4\" font-size=\"6.1\" font-weight=\"700\" class=\"i-fill\" text-anchor=\"middle\" dominant-baseline=\"middle\">A</text>
            <text x=\"5.2\" y=\"16.6\" font-size=\"6.1\" font-weight=\"700\" class=\"i-fill\" text-anchor=\"middle\" dominant-baseline=\"middle\">Z</text>
            ${arrow}
          </svg>
        `.trim();
      }
      if (key === "rating") {
        return `
          <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\">
            <path class=\"i-stroke\" d=\"M8 5.2l1.3 2.7 3 .5-2.1 2.1.5 2.9-2.7-1.5-2.7 1.5.5-2.9-2.1-2.1 3-.5z\"></path>
            ${arrow}
          </svg>
        `.trim();
      }
      return `
        <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\">
          <rect x=\"3.8\" y=\"5.8\" width=\"8\" height=\"7.6\" rx=\"1.6\" class=\"i-stroke\"></rect>
          <circle cx=\"7.3\" cy=\"8.6\" r=\"0.9\" class=\"i-fill\"></circle>
          <path class=\"i-stroke\" d=\"M4.9 11.8l2.2-1.8 1.8 1.5 2.3-2\"></path>
          ${arrow}
        </svg>
      `.trim();
    }
    function renderStoreSortButtons() {
      const btns = [
        { node: el.sortNameBtn, key: "name" },
        { node: el.sortRatingBtn, key: "rating" },
        { node: el.sortLogoBtn, key: "logo" },
      ];
      btns.forEach(({ node, key }) => {
        if (!node) return;
        const mode = getSortStateForKey(key);
        node.dataset.state = mode;
        node.innerHTML = sortIconSvg(key, mode);
      });
    }
    function cycleStoreSort(key) {
      const curr = getSortStateForKey(key);
      const next = curr === "off" ? "asc" : curr === "asc" ? "desc" : "off";
      if (next === "off") {
        state.storeSortKey = "";
        state.storeSortDir = "off";
      } else {
        state.storeSortKey = key;
        state.storeSortDir = next;
      }
      state.storePage = 1;
      state.storePageManual = false;
      state.storeId = null;
      state.reviewIndex = 0;
      renderAll();
    }
    function starRatingHtml(ratingValue, ratingCount, options = {}) {
      const cls = String(options.cls || "").trim();
      const val = asNum(ratingValue, NaN);
      if (!Number.isFinite(val)) {
        return `<span class=\"star-row ${esc(cls)}\"><span class=\"star-track\">★★★★★</span><span class=\"star-na\">No Google rating</span></span>`;
      }
      const clipped = Math.max(0, Math.min(5, val));
      const pct = Math.max(0, Math.min(100, (clipped / 5) * 100));
      const countNum = asNum(ratingCount, 0);
      return `
        <span class=\"star-row ${esc(cls)}\">
          <span class=\"star-track\">★★★★★<span class=\"star-fill\" style=\"width:${pct}%;\">★★★★★</span></span>
          <span class=\"star-num\">${clipped.toFixed(1)}</span>
          <span class=\"star-count\">(${fmtInt(countNum)})</span>
        </span>
      `.trim();
    }
    function getStoreById(storeId) {
      return storeData.find((s) => String(s.store_id || "") === String(storeId || "")) || null;
    }
    function getStoresWithCoordinates(rows) {
      return (rows || []).filter((s) => {
        const lat = asNum(s.latitude, NaN);
        const lon = asNum(s.longitude, NaN);
        return Number.isFinite(lat) && Number.isFinite(lon);
      });
    }
    function estimateMapZoom(rows) {
      if (!rows.length) return 9.2;
      const lats = rows.map((s) => asNum(s.latitude));
      const lons = rows.map((s) => asNum(s.longitude));
      const latSpan = Math.max(...lats) - Math.min(...lats);
      const lonSpan = Math.max(...lons) - Math.min(...lons);
      const span = Math.max(latSpan, lonSpan);
      if (span < 0.03) return 12.2;
      if (span < 0.06) return 11.5;
      if (span < 0.12) return 10.8;
      if (span < 0.20) return 10.1;
      return 9.3;
    }
    function mapCenter(rows) {
      if (!rows.length) return { lat: 47.6062, lon: -122.3321 };
      const lats = rows.map((s) => asNum(s.latitude));
      const lons = rows.map((s) => asNum(s.longitude));
      return {
        lat: lats.reduce((a, b) => a + b, 0) / lats.length,
        lon: lons.reduce((a, b) => a + b, 0) / lons.length,
      };
    }
    function getStorePageSize() {
      if (window.innerWidth <= 760) return 6;
      if (window.innerWidth <= 1180) return 8;
      return 10;
    }
    function totalStorePages(totalRows) {
      const size = getStorePageSize();
      return Math.max(1, Math.ceil(asNum(totalRows, 0) / size));
    }
    function miniPlotHeight() {
      if (window.innerWidth <= 760) return 260;
      if (window.innerWidth <= 1180) return 300;
      return 420;
    }
    function boundaryGeoToLineCoords(geojson) {
      const lat = [];
      const lon = [];
      if (!geojson || !Array.isArray(geojson.features)) return { lat, lon };

      function pushRing(ring) {
        if (!Array.isArray(ring)) return;
        for (const pt of ring) {
          if (!Array.isArray(pt) || pt.length < 2) continue;
          const x = Number(pt[0]);
          const y = Number(pt[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          lon.push(x);
          lat.push(y);
        }
        lon.push(null);
        lat.push(null);
      }

      function walkGeom(geom) {
        if (!geom || !geom.type) return;
        const t = String(geom.type);
        const c = geom.coordinates;
        if (!Array.isArray(c)) return;
        if (t === "Polygon") {
          for (const ring of c) pushRing(ring);
        } else if (t === "MultiPolygon") {
          for (const poly of c) {
            if (!Array.isArray(poly)) continue;
            for (const ring of poly) pushRing(ring);
          }
        } else if (t === "LineString") {
          pushRing(c);
        } else if (t === "MultiLineString") {
          for (const ring of c) pushRing(ring);
        }
      }

      for (const ft of geojson.features) {
        if (!ft || typeof ft !== "object") continue;
        walkGeom(ft.geometry);
      }
      return { lat, lon };
    }
    function featureZipCode(ft) {
      const props = (ft && typeof ft === "object" && ft.properties && typeof ft.properties === "object")
        ? ft.properties
        : {};
      const keys = ["ZCTA5CE10", "ZCTA5CE20", "ZIPCODE", "zip", "zip5", "GEOID10", "GEOID20"];
      for (const key of keys) {
        const raw = String(props[key] ?? "").trim();
        const match = raw.match(/([0-9]{5})/);
        if (match) return match[1];
      }
      const fid = String((ft && ft.id) || "");
      const match = fid.match(/([0-9]{5})/);
      return match ? match[1] : "";
    }
    function ringToMapCoords(ring) {
      const lat = [];
      const lon = [];
      if (!Array.isArray(ring)) return { lat, lon };
      for (const pt of ring) {
        if (!Array.isArray(pt) || pt.length < 2) continue;
        const x = Number(pt[0]);
        const y = Number(pt[1]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        lon.push(x);
        lat.push(y);
      }
      if (lat.length > 2) {
        const firstLat = lat[0];
        const firstLon = lon[0];
        const lastLat = lat[lat.length - 1];
        const lastLon = lon[lon.length - 1];
        if (firstLat !== lastLat || firstLon !== lastLon) {
          lat.push(firstLat);
          lon.push(firstLon);
        }
      }
      return { lat, lon };
    }
    function getZipFocusFillTraces(zipInput) {
      const targets = Array.isArray(zipInput)
        ? zipInput.map((z) => String(z || "").trim())
        : [String(zipInput || "").trim()];
      const targetSet = new Set(targets.filter((z) => z && z !== "ALL"));
      if (!targetSet.size) return [];
      const features = (mapLayers.zip_boundaries && Array.isArray(mapLayers.zip_boundaries.features))
        ? mapLayers.zip_boundaries.features
        : [];
      const traces = [];
      for (const ft of features) {
        if (!ft || typeof ft !== "object") continue;
        const zip5 = featureZipCode(ft);
        if (!targetSet.has(zip5)) continue;
        const geom = ft.geometry || {};
        const type = String(geom.type || "");
        const coords = geom.coordinates;
        if (!Array.isArray(coords)) continue;
        if (type === "Polygon") {
          const outline = ringToMapCoords(coords[0]);
          if (outline.lat.length > 2) {
            traces.push({
              type: "scattermapbox",
              mode: "lines",
              lat: outline.lat,
              lon: outline.lon,
              line: { color: "rgba(75,85,99,0.95)", width: 1.35 },
              fill: "toself",
              fillcolor: "rgba(156,163,175,0.42)",
              hoverinfo: "skip",
              showlegend: false,
              name: `ZIP ${zip5}`,
            });
          }
        } else if (type === "MultiPolygon") {
          for (const poly of coords) {
            if (!Array.isArray(poly) || !Array.isArray(poly[0])) continue;
            const outline = ringToMapCoords(poly[0]);
            if (outline.lat.length > 2) {
              traces.push({
                type: "scattermapbox",
                mode: "lines",
                lat: outline.lat,
                lon: outline.lon,
                line: { color: "rgba(75,85,99,0.95)", width: 1.35 },
                fill: "toself",
                fillcolor: "rgba(156,163,175,0.42)",
                hoverinfo: "skip",
                showlegend: false,
                name: `ZIP ${zip5}`,
              });
            }
          }
        }
      }
      return traces;
    }
    const zipBoundaryLines = boundaryGeoToLineCoords(mapLayers.zip_boundaries || null);
    const kingCountyBoundaryLines = boundaryGeoToLineCoords(mapLayers.king_county_boundary || null);

    function clampMapZoom(z) {
      return Math.max(7.1, Math.min(13.6, asNum(z, 9.2)));
    }
    function getBoundaryViewport(lines, fallback = { lat: 47.47, lon: -121.95, zoom: 8.35 }) {
      const lats = (lines?.lat || [])
        .filter((v) => v !== null && v !== undefined)
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v));
      const lons = (lines?.lon || [])
        .filter((v) => v !== null && v !== undefined)
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v));
      if (!lats.length || !lons.length) return fallback;
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLon = Math.min(...lons);
      const maxLon = Math.max(...lons);
      const span = Math.max(maxLat - minLat, maxLon - minLon);
      let zoom = 8.35;
      if (span > 2.0) zoom = 7.2;
      else if (span > 1.5) zoom = 7.7;
      else if (span > 1.1) zoom = 8.15;
      else if (span > 0.8) zoom = 8.55;
      else if (span > 0.55) zoom = 8.9;
      else if (span > 0.35) zoom = 9.45;
      else zoom = 10.2;
      return {
        lat: (minLat + maxLat) / 2,
        lon: (minLon + maxLon) / 2,
        zoom: clampMapZoom(zoom),
      };
    }
    const countyViewport = getBoundaryViewport(kingCountyBoundaryLines);
    function getZipBoundaryViewport(zip5) {
      const target = String(zip5 || "");
      if (!target || target === "ALL") return null;
      const features = (mapLayers.zip_boundaries && Array.isArray(mapLayers.zip_boundaries.features))
        ? mapLayers.zip_boundaries.features
        : [];
      const lats = [];
      const lons = [];

      function pushCoordPoint(pt) {
        if (!Array.isArray(pt) || pt.length < 2) return;
        const x = Number(pt[0]);
        const y = Number(pt[1]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        lons.push(x);
        lats.push(y);
      }

      function walkRing(ring) {
        if (!Array.isArray(ring)) return;
        ring.forEach(pushCoordPoint);
      }

      for (const ft of features) {
        if (!ft || typeof ft !== "object") continue;
        if (featureZipCode(ft) !== target) continue;
        const geom = ft.geometry || {};
        const type = String(geom.type || "");
        const coords = geom.coordinates;
        if (!Array.isArray(coords)) continue;
        if (type === "Polygon") {
          walkRing(coords[0] || []);
        } else if (type === "MultiPolygon") {
          for (const poly of coords) {
            if (!Array.isArray(poly) || !Array.isArray(poly[0])) continue;
            walkRing(poly[0]);
          }
        }
      }
      if (!lats.length || !lons.length) return null;

      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLon = Math.min(...lons);
      const maxLon = Math.max(...lons);
      const span = Math.max(maxLat - minLat, maxLon - minLon);
      let zoom = 11.2;
      if (span > 0.60) zoom = 8.9;
      else if (span > 0.42) zoom = 9.3;
      else if (span > 0.30) zoom = 9.8;
      else if (span > 0.20) zoom = 10.3;
      else if (span > 0.13) zoom = 10.8;
      else if (span > 0.08) zoom = 11.3;
      else if (span > 0.05) zoom = 11.8;
      else zoom = 12.3;

      return {
        lat: (minLat + maxLat) / 2,
        lon: (minLon + maxLon) / 2,
        // keep a little nearby context around the ZIP.
        zoom: clampMapZoom(zoom - 0.2),
      };
    }
    function applyZipViewport(zip5) {
      const vp = getZipBoundaryViewport(zip5);
      if (!vp) return false;
      state.mapCenterLat = vp.lat;
      state.mapCenterLon = vp.lon;
      state.mapZoom = vp.zoom;
      return true;
    }
    function getCurrentMapViewport(fallbackRows = []) {
      const fallbackCenter = mapCenter(fallbackRows);
      const fallbackZoom = estimateMapZoom(fallbackRows);
      const map = document.getElementById("storeMap");
      const mapLayout = map?.layout?.mapbox || {};
      const lat = asNumSafe(state.mapCenterLat, asNumSafe(mapLayout?.center?.lat, fallbackCenter.lat));
      const lon = asNumSafe(state.mapCenterLon, asNumSafe(mapLayout?.center?.lon, fallbackCenter.lon));
      const zoom = clampMapZoom(asNumSafe(state.mapZoom, asNumSafe(mapLayout?.zoom, fallbackZoom)));
      return { lat, lon, zoom };
    }
    function setMapViewport(lat, lon, zoom) {
      const next = {
        lat: asNumSafe(lat, countyViewport.lat),
        lon: asNumSafe(lon, countyViewport.lon),
        zoom: clampMapZoom(zoom),
      };
      state.mapCenterLat = next.lat;
      state.mapCenterLon = next.lon;
      state.mapZoom = next.zoom;
      const map = document.getElementById("storeMap");
      if (!HAS_PLOTLY || !map || typeof Plotly?.relayout !== "function") return;
      Plotly.relayout(map, {
        "mapbox.center.lat": next.lat,
        "mapbox.center.lon": next.lon,
        "mapbox.zoom": next.zoom,
      });
    }

    function getFilterStores() {
      const selectedZipSet = new Set((state.zipFilters || []).map((z) => String(z || "")));
      return storeData.filter((s) => {
        if (state.city !== "ALL" && String(s.city) !== state.city) return false;
        if (selectedZipSet.size) {
          const zip = String(s.zip5 || "");
          if (!selectedZipSet.has(zip)) return false;
        }
        if (state.categories.length) {
          const cats = getStoreCategories(s);
          if (!cats.some((c) => state.categories.includes(c))) return false;
        }
        if (state.brand !== "ALL" && String(s.brand) !== state.brand) return false;
        return true;
      });
    }
    function getVisibleStores() {
      const q = state.search.trim().toLowerCase();
      const arr = getFilterStores().filter((s) => {
        if (!q) return true;
        return `${String(s.brand || "")} ${String(s.store_name || "")}`.toLowerCase().includes(q);
      });
      const sortKey = String(state.storeSortKey || "");
      const sortDir = String(state.storeSortDir || "off");
      const collator = { sensitivity: "base", numeric: true };
      const defaultCompare = (a, b) => {
        let x = String(a.brand || "").localeCompare(String(b.brand || ""), undefined, collator);
        if (x !== 0) return x;
        x = String(a.store_name || "").localeCompare(String(b.store_name || ""), undefined, collator);
        if (x !== 0) return x;
        return String(a.store_id || "").localeCompare(String(b.store_id || ""), undefined, collator);
      };
      const compareByKey = (a, b) => {
        if (sortKey === "name") {
          return String(a.store_name || "").localeCompare(String(b.store_name || ""), undefined, collator);
        }
        if (sortKey === "rating") {
          const ra = asNum(a.google_rating, NaN);
          const rb = asNum(b.google_rating, NaN);
          const aHas = Number.isFinite(ra);
          const bHas = Number.isFinite(rb);
          if (aHas && !bHas) return -1;
          if (!aHas && bHas) return 1;
          if (aHas && bHas && ra !== rb) {
            return sortDir === "desc" ? (rb - ra) : (ra - rb);
          }
          const ca = asNum(a.google_user_ratings_total, 0);
          const cb = asNum(b.google_user_ratings_total, 0);
          if (ca !== cb) return sortDir === "desc" ? (cb - ca) : (ca - cb);
          return 0;
        }
        if (sortKey === "logo") {
          const la = hasRealStoreLogo(a) ? 1 : 0;
          const lb = hasRealStoreLogo(b) ? 1 : 0;
          if (la !== lb) return sortDir === "desc" ? (lb - la) : (la - lb);
          return 0;
        }
        return 0;
      };
      arr.sort((a, b) => {
        let x = 0;
        if (sortKey && sortDir !== "off") {
          x = compareByKey(a, b);
          if (sortKey === "name" && x !== 0 && sortDir === "desc") x = -x;
        }
        if (x !== 0) return x;
        return defaultCompare(a, b);
      });
      return arr;
    }
    function getFilteredZipRows(filterStores) {
      const selectedZipSet = new Set((state.zipFilters || []).map((z) => String(z || "")).filter(Boolean));
      if (selectedZipSet.size) {
        let rows = zipData.filter((z) => selectedZipSet.has(String(z.zip5 || "")));
        if (!rows.length) rows = zipData.slice();
        rows.sort((a, b) => asNum(a.rank, 9999) - asNum(b.rank, 9999));
        return rows;
      }
      const zipSet = new Set(filterStores.map((s) => String(s.zip5 || "")).filter(Boolean));
      let rows = zipData.filter((z) => zipSet.size === 0 || zipSet.has(String(z.zip5 || "")));
      if (!rows.length) rows = zipData.slice();
      rows.sort((a, b) => asNum(a.rank, 9999) - asNum(b.rank, 9999));
      return rows;
    }

    function buildReviewKeywords(store, zipRow) {
      const topic = String(zipRow?.top_topic_terms || "")
        .split(",")
        .map((x) => x.trim().toLowerCase())
        .filter(Boolean);
      const category = getStoreCategories(store)
        .map((x) => String(x || "").trim().toLowerCase())
        .filter(Boolean);
      const base = [
        "boba", "tea", "milk tea", "fruit tea", "service", "staff", "sweet", "quality",
        "flavor", "fresh", "line", "wait", "price", "clean", "friendly", "parking"
      ];
      return [...new Set([...topic, ...category, ...base])]
        .filter((x) => x.length >= 3)
        .slice(0, 14);
    }
    function normalizeKeywordTerm(term) {
      return String(term || "")
        .toLowerCase()
        .replace(new RegExp("\\\\s+", "g"), " ")
        .trim();
    }
    function buildKeywordRegex(term) {
      const kw = normalizeKeywordTerm(term);
      if (!kw) return null;
      if (kw.includes(" ")) return new RegExp(escRegex(kw), "ig");
      return new RegExp(`\\\\b${escRegex(kw)}\\\\b`, "ig");
    }
    function rangesOverlap(aStart, aEnd, bStart, bEnd) {
      return aStart < bEnd && bStart < aEnd;
    }
    function spanOverlapsAny(start, end, spans) {
      for (const sp of (spans || [])) {
        if (!Array.isArray(sp) || sp.length < 2) continue;
        if (rangesOverlap(start, end, Number(sp[0]), Number(sp[1]))) return true;
      }
      return false;
    }
    function collectTermHitsWithSpans(text, terms, options = {}) {
      const maxLen = asNum(options.maxLen, 8);
      const blocked = Array.isArray(options.blockedSpans) ? options.blockedSpans : [];
      const src = String(text || "");
      const out = [];
      const spans = [];
      const seen = new Set();
      for (const item of (terms || [])) {
        const kw = normalizeKeywordTerm(item);
        if (!kw || seen.has(kw)) continue;
        const re = buildKeywordRegex(kw);
        if (!re) continue;
        let matchedThis = false;
        let m;
        while ((m = re.exec(src)) !== null) {
          const start = m.index;
          const end = m.index + String(m[0] || "").length;
          if (spanOverlapsAny(start, end, blocked) || spanOverlapsAny(start, end, spans)) continue;
          spans.push([start, end]);
          matchedThis = true;
        }
        if (matchedThis) {
          out.push(kw);
          seen.add(kw);
        }
        if (out.length >= maxLen) break;
      }
      return { hits: out, spans };
    }
    function collectNegatedPhraseHits(text, terms, options = {}) {
      const maxLen = asNum(options.maxLen, 8);
      const blocked = Array.isArray(options.blockedSpans) ? options.blockedSpans : [];
      const src = String(text || "").toLowerCase();
      const out = [];
      const spans = [];
      const seen = new Set();
      const neg = `(?:${NEGATION_TERMS.map((t) => escRegex(t)).join("|")})`;
      for (const item of (terms || [])) {
        const kw = normalizeKeywordTerm(item);
        if (!kw || seen.has(kw)) continue;
        const re = new RegExp(`\\\\b${neg}\\\\b(?:\\\\s+(?:really|very|so|too|quite|that|more|less))?(?:\\\\s+\\\\w+)?\\\\s+${escRegex(kw)}\\\\b`, "ig");
        let m;
        let matchedThis = false;
        while ((m = re.exec(src)) !== null) {
          const phrase = normalizeKeywordTerm(m[0]);
          if (!phrase) continue;
          const start = m.index;
          const end = m.index + String(m[0] || "").length;
          if (spanOverlapsAny(start, end, blocked) || spanOverlapsAny(start, end, spans)) continue;
          out.push(phrase);
          spans.push([start, end]);
          matchedThis = true;
        }
        if (matchedThis) seen.add(kw);
        if (out.length >= maxLen) break;
      }
      return { hits: [...new Set(out)], spans };
    }
    function keywordExistsInText(textLower, keyword) {
      const kw = normalizeKeywordTerm(keyword);
      if (!kw) return false;
      if (kw.includes(" ")) return textLower.includes(kw);
      const re = new RegExp(`\\\\b${escRegex(kw)}\\\\b`, "i");
      return re.test(textLower);
    }
    function matchedKeywords(text, candidates, maxLen = 8) {
      const src = String(text || "").toLowerCase();
      const out = [];
      const seen = new Set();
      for (const item of (candidates || [])) {
        const kw = normalizeKeywordTerm(item);
        if (!kw || seen.has(kw)) continue;
        if (!keywordExistsInText(src, kw)) continue;
        out.push(kw);
        seen.add(kw);
        if (out.length >= maxLen) break;
      }
      return out;
    }
    function buildKeywordGroupsForReview(text, signalCandidates = []) {
      const src = String(text || "");
      const negativeNoPrefix = NEGATIVE_WORDS.filter((w) => !normalizeKeywordTerm(w).startsWith("not "));
      const negatedPositivePhrases = collectNegatedPhraseHits(src, POSITIVE_PHRASE_OVERRIDES, { maxLen: 6 });
      const positivePhraseHits = collectTermHitsWithSpans(src, POSITIVE_PHRASE_OVERRIDES, {
        maxLen: 8,
        blockedSpans: negatedPositivePhrases.spans,
      });
      const negatedPositive = collectNegatedPhraseHits(src, POSITIVE_WORDS, {
        maxLen: 8,
        blockedSpans: positivePhraseHits.spans,
      });
      const negatedNegative = collectNegatedPhraseHits(src, negativeNoPrefix, { maxLen: 6 });
      const positiveRaw = collectTermHitsWithSpans(src, POSITIVE_WORDS, {
        maxLen: 10,
        blockedSpans: [...negatedPositive.spans, ...negatedPositivePhrases.spans],
      }).hits;
      const negativeRaw = collectTermHitsWithSpans(src, NEGATIVE_WORDS, {
        maxLen: 10,
        blockedSpans: [...negatedNegative.spans, ...positivePhraseHits.spans],
      }).hits;

      const positiveHits = [...new Set([...positivePhraseHits.hits, ...positiveRaw, ...negatedNegative.hits])].slice(0, 8);
      const negativeHits = [...new Set([...negativeRaw, ...negatedPositive.hits, ...negatedPositivePhrases.hits])].slice(0, 8);

      const sentimentRoots = new Set();
      for (const term of [...positiveHits, ...negativeHits]) {
        const norm = normalizeKeywordTerm(term);
        if (!norm) continue;
        sentimentRoots.add(norm);
        const parts = norm.split(" ").map((p) => p.replaceAll("'", "").trim()).filter(Boolean);
        for (const p of parts) {
          if (!NEGATION_TOKEN_SET.has(p)) sentimentRoots.add(p);
        }
      }

      const signalHits = matchedKeywords(src, signalCandidates, 14).filter((k) => {
        const norm = normalizeKeywordTerm(k);
        return norm && !sentimentRoots.has(norm);
      });
      const signalFallback = (signalCandidates || [])
        .map(normalizeKeywordTerm)
        .filter((k) => k && !sentimentRoots.has(k))
        .slice(0, 10);

      return {
        positive: positiveHits,
        negative: negativeHits,
        signal: signalHits.length ? signalHits : signalFallback,
      };
    }
    function classifyReviewTone(review) {
      const rating = asNum(review?.rating, NaN);
      const text = String(review?.review_text || review?.review_text_short || "");
      const groups = buildKeywordGroupsForReview(text, []);
      let score = 0;
      if (Number.isFinite(rating)) score += (rating - 3) * 1.05;
      score += (groups.positive || []).length * 0.34;
      score -= (groups.negative || []).length * 0.40;
      if (score >= 0.8) return { label: "Positive", cls: "tone-positive" };
      if (score <= -0.8) return { label: "Negative", cls: "tone-negative" };
      return { label: "Neutral", cls: "tone-neutral" };
    }
    function renderKeywordRows(groups) {
      const cfg = [
        { key: "positive", label: "Positive Cues", cls: "kw-token kw-token-positive" },
        { key: "negative", label: "Risk Cues", cls: "kw-token kw-token-negative" },
        { key: "signal", label: "Product & Service Signals", cls: "kw-token kw-token-signal" },
      ];
      return cfg.map((row) => {
        const vals = (groups?.[row.key] || []).filter(Boolean);
        const body = vals.length
          ? vals.map((k) => `<span class="${row.cls}">${esc(k)}</span>`).join("")
          : `<span class="kw-none">none</span>`;
        return `<div class="kw-row"><span class="kw-label">${row.label}:</span><span class="kw-values">${body}</span></div>`;
      }).join("");
    }
    function highlightReviewText(text, keywordGroups) {
      const raw = String(text || "");
      if (!raw) return "No text.";
      const positive = (keywordGroups?.positive || []).map(normalizeKeywordTerm).filter(Boolean);
      const negative = (keywordGroups?.negative || []).map(normalizeKeywordTerm).filter(Boolean);
      const signal = (keywordGroups?.signal || []).map(normalizeKeywordTerm).filter(Boolean);
      const list = [...new Set([...negative, ...positive, ...signal])].sort((a, b) => b.length - a.length);
      if (!list.length) return esc(raw);
      const positiveSet = new Set(positive);
      const negativeSet = new Set(negative);
      const signalSet = new Set(signal);
      const matches = [];
      for (const term of list) {
        const re = buildKeywordRegex(term);
        if (!re) continue;
        let m;
        while ((m = re.exec(raw)) !== null) {
          const start = m.index;
          const end = m.index + String(m[0] || "").length;
          if (spanOverlapsAny(start, end, matches.map((x) => [x.start, x.end]))) continue;
          const key = normalizeKeywordTerm(m[0]);
          let cls = "kw-highlight";
          if (negativeSet.has(key) || negativeSet.has(term)) cls = "kw-highlight-negative";
          else if (positiveSet.has(key) || positiveSet.has(term)) cls = "kw-highlight-positive";
          else if (signalSet.has(key) || signalSet.has(term)) cls = "kw-highlight";
          matches.push({ start, end, text: m[0], cls });
        }
      }
      matches.sort((a, b) => a.start - b.start);
      let out = "";
      let last = 0;
      for (const m of matches) {
        out += esc(raw.slice(last, m.start));
        out += `<mark class="${m.cls}">${esc(m.text)}</mark>`;
        last = m.end;
      }
      out += esc(raw.slice(last));
      return out;
    }
    function snippet(text, maxLen = 110) {
      const t = String(text || "").replace(new RegExp("\\\\s+", "g"), " ").trim();
      if (t.length <= maxLen) return t;
      return `${t.slice(0, maxLen - 1)}...`;
    }

    function syncState() {
      const zipOptions = allZipOptions();
      state.zipFilters = (state.zipFilters || []).filter((z) => zipOptions.includes(String(z || "")));
      const filterStores = getFilterStores();
      const zipSet = new Set(filterStores.map((s) => String(s.zip5 || "")));
      if (state.zipFilters.length && state.zip === "ALL") {
        state.zip = String(state.zipFilters[0] || "ALL");
      }
      if (state.zip !== "ALL" && !zipSet.has(state.zip)) {
        const fallbackZip = (state.zipFilters || []).find((z) => zipSet.has(String(z || "")));
        state.zip = fallbackZip ? String(fallbackZip) : "ALL";
      }

      const visible = getVisibleStores();
      const idSet = new Set(visible.map((s) => String(s.store_id || "")));
      if (!state.storeId || !idSet.has(state.storeId)) {
        state.storeId = visible.length ? String(visible[0].store_id) : null;
        state.reviewIndex = 0;
        state.storePageManual = false;
      }

      const pageSize = getStorePageSize();
      const pages = Math.max(1, Math.ceil(visible.length / pageSize));
      if (!Number.isFinite(asNum(state.storePage, NaN)) || asNum(state.storePage, NaN) < 1) state.storePage = 1;
      state.storePage = Math.min(state.storePage, pages);
      if (state.storeId && !state.storePageManual) {
        const idx = visible.findIndex((s) => String(s.store_id || "") === String(state.storeId || ""));
        if (idx >= 0) {
          state.storePage = Math.floor(idx / pageSize) + 1;
        }
      }
    }

    function renderFilters() {
      el.cityFilter.innerHTML = optionList(
        uniqSorted(storeData, "city"),
        state.city,
        "All Cities",
        (v) => toTitleCaseCity(v),
      );
      const zipOptions = allZipOptions();
      state.zipFilters = (state.zipFilters || []).filter((z) => zipOptions.includes(String(z || "")));
      const selectedZipSet = new Set(state.zipFilters);
      const zipMenuRows = zipOptions.map((zip5) => {
        const active = selectedZipSet.has(zip5);
        return `
          <button class=\"multi-option${active ? " active" : ""}\" data-zip=\"${esc(zip5)}\" type=\"button\" role=\"option\" aria-selected=\"${active ? "true" : "false"}\">
            <input class=\"multi-checkbox\" type=\"checkbox\" ${active ? "checked" : ""} tabindex=\"-1\" aria-hidden=\"true\" />
            <span>${esc(zip5)}</span>
          </button>
        `;
      }).join("");
      el.zipFilterMenu.innerHTML = `
        <div class=\"multi-actions\">
          <button class=\"multi-mini-btn\" type=\"button\" data-action=\"all\">Select All</button>
          <button class=\"multi-mini-btn\" type=\"button\" data-action=\"clear\">Clear</button>
        </div>
        ${zipMenuRows || `<div class=\"tiny\" style=\"padding:6px 8px;\">No ZIP codes.</div>`}
      `;
      el.zipFilterText.textContent = zipSummaryText(state.zipFilters, zipOptions);
      el.zipFilterBtn.title = state.zipFilters.length
        ? `Selected: ${state.zipFilters.join(", ")}`
        : "All ZIP codes";

      const catOptions = allCategoryOptions(storeData);
      state.categories = (state.categories || []).filter((c) => catOptions.includes(c));
      const selectedSet = new Set(state.categories);
      const menuRows = catOptions.map((cat) => {
        const active = selectedSet.has(cat);
        return `
          <button class=\"multi-option${active ? " active" : ""}\" data-category=\"${esc(cat)}\" type=\"button\" role=\"option\" aria-selected=\"${active ? "true" : "false"}\">
            <input class=\"multi-checkbox\" type=\"checkbox\" ${active ? "checked" : ""} tabindex=\"-1\" aria-hidden=\"true\" />
            <span>${esc(cat)}</span>
          </button>
        `;
      }).join("");
      el.categoryFilterMenu.innerHTML = `
        <div class=\"multi-actions\">
          <button class=\"multi-mini-btn\" type=\"button\" data-action=\"all\">Select All</button>
          <button class=\"multi-mini-btn\" type=\"button\" data-action=\"clear\">Clear</button>
        </div>
        ${menuRows || `<div class=\"tiny\" style=\"padding:6px 8px;\">No categories.</div>`}
      `;
      el.categoryFilterText.textContent = categorySummaryText(state.categories, catOptions);
      el.categoryFilterBtn.title = state.categories.length
        ? `Selected: ${state.categories.join(", ")}`
        : "All categories";
      el.brandFilter.innerHTML = optionList(uniqSorted(storeData, "brand"), state.brand, "All Brands");
      el.cityFilter.value = state.city;
      el.brandFilter.value = state.brand;
      el.storeSearch.value = state.search;
    }

    function renderKpis(visibleStores, zipRows) {
      const visibleStoreCount = visibleStores.length;
      const visibleBrands = new Set(visibleStores.map((s) => String(s.brand || ""))).size;
      const topZip = zipRows.slice().sort((a, b) => asNum(b.opportunity_score) - asNum(a.opportunity_score))[0] || null;
      const focusedZip = state.zip !== "ALL" ? state.zip : (topZip ? `Top ${topZip.zip5}` : "-");
      const ratings = visibleStores.map((s) => asNum(s.google_rating, NaN)).filter((x) => Number.isFinite(x));
      const avgRating = ratings.length ? ratings.reduce((a, b) => a + b, 0) / ratings.length : NaN;

      el.kpiStores.textContent = fmtInt(visibleStoreCount);
      el.kpiBrands.textContent = fmtInt(visibleBrands);
      el.kpiZip.textContent = focusedZip;
      el.kpiRating.textContent = Number.isFinite(avgRating) ? avgRating.toFixed(2) : "-";
    }

    function showEmptyPlot(id, message) {
      const node = document.getElementById(id);
      if (!node) return;
      node.innerHTML = `<div class=\"empty-plot\">${esc(message)}</div>`;
    }

    function renderOpportunityHeatmap(zipRows) {
      if (!HAS_PLOTLY) {
        showEmptyPlot("oppHeatmap", "Plotly failed to load. Please check network.");
        return;
      }
      const top = zipRows
        .slice()
        .sort((a, b) => asNum(b.opportunity_score) - asNum(a.opportunity_score))
        .slice(0, 12);
      const metrics = [
        { label: "Opportunity", getter: (r) => asNum(r.opportunity_score) },
        { label: "Demand", getter: (r) => asNum(r.demand_score) },
        {
          label: "Gap",
          getter: (r) => {
            const comp = asNumSafe(r.gap_component_score, NaN);
            if (Number.isFinite(comp)) return comp;
            const raw = asNumSafe(r.gap_score, 0);
            return Math.max(0, Math.min(100, raw + 50));
          },
        },
        { label: "Momentum", getter: (r) => asNum(r.momentum_score) },
        { label: "Text", getter: (r) => asNum(r.text_signal_score) },
        { label: "Feasibility", getter: (r) => asNum(r.feasibility_score, 50) },
      ];
      const x = metrics.map((m) => m.label);
      const y = top.map((r) => String(r.zip5 || ""));
      const z = top.map((r) => metrics.map((m) => m.getter(r)));
      const trace = {
        type: "heatmap",
        x,
        y,
        z,
        zmin: 0,
        zmax: 100,
        colorscale: [
          [0, "#f8e4cf"],
          [0.35, "#f3c9a7"],
          [0.65, "#9dbfdd"],
          [1, "#1f3d5a"],
        ],
        hovertemplate: "%{y}<br>%{x}: %{z:.1f}<extra></extra>",
      };
      const layout = {
        margin: { l: 58, r: 10, t: 8, b: 28 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Manrope, sans-serif", size: 11, color: "#1f2937" },
        xaxis: { side: "bottom" },
        yaxis: { type: "category", automargin: true, autorange: "reversed" },
        height: 300,
      };
      Plotly.react("oppHeatmap", [trace], layout, { displayModeBar: false, responsive: true });
      if (el.oppFormulaNote) {
        el.oppFormulaNote.textContent = String(
          DATA?.meta?.scoring_formula_note
          || "Opportunity = weighted blend of Demand, Gap, Momentum, Text, and Feasibility."
        );
      }
    }

    function renderGapScatter(zipRows) {
      if (!HAS_PLOTLY) {
        showEmptyPlot("gapScatter", "Plotly failed to load.");
        if (el.gapMedianNote) el.gapMedianNote.textContent = "";
        return;
      }
      const demandVals = zipRows.map((d) => asNum(d.demand_score, NaN)).filter((v) => Number.isFinite(v));
      const supplyVals = zipRows.map((d) => asNum(d.supply_score, NaN)).filter((v) => Number.isFinite(v));
      const demandMedian = medianOf(demandVals, 50);
      const supplyMedian = medianOf(supplyVals, 50);
      const xMinRaw = demandVals.length ? Math.min(...demandVals) : 0;
      const xMaxRaw = demandVals.length ? Math.max(...demandVals) : 100;
      const yMinRaw = supplyVals.length ? Math.min(...supplyVals) : 0;
      const yMaxRaw = supplyVals.length ? Math.max(...supplyVals) : 100;
      const xPad = Math.max(3, (xMaxRaw - xMinRaw) * 0.08);
      const yPad = Math.max(3, (yMaxRaw - yMinRaw) * 0.08);
      let xRange = [Math.max(0, xMinRaw - xPad), Math.min(100, xMaxRaw + xPad)];
      let yRange = [Math.max(0, yMinRaw - yPad), Math.min(100, yMaxRaw + yPad)];
      if (xRange[1] - xRange[0] < 8) xRange = [Math.max(0, demandMedian - 4), Math.min(100, demandMedian + 4)];
      if (yRange[1] - yRange[0] < 8) yRange = [Math.max(0, supplyMedian - 4), Math.min(100, supplyMedian + 4)];
      const xLowerCenter = xRange[0] + (demandMedian - xRange[0]) * 0.5;
      const xUpperCenter = demandMedian + (xRange[1] - demandMedian) * 0.5;
      const yLowerCenter = yRange[0] + (supplyMedian - yRange[0]) * 0.5;
      const yUpperCenter = supplyMedian + (yRange[1] - supplyMedian) * 0.5;
      const yTicks = [];
      for (let v = 0; v <= 100; v += 20) {
        if (v > yRange[0] - 0.01 && v < yRange[1] + 0.01 && v !== 0) yTicks.push(v);
      }

      const allTrace = {
        type: "scatter",
        mode: "markers",
        x: zipRows.map((d) => asNum(d.demand_score)),
        y: zipRows.map((d) => asNum(d.supply_score)),
        customdata: zipRows.map((d) => [String(d.zip5), asNum(d.opportunity_score), asNum(d.active_shop_count), String(d.quadrant || "")]),
        marker: {
          size: zipRows.map((d) => Math.max(8, asNum(d.active_shop_count) * 2.1)),
          color: zipRows.map((d) => asNum(d.opportunity_score)),
          colorscale: [[0, "#f3c9a7"], [0.45, "#b5d0e8"], [1, "#1f3d5a"]],
          line: { width: 0.8, color: "#fff" },
          showscale: false,
          opacity: 0.88,
        },
        hovertemplate: "ZIP %{customdata[0]}<br>Demand %{x:.1f}<br>Supply %{y:.1f}<br>Opportunity %{customdata[1]:.1f}<br>Shops %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
      };

      const traces = [allTrace];
      if (state.zip !== "ALL") {
        const row = zipRows.find((d) => String(d.zip5) === state.zip);
        if (row) {
          traces.push({
            type: "scatter",
            mode: "markers",
            x: [asNum(row.demand_score)],
            y: [asNum(row.supply_score)],
            marker: { size: 16, color: "#111827", symbol: "diamond", line: { width: 1.2, color: "#fff" } },
            hovertemplate: `Focused ZIP ${row.zip5}<extra></extra>`,
            showlegend: false,
          });
        }
      }

      const layout = {
        margin: { l: 44, r: 10, t: 8, b: 32 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Manrope, sans-serif", size: 11, color: "#1f2937" },
        xaxis: { title: "Demand", gridcolor: "#e9e1d6", zeroline: false, range: xRange },
        yaxis: {
          title: "Supply",
          gridcolor: "#e9e1d6",
          zeroline: false,
          range: yRange,
          tickmode: yTicks.length ? "array" : "auto",
          tickvals: yTicks.length ? yTicks : undefined,
          ticktext: yTicks.length ? yTicks.map((v) => String(v)) : undefined,
        },
        shapes: [
          {
            type: "line",
            xref: "x",
            yref: "y",
            x0: demandMedian,
            x1: demandMedian,
            y0: yRange[0],
            y1: yRange[1],
            line: { color: "#475467", width: 1.2, dash: "dash" },
          },
          {
            type: "line",
            xref: "x",
            yref: "y",
            x0: xRange[0],
            x1: xRange[1],
            y0: supplyMedian,
            y1: supplyMedian,
            line: { color: "#475467", width: 1.2, dash: "dash" },
          },
        ],
        annotations: [
          {
            x: xUpperCenter,
            y: yUpperCenter,
            xref: "x",
            yref: "y",
            text: "High Demand<br>High Supply",
            showarrow: false,
            font: { size: 9, color: "#667085" },
          },
          {
            x: xUpperCenter,
            y: yLowerCenter,
            xref: "x",
            yref: "y",
            text: "High Demand<br>Low Supply",
            showarrow: false,
            font: { size: 9, color: "#667085" },
          },
          {
            x: xLowerCenter,
            y: yUpperCenter,
            xref: "x",
            yref: "y",
            text: "Low Demand<br>High Supply",
            showarrow: false,
            font: { size: 9, color: "#667085" },
          },
          {
            x: xLowerCenter,
            y: yLowerCenter,
            xref: "x",
            yref: "y",
            text: "Low Demand<br>Low Supply",
            showarrow: false,
            font: { size: 9, color: "#667085" },
          },
        ],
        height: 300,
      };
      Plotly.react("gapScatter", traces, layout, { displayModeBar: false, responsive: true });
      if (el.gapMedianNote) {
        el.gapMedianNote.innerHTML = `Demand median = ${demandMedian.toFixed(1)}<br>Supply median = ${supplyMedian.toFixed(1)}`;
      }
    }

    function renderBrandMix(filterStores) {
      if (!HAS_PLOTLY) {
        showEmptyPlot("miniChart", "Plotly failed to load.");
        return;
      }
      const node = document.getElementById("miniChart");
      node?.classList.remove("logo-wall");
      if (node) node.innerHTML = "";
      const count = {};
      filterStores.forEach((s) => {
        const k = String(s.brand || "Unknown");
        count[k] = (count[k] || 0) + 1;
      });
      let rows = Object.entries(count).map(([brand, v]) => ({ brand, v }));
      rows.sort((a, b) => b.v - a.v);
      const top = rows.slice(0, 8);
      if (rows.length > 8) {
        const other = rows.slice(8).reduce((a, b) => a + b.v, 0);
        if (other > 0) top.push({ brand: "Others", v: other });
      }
      const trace = {
        type: "pie",
        labels: top.map((d) => d.brand),
        values: top.map((d) => d.v),
        hole: 0.56,
        textinfo: "label+percent",
        textfont: { size: 10 },
        sort: false,
        marker: { colors: ["#1f3d5a", "#4f7ca7", "#7ea6cb", "#abc7e2", "#f0b694", "#f5cfb0", "#e6ddd2", "#c6d4e2", "#dde6ef"] },
        hovertemplate: "%{label}<br>Stores %{value}<extra></extra>",
      };
      const layout = {
        margin: { l: 8, r: 8, t: 6, b: 8 },
        paper_bgcolor: "rgba(0,0,0,0)",
        showlegend: false,
        height: miniPlotHeight(),
      };
      Plotly.react("miniChart", [trace], layout, { displayModeBar: false, responsive: true });
    }

    function renderCityCategory(filterStores) {
      if (!HAS_PLOTLY) {
        showEmptyPlot("miniChart", "Plotly failed to load.");
        return;
      }
      const node = document.getElementById("miniChart");
      node?.classList.remove("logo-wall");
      if (node) node.innerHTML = "";
      const cityRaw = filterStores.map((s) => String(s.city || "").trim()).filter(Boolean);
      const catRaw = [];
      filterStores.forEach((s) => {
        getStoreCategories(s).forEach((c) => catRaw.push(String(c || "").trim()));
      });
      if (!cityRaw.length || !catRaw.length) {
        showEmptyPlot("miniChart", "No city/category rows.");
        return;
      }
      const cityTotals = {};
      const catTotals = {};
      const matrix = {};
      filterStores.forEach((s) => {
        const city = toTitleCaseCity(String(s.city || "").trim());
        const cats = getStoreCategories(s);
        if (!city || !cats.length) return;
        cityTotals[city] = (cityTotals[city] || 0) + 1;
        cats.forEach((cat) => {
          if (!cat) return;
          catTotals[cat] = (catTotals[cat] || 0) + 1;
          const key = `${city}|||${cat}`;
          matrix[key] = (matrix[key] || 0) + 1;
        });
      });
      const topCities = Object.entries(cityTotals)
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .slice(0, 9)
        .map(([city]) => city);
      const topCats = Object.entries(catTotals)
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .slice(0, 5)
        .map(([cat]) => cat);
      if (!topCities.length || !topCats.length) {
        showEmptyPlot("miniChart", "No city/category rows.");
        return;
      }
      const palette = ["#1f3d5a", "#4f7ca7", "#7ea6cb", "#f0b694", "#e0c7ad", "#9ab3c9"];
      const traces = topCats.map((cat, i) => {
        const catLabel = String(cat || "").replaceAll("_", " / ");
        return {
          type: "bar",
          orientation: "h",
          name: catLabel,
          y: topCities,
          x: topCities.map((city) => matrix[`${city}|||${cat}`] || 0),
          marker: { color: palette[i % palette.length] },
          hovertemplate: `City %{y}<br>Category ${esc(catLabel)}<br>Stores %{x}<extra></extra>`,
        };
      });
      const layout = {
        margin: { l: 122, r: 12, t: 8, b: 36 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        barmode: "stack",
        xaxis: { title: "Store count", gridcolor: "#e9e1d6", zerolinecolor: "#e9e1d6" },
        yaxis: { automargin: true, tickfont: { size: 11 } },
        legend: { orientation: "h", y: 1.14, x: 0, font: { size: 10 } },
        height: miniPlotHeight(),
        font: { family: "Manrope, sans-serif", size: 10 },
      };
      Plotly.react("miniChart", traces, layout, { displayModeBar: false, responsive: true });
    }

    function renderLicense(zipRows) {
      if (!HAS_PLOTLY) {
        showEmptyPlot("miniChart", "Plotly failed to load.");
        return;
      }
      const node = document.getElementById("miniChart");
      node?.classList.remove("logo-wall");
      if (node) node.innerHTML = "";
      const hasScopedFilter =
        state.city !== "ALL" ||
        state.brand !== "ALL" ||
        (state.zipFilters || []).length > 0 ||
        (state.categories || []).length > 0 ||
        String(state.search || "").trim().length > 0 ||
        String(state.zip || "ALL") !== "ALL";

      const candidateZips = hasScopedFilter
        ? zipRows
            .map((z) => String(z.zip5 || ""))
            .filter(Boolean)
        : zipRows
            .slice()
            .sort((a, b) => asNum(b.opportunity_score) - asNum(a.opportunity_score))
            .slice(0, 3)
            .map((z) => String(z.zip5 || ""))
            .filter(Boolean);

      const uniqueZips = [...new Set(candidateZips)];
      const lines = uniqueZips
        .map((zip) => ({
          zip,
          rows: licenseData
            .filter((d) => String(d.zip5) === zip)
            .sort((a, b) => asNum(a.year) - asNum(b.year)),
        }))
        .filter((line) => (line.rows || []).length > 0);

      if (!lines.length) {
        showEmptyPlot("miniChart", "No license trend rows for current filters.");
        return;
      }

      const palette = ["#1f3d5a", "#5b89b5", "#f2b08b", "#8fbc8f"];
      const traces = lines.map((line, i) => ({
        type: "scatter",
        mode: "lines+markers",
        name: line.zip,
        x: line.rows.map((r) => asNum(r.year)),
        y: line.rows.map((r) => asNum(r.shops_started)),
        line: { width: 2.2, color: palette[i % palette.length] },
        marker: { size: 6 },
        hovertemplate: `ZIP ${line.zip}<br>Year %{x}<br>Shops started %{y}<extra></extra>`,
      }));
      const layout = {
        margin: { l: 40, r: 10, t: 8, b: 34 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { dtick: 1, gridcolor: "#e9e1d6" },
        yaxis: { title: "Openings", gridcolor: "#e9e1d6" },
        legend: { orientation: "h", y: 1.15, x: 0, font: { size: 10 } },
        height: miniPlotHeight(),
        font: { family: "Manrope, sans-serif", size: 10 },
      };
      Plotly.react("miniChart", traces, layout, { displayModeBar: false, responsive: true });
    }

    function renderBrandLogos(filterStores) {
      const node = document.getElementById("miniChart");
      if (!node) return;
      if (HAS_PLOTLY && typeof Plotly?.purge === "function") {
        try { Plotly.purge(node); } catch (_e) {}
      }
      node.classList.add("logo-wall");

      const rows = filterStores
        .filter((s) => hasCuratedBrandLogo(s))
        .map((s) => ({
          brand: String(s.brand || "Unknown").trim(),
          logo: String(s.logo_url || "").trim(),
        }))
        .filter((r) => r.logo);

      rows.sort((a, b) => a.brand.localeCompare(b.brand));

      const seen = new Set();
      const unique = [];
      for (const r of rows) {
        if (seen.has(r.logo)) continue;
        seen.add(r.logo);
        unique.push(r);
      }

      if (!unique.length) {
        node.classList.remove("plotly", "js-plotly-plot");
        node.innerHTML = `<div class=\"empty-plot\">No logo images in current filters.</div>`;
        return;
      }

      node.classList.remove("plotly", "js-plotly-plot");
      node.innerHTML = `
        <div class=\"logo-wall-grid\">
          ${unique.map((r) => `
            <div class=\"logo-node\" title=\"${esc(r.brand)}\">
              <div class=\"logo-circle\">
                <img src=\"${esc(r.logo)}\" alt=\"${esc(r.brand)} logo\" />
              </div>
              <div class=\"logo-name\">${esc(r.brand)}</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    function renderMiniChart(filterStores, visibleStores, zipRows) {
      if (state.miniTab !== "logos") {
        const node = document.getElementById("miniChart");
        if (node) {
          node.classList.remove("logo-wall");
          if (HAS_PLOTLY && typeof Plotly?.purge === "function") {
            try { Plotly.purge(node); } catch (_e) {}
          }
        }
      }
      if (state.miniTab === "brand") {
        renderBrandMix(filterStores);
      } else if (state.miniTab === "city") {
        renderCityCategory(filterStores);
      } else if (state.miniTab === "logos") {
        renderBrandLogos(visibleStores);
      } else {
        renderLicense(zipRows);
      }
      const tabs = el.miniTabs.querySelectorAll(".tab-btn");
      tabs.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === state.miniTab));
    }

    function renderStoreMap(visibleStores) {
      if (!HAS_PLOTLY) {
        el.mapSummary.textContent = "Heatmap unavailable";
        showEmptyPlot("storeMap", "Plotly failed to load.");
        return;
      }

      const rows = getStoresWithCoordinates(visibleStores);
      const total = visibleStores.length;
      el.mapSummary.textContent = `Point + Heatmap · ${rows.length}/${total} stores with coordinates`;

      if (!rows.length) {
        showEmptyPlot("storeMap", "No coordinates available in current filters.");
        return;
      }

      const viewport = getCurrentMapViewport(rows);
      const center = { lat: viewport.lat, lon: viewport.lon };
      const zoom = viewport.zoom;
      const customdata = rows.map((s) => [
        String(s.store_id || ""),
        String(s.store_name || ""),
        String(s.brand || ""),
        toTitleCaseCity(String(s.city || "")),
        String(s.zip5 || ""),
        categoryDisplayText(s),
        Number.isFinite(asNum(s.google_rating, NaN)) ? asNum(s.google_rating).toFixed(1) : "-",
        fmtInt(s.google_user_ratings_total),
        fmt1(s.opportunity_score),
      ]);

      const traces = [];
      traces.push({
        type: "densitymapbox",
        lat: rows.map((s) => asNum(s.latitude)),
        lon: rows.map((s) => asNum(s.longitude)),
        z: rows.map((s) => Math.max(1, asNum(s.google_user_ratings_total, 1))),
        radius: 24,
        colorscale: [[0, "rgba(247,224,220,0.08)"], [0.35, "rgba(232,154,138,0.44)"], [0.7, "rgba(170,66,66,0.75)"], [1, "rgba(126,31,36,0.95)"]],
        hoverinfo: "skip",
        name: "Density",
      });

      const focusZips = (state.zipFilters || []).map((z) => String(z || "").trim()).filter(Boolean);
      if (!focusZips.length && state.zip !== "ALL") focusZips.push(String(state.zip));
      const zipFocusFillTraces = getZipFocusFillTraces(focusZips);
      zipFocusFillTraces.forEach((t) => traces.push(t));

      traces.push({
        type: "scattermapbox",
        mode: "markers",
        lat: rows.map((s) => asNum(s.latitude)),
        lon: rows.map((s) => asNum(s.longitude)),
        customdata,
        marker: {
          size: 6.2,
          color: rows.map((s) => asNum(s.opportunity_score)),
          cmin: 0,
          cmax: 100,
          colorscale: [[0, "#f1bbb0"], [0.45, "#cc6a62"], [1, "#7e1f24"]],
          opacity: 0.74,
          line: { width: 0.45, color: "rgba(255,255,255,0.95)" },
          showscale: false,
        },
        hovertemplate: "%{customdata[1]}<br>%{customdata[3]}, %{customdata[4]}<br>%{customdata[5]}<br>Google %{customdata[6]} (%{customdata[7]})<br>Opportunity %{customdata[8]}<extra></extra>",
        name: "Stores",
      });

      if (zipBoundaryLines.lat.length) {
        traces.push({
          type: "scattermapbox",
          mode: "lines",
          lat: zipBoundaryLines.lat,
          lon: zipBoundaryLines.lon,
          line: { color: "rgba(124,124,124,0.82)", width: 1.05 },
          hoverinfo: "skip",
          showlegend: false,
          name: "ZIP boundaries",
        });
      }
      if (kingCountyBoundaryLines.lat.length) {
        traces.push({
          type: "scattermapbox",
          mode: "lines",
          lat: kingCountyBoundaryLines.lat,
          lon: kingCountyBoundaryLines.lon,
          line: { color: "#b91c1c", width: 2.2 },
          hoverinfo: "skip",
          showlegend: false,
          name: "King County boundary",
        });
      }

      const selected = rows.find((s) => String(s.store_id || "") === String(state.storeId || ""));
      if (selected) {
        traces.push({
          type: "scattermapbox",
          mode: "markers",
          lat: [asNum(selected.latitude)],
          lon: [asNum(selected.longitude)],
          customdata: [[String(selected.store_id || "")]],
          marker: { size: 20, color: "#7e1f24", symbol: "star", line: { width: 1.2, color: "#ffffff" } },
          hovertemplate: `${esc(String(selected.store_name || ""))}<extra>Selected</extra>`,
          name: "Selected",
        });
      }

      const layout = {
        margin: { l: 4, r: 4, t: 4, b: 4 },
        paper_bgcolor: "rgba(0,0,0,0)",
        showlegend: false,
        mapbox: {
          style: "carto-positron",
          center,
          zoom,
        },
        height: 430,
      };
      Plotly.react("storeMap", traces, layout, { displayModeBar: false, responsive: true });
      state.mapCenterLat = center.lat;
      state.mapCenterLon = center.lon;
      state.mapZoom = zoom;
    }

    function parseZipTopicTerms(row) {
      return String(row?.top_topic_terms || "")
        .split(",")
        .map((x) => normalizeKeywordTerm(x))
        .filter(Boolean);
    }

    function inferProductAngle(row) {
      const terms = parseZipTopicTerms(row);
      const termSet = new Set(terms);
      const picks = [];
      const addPick = (label) => { if (label && !picks.includes(label)) picks.push(label); };
      const matchAny = (keys) => keys.some((k) => termSet.has(normalizeKeywordTerm(k)));

      if (matchAny(["milk tea", "milktea", "milk"])) addPick("milk tea");
      if (matchAny(["fruit tea", "fruittea", "fruit"])) addPick("fruit tea");
      if (matchAny(["pure tea", "puretea"])) addPick("pure tea");
      if (matchAny(["matcha"])) addPick("matcha line");
      if (matchAny(["cheese tea", "cheesetea"])) addPick("cheese tea");
      if (matchAny(["boba", "bubble"])) addPick("boba toppings");

      if (!picks.length) addPick("milk tea");
      if (picks.length < 2 && !picks.includes("fruit tea")) addPick("fruit tea");
      return picks.slice(0, 2).join(" + ");
    }

    function buildZipSpecificStrategy(row) {
      const demand = asNum(row.demand_score);
      const gap = asNumSafe(row.gap_component_score, Math.max(0, Math.min(100, asNum(row.gap_score) + 50)));
      const momentum = asNum(row.momentum_score);
      const text = asNum(row.text_signal_score);
      const feasibility = asNum(row.feasibility_score, 50);
      const productAngle = inferProductAngle(row);

      let entryPlan = "Run a controlled pilot first and expand only after KPIs are stable.";
      if (demand >= 75 && gap >= 80 && feasibility >= 68) {
        entryPlan = "Open one pilot store in this ZIP within the next quarter, then evaluate a second point after a 12-week KPI review.";
      } else if (demand >= 75 && gap >= 80 && feasibility < 68) {
        entryPlan = "Enter with a low-capex kiosk first and delay a full-size lease until unit economics are proven.";
      } else if (demand >= 70 && gap < 80) {
        entryPlan = "Open only one differentiated flagship in phase 1 and avoid multi-store rollout in this crowded ZIP.";
      } else if (demand < 70 && text >= 65) {
        entryPlan = "Use a 6-8 week pop-up or shared-counter test first, and sign a long lease only if conversion is validated.";
      } else if (demand < 70) {
        entryPlan = "Keep this ZIP in test mode and validate demand with low-capex pilots before committing to a permanent store.";
      }

      let acquisitionPlan = "";
      if (momentum >= 45) {
        acquisitionPlan = "Launch with opening-week bundles and referral mechanics to capture existing tea traffic quickly.";
      } else if (momentum >= 25) {
        acquisitionPlan = "Prioritize 2-mile local ads (Google Maps + social) and partner with nearby schools/offices for weekday bundles.";
      } else {
        acquisitionPlan = "Compensate for low opening momentum with heavy local sampling, apartment/community partnerships, and review seeding in the first 6 weeks.";
      }

      let operationsPlan = "";
      if (text >= 70) {
        operationsPlan = "Keep service speed under 4 minutes and maintain daily review replies to protect strong word-of-mouth.";
      } else if (text >= 55) {
        operationsPlan = "Tighten consistency first: simplify to 10-14 core SKUs and standardize recipes before scaling marketing spend.";
      } else {
        operationsPlan = "Fix service quality before expansion: staff training, queue management, and complaint-response SLA first.";
      }

      const guardrails = [];
      if (feasibility < 60) guardrails.push("set a strict rent cap and keep footprint under ~900 sq ft");
      if (momentum < 25) guardrails.push("do not commit to site #2 before 12-week repeat-customer target is met");
      if (!guardrails.length) guardrails.push("use phase-gate KPIs (weekly sales, repeat rate, rating) before opening site #2");

      return [
        { label: "Expansion Strategy", labelClass: "entry", text: entryPlan },
        { label: "Menu Focus", labelClass: "menu", text: `${productAngle}, with low-sugar options and one rotating seasonal SKU.` },
        { label: "Go-To-Market", labelClass: "gtm", text: acquisitionPlan },
        { label: "Operating Focus", labelClass: "ops", text: operationsPlan },
        { label: "Risk Guardrail", labelClass: "risk", text: `${guardrails.join("; ")}.` },
      ];
    }

    function renderZipInsight(zipRows) {
      let row = null;
      if (state.zip !== "ALL") row = zipRows.find((z) => String(z.zip5) === state.zip) || null;
      if (!row) row = zipRows.slice().sort((a, b) => asNum(b.opportunity_score) - asNum(a.opportunity_score))[0] || null;

      if (!row) {
        el.zipHead.textContent = "No ZIP data";
        el.zipMeta.textContent = "-";
        el.zipChips.innerHTML = "";
        el.zipBars.innerHTML = "";
        el.zipFormulaLines.innerHTML = "";
        el.zipAction.innerHTML = '<div class="zip-action-text">No recommendation.</div>';
        return;
      }

      el.zipHead.textContent = `ZIP ${row.zip5} · Rank #${asNum(row.rank, 0)}`;
      el.zipMeta.textContent = `${String(row.quadrant || "NA")} · ${String(row.opportunity_tier || "Tier NA")}`;
      el.zipChips.innerHTML = [
        `Opportunity ${fmt1(row.opportunity_score)}`,
        `Shops ${fmtInt(row.active_shop_count)}`,
        `Income $${fmtInt(row.median_income)}`,
        `Rent $${fmtInt(row.median_gross_rent)}/mo`,
        `Young share ${(asNum(row.young_adult_share) * 100).toFixed(1)}%`
      ].map((x) => `<span class=\"chip\">${esc(x)}</span>`).join("");

      const bars = [
        { label: "Demand", value: asNum(row.demand_score) },
        { label: "Gap", value: asNumSafe(row.gap_component_score, Math.max(0, Math.min(100, asNum(row.gap_score) + 50))) },
        { label: "Momentum", value: asNum(row.momentum_score) },
        { label: "Text", value: asNum(row.text_signal_score) },
        { label: "Feasibility", value: asNum(row.feasibility_score, 50) },
      ];
      el.zipBars.innerHTML = bars.map((b) => {
        const pct = Math.max(0, Math.min(100, b.value));
        return `<div class=\"bar-row\"><div class=\"bar-label\">${esc(b.label)}</div><div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:${pct}%;\"></div></div><div class=\"bar-value\">${b.value.toFixed(1)}</div></div>`;
      }).join("");

      const formulaLines = [
        {
          label: "Demand",
          expr: "MinMax(0.50*Z(Population) + 0.30*Z(MedianIncome) + 0.20*Z(YoungAdultShare))",
          value: asNum(row.demand_score),
        },
        {
          label: "Gap",
          expr: "MinMax(DemandScore - SupplyScore)",
          value: asNumSafe(row.gap_component_score, Math.max(0, Math.min(100, asNum(row.gap_score) + 50))),
        },
        {
          label: "Momentum",
          expr: "MinMax(Avg Openings last 2y - baseline openings)",
          value: asNum(row.momentum_score),
        },
        {
          label: "Text",
          expr: "((MeanSentiment + 1) / 2) * 100",
          value: asNum(row.text_signal_score),
        },
        {
          label: "Feasibility",
          expr: "0.65*RegionalFeasibility + 0.35*StoreAffordability",
          value: asNum(row.feasibility_score, 50),
        },
      ];
      el.zipFormulaLines.innerHTML = formulaLines
        .map((line, idx) => (
          `<div class=\"zip-formula-line\">` +
            `<span class=\"zip-formula-idx\">${idx + 1}.</span>` +
            `<span class=\"zip-formula-text\"><span class=\"zip-formula-label\">${esc(String(line.label || ""))}:</span> ${esc(String(line.expr || ""))} = ${asNum(line.value).toFixed(1)}</span>` +
          `</div>`
        ))
        .join("");
      const strategyLines = buildZipSpecificStrategy(row);
      el.zipAction.innerHTML = strategyLines
        .map((item, idx) => {
          const label = String(item?.label || "");
          const labelClass = String(item?.labelClass || "");
          const text = String(item?.text || "");
          return (
            `<div class="zip-action-line">` +
              `<span class="zip-action-idx">${idx + 1}.</span>` +
              `<span class="zip-action-text">` +
                `<span class="zip-action-label ${esc(labelClass)}">${esc(label)}:</span> ${esc(text)}` +
              `</span>` +
            `</div>`
          );
        })
        .join("");
    }

    function renderStoreGrid(visibleStores) {
      const pageSize = getStorePageSize();
      const pages = totalStorePages(visibleStores.length);
      state.storePage = Math.max(1, Math.min(state.storePage, pages));
      const start = (state.storePage - 1) * pageSize;
      const end = Math.min(start + pageSize, visibleStores.length);
      const pageRows = visibleStores.slice(start, end);

      renderStoreSortButtons();
      el.storePageText.textContent = `${visibleStores.length} stores · page ${state.storePage}/${pages}`;
      el.storePrevBtn.disabled = state.storePage <= 1;
      el.storeNextBtn.disabled = state.storePage >= pages;

      el.storeGrid.innerHTML = pageRows.map((s) => {
        const active = String(s.store_id) === state.storeId ? " active" : "";
        const ratingStars = starRatingHtml(s.google_rating, s.google_user_ratings_total, { cls: "store-stars" });
        const logo = getStoreLogo(s);
        const fallback = String(s.logo_fallback || "");
        return `
          <button class=\"store-item${active}\" data-store-id=\"${esc(String(s.store_id || ""))}\">
            <img class=\"logo\" src=\"${esc(logo)}\" data-fallback=\"${esc(fallback)}\" alt=\"${esc(String(s.brand || "logo"))}\"
              onerror=\"if(this.dataset.fallback && this.src!==this.dataset.fallback){this.src=this.dataset.fallback;}else{this.style.visibility='hidden';}\" />
            <div>
              <div class=\"store-name\">${esc(String(s.store_name || ""))}</div>
              <div class=\"store-meta\">${esc(toTitleCaseCity(String(s.city || "")))}, ${esc(String(s.zip5 || ""))} · ${esc(categoryDisplayText(s))}</div>
              <div class=\"store-meta\">${ratingStars} · ZIP rank #${asNum(s.rank, 0)}</div>
            </div>
          </button>
        `;
      }).join("") || `<div class=\"tiny\">No stores in current filters.</div>`;
    }

    function renderStorePanel(visibleStores) {
      const store = visibleStores.find((s) => String(s.store_id) === state.storeId) || null;
      if (!store) {
        el.focusStoreLogo.src = "";
        el.focusStoreLogo.dataset.fallback = "";
        el.focusStoreName.textContent = "-";
        el.focusStoreMeta.textContent = "No store under this filter.";
        el.focusStoreRatingVisual.innerHTML = "";
        el.focusStoreChips.innerHTML = "";
        el.focusStoreLinkIcon.classList.add("hidden");
        el.focusStoreLinkIcon.removeAttribute("href");
        return;
      }
      const logo = getStoreLogo(store);
      const fallback = String(store.logo_fallback || "");
      el.focusStoreLogo.dataset.fallback = fallback;
      el.focusStoreLogo.onerror = function () {
        if (this.dataset.fallback && this.src !== this.dataset.fallback) {
          this.src = this.dataset.fallback;
        } else {
          this.style.visibility = "hidden";
        }
      };
      el.focusStoreLogo.src = logo;
      el.focusStoreLogo.style.visibility = "visible";

      el.focusStoreName.textContent = String(store.store_name || "");
      el.focusStoreMeta.textContent = `${toTitleCaseCity(String(store.city || ""))}, ${String(store.zip5 || "")} · ${categoryDisplayText(store)}`;
      el.focusStoreRatingVisual.innerHTML = starRatingHtml(store.google_rating, store.google_user_ratings_total, { cls: "focus-stars" });
      el.focusStoreChips.innerHTML = [
        `Reviews ${fmtInt(store.review_count)}`,
        `Model ${String(store.business_model || "NA")}`,
      ].map((x) => `<span class=\"chip\">${esc(x)}</span>`).join("");

      const url = String(store.matched_google_url || "").trim();
      if (url) {
        el.focusStoreLinkIcon.href = url;
        el.focusStoreLinkIcon.classList.remove("hidden");
      } else {
        el.focusStoreLinkIcon.classList.add("hidden");
        el.focusStoreLinkIcon.removeAttribute("href");
      }
    }

    function renderReviewPanel() {
      if (!state.storeId) {
        el.reviewText.textContent = "Select store to inspect one review at a time.";
        el.reviewMeta.textContent = "";
        el.reviewPager.textContent = "0 / 0";
        el.reviewKeywords.innerHTML = renderKeywordRows({ positive: [], negative: [], signal: [] });
        el.reviewTimeline.innerHTML = "";
        return;
      }

      const pool = reviewsByStore[state.storeId] || [];
      const store = storeData.find((s) => String(s.store_id) === state.storeId) || null;
      const zipRow = getZipRow(store?.zip5 || state.zip);
      const keywords = buildReviewKeywords(store, zipRow);

      if (!pool.length) {
        el.reviewText.textContent = "This store currently has no sampled review row in the curated pool.";
        el.reviewMeta.textContent = "";
        el.reviewPager.textContent = "0 / 0";
        el.reviewKeywords.innerHTML = renderKeywordRows({ positive: [], negative: [], signal: [] });
        el.reviewTimeline.innerHTML = "";
        return;
      }

      if (state.reviewIndex >= pool.length) state.reviewIndex = 0;
      if (state.reviewIndex < 0) state.reviewIndex = pool.length - 1;

      const r = pool[state.reviewIndex];
      const text = String(r.review_text || r.review_text_short || "").trim();
      const tone = classifyReviewTone(r);
      const srcUrl = String(r.source_url || "").trim();
      const srcPart = srcUrl ? ` · <a class=\"link\" href=\"${esc(srcUrl)}\" target=\"_blank\" rel=\"noopener noreferrer\">source</a>` : "";
      const keywordGroups = buildKeywordGroupsForReview(text, keywords);

      el.reviewText.innerHTML = highlightReviewText(text, keywordGroups);
      el.reviewMeta.innerHTML = `<span class=\"review-tone ${tone.cls}\">${tone.label}</span> · ${esc(String(r.review_date || "NA"))} · Rating ${esc(String(r.rating ?? "NA"))} · ${esc(String(r.source_platform || "NA"))}${srcPart}`;
      el.reviewPager.textContent = `${state.reviewIndex + 1} / ${pool.length}`;
      el.reviewKeywords.innerHTML = renderKeywordRows(keywordGroups);

      el.reviewTimeline.innerHTML = pool.map((item, idx) => {
        const active = idx === state.reviewIndex ? " active" : "";
        const itemTone = classifyReviewTone(item);
        const d = String(item.review_date || "NA");
        const rt = String(item.rating ?? "NA");
        return `
          <button class=\"timeline-item${active}\" data-review-idx=\"${idx}\">
            <div class=\"timeline-head\"><span>${esc(d)} · ${esc(rt)}★</span><span class=\"review-tone ${itemTone.cls}\">${itemTone.label}</span></div>
            <div class=\"timeline-snippet\">${esc(snippet(item.review_text || item.review_text_short || ""))}</div>
          </button>
        `;
      }).join("");
    }

    function renderScenarioButtons() {
      const nodes = el.scenarioRow.querySelectorAll(".scenario");
      nodes.forEach((n) => n.classList.toggle("active", n.dataset.scenario === state.scenario));
    }

    function renderAll() {
      syncState();
      const filterStores = getFilterStores();
      const visibleStores = getVisibleStores();
      const zipRows = getFilteredZipRows(filterStores);

      renderKpis(visibleStores, zipRows);
      renderOpportunityHeatmap(zipRows);
      renderGapScatter(zipRows);
      renderZipInsight(zipRows);
      renderMiniChart(filterStores, visibleStores, zipRows);
      renderStoreGrid(visibleStores);
      renderStorePanel(visibleStores);
      renderReviewPanel();
      renderStoreMap(visibleStores);
      renderScenarioButtons();
      bindPlotClickOnce();
    }

    function focusZip(zip, options = {}) {
      if (!zip) return;
      state.zip = String(zip);
      if (options && options.applyFilter === true) {
        state.zipFilters = [state.zip];
      }
      state.storePage = 1;
      state.storePageManual = false;
      state.storeId = null;
      state.reviewIndex = 0;
      applyZipViewport(state.zip);
      renderFilters();
      renderAll();
    }

    function applyScenario(key) {
      closeCategoryMenu();
      closeZipMenu();
      state.scenario = key;
      state.storePageManual = false;
      if (key === "top") {
        state.city = "ALL";
        state.zipFilters = [];
        state.categories = [];
        state.brand = "ALL";
        state.search = "";
        const top = zipData.slice().sort((a, b) => asNum(b.opportunity_score) - asNum(a.opportunity_score))[0];
        state.zip = top ? String(top.zip5) : "ALL";
        state.zipFilters = state.zip !== "ALL" ? [state.zip] : [];
      } else if (key === "fruit") {
        state.city = "ALL";
        state.zipFilters = [];
        state.brand = "ALL";
        const opts = allCategoryOptions(storeData);
        const target = opts.find((x) => x.toLowerCase().includes("fruittea")) || opts.find((x) => x.toLowerCase().includes("fruit tea")) || opts.find((x) => x.toLowerCase().includes("fruit")) || "ALL";
        state.categories = target === "ALL" ? [] : [target];
        state.zip = "ALL";
      } else if (key === "gap") {
        state.city = "ALL";
        state.zipFilters = [];
        state.categories = [];
        state.brand = "ALL";
        state.search = "";
        const cand = zipData
          .filter((z) => String(z.quadrant || "") === "HighDemand_LowSupply")
          .sort((a, b) => asNum(b.gap_score) - asNum(a.gap_score))[0];
        state.zip = cand ? String(cand.zip5) : "ALL";
        state.zipFilters = state.zip !== "ALL" ? [state.zip] : [];
      }
      state.storePage = 1;
      state.storeId = null;
      state.reviewIndex = 0;
      if (state.zip !== "ALL") {
        applyZipViewport(state.zip);
      } else {
        state.mapCenterLat = countyViewport.lat;
        state.mapCenterLon = countyViewport.lon;
        state.mapZoom = countyViewport.zoom;
      }
      renderFilters();
      renderAll();
    }

    function resetAllFilters() {
      closeCategoryMenu();
      closeZipMenu();
      state.city = "ALL";
      state.zipFilters = [];
      state.categories = [];
      state.brand = "ALL";
      state.zip = "ALL";
      state.search = "";
      state.scenario = null;
      state.storePage = 1;
      state.storePageManual = false;
      state.storeSortKey = "";
      state.storeSortDir = "off";
      state.storeId = null;
      state.reviewIndex = 0;
      state.miniTab = "brand";
      state.mapCenterLat = countyViewport.lat;
      state.mapCenterLon = countyViewport.lon;
      state.mapZoom = countyViewport.zoom;
      renderFilters();
      renderAll();
      setMapViewport(countyViewport.lat, countyViewport.lon, countyViewport.zoom);
    }

    function bindUi() {
      el.cityFilter.addEventListener("change", (e) => {
        state.city = e.target.value;
        state.scenario = null;
        state.storePage = 1;
        state.storePageManual = false;
        state.reviewIndex = 0;
        closeZipMenu();
        closeCategoryMenu();
        renderAll();
      });
      el.zipFilterBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        closeCategoryMenu();
        toggleZipMenu();
      });
      el.zipFilterMenu.addEventListener("click", (e) => {
        const actionBtn = e.target.closest("[data-action]");
        const optionBtn = e.target.closest(".multi-option");
        const zipOptions = allZipOptions();

        if (actionBtn) {
          const action = String(actionBtn.dataset.action || "");
          if (action === "all") state.zipFilters = zipOptions.slice();
          if (action === "clear") state.zipFilters = [];
        } else if (optionBtn) {
          const zip = String(optionBtn.dataset.zip || "");
          if (!zip) return;
          const set = new Set(state.zipFilters || []);
          if (set.has(zip)) set.delete(zip);
          else set.add(zip);
          state.zipFilters = zipOptions.filter((z) => set.has(z));
        } else {
          return;
        }
        state.scenario = null;
        state.storePage = 1;
        state.storePageManual = false;
        state.reviewIndex = 0;
        const keepOpen = isZipMenuOpen();
        renderFilters();
        if (keepOpen) openZipMenu();
        renderAll();
      });
      el.categoryFilterBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        closeZipMenu();
        toggleCategoryMenu();
      });
      el.categoryFilterMenu.addEventListener("click", (e) => {
        const actionBtn = e.target.closest("[data-action]");
        const optionBtn = e.target.closest(".multi-option");
        const catOptions = allCategoryOptions(storeData);

        if (actionBtn) {
          const action = String(actionBtn.dataset.action || "");
          if (action === "all") state.categories = catOptions.slice();
          if (action === "clear") state.categories = [];
        } else if (optionBtn) {
          const cat = String(optionBtn.dataset.category || "");
          if (!cat) return;
          const set = new Set(state.categories);
          if (set.has(cat)) set.delete(cat);
          else set.add(cat);
          state.categories = catOptions.filter((c) => set.has(c));
        } else {
          return;
        }
        state.scenario = null;
        state.storePage = 1;
        state.storePageManual = false;
        state.reviewIndex = 0;
        const keepOpen = isCategoryMenuOpen();
        renderFilters();
        if (keepOpen) openCategoryMenu();
        renderAll();
      });
      document.addEventListener("click", (e) => {
        if (isZipMenuOpen() && el.zipFilterWrap && !el.zipFilterWrap.contains(e.target)) {
          closeZipMenu();
        }
        if (isCategoryMenuOpen() && el.categoryFilterWrap && !el.categoryFilterWrap.contains(e.target)) {
          closeCategoryMenu();
        }
      });
      window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          closeZipMenu();
          closeCategoryMenu();
        }
      });
      el.brandFilter.addEventListener("change", (e) => {
        state.brand = e.target.value;
        state.scenario = null;
        state.storePage = 1;
        state.storePageManual = false;
        state.reviewIndex = 0;
        closeZipMenu();
        closeCategoryMenu();
        renderAll();
      });
      el.storeSearch.addEventListener("input", (e) => {
        state.search = e.target.value || "";
        state.storePage = 1;
        state.storePageManual = false;
        state.reviewIndex = 0;
        closeZipMenu();
        closeCategoryMenu();
        renderAll();
      });
      el.resetFocusBtn.addEventListener("click", () => {
        resetAllFilters();
      });

      el.scenarioRow.addEventListener("click", (e) => {
        const btn = e.target.closest(".scenario");
        if (!btn) return;
        applyScenario(btn.dataset.scenario || "top");
      });

      el.miniTabs.addEventListener("click", (e) => {
        const btn = e.target.closest(".tab-btn");
        if (!btn) return;
        state.miniTab = btn.dataset.tab || "brand";
        renderAll();
      });

      el.storeGrid.addEventListener("click", (e) => {
        const card = e.target.closest(".store-item");
        if (!card) return;
        state.storeId = card.dataset.storeId || null;
        state.storePageManual = true;
        state.reviewIndex = 0;
        renderAll();
      });
      el.storeSortRow.addEventListener("click", (e) => {
        const btn = e.target.closest(".store-sort-btn");
        if (!btn) return;
        const key = String(btn.dataset.sortKey || "");
        if (!key) return;
        cycleStoreSort(key);
      });
      el.storePrevBtn.addEventListener("click", () => {
        state.storePageManual = true;
        state.storePage = Math.max(1, state.storePage - 1);
        renderAll();
      });
      el.storeNextBtn.addEventListener("click", () => {
        state.storePageManual = true;
        state.storePage = state.storePage + 1;
        renderAll();
      });

      el.reviewPrevBtn.addEventListener("click", () => {
        state.reviewIndex -= 1;
        renderReviewPanel();
      });
      el.reviewNextBtn.addEventListener("click", () => {
        state.reviewIndex += 1;
        renderReviewPanel();
      });
      el.reviewTimeline.addEventListener("click", (e) => {
        const node = e.target.closest(".timeline-item");
        if (!node) return;
        state.reviewIndex = asNum(node.dataset.reviewIdx, 0);
        renderReviewPanel();
      });

      el.mapLocateBtn.addEventListener("click", () => {
        if (!navigator?.geolocation) return;
        navigator.geolocation.getCurrentPosition(
          (pos) => {
            const lat = Math.round(asNum(pos?.coords?.latitude) * 20) / 20;
            const lon = Math.round(asNum(pos?.coords?.longitude) * 20) / 20;
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
            setMapViewport(lat, lon, 10.2);
          },
          () => {
            setMapViewport(countyViewport.lat, countyViewport.lon, countyViewport.zoom);
          },
          { enableHighAccuracy: false, timeout: 8500, maximumAge: 120000 }
        );
      });
      el.mapCountyBtn.addEventListener("click", () => {
        setMapViewport(countyViewport.lat, countyViewport.lon, countyViewport.zoom);
      });
      el.mapZoomInBtn.addEventListener("click", () => {
        const rows = getStoresWithCoordinates(getVisibleStores());
        const curr = getCurrentMapViewport(rows);
        setMapViewport(curr.lat, curr.lon, curr.zoom + 0.65);
      });
      el.mapZoomOutBtn.addEventListener("click", () => {
        const rows = getStoresWithCoordinates(getVisibleStores());
        const curr = getCurrentMapViewport(rows);
        setMapViewport(curr.lat, curr.lon, curr.zoom - 0.65);
      });

      let resizeTimer = null;
      window.addEventListener("resize", () => {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => renderAll(), 120);
      });
    }

    function bindPlotClickOnce() {
      if (!HAS_PLOTLY) return;

      const opp = document.getElementById("oppHeatmap");
      if (opp && opp.on && !opp.dataset.bound) {
        opp.on("plotly_click", (ev) => {
          const y = String(ev?.points?.[0]?.y || "");
          const match = y.match(new RegExp("([0-9]{5})"));
          const zip = match ? match[1] : "";
          if (!zip) return;
          state.scenario = null;
          focusZip(zip);
        });
        opp.dataset.bound = "1";
      }

      const scatter = document.getElementById("gapScatter");
      if (scatter && scatter.on && !scatter.dataset.bound) {
        scatter.on("plotly_click", (ev) => {
          const zip = String(ev?.points?.[0]?.customdata?.[0] || "");
          if (!zip) return;
          state.scenario = null;
          focusZip(zip);
        });
        scatter.dataset.bound = "1";
      }

      const map = document.getElementById("storeMap");
      if (map && map.on && !map.dataset.bound) {
        map.on("plotly_click", (ev) => {
          const storeId = String(ev?.points?.[0]?.customdata?.[0] || "");
          if (!storeId) return;
          state.storeId = storeId;
          state.storePageManual = false;
          state.reviewIndex = 0;
          const s = getStoreById(storeId);
          if (s && s.zip5) {
            state.zip = String(s.zip5);
            state.scenario = null;
            applyZipViewport(state.zip);
          }
          renderAll();
        });
        map.on("plotly_relayout", (ev) => {
          if (!ev || typeof ev !== "object") return;
          const lat = asNum(ev["mapbox.center.lat"], asNum(state.mapCenterLat, NaN));
          const lon = asNum(ev["mapbox.center.lon"], asNum(state.mapCenterLon, NaN));
          const zoom = asNum(ev["mapbox.zoom"], asNum(state.mapZoom, NaN));
          if (Number.isFinite(lat)) state.mapCenterLat = lat;
          if (Number.isFinite(lon)) state.mapCenterLon = lon;
          if (Number.isFinite(zoom)) state.mapZoom = clampMapZoom(zoom);
        });
        map.dataset.bound = "1";
      }
    }

    function bootstrap() {
      el.generatedAt.textContent = String(DATA.meta?.generated_at || "-");
      el.scopeText.textContent = String(DATA.meta?.scope || "King County, WA");
      el.metaRows.textContent = `${fmtInt(DATA.meta?.zips_total || zipData.length)} ZIPs · ${fmtInt(DATA.meta?.stores_total || storeData.length)} stores · ${fmtInt(DATA.meta?.review_rows_total || reviewData.length)} reviews`;

      state.mapCenterLat = countyViewport.lat;
      state.mapCenterLon = countyViewport.lon;
      state.mapZoom = countyViewport.zoom;
      renderFilters();
      renderAll();
      bindUi();
    }

    bootstrap();
  </script>
</body>
</html>
"""


def write_dashboard(payload: dict[str, Any], output_path: Path, hero_logo_uri: str) -> None:
    html = dashboard_template()
    html = html.replace("__HERO_LOGO_URI__", hero_logo_uri)
    html = html.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    payload = build_payload(args, root)
    hero_logo_uri = resolve_hero_logo_data_uri(root, args.hero_logo_input)
    out = root / args.output
    write_dashboard(payload, out, hero_logo_uri)
    print(f"[ok] Dashboard v7 generated: {out}")
    print(
        "[meta] "
        f"zips={payload['meta']['zips_total']} "
        f"stores={payload['meta']['stores_total']} "
        f"reviews={payload['meta']['review_rows_total']} "
        f"coords={payload['meta']['stores_with_coordinates']}"
    )


if __name__ == "__main__":
    main()
