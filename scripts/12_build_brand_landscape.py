#!/usr/bin/env python3
"""Build King County bubble-tea brand landscape from shop master."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shop-input",
        default="data/interim/supply_shop_master.csv",
        help="Shop-level input from script 02.",
    )
    parser.add_argument(
        "--taxonomy-input",
        default="configs/brand_taxonomy.csv",
        help="Brand mapping taxonomy CSV.",
    )
    parser.add_argument(
        "--brand-output",
        default="data/processed/brand_landscape_king_county.csv",
        help="Aggregated brand landscape output.",
    )
    parser.add_argument(
        "--unmapped-output",
        default="data/interim/brand_unmapped_shops.csv",
        help="Unmapped shops output for manual review.",
    )
    parser.add_argument(
        "--brand-city-output",
        default="outputs/tables/brand_city_presence.csv",
        help="Brand x city presence table for slides.",
    )
    parser.add_argument(
        "--shop-brand-output",
        default="outputs/tables/shop_brand_category_king_county.csv",
        help="Shop-level brand/category table output.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_brand_id(brand: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", str(brand).lower()).strip("_")
    return clean or "unknown_brand"


def to_level1_category(raw: str) -> str:
    """Collapse detailed taxonomy category into Level-1 category."""
    s = str(raw or "").strip()
    if not s:
        return "Unknown"

    token = s.split("_")[0].strip()
    key = re.sub(r"[^a-z0-9]+", "", token.lower())
    mapping = {
        "milktea": "MilkTea",
        "fruittea": "FruitTea",
        "puretea": "PureTea",
        "cheesetea": "CheeseTea",
        "matcha": "Matcha",
        # "Tea" is intentionally not a standalone class at Level 1.
        # If only generic tea signal exists, keep it Unknown and let
        # menu-evidence script decide specific tags.
        "tea": "Unknown",
        "mixed": "Unknown",
        "unknown": "Unknown",
    }
    if key in mapping:
        return mapping[key]

    # Fallback heuristics for irregular labels.
    lower = s.lower()
    if "matcha" in lower:
        return "Matcha"
    if "cheese" in lower and "tea" in lower:
        return "CheeseTea"
    if "milk" in lower and "tea" in lower:
        return "MilkTea"
    if "fruit" in lower and "tea" in lower:
        return "FruitTea"
    if "pure" in lower and "tea" in lower:
        return "PureTea"
    if "tea" in lower:
        return "Unknown"
    return "Unknown"


def load_inputs(shop_path: Path, taxonomy_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not shop_path.exists():
        raise FileNotFoundError(f"Shop input not found: {shop_path}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy input not found: {taxonomy_path}")

    shop = pd.read_csv(shop_path, dtype=str)
    tax = pd.read_csv(taxonomy_path, dtype=str)
    return shop, tax


def assign_brand(shop: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    out = shop.copy()
    out["trade_name"] = out["trade_name"].fillna("").astype(str)
    out["city"] = out["city"].fillna("").astype(str).str.upper().str.strip()
    out["zip5"] = out["zip5"].fillna("").astype(str).str.extract(r"(\d{5})", expand=False)
    out["license_start_year"] = pd.to_numeric(out.get("license_start_year"), errors="coerce")
    out["text_for_match"] = (
        out["trade_name"].fillna("")
        + " "
        + out.get("business_legal_name", pd.Series("", index=out.index)).fillna("")
    ).str.lower()

    out["brand"] = "Independent/Other"
    out["origin_region"] = "Mixed"
    out["positioning"] = "Long-tail independent shops"
    out["business_model"] = "Independent"
    out["primary_beverage_category"] = "Unknown"
    out["hero_products"] = "n/a"
    out["evidence_url"] = ""

    for _, row in taxonomy.iterrows():
        brand = str(row["brand"]).strip()
        pattern = str(row["regex_pattern"]).strip()
        if not brand or not pattern:
            continue
        mask = out["text_for_match"].str.contains(re.compile(pattern, flags=re.IGNORECASE), regex=True)
        out.loc[mask, "brand"] = brand
        out.loc[mask, "origin_region"] = str(row["origin_region"]).strip()
        out.loc[mask, "positioning"] = str(row["positioning"]).strip()
        out.loc[mask, "business_model"] = str(row["business_model"]).strip()
        out.loc[mask, "primary_beverage_category"] = to_level1_category(
            str(row.get("primary_beverage_category", "Unknown")).strip()
        )
        out.loc[mask, "hero_products"] = str(row.get("hero_products", "n/a")).strip()
        out.loc[mask, "evidence_url"] = str(row.get("evidence_url", "")).strip()

    out["primary_beverage_category"] = out["primary_beverage_category"].map(to_level1_category)
    return out


def build_brand_landscape(assigned: pd.DataFrame) -> pd.DataFrame:
    agg = (
        assigned.groupby(
            [
                "brand",
                "origin_region",
                "positioning",
                "business_model",
                "primary_beverage_category",
                "hero_products",
            ],
            as_index=False,
        )
        .agg(
            shop_count=("shop_key", "nunique"),
            city_count=("city", "nunique"),
            zip_count=("zip5", "nunique"),
            first_seen_year=("license_start_year", "min"),
            latest_seen_year=("license_start_year", "max"),
        )
        .sort_values(["shop_count", "city_count", "zip_count"], ascending=False)
        .reset_index(drop=True)
    )
    agg["shop_share_pct"] = (agg["shop_count"] / agg["shop_count"].sum() * 100.0).round(2)
    agg["rank"] = agg.index + 1
    cols = [
        "rank",
        "brand",
        "origin_region",
        "positioning",
        "business_model",
        "primary_beverage_category",
        "hero_products",
        "shop_count",
        "shop_share_pct",
        "city_count",
        "zip_count",
        "first_seen_year",
        "latest_seen_year",
    ]
    return agg[cols]


def build_brand_city_presence(assigned: pd.DataFrame) -> pd.DataFrame:
    city_counts = (
        assigned.groupby(["brand", "city"], as_index=False)
        .agg(shop_count=("shop_key", "nunique"))
        .sort_values(["shop_count", "brand", "city"], ascending=[False, True, True])
    )
    return city_counts.reset_index(drop=True)


def build_unmapped(assigned: pd.DataFrame) -> pd.DataFrame:
    unknown = assigned[assigned["brand"] == "Independent/Other"].copy()
    keep = [
        "trade_name",
        "business_legal_name",
        "city",
        "zip5",
        "license_start_year",
        "primary_beverage_category",
    ]
    for col in keep:
        if col not in unknown.columns:
            unknown[col] = pd.NA
    return unknown[keep].sort_values(["city", "zip5", "trade_name"]).reset_index(drop=True)


def build_shop_brand_table(assigned: pd.DataFrame) -> pd.DataFrame:
    out = assigned.copy()
    out["store_id"] = out.get("shop_key", pd.Series("", index=out.index)).astype(str)
    out["brand_id"] = out["brand"].apply(make_brand_id)
    out["store_name"] = out.get("trade_name", pd.Series("", index=out.index)).astype(str)

    keep = [
        "store_id",
        "brand_id",
        "shop_key",
        "store_name",
        "trade_name",
        "business_legal_name",
        "street_address",
        "city",
        "zip5",
        "license_start_year",
        "brand",
        "origin_region",
        "primary_beverage_category",
        "hero_products",
        "positioning",
        "business_model",
        "evidence_url",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].sort_values(
        ["brand", "store_name", "city", "zip5"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = project_root()
    shop_path = root / args.shop_input
    tax_path = root / args.taxonomy_input
    brand_out = root / args.brand_output
    unmapped_out = root / args.unmapped_output
    city_out = root / args.brand_city_output
    shop_brand_out = root / args.shop_brand_output

    shop, tax = load_inputs(shop_path, tax_path)
    assigned = assign_brand(shop, tax)
    brand_df = build_brand_landscape(assigned)
    city_df = build_brand_city_presence(assigned)
    unknown_df = build_unmapped(assigned)
    shop_brand_df = build_shop_brand_table(assigned)

    ensure_parent(brand_out)
    ensure_parent(unmapped_out)
    ensure_parent(city_out)
    ensure_parent(shop_brand_out)
    brand_df.to_csv(brand_out, index=False)
    unknown_df.to_csv(unmapped_out, index=False)
    city_df.to_csv(city_out, index=False)
    shop_brand_df.to_csv(shop_brand_out, index=False)

    print(
        "[12] Completed. "
        f"brands={brand_df.shape[0]}, shops={assigned['shop_key'].nunique()}, "
        f"top_brand={brand_df.iloc[0]['brand'] if not brand_df.empty else 'n/a'}"
    )
    print(f"[12] brand landscape: {brand_out}")
    print(f"[12] brand city presence: {city_out}")
    print(f"[12] shop brand-category table: {shop_brand_out}")
    print(f"[12] unmapped shops: {unmapped_out}")


if __name__ == "__main__":
    main()
