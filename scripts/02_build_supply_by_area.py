#!/usr/bin/env python3
"""Build supply dataset by district.
Input: data/raw/king_county_food_inspections.csv (preferred) or Seattle legacy license CSV
Output: data/interim/supply_shop_master.csv, data/processed/supply_by_district.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


BOBA_PATTERN = re.compile(
    r"\bboba\b|bubble\s*tea|milk\s*tea|milktea|tapioca|fruit\s*tea|thai\s*tea|"
    r"cheese\s*tea|tea\s*zone|brown\s*sugar",
    flags=re.IGNORECASE,
)
BRAND_PATTERN = re.compile(
    r"share\s*tea|gong\s*cha|ding\s*tea|kung\s*fu\s*tea|coco\s*fresh|hey\s*tea|heytea|"
    r"yi\s*fang|yifang|tp\s*tea|sunright|happy\s*lemon|teazzi|t4\s*tea|the\s*alley|"
    r"come\s*buy|tab\s*milk\s*tea|tea\s*addicts|macu\s*tea|molly\s*tea|i-?tea|bobae",
    flags=re.IGNORECASE,
)
EXCLUDE_PATTERN = re.compile(
    r"tree\s*oil|tea\s*tree|transport|limousine|transit|trucking|construction|"
    r"plumbing|electrical|consulting|insurance|real\s*estate|law|attorney|clinic|"
    r"dental|salon|barber|auto\s*repair|tea\s*room|afternoon\s*tea|peet'?s|zoka|"
    r"microsoft.*tea\s*house",
    flags=re.IGNORECASE,
)
KING_HINT_COLUMNS = {"inspection_date", "city", "zip_code", "business_id"}
SEATTLE_HINT_COLUMNS = {
    "trade_name",
    "business_legal_name",
    "naics_description",
    "license_start_date",
    "street_address",
    "city",
    "zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/raw/king_county_food_inspections.csv",
        help="Input raw license CSV path relative to project root.",
    )
    parser.add_argument(
        "--shop-master-output",
        default="data/interim/supply_shop_master.csv",
        help="Output filtered shop-level dataset.",
    )
    parser.add_argument(
        "--supply-output",
        default="data/processed/supply_by_district.csv",
        help="Output area-level supply aggregate.",
    )
    parser.add_argument(
        "--area-column",
        default="zip5",
        choices=["zip5"],
        help="Area aggregation key for MVP.",
    )
    parser.add_argument(
        "--taxonomy-input",
        default="configs/brand_taxonomy.csv",
        help="Brand taxonomy file with regex_pattern column.",
    )
    parser.add_argument(
        "--manual-additions-input",
        default="configs/manual_shop_additions.csv",
        help="Optional manual shop additions CSV to append before final de-dup.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_zip(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.extract(r"(\d{5})", expand=False)
        .fillna("")
    )


def load_taxonomy_brand_pattern(taxonomy_path: Path) -> re.Pattern[str] | None:
    if not taxonomy_path.exists():
        return None
    tax = pd.read_csv(taxonomy_path, dtype=str).fillna("")
    if "regex_pattern" not in tax.columns:
        return None
    patterns = [str(p).strip() for p in tax["regex_pattern"].tolist() if str(p).strip()]
    if not patterns:
        return None
    merged = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(merged, flags=re.IGNORECASE)


def load_manual_additions(path: Path) -> pd.DataFrame:
    keep_cols = [
        "shop_key",
        "trade_name",
        "business_legal_name",
        "naics_code",
        "naics_description",
        "license_start_date",
        "license_start_year",
        "street_address",
        "city",
        "state",
        "zip5",
        "business_id",
        "city_account_number",
        "match_boba_keyword",
        "match_brand_keyword",
        "match_taxonomy_keyword",
        "source_dataset",
    ]
    if not path.exists():
        return pd.DataFrame(columns=keep_cols)

    raw = pd.read_csv(path, dtype=str).fillna("")
    for col in [
        "store_id",
        "trade_name",
        "business_legal_name",
        "street_address",
        "city",
        "state",
        "zip5",
        "business_id",
        "license_start_date",
        "naics_code",
        "naics_description",
        "city_account_number",
    ]:
        if col not in raw.columns:
            raw[col] = ""

    out = pd.DataFrame(index=raw.index)
    out["shop_key"] = raw["store_id"].fillna("").astype(str).str.strip()
    out["trade_name"] = raw["trade_name"].fillna("").astype(str).str.strip()
    out["business_legal_name"] = raw["business_legal_name"].fillna("").astype(str).str.strip()
    empty_legal = out["business_legal_name"] == ""
    out.loc[empty_legal, "business_legal_name"] = out.loc[empty_legal, "trade_name"]
    out["naics_code"] = raw["naics_code"].fillna("").astype(str).str.strip()
    out["naics_description"] = raw["naics_description"].fillna("").astype(str).str.strip()
    out["street_address"] = raw["street_address"].fillna("").astype(str).str.strip()
    out["city"] = raw["city"].fillna("").astype(str).str.upper().str.strip()
    out["state"] = raw["state"].fillna("WA").astype(str).str.upper().str.strip().replace("", "WA")
    out["zip5"] = clean_zip(raw["zip5"])
    out["business_id"] = raw["business_id"].fillna("").astype(str).str.strip()
    out["city_account_number"] = raw["city_account_number"].fillna("").astype(str).str.strip()
    out["license_start_date"] = pd.to_datetime(raw["license_start_date"], errors="coerce")
    out["license_start_year"] = out["license_start_date"].dt.year
    out["match_boba_keyword"] = True
    out["match_brand_keyword"] = True
    out["match_taxonomy_keyword"] = True
    out["source_dataset"] = "manual_additions"

    no_key = out["shop_key"] == ""
    with_bid = out["business_id"] != ""
    out.loc[no_key & with_bid, "shop_key"] = "BID|" + out.loc[no_key & with_bid, "business_id"]

    still_no_key = out["shop_key"] == ""
    out.loc[still_no_key, "shop_key"] = (
        out.loc[still_no_key, "trade_name"].str.upper().str.strip()
        + "|"
        + out.loc[still_no_key, "street_address"].str.upper().str.strip()
        + "|"
        + out.loc[still_no_key, "zip5"].astype(str)
    )

    out = out[(out["trade_name"] != "") & (out["zip5"] != "")].copy()
    out = out.drop_duplicates(subset=["shop_key"], keep="first")
    for col in keep_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep_cols]


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing all candidate columns: {list(candidates)}")


def detect_schema(df: pd.DataFrame) -> str:
    if KING_HINT_COLUMNS.issubset(df.columns) and (
        "inspection_business_name" in df.columns or "name" in df.columns
    ):
        return "king_county_inspections"
    if SEATTLE_HINT_COLUMNS.issubset(df.columns):
        return "seattle_business_license"
    raise ValueError("Unsupported schema for supply build. Use script 01 outputs.")


def _build_from_king_county(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    biz_name_col = _first_existing(df, ["inspection_business_name", "name"])
    out["business_legal_name"] = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    out["trade_name"] = df[biz_name_col].fillna("").astype(str)
    fallback_name = out["business_legal_name"].fillna("").astype(str)
    empty_trade = out["trade_name"].str.strip() == ""
    out.loc[empty_trade, "trade_name"] = fallback_name[empty_trade]
    out["naics_code"] = pd.NA
    out["naics_description"] = df.get("program_identifier", pd.Series("", index=df.index)).fillna("").astype(str)
    out["street_address"] = df.get("address", pd.Series("", index=df.index)).fillna("").astype(str)
    out["city"] = df.get("city", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.strip()
    out["state"] = "WA"
    out["zip5"] = clean_zip(df.get("zip_code", pd.Series("", index=df.index)))
    out["business_id"] = df.get("business_id", pd.Series("", index=df.index)).fillna("").astype(str)
    out["city_account_number"] = pd.NA
    out["license_start_date"] = pd.to_datetime(
        df.get("inspection_date", pd.Series("", index=df.index)),
        errors="coerce",
    )
    return out


def _build_from_seattle(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["trade_name"] = df.get("trade_name", pd.Series("", index=df.index)).fillna("").astype(str)
    out["business_legal_name"] = (
        df.get("business_legal_name", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["naics_code"] = df.get("naics_code", pd.Series("", index=df.index)).fillna("").astype(str)
    out["naics_description"] = (
        df.get("naics_description", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["street_address"] = (
        df.get("street_address", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["city"] = df.get("city", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.strip()
    out["state"] = df.get("state", pd.Series("WA", index=df.index)).fillna("WA").astype(str)
    out["zip5"] = clean_zip(df.get("zip", pd.Series("", index=df.index)))
    out["business_id"] = pd.NA
    out["city_account_number"] = (
        df.get("city_account_number", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["license_start_date"] = pd.to_datetime(
        df.get("license_start_date", pd.Series("", index=df.index)),
        errors="coerce",
        format="%Y%m%d",
    )
    return out


def build_shop_master(
    df: pd.DataFrame,
    taxonomy_brand_pattern: re.Pattern[str] | None = None,
    manual_additions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    schema = detect_schema(df)
    if schema == "king_county_inspections":
        shops = _build_from_king_county(df)
    else:
        shops = _build_from_seattle(df)

    shops = shops[shops["zip5"] != ""].copy()

    text_blob = (
        shops["trade_name"] + " " + shops["business_legal_name"] + " " + shops["naics_description"]
    )
    boba_mask = text_blob.str.contains(BOBA_PATTERN, regex=True)
    brand_mask = text_blob.str.contains(BRAND_PATTERN, regex=True)
    taxonomy_mask = (
        text_blob.str.contains(taxonomy_brand_pattern, regex=True)
        if taxonomy_brand_pattern is not None
        else pd.Series(False, index=shops.index)
    )
    exclude_mask = text_blob.str.contains(EXCLUDE_PATTERN, regex=True)
    shops["is_boba_candidate"] = (boba_mask | brand_mask | taxonomy_mask) & (~exclude_mask)
    shops["match_boba_keyword"] = boba_mask
    shops["match_brand_keyword"] = brand_mask
    shops["match_taxonomy_keyword"] = taxonomy_mask
    shops = shops[shops["is_boba_candidate"]].copy()

    if manual_additions is not None and not manual_additions.empty:
        shops = pd.concat([shops, manual_additions], ignore_index=True, sort=False)

    shops["license_start_year"] = shops["license_start_date"].dt.year
    shops["business_id_clean"] = shops["business_id"].fillna("").astype(str).str.strip()

    if "shop_key" not in shops.columns:
        shops["shop_key"] = ""
    shops["shop_key"] = shops["shop_key"].fillna("").astype(str)
    missing_key = shops["shop_key"].str.strip() == ""
    shops.loc[missing_key, "shop_key"] = (
        shops.loc[missing_key, "trade_name"].str.upper().str.strip()
        + "|"
        + shops.loc[missing_key, "street_address"].str.upper().str.strip()
        + "|"
        + shops.loc[missing_key, "zip5"].astype(str)
    )
    has_bid = missing_key & (shops["business_id_clean"] != "")
    shops.loc[has_bid, "shop_key"] = "BID|" + shops.loc[has_bid, "business_id_clean"]
    shops = shops.sort_values(["shop_key", "license_start_date"], na_position="last")
    shops = shops.drop_duplicates(subset=["shop_key"], keep="first").copy()
    if "source_dataset" not in shops.columns:
        shops["source_dataset"] = schema
    else:
        shops["source_dataset"] = shops["source_dataset"].fillna("").astype(str).str.strip()
        shops["source_dataset"] = shops["source_dataset"].where(shops["source_dataset"] != "", schema)

    keep_cols = [
        "shop_key",
        "trade_name",
        "business_legal_name",
        "naics_code",
        "naics_description",
        "license_start_date",
        "license_start_year",
        "street_address",
        "city",
        "state",
        "zip5",
        "business_id",
        "city_account_number",
        "match_boba_keyword",
        "match_brand_keyword",
        "match_taxonomy_keyword",
        "source_dataset",
    ]
    for col in keep_cols:
        if col not in shops.columns:
            shops[col] = pd.NA
    out = shops[keep_cols].sort_values(["zip5", "trade_name"]).reset_index(drop=True)
    return out, schema


def aggregate_supply(shop_df: pd.DataFrame, area_column: str) -> pd.DataFrame:
    agg = (
        shop_df.groupby(area_column, dropna=False)
        .agg(
            active_shop_count=("shop_key", "nunique"),
            unique_trade_names=("trade_name", "nunique"),
            naics_diversity=("naics_description", "nunique"),
            median_license_start_year=("license_start_year", "median"),
        )
        .reset_index()
        .rename(columns={area_column: "district_id"})
    )
    agg["district_level"] = area_column
    return agg.sort_values("active_shop_count", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = project_root()
    inp = root / args.input
    shop_out = root / args.shop_master_output
    supply_out = root / args.supply_output
    taxonomy_path = root / args.taxonomy_input
    manual_additions_path = root / args.manual_additions_input

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    raw_df = pd.read_csv(inp, dtype=str)
    df = normalize_columns(raw_df)
    taxonomy_brand_pattern = load_taxonomy_brand_pattern(taxonomy_path)
    manual_additions_df = load_manual_additions(manual_additions_path)
    shop_df, schema = build_shop_master(
        df,
        taxonomy_brand_pattern=taxonomy_brand_pattern,
        manual_additions=manual_additions_df,
    )
    supply_df = aggregate_supply(shop_df, args.area_column)

    ensure_parent(shop_out)
    ensure_parent(supply_out)
    shop_df.to_csv(shop_out, index=False)
    supply_df.to_csv(supply_out, index=False)

    print(
        "[02] Completed. "
        f"schema={schema}, shop_rows={shop_df.shape[0]}, district_rows={supply_df.shape[0]}, "
        f"manual_additions_loaded={manual_additions_df.shape[0]}"
    )
    print(f"[02] shop master: {shop_out}")
    print(f"[02] supply by district: {supply_out}")


if __name__ == "__main__":
    main()
