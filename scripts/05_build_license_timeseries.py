#!/usr/bin/env python3
"""Build district license open/close momentum time-series.
Input: data/raw/king_county_food_inspections.csv (preferred) or Seattle legacy license CSV
Output: data/processed/license_timeseries_by_district.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
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
        help="Raw license CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/license_timeseries_by_district.csv",
        help="Output time-series table path.",
    )
    parser.add_argument(
        "--figure-output",
        default="outputs/figures/fig_license_timeseries.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="Minimum year in output.",
    )
    parser.add_argument(
        "--taxonomy-input",
        default="configs/brand_taxonomy.csv",
        help="Brand taxonomy file with regex_pattern column.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out


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
    raise ValueError("Unsupported schema for time-series build. Use script 01 outputs.")


def _prep_king_county(df: pd.DataFrame) -> pd.DataFrame:
    name_col = _first_existing(df, ["inspection_business_name", "name"])
    out = pd.DataFrame(index=df.index)
    out["business_legal_name"] = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    out["trade_name"] = df[name_col].fillna("").astype(str)
    fallback_name = out["business_legal_name"].fillna("").astype(str)
    empty_trade = out["trade_name"].str.strip() == ""
    out.loc[empty_trade, "trade_name"] = fallback_name[empty_trade]
    out["naics_description"] = df.get("program_identifier", pd.Series("", index=df.index)).fillna("").astype(str)
    out["street_address"] = df.get("address", pd.Series("", index=df.index)).fillna("").astype(str)
    out["zip5"] = df.get("zip_code", pd.Series("", index=df.index)).fillna("").astype(str).str.extract(r"(\d{5})", expand=False)
    out["business_id"] = df.get("business_id", pd.Series("", index=df.index)).fillna("").astype(str)
    out["license_start_date"] = pd.to_datetime(
        df.get("inspection_date", pd.Series("", index=df.index)),
        errors="coerce",
    )
    return out


def _prep_seattle(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["trade_name"] = df.get("trade_name", pd.Series("", index=df.index)).fillna("").astype(str)
    out["business_legal_name"] = (
        df.get("business_legal_name", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["naics_description"] = (
        df.get("naics_description", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["street_address"] = (
        df.get("street_address", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    out["zip5"] = df.get("zip", pd.Series("", index=df.index)).fillna("").astype(str).str.extract(r"(\d{5})", expand=False)
    out["business_id"] = pd.NA
    out["license_start_date"] = pd.to_datetime(
        df.get("license_start_date", pd.Series("", index=df.index)),
        errors="coerce",
        format="%Y%m%d",
    )
    return out


def filter_boba(
    df: pd.DataFrame,
    taxonomy_brand_pattern: re.Pattern[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    schema = detect_schema(df)
    work = _prep_king_county(df) if schema == "king_county_inspections" else _prep_seattle(df)

    work = work[work["zip5"].notna()].copy()
    text_blob = work["trade_name"] + " " + work["business_legal_name"] + " " + work["naics_description"]
    boba_mask = text_blob.str.contains(BOBA_PATTERN, regex=True)
    brand_mask = text_blob.str.contains(BRAND_PATTERN, regex=True)
    taxonomy_mask = (
        text_blob.str.contains(taxonomy_brand_pattern, regex=True)
        if taxonomy_brand_pattern is not None
        else pd.Series(False, index=work.index)
    )
    mask = (boba_mask | brand_mask | taxonomy_mask) & (
        ~text_blob.str.contains(EXCLUDE_PATTERN, regex=True)
    )
    work = work[mask].copy()

    work["business_id_clean"] = work["business_id"].fillna("").astype(str).str.strip()
    work["shop_key"] = (
        work["trade_name"].str.upper().str.strip()
        + "|"
        + work["street_address"].str.upper().str.strip()
        + "|"
        + work["zip5"].astype(str)
    )
    has_bid = work["business_id_clean"] != ""
    work.loc[has_bid, "shop_key"] = "BID|" + work.loc[has_bid, "business_id_clean"]

    # King County inspections are repeated records; using first appearance as store entry proxy.
    work = work.sort_values(["shop_key", "license_start_date"], na_position="last")
    first_seen = work.drop_duplicates(subset=["shop_key"], keep="first").copy()
    first_seen["license_start_year"] = first_seen["license_start_date"].dt.year
    first_seen = first_seen[first_seen["license_start_year"].notna()].copy()
    first_seen["license_start_year"] = first_seen["license_start_year"].astype(int)
    return first_seen, schema


def build_timeseries(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    ts = (
        df.groupby(["zip5", "license_start_year"], as_index=False)
        .agg(shops_started=("shop_key", "nunique"))
        .rename(columns={"license_start_year": "year"})
    )
    ts = ts[ts["year"] >= min_year].copy()
    if ts.empty:
        return ts

    all_years = pd.Index(range(ts["year"].min(), ts["year"].max() + 1), name="year")
    expanded = []
    for zip5, g in ts.groupby("zip5"):
        g2 = (
            g[["year", "shops_started"]]
            .set_index("year")
            .reindex(all_years, fill_value=0)
            .reset_index()
        )
        g2["zip5"] = zip5
        expanded.append(g2)
    ts = pd.concat(expanded, ignore_index=True)
    ts["shops_started"] = ts["shops_started"].astype(int)
    ts["rolling3_started"] = (
        ts.sort_values(["zip5", "year"])
        .groupby("zip5")["shops_started"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    return ts.sort_values(["year", "zip5"]).reset_index(drop=True)


def make_figure(ts: pd.DataFrame, fig_path: Path) -> None:
    if ts.empty:
        return
    ensure_parent(fig_path)
    top_zips = (
        ts.groupby("zip5", as_index=False)["shops_started"]
        .sum()
        .sort_values("shops_started", ascending=False)
        .head(5)["zip5"]
        .tolist()
    )
    plot_df = ts[ts["zip5"].isin(top_zips)].copy()

    plt.figure(figsize=(11, 7))
    for zip5, g in plot_df.groupby("zip5"):
        g = g.sort_values("year")
        plt.plot(g["year"], g["shops_started"], marker="o", linewidth=1.5, label=zip5)
    plt.title("Store Entry Proxy Activity for Bubble Tea Candidates (Top ZIPs, King County)")
    plt.xlabel("Year")
    plt.ylabel("Shops Started")
    plt.legend(title="ZIP", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    root = project_root()
    inp = root / args.input
    out = root / args.output
    fig = root / args.figure_output
    taxonomy_path = root / args.taxonomy_input

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    raw = pd.read_csv(inp, dtype=str)
    df = normalize_columns(raw)
    taxonomy_brand_pattern = load_taxonomy_brand_pattern(taxonomy_path)
    filtered, schema = filter_boba(df, taxonomy_brand_pattern=taxonomy_brand_pattern)
    ts = build_timeseries(filtered, args.min_year)

    ensure_parent(out)
    ts.to_csv(out, index=False)
    make_figure(ts, fig)

    print(
        "[05] Completed. "
        f"schema={schema}, filtered_rows={filtered.shape[0]}, "
        f"ts_rows={ts.shape[0]}, zip_count={ts['zip5'].nunique() if not ts.empty else 0}"
    )
    print(f"[05] output table: {out}")
    print(f"[05] output figure: {fig}")


if __name__ == "__main__":
    main()
