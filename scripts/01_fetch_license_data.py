#!/usr/bin/env python3
"""Fetch or load King County food establishment inspection data.
Input: local CSV export path or URL
Output: data/raw/king_county_food_inspections.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copyfile

import pandas as pd
import requests


DEFAULT_LICENSE_CSV_URL = (
    "https://data.kingcounty.gov/resource/f29f-zza5.csv?$limit=500000"
)
KING_REQUIRED_COLUMNS = {
    "inspection_business_name",
    "inspection_date",
    "address",
    "city",
    "zip_code",
    "business_id",
}
SEATTLE_REQUIRED_COLUMNS = {
    "Business Legal Name",
    "Trade Name",
    "NAICS Description",
    "License Start Date",
    "Street Address",
    "City",
    "State",
    "Zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-url",
        default=DEFAULT_LICENSE_CSV_URL,
        help="CSV URL used when --input-file is not provided.",
    )
    parser.add_argument(
        "--input-file",
        default="",
        help="Local CSV path. If provided, this file is copied instead of downloading.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/king_county_food_inspections.csv",
        help="Output CSV path relative to project root.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_and_validate(csv_path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        raise ValueError("License file is empty.")

    if KING_REQUIRED_COLUMNS.issubset(df.columns):
        return df, "king_county_inspections"
    if SEATTLE_REQUIRED_COLUMNS.issubset(df.columns):
        return df, "seattle_business_license"

    missing_king = sorted(KING_REQUIRED_COLUMNS.difference(df.columns))
    missing_seattle = sorted(SEATTLE_REQUIRED_COLUMNS.difference(df.columns))
    raise ValueError(
        "Input schema not recognized. "
        f"Missing (KingCounty schema): {', '.join(missing_king)}. "
        f"Missing (Seattle schema): {', '.join(missing_seattle)}."
    )


def write_profile(df: pd.DataFrame, profile_path: Path, dataset_type: str) -> None:
    if dataset_type == "king_county_inspections":
        city_top5 = df["city"].fillna("NA").value_counts().head(10).to_dict()
        sample_names = df["inspection_business_name"].fillna("").head(10).tolist()
        min_date = str(pd.to_datetime(df["inspection_date"], errors="coerce").min())
        max_date = str(pd.to_datetime(df["inspection_date"], errors="coerce").max())
    else:
        city_top5 = df["City"].fillna("NA").value_counts().head(10).to_dict()
        sample_names = df["Trade Name"].fillna("").head(10).tolist()
        min_date = str(pd.to_datetime(df["License Start Date"], errors="coerce").min())
        max_date = str(pd.to_datetime(df["License Start Date"], errors="coerce").max())

    profile = {
        "dataset_type": dataset_type,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "city_top10": city_top5,
        "sample_business_names": sample_names,
        "date_range": {"min": min_date, "max": max_date},
    }
    ensure_parent(profile_path)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def download_csv(url: str, out_path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    ensure_parent(out_path)
    out_path.write_bytes(response.content)


def main() -> None:
    args = parse_args()
    root = project_root()
    out = root / args.output
    profile_path = root / "data/interim/license_schema_profile.json"

    if args.input_file:
        src = Path(args.input_file).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")
        ensure_parent(out)
        copyfile(src, out)
        mode = "local-file"
    else:
        download_csv(args.source_url, out)
        mode = "download-url"

    df, dataset_type = read_and_validate(out)
    write_profile(df, profile_path, dataset_type)

    print(
        f"[01] Completed ({mode}). "
        f"dataset={dataset_type}, rows={df.shape[0]}, cols={df.shape[1]}, output={out}"
    )
    print(f"[01] Wrote schema profile: {profile_path}")


if __name__ == "__main__":
    main()
