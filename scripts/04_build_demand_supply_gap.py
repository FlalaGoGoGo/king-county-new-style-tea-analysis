#!/usr/bin/env python3
"""Build district-level demand-vs-supply gap panel.
Input: processed supply + demographics + demand trend
Output: data/processed/district_gap_panel.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import requests


ACS_VARIABLES = [
    "NAME",
    "B01003_001E",  # total population
    "B19013_001E",  # median household income
    "B25064_001E",  # median gross rent (monthly)
    "B25077_001E",  # median home value
    "B01001_008E",  # male 20-24
    "B01001_009E",  # male 25-29
    "B01001_010E",  # male 30-34
    "B01001_032E",  # female 20-24
    "B01001_033E",  # female 25-29
    "B01001_034E",  # female 30-34
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supply-input",
        default="data/processed/supply_by_district.csv",
        help="Supply table path.",
    )
    parser.add_argument(
        "--acs-input",
        default="data/raw/acs_demographics_by_zip.csv",
        help="ACS demographics CSV path. If missing, script attempts to fetch ACS.",
    )
    parser.add_argument(
        "--trends-input",
        default="data/processed/demand_trend_weekly.csv",
        help="Standardized trends path. If missing, fallback trend index is used.",
    )
    parser.add_argument(
        "--output-panel",
        default="data/processed/district_gap_panel.csv",
        help="Output panel path.",
    )
    parser.add_argument(
        "--output-fig",
        default="outputs/figures/fig_gap_quadrant.png",
        help="Output scatter figure path.",
    )
    parser.add_argument(
        "--acs-year",
        default="2023",
        help="ACS 5-year data vintage (e.g., 2023).",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def minmax_0_100(series: pd.Series) -> pd.Series:
    lo = series.min()
    hi = series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=series.index)
    return 100.0 * (series - lo) / (hi - lo)


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of candidate columns found: {list(candidates)}")


def load_supply(supply_path: Path) -> pd.DataFrame:
    df = pd.read_csv(supply_path)
    key_col = _first_existing(df, ["district_id", "zip5", "zip"])
    cnt_col = _first_existing(df, ["active_shop_count", "shop_count"])
    out = df[[key_col, cnt_col]].copy()
    out.columns = ["zip5", "active_shop_count"]
    out["zip5"] = out["zip5"].astype(str).str.extract(r"(\d{5})", expand=False)
    out["active_shop_count"] = pd.to_numeric(out["active_shop_count"], errors="coerce")
    out = out.dropna(subset=["zip5", "active_shop_count"]).copy()
    return out


def fetch_acs_zip(year: str) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": ",".join(ACS_VARIABLES),
        "for": "zip code tabulation area:*",
    }
    resp = requests.get(base, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    header, rows = payload[0], payload[1:]
    df = pd.DataFrame(rows, columns=header)
    return df


def standardize_acs(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "zip code tabulation area": "zip5",
        "B01003_001E": "population",
        "B19013_001E": "median_income",
        "B25064_001E": "median_gross_rent",
        "B25077_001E": "median_home_value",
        "B01001_008E": "m_20_24",
        "B01001_009E": "m_25_29",
        "B01001_010E": "m_30_34",
        "B01001_032E": "f_20_24",
        "B01001_033E": "f_25_29",
        "B01001_034E": "f_30_34",
    }
    df = df.rename(columns=rename_map).copy()
    for col in [
        "population",
        "median_income",
        "median_gross_rent",
        "median_home_value",
        "m_20_24",
        "m_25_29",
        "m_30_34",
        "f_20_24",
        "f_25_29",
        "f_30_34",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["zip5"] = df["zip5"].astype(str).str.extract(r"(\d{5})", expand=False)
    young_sum = (
        df[["m_20_24", "m_25_29", "m_30_34", "f_20_24", "f_25_29", "f_30_34"]]
        .fillna(0)
        .sum(axis=1)
    )
    df["young_adult_share"] = young_sum / df["population"].replace(0, pd.NA)
    return df[
        [
            "zip5",
            "population",
            "median_income",
            "young_adult_share",
            "median_gross_rent",
            "median_home_value",
        ]
    ]


def _fallback_acs_from_supply_zips(supply_zips: pd.Series) -> pd.DataFrame:
    zips = supply_zips.dropna().astype(str).str.extract(r"(\d{5})", expand=False).dropna().unique()
    return pd.DataFrame(
        {
            "zip5": zips,
            "population": 10000.0,
            "median_income": 80000.0,
            "young_adult_share": 0.22,
            "median_gross_rent": 2200.0,
            "median_home_value": 700000.0,
        }
    )


def apply_cost_fallbacks(acs_df: pd.DataFrame) -> pd.DataFrame:
    df = acs_df.copy()
    for col in [
        "zip5",
        "population",
        "median_income",
        "young_adult_share",
        "median_gross_rent",
        "median_home_value",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    df["zip5"] = df["zip5"].astype(str).str.extract(r"(\d{5})", expand=False)
    for col in [
        "population",
        "median_income",
        "young_adult_share",
        "median_gross_rent",
        "median_home_value",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If direct ACS cost fields are missing, infer conservative ZIP-level proxies from income.
    income_for_proxy = df["median_income"].clip(lower=15000)
    inferred_rent = (income_for_proxy * 0.30 / 12.0).clip(lower=1100, upper=4200)
    inferred_home = (income_for_proxy * 4.5).clip(lower=200000, upper=1800000)

    df["median_gross_rent"] = df["median_gross_rent"].where(df["median_gross_rent"] > 0, pd.NA)
    df["median_home_value"] = df["median_home_value"].where(df["median_home_value"] > 0, pd.NA)

    df["median_gross_rent"] = df["median_gross_rent"].fillna(inferred_rent)
    df["median_home_value"] = df["median_home_value"].fillna(inferred_home)

    rent_med = pd.to_numeric(df["median_gross_rent"], errors="coerce").median()
    home_med = pd.to_numeric(df["median_home_value"], errors="coerce").median()
    if pd.isna(rent_med):
        rent_med = 2200.0
    if pd.isna(home_med):
        home_med = 700000.0
    df["median_gross_rent"] = pd.to_numeric(df["median_gross_rent"], errors="coerce").fillna(rent_med)
    df["median_home_value"] = pd.to_numeric(df["median_home_value"], errors="coerce").fillna(home_med)

    return df[
        [
            "zip5",
            "population",
            "median_income",
            "young_adult_share",
            "median_gross_rent",
            "median_home_value",
        ]
    ]


def load_or_fetch_acs(acs_path: Path, year: str, supply_zips: pd.Series) -> pd.DataFrame:
    if acs_path.exists():
        raw = pd.read_csv(acs_path, dtype=str)
        if "zip5" in raw.columns and "population" in raw.columns:
            existing = raw.copy()
            for col in [
                "median_income",
                "young_adult_share",
                "median_gross_rent",
                "median_home_value",
            ]:
                if col not in existing.columns:
                    existing[col] = pd.NA

            missing_cost_cols = (
                pd.to_numeric(existing["median_gross_rent"], errors="coerce").isna().all()
                or pd.to_numeric(existing["median_home_value"], errors="coerce").isna().all()
            )
            if missing_cost_cols:
                try:
                    fetched = standardize_acs(fetch_acs_zip(year))
                    existing["zip5"] = existing["zip5"].astype(str).str.extract(r"(\d{5})", expand=False)
                    fetched = fetched[["zip5", "median_gross_rent", "median_home_value"]].drop_duplicates("zip5")
                    existing = existing.merge(
                        fetched,
                        on="zip5",
                        how="left",
                        suffixes=("", "_acs"),
                    )
                    existing["median_gross_rent"] = pd.to_numeric(
                        existing["median_gross_rent"], errors="coerce"
                    ).fillna(pd.to_numeric(existing["median_gross_rent_acs"], errors="coerce"))
                    existing["median_home_value"] = pd.to_numeric(
                        existing["median_home_value"], errors="coerce"
                    ).fillna(pd.to_numeric(existing["median_home_value_acs"], errors="coerce"))
                    existing = existing.drop(
                        columns=["median_gross_rent_acs", "median_home_value_acs"],
                        errors="ignore",
                    )
                    ensure_parent(acs_path)
                    apply_cost_fallbacks(existing).to_csv(acs_path, index=False)
                except Exception as exc:
                    print(f"[04] Warning: ACS rent/home fetch failed ({exc}). Using cost proxies.")
            return apply_cost_fallbacks(existing)

    try:
        fetched = fetch_acs_zip(year)
        std = standardize_acs(fetched)
        std = apply_cost_fallbacks(std)
        ensure_parent(acs_path)
        std.to_csv(acs_path, index=False)
        return std
    except Exception as exc:
        print(f"[04] Warning: ACS fetch failed ({exc}). Falling back to neutral defaults.")
        return apply_cost_fallbacks(_fallback_acs_from_supply_zips(supply_zips))


def load_city_trend_index(trends_path: Path) -> float:
    if not trends_path.exists():
        return 50.0
    trends = pd.read_csv(trends_path)
    if "trend_index_ma4" in trends.columns:
        val = pd.to_numeric(trends["trend_index_ma4"], errors="coerce").dropna()
    elif "trend_index" in trends.columns:
        val = pd.to_numeric(trends["trend_index"], errors="coerce").dropna()
    else:
        return 50.0
    if val.empty:
        return 50.0
    return float(val.tail(4).mean())


def build_panel(supply_df: pd.DataFrame, acs_df: pd.DataFrame, city_trend_index: float) -> pd.DataFrame:
    panel = supply_df.merge(acs_df, how="left", on="zip5")

    panel["population"] = panel["population"].fillna(panel["population"].median())
    panel["median_income"] = panel["median_income"].fillna(panel["median_income"].median())
    panel["young_adult_share"] = panel["young_adult_share"].fillna(panel["young_adult_share"].median())
    panel["median_gross_rent"] = pd.to_numeric(panel["median_gross_rent"], errors="coerce")
    panel["median_home_value"] = pd.to_numeric(panel["median_home_value"], errors="coerce")
    panel["median_gross_rent"] = panel["median_gross_rent"].fillna(panel["median_gross_rent"].median())
    panel["median_home_value"] = panel["median_home_value"].fillna(panel["median_home_value"].median())

    panel["supply_per_10k"] = (
        panel["active_shop_count"] / panel["population"].replace(0, pd.NA)
    ) * 10000
    panel["supply_per_10k"] = panel["supply_per_10k"].fillna(panel["supply_per_10k"].median())

    panel["demand_raw"] = (
        0.50 * zscore(panel["population"])
        + 0.30 * zscore(panel["median_income"])
        + 0.20 * zscore(panel["young_adult_share"])
    )
    panel["supply_raw"] = (
        0.40 * zscore(panel["active_shop_count"])
        + 0.60 * zscore(panel["supply_per_10k"])
    )

    panel["demand_score"] = minmax_0_100(panel["demand_raw"])
    panel["supply_score"] = minmax_0_100(panel["supply_raw"])
    panel["gap_score"] = panel["demand_score"] - panel["supply_score"]
    panel["regional_cost_raw"] = (
        0.55 * zscore(panel["median_gross_rent"])
        + 0.45 * zscore(panel["median_home_value"])
    )
    panel["regional_cost_score"] = minmax_0_100(panel["regional_cost_raw"])
    panel["regional_feasibility_score"] = 100.0 - panel["regional_cost_score"]
    panel["city_trend_index"] = city_trend_index

    demand_med = panel["demand_score"].median()
    supply_med = panel["supply_score"].median()

    def quadrant(row: pd.Series) -> str:
        high_d = row["demand_score"] >= demand_med
        low_s = row["supply_score"] < supply_med
        if high_d and low_s:
            return "HighDemand_LowSupply"
        if high_d and not low_s:
            return "HighDemand_HighSupply"
        if not high_d and low_s:
            return "LowDemand_LowSupply"
        return "LowDemand_HighSupply"

    panel["quadrant"] = panel.apply(quadrant, axis=1)
    panel = panel.sort_values("gap_score", ascending=False).reset_index(drop=True)
    return panel[
        [
            "zip5",
            "active_shop_count",
            "population",
            "median_income",
            "median_gross_rent",
            "median_home_value",
            "young_adult_share",
            "supply_per_10k",
            "demand_score",
            "supply_score",
            "gap_score",
            "regional_cost_score",
            "regional_feasibility_score",
            "city_trend_index",
            "quadrant",
        ]
    ]


def make_gap_scatter(panel: pd.DataFrame, fig_path: Path) -> None:
    ensure_parent(fig_path)
    top = panel.head(10)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        panel["supply_score"],
        panel["demand_score"],
        c=panel["gap_score"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    plt.axhline(panel["demand_score"].median(), linestyle="--", linewidth=1)
    plt.axvline(panel["supply_score"].median(), linestyle="--", linewidth=1)
    for _, row in top.iterrows():
        plt.text(
            row["supply_score"] + 0.6,
            row["demand_score"] + 0.6,
            str(row["zip5"]),
            fontsize=8,
        )
    plt.xlabel("Supply Pressure Score (Higher = More Saturated)")
    plt.ylabel("Demand Potential Score (Higher = Stronger Demand Proxy)")
    plt.title("District Gap Quadrant (ZIP-level MVP)")
    cbar = plt.colorbar()
    cbar.set_label("Gap Score (Demand - Supply)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    root = project_root()
    supply_path = root / args.supply_input
    acs_path = root / args.acs_input
    trends_path = root / args.trends_input
    panel_out = root / args.output_panel
    fig_out = root / args.output_fig

    if not supply_path.exists():
        raise FileNotFoundError(f"Supply input not found: {supply_path}")

    supply_df = load_supply(supply_path)
    acs_df = load_or_fetch_acs(acs_path, args.acs_year, supply_df["zip5"])
    city_trend_index = load_city_trend_index(trends_path)

    panel = build_panel(supply_df, acs_df, city_trend_index)
    ensure_parent(panel_out)
    panel.to_csv(panel_out, index=False)
    make_gap_scatter(panel, fig_out)

    print(
        "[04] Completed. "
        f"rows={panel.shape[0]}, top_zip={panel.iloc[0]['zip5']}, top_gap={panel.iloc[0]['gap_score']:.2f}"
    )
    print(f"[04] output panel: {panel_out}")
    print(f"[04] output figure: {fig_out}")


if __name__ == "__main__":
    main()
