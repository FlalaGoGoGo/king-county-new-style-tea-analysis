#!/usr/bin/env python3
"""Fetch Google Trends weekly data for last 5 years (Seattle metro proxy).

Output:
  - data/raw/google_trends_weekly.csv
  - data/interim/google_trends_fetch_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd


DEFAULT_KEYWORDS = ["bubble tea", "boba tea", "milk tea", "fruit tea"]
DEFAULT_GEO_CANDIDATES = ["US-WA-819", "US-WA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/raw/google_trends_weekly.csv",
        help="Raw trends output CSV path.",
    )
    parser.add_argument(
        "--meta-output",
        default="data/interim/google_trends_fetch_meta.json",
        help="Metadata output path.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Keyword list for Google Trends query.",
    )
    parser.add_argument(
        "--timeframe",
        default="today 5-y",
        help="Google Trends timeframe, e.g., 'today 5-y'.",
    )
    parser.add_argument(
        "--geo",
        default="US-WA-819,US-WA",
        help="Comma-separated geo fallbacks. Default tries Seattle DMA then WA state.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fetch_once(keywords: list[str], timeframe: str, geo: str) -> pd.DataFrame:
    from pytrends.request import TrendReq  # local import to keep script dependency explicit

    pytrends = TrendReq(
        hl="en-US",
        tz=480,  # PST/PDT
        timeout=(10, 30),
        retries=2,
        backoff_factor=0.4,
    )
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df.empty:
        raise ValueError(f"Empty Google Trends response for geo={geo}.")
    return df


def fetch_with_fallback(keywords: list[str], timeframe: str, geos: list[str]) -> tuple[pd.DataFrame, str]:
    last_err: Exception | None = None
    for geo in geos:
        for attempt in range(1, 4):
            try:
                df = _fetch_once(keywords, timeframe, geo)
                return df, geo
            except Exception as exc:
                last_err = exc
                if attempt < 3:
                    time.sleep(2.0 * attempt)
    if last_err is None:
        raise RuntimeError("Google Trends fetch failed for unknown reason.")
    raise RuntimeError(f"Google Trends fetch failed: {last_err}") from last_err


def standardize_raw(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "isPartial" in out.columns:
        out = out[~out["isPartial"].fillna(False)].copy()
        out = out.drop(columns=["isPartial"])
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()].copy()
    out = out.sort_index().reset_index().rename(columns={"date": "week_start"})

    keep_cols = ["week_start"] + [k for k in keywords if k in out.columns]
    out = out[keep_cols].copy()
    for k in keywords:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")
    metric_cols = [k for k in keywords if k in out.columns]
    out["trend_index_raw"] = out[metric_cols].mean(axis=1)
    out = out[["week_start", "trend_index_raw"]]
    return out.dropna(subset=["week_start", "trend_index_raw"]).reset_index(drop=True)


def write_meta(path: Path, meta: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    out_path = root / args.output
    meta_path = root / args.meta_output
    keywords = [k.strip() for k in args.keywords if str(k).strip()]
    geos = [g.strip() for g in args.geo.split(",") if g.strip()]
    if not geos:
        geos = DEFAULT_GEO_CANDIDATES

    raw_df, used_geo = fetch_with_fallback(
        keywords=keywords,
        timeframe=args.timeframe,
        geos=geos,
    )
    std_df = standardize_raw(raw_df, keywords)
    if std_df.empty:
        raise ValueError("No usable rows after Trends standardization.")

    ensure_parent(out_path)
    std_df.to_csv(out_path, index=False)

    meta = {
        "source": "Google Trends (pytrends)",
        "keywords": keywords,
        "timeframe": args.timeframe,
        "geo_candidates": geos,
        "geo_used": used_geo,
        "rows": int(std_df.shape[0]),
        "date_min": str(std_df["week_start"].min()),
        "date_max": str(std_df["week_start"].max()),
    }
    write_meta(meta_path, meta)

    print(
        "[03a] Completed. "
        f"geo={used_geo}, rows={std_df.shape[0]}, "
        f"range={std_df['week_start'].min()}..{std_df['week_start'].max()}"
    )
    print(f"[03a] raw trends output: {out_path}")
    print(f"[03a] meta output: {meta_path}")


if __name__ == "__main__":
    main()
