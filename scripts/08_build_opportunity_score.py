#!/usr/bin/env python3
"""Build final opportunity ranking using weighted score.
Input: gap + momentum + text evidence + store cost + configs/opportunity_weights.yaml
Output: data/processed/opportunity_ranked_districts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_WEIGHTS = {
    "demand_potential": 0.28,
    "supply_gap": 0.25,
    "growth_momentum": 0.17,
    "text_signal": 0.10,
    "feasibility": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gap-input",
        default="data/processed/district_gap_panel.csv",
        help="Gap panel input path.",
    )
    parser.add_argument(
        "--license-ts-input",
        default="data/processed/license_timeseries_by_district.csv",
        help="License timeseries input path.",
    )
    parser.add_argument(
        "--sentiment-input",
        default="data/processed/text_sentiment_by_district.csv",
        help="Sentiment input path.",
    )
    parser.add_argument(
        "--store-meta-input",
        default="data/raw/google_places_store_meta_king_county.csv",
        help="Store metadata path (Google price_level proxy for store cost).",
    )
    parser.add_argument(
        "--topics-input",
        default="data/processed/text_topics_by_district.csv",
        help="Topics input path.",
    )
    parser.add_argument(
        "--weights-config",
        default="configs/opportunity_weights.yaml",
        help="Weights config path.",
    )
    parser.add_argument(
        "--output-ranked",
        default="data/processed/opportunity_ranked_districts.csv",
        help="Output ranked table path.",
    )
    parser.add_argument(
        "--output-actions",
        default="outputs/tables/top_district_actions.csv",
        help="Output actions table path.",
    )
    parser.add_argument(
        "--top-n-actions",
        type=int,
        default=10,
        help="How many top districts to include in action table.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def minmax_0_100(series: pd.Series) -> pd.Series:
    lo = series.min()
    hi = series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=series.index)
    return 100 * (series - lo) / (hi - lo)


def load_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return DEFAULT_WEIGHTS.copy()
    weights = {}
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

    # Keep only known keys for stable normalization.
    keep = set(DEFAULT_WEIGHTS.keys())
    weights = {k: v for k, v in weights.items() if k in keep}
    if not weights:
        return DEFAULT_WEIGHTS.copy()
    # Normalize to sum=1
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in weights.items()}


def load_store_cost(store_meta_path: Path) -> pd.DataFrame:
    cols = [
        "zip5",
        "median_google_price_level",
        "store_cost_score",
        "store_affordability_score",
        "store_cost_coverage",
    ]
    if not store_meta_path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(store_meta_path, dtype=str).fillna("")
    zip_col = "zip5_input" if "zip5_input" in df.columns else ("zip5" if "zip5" in df.columns else "")
    if not zip_col or "google_price_level" not in df.columns:
        return pd.DataFrame(columns=cols)

    df["zip5"] = df[zip_col].astype(str).str.extract(r"(\d{5})", expand=False)
    df["google_price_level"] = pd.to_numeric(df["google_price_level"], errors="coerce")
    df = df.dropna(subset=["zip5"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    grouped = (
        df.groupby("zip5", as_index=False)
        .agg(
            median_google_price_level=("google_price_level", "median"),
            store_count=("zip5", "size"),
            priced_store_count=("google_price_level", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
    )
    grouped["store_cost_coverage"] = (
        grouped["priced_store_count"] / grouped["store_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["median_google_price_level"] = pd.to_numeric(
        grouped["median_google_price_level"], errors="coerce"
    )
    # If all ZIPs missing price_level, use neutral 50.
    if grouped["median_google_price_level"].notna().sum() == 0:
        grouped["store_cost_score"] = 50.0
    else:
        grouped["store_cost_score"] = minmax_0_100(
            grouped["median_google_price_level"].fillna(grouped["median_google_price_level"].median())
        )
    grouped["store_affordability_score"] = 100.0 - grouped["store_cost_score"]
    return grouped[cols]


def load_momentum(license_ts_path: Path) -> pd.DataFrame:
    if not license_ts_path.exists():
        return pd.DataFrame(columns=["zip5", "momentum_score", "momentum_raw"])
    ts = pd.read_csv(license_ts_path)
    if ts.empty:
        return pd.DataFrame(columns=["zip5", "momentum_score", "momentum_raw"])
    ts["year"] = pd.to_numeric(ts["year"], errors="coerce")
    ts["shops_started"] = pd.to_numeric(ts["shops_started"], errors="coerce")
    ts = ts.dropna(subset=["zip5", "year", "shops_started"]).copy()
    ts["year"] = ts["year"].astype(int)

    rows = []
    for zip5, g in ts.groupby("zip5"):
        g = g.sort_values("year")
        if g.empty:
            continue
        max_year = int(g["year"].max())
        recent = g[g["year"] >= max_year - 1]["shops_started"].mean()
        baseline = g[g["year"] < max_year - 1]["shops_started"].tail(3).mean()
        if pd.isna(baseline):
            baseline = g["shops_started"].mean()
        momentum_raw = float(recent - baseline)
        rows.append({"zip5": str(zip5), "momentum_raw": momentum_raw})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["zip5", "momentum_score", "momentum_raw"])
    out["momentum_score"] = minmax_0_100(out["momentum_raw"])
    return out


def load_text_signal(sentiment_path: Path, topics_path: Path) -> pd.DataFrame:
    if not sentiment_path.exists():
        return pd.DataFrame(columns=["zip5", "text_signal_score", "top_topic_terms"])

    s = pd.read_csv(sentiment_path)
    if s.empty:
        return pd.DataFrame(columns=["zip5", "text_signal_score", "top_topic_terms"])
    s["zip5"] = s["zip5"].astype(str)
    if "text_signal_score" not in s.columns:
        if "mean_sentiment" in s.columns:
            ms = pd.to_numeric(s["mean_sentiment"], errors="coerce").fillna(0.0)
            s["text_signal_score"] = ((ms + 1) / 2) * 100
        else:
            s["text_signal_score"] = 50.0
    s["text_signal_score"] = pd.to_numeric(s["text_signal_score"], errors="coerce").fillna(50.0)

    out = s[["zip5", "text_signal_score"]].copy()
    out["top_topic_terms"] = "n/a"

    if topics_path.exists():
        t = pd.read_csv(topics_path)
        if not t.empty and "zip5" in t.columns and "topic_terms" in t.columns:
            t = t.copy()
            t["zip5"] = t["zip5"].astype(str)
            if "topic_share" in t.columns:
                t["topic_share"] = pd.to_numeric(t["topic_share"], errors="coerce").fillna(0)
                t = (
                    t.sort_values(["zip5", "topic_share"], ascending=[True, False])
                    .groupby("zip5", as_index=False)
                    .head(1)
                )
            else:
                t = t.drop_duplicates(subset=["zip5"])
            out = out.merge(
                t[["zip5", "topic_terms"]].rename(columns={"topic_terms": "top_topic_terms"}),
                on="zip5",
                how="left",
                suffixes=("", "_from_topics"),
            )
            out["top_topic_terms"] = out["top_topic_terms_from_topics"].fillna(out["top_topic_terms"])
            out = out.drop(columns=["top_topic_terms_from_topics"])
    return out


def build_actions(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top = df.head(top_n).copy()

    def classify(row: pd.Series) -> tuple[str, str]:
        if row["feasibility_score"] < 40:
            return (
                "Cost-Control Entry Strategy",
                "Opportunity exists but cost feasibility is weak: start with lower-capex formats and strict rent discipline.",
            )
        if row["supply_score"] >= 70 and row["demand_score"] >= 70:
            return (
                "Differentiation Strategy",
                "High demand but crowded area: focus on product differentiation and service speed.",
            )
        if row["demand_score"] >= 70 and row["supply_score"] < 50:
            return (
                "Expansion Strategy",
                "Strong demand and lower saturation: prioritize pilot store and local digital acquisition.",
            )
        if row["text_signal_score"] < 45:
            return (
                "Operations Fix Strategy",
                "Text signal is weak: prioritize wait-time reduction and quality consistency before expansion.",
            )
        return (
            "Selective Growth Strategy",
            "Moderate opportunity: test low-risk campaigns and monitor momentum before scaling.",
        )

    strategy = top.apply(lambda r: classify(r), axis=1)
    top["recommended_strategy"] = [s[0] for s in strategy]
    top["action_note"] = [s[1] for s in strategy]
    keep = [
        "rank",
        "zip5",
        "opportunity_score",
        "demand_score",
        "supply_score",
        "momentum_score",
        "text_signal_score",
        "feasibility_score",
        "regional_feasibility_score",
        "store_affordability_score",
        "top_topic_terms",
        "recommended_strategy",
        "action_note",
    ]
    return top[keep]


def main() -> None:
    args = parse_args()
    root = project_root()
    gap_path = root / args.gap_input
    ts_path = root / args.license_ts_input
    sent_path = root / args.sentiment_input
    store_meta_path = root / args.store_meta_input
    topics_path = root / args.topics_input
    cfg_path = root / args.weights_config
    ranked_out = root / args.output_ranked
    actions_out = root / args.output_actions

    if not gap_path.exists():
        raise FileNotFoundError(f"Gap input not found: {gap_path}")

    weights = load_weights(cfg_path)
    gap = pd.read_csv(gap_path)
    gap["zip5"] = gap["zip5"].astype(str)
    for col in ["demand_score", "supply_score", "gap_score"]:
        gap[col] = pd.to_numeric(gap[col], errors="coerce").fillna(50.0)
    if "regional_feasibility_score" not in gap.columns:
        if "median_income" in gap.columns:
            gap["regional_cost_score"] = minmax_0_100(
                pd.to_numeric(gap["median_income"], errors="coerce").fillna(
                    pd.to_numeric(gap["median_income"], errors="coerce").median()
                )
            )
            gap["regional_feasibility_score"] = 100.0 - gap["regional_cost_score"]
        else:
            gap["regional_feasibility_score"] = 50.0
    gap["regional_feasibility_score"] = pd.to_numeric(
        gap["regional_feasibility_score"], errors="coerce"
    ).fillna(50.0)
    if "regional_cost_score" not in gap.columns:
        gap["regional_cost_score"] = 100.0 - gap["regional_feasibility_score"]
    gap["regional_cost_score"] = pd.to_numeric(gap["regional_cost_score"], errors="coerce").fillna(50.0)

    momentum = load_momentum(ts_path)
    text_df = load_text_signal(sent_path, topics_path)
    store_cost_df = load_store_cost(store_meta_path)

    merged = gap.merge(momentum[["zip5", "momentum_score"]], on="zip5", how="left")
    merged = merged.merge(text_df, on="zip5", how="left")
    merged = merged.merge(store_cost_df, on="zip5", how="left")
    merged["momentum_score"] = pd.to_numeric(merged["momentum_score"], errors="coerce").fillna(50.0)
    merged["text_signal_score"] = pd.to_numeric(merged["text_signal_score"], errors="coerce").fillna(50.0)
    merged["store_affordability_score"] = pd.to_numeric(
        merged["store_affordability_score"], errors="coerce"
    ).fillna(50.0)
    merged["store_cost_score"] = pd.to_numeric(merged["store_cost_score"], errors="coerce").fillna(50.0)
    merged["store_cost_coverage"] = pd.to_numeric(merged["store_cost_coverage"], errors="coerce").fillna(0.0)
    merged["median_google_price_level"] = pd.to_numeric(
        merged["median_google_price_level"], errors="coerce"
    ).fillna(pd.NA)
    merged["gap_component_score"] = minmax_0_100(pd.to_numeric(merged["gap_score"], errors="coerce").fillna(0.0))
    merged["feasibility_score"] = (
        0.65 * merged["regional_feasibility_score"]
        + 0.35 * merged["store_affordability_score"]
    )
    merged["top_topic_terms"] = merged["top_topic_terms"].fillna("n/a")

    merged["opportunity_score"] = (
        weights.get("demand_potential", DEFAULT_WEIGHTS["demand_potential"]) * merged["demand_score"]
        + weights.get("supply_gap", DEFAULT_WEIGHTS["supply_gap"]) * merged["gap_component_score"]
        + weights.get("growth_momentum", DEFAULT_WEIGHTS["growth_momentum"]) * merged["momentum_score"]
        + weights.get("text_signal", DEFAULT_WEIGHTS["text_signal"]) * merged["text_signal_score"]
        + weights.get("feasibility", DEFAULT_WEIGHTS["feasibility"]) * merged["feasibility_score"]
    )
    merged = merged.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    ensure_parent(ranked_out)
    merged.to_csv(ranked_out, index=False)

    actions = build_actions(merged, args.top_n_actions)
    ensure_parent(actions_out)
    actions.to_csv(actions_out, index=False)

    print(
        "[08] Completed. "
        f"rows={merged.shape[0]}, top_zip={merged.iloc[0]['zip5']}, "
        f"top_score={merged.iloc[0]['opportunity_score']:.2f}"
    )
    print(f"[08] ranked output: {ranked_out}")
    print(f"[08] actions output: {actions_out}")


if __name__ == "__main__":
    main()
