#!/usr/bin/env python3
"""Run weight sensitivity analysis for opportunity ranking stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
        "--ranked-input",
        default="data/processed/opportunity_ranked_districts.csv",
        help="Ranked opportunity table with component score columns.",
    )
    parser.add_argument(
        "--weights-config",
        default="configs/opportunity_weights.yaml",
        help="Base weights config path.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=2000,
        help="Number of random weight simulations.",
    )
    parser.add_argument(
        "--concentration",
        type=float,
        default=60.0,
        help="Dirichlet concentration around base weights (higher=less perturbation).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k frequency table size.",
    )
    parser.add_argument(
        "--top1-output",
        default="outputs/tables/weight_sensitivity_top1_frequency.csv",
        help="Top1 frequency output path.",
    )
    parser.add_argument(
        "--top5-output",
        default="outputs/tables/weight_sensitivity_top5_frequency.csv",
        help="Top5 frequency output path.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/interim/weight_sensitivity_summary.json",
        help="Summary json output path.",
    )
    parser.add_argument(
        "--figure-output",
        default="outputs/figures/fig_weight_sensitivity_top1.png",
        help="Top1 frequency figure output path.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    weights = {k: v for k, v in weights.items() if k in DEFAULT_WEIGHTS}
    if not weights:
        return DEFAULT_WEIGHTS.copy()
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in weights.items()}


def compute_score(df: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    return (
        w["demand_potential"] * df["demand_score"]
        + w["supply_gap"] * df["gap_component_score"]
        + w["growth_momentum"] * df["momentum_score"]
        + w["text_signal"] * df["text_signal_score"]
        + w["feasibility"] * df["feasibility_score"]
    )


def run_simulation(
    df: pd.DataFrame,
    base_weights: dict[str, float],
    simulations: int,
    concentration: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zips = df["zip5"].astype(str).tolist()
    idx_map = {z: i for i, z in enumerate(zips)}
    n = len(zips)
    top1_counts = np.zeros(n, dtype=int)
    top5_counts = np.zeros(n, dtype=int)
    rank_sum = np.zeros(n, dtype=float)

    bw = np.array(
        [
            base_weights["demand_potential"],
            base_weights["supply_gap"],
            base_weights["growth_momentum"],
            base_weights["text_signal"],
            base_weights["feasibility"],
        ],
        dtype=float,
    )
    alpha = np.maximum(bw * concentration, 1e-6)
    rng = np.random.default_rng(2026)

    for _ in range(simulations):
        w = rng.dirichlet(alpha)
        w_map = {
            "demand_potential": float(w[0]),
            "supply_gap": float(w[1]),
            "growth_momentum": float(w[2]),
            "text_signal": float(w[3]),
            "feasibility": float(w[4]),
        }
        score = compute_score(df, w_map)
        order = np.argsort(-score.values)  # descending
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        rank_sum += ranks
        top1_counts[order[0]] += 1
        top5_counts[order[: min(5, n)]] += 1

    out = pd.DataFrame(
        {
            "zip5": zips,
            "top1_count": top1_counts,
            "top1_freq_pct": top1_counts / simulations * 100.0,
            "top5_count": top5_counts,
            "top5_freq_pct": top5_counts / simulations * 100.0,
            "mean_rank": rank_sum / simulations,
        }
    ).sort_values(["top1_freq_pct", "top5_freq_pct"], ascending=False).reset_index(drop=True)
    top1 = out[["zip5", "top1_count", "top1_freq_pct", "mean_rank"]].copy()
    top5 = out[["zip5", "top5_count", "top5_freq_pct", "mean_rank"]].copy().sort_values(
        "top5_freq_pct", ascending=False
    )
    return out, top1.reset_index(drop=True), top5.reset_index(drop=True)


def make_figure(top1_df: pd.DataFrame, fig_path: Path, top_k: int) -> None:
    show = top1_df.head(top_k).sort_values("top1_freq_pct", ascending=True)
    ensure_parent(fig_path)
    plt.figure(figsize=(10, 6))
    plt.barh(show["zip5"], show["top1_freq_pct"], color="#264653")
    plt.xlabel("Top-1 Frequency (%)")
    plt.ylabel("ZIP")
    plt.title("Weight Sensitivity: Top-1 ZIP Frequency")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    root = project_root()
    ranked_path = root / args.ranked_input
    cfg_path = root / args.weights_config
    top1_out = root / args.top1_output
    top5_out = root / args.top5_output
    summary_out = root / args.summary_output
    fig_out = root / args.figure_output

    if not ranked_path.exists():
        raise FileNotFoundError(f"Ranked input not found: {ranked_path}")
    df = pd.read_csv(ranked_path)
    required = ["zip5", "demand_score", "momentum_score", "text_signal_score", "feasibility_score"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column in ranked input: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce") if c != "zip5" else df[c]
    if "gap_component_score" not in df.columns:
        if "gap_score" in df.columns:
            gap = pd.to_numeric(df["gap_score"], errors="coerce")
            lo, hi = gap.min(), gap.max()
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                df["gap_component_score"] = 50.0
            else:
                df["gap_component_score"] = 100 * (gap - lo) / (hi - lo)
        elif "supply_score" in df.columns:
            df["gap_component_score"] = 100 - pd.to_numeric(df["supply_score"], errors="coerce")
        else:
            df["gap_component_score"] = 50.0
    df["gap_component_score"] = pd.to_numeric(df["gap_component_score"], errors="coerce")
    df = df.dropna(
        subset=["demand_score", "gap_component_score", "momentum_score", "text_signal_score", "feasibility_score"]
    ).copy()
    df["zip5"] = df["zip5"].astype(str)

    base_weights = load_weights(cfg_path)
    _, top1_df, top5_df = run_simulation(
        df,
        base_weights=base_weights,
        simulations=args.simulations,
        concentration=args.concentration,
    )

    ensure_parent(top1_out)
    ensure_parent(top5_out)
    top1_df.to_csv(top1_out, index=False)
    top5_df.to_csv(top5_out, index=False)
    make_figure(top1_df, fig_out, top_k=args.top_k)

    summary = pd.DataFrame(
        [
            {"metric": "simulations", "value": args.simulations},
            {"metric": "zip_count", "value": int(df["zip5"].nunique())},
            {"metric": "base_weight_demand", "value": base_weights["demand_potential"]},
            {"metric": "base_weight_gap", "value": base_weights["supply_gap"]},
            {"metric": "base_weight_momentum", "value": base_weights["growth_momentum"]},
            {"metric": "base_weight_text", "value": base_weights["text_signal"]},
            {"metric": "base_weight_feasibility", "value": base_weights["feasibility"]},
            {"metric": "most_stable_top1_zip", "value": str(top1_df.iloc[0]["zip5"]) if not top1_df.empty else "n/a"},
            {"metric": "most_stable_top1_freq_pct", "value": float(top1_df.iloc[0]["top1_freq_pct"]) if not top1_df.empty else 0.0},
        ]
    )
    ensure_parent(summary_out)
    summary.to_json(summary_out, orient="records", indent=2)

    print(
        "[17] Completed. "
        f"simulations={args.simulations}, zip_count={df['zip5'].nunique()}, "
        f"top1_zip={top1_df.iloc[0]['zip5'] if not top1_df.empty else 'n/a'}, "
        f"top1_freq={top1_df.iloc[0]['top1_freq_pct'] if not top1_df.empty else 0:.2f}%"
    )
    print(f"[17] top1 table: {top1_out}")
    print(f"[17] top5 table: {top5_out}")
    print(f"[17] summary: {summary_out}")
    print(f"[17] figure: {fig_out}")


if __name__ == "__main__":
    main()
