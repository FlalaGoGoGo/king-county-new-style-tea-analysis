#!/usr/bin/env python3
"""Run sentiment + topic extraction for focus districts.
Input: data/raw/focus_district_reviews.csv
Output: text sentiment/topic tables in data/processed/
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


POS_WORDS = {
    "great", "good", "excellent", "fresh", "friendly", "clean", "balanced",
    "quick", "solid", "love", "amazing", "best", "nice", "recommend",
    "recommended", "awesome", "fast", "perfect", "smooth", "tasty",
    "flavorful", "authentic", "cozy", "quality", "delicious",
}
NEG_WORDS = {
    "bad", "slow", "long", "wait", "waiting", "overcooked", "inconsistent",
    "watery", "watered", "worse", "worst", "noisy", "dirty", "expensive",
    "overpriced", "bland", "disappoint", "disappointed", "issue", "problem",
    "wrong", "soggy", "cold",
}
STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with",
    "is", "are", "was", "were", "it", "this", "that", "at", "as", "be",
    "by", "from", "overall", "very", "really", "would", "could",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/raw/focus_district_reviews.csv",
        help="Input review file.",
    )
    parser.add_argument(
        "--reference-input",
        default="data/raw/yelp_open_boba_reference_reviews.csv",
        help="Optional external reference corpus for topic model training.",
    )
    parser.add_argument(
        "--sentiment-output",
        default="data/processed/text_sentiment_by_district.csv",
        help="Output path for district-level sentiment table.",
    )
    parser.add_argument(
        "--topics-output",
        default="data/processed/text_topics_by_district.csv",
        help="Output path for district-level topic table.",
    )
    parser.add_argument(
        "--figure-output",
        default="outputs/figures/fig_text_sentiment_topics.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=4,
        help="Max topics for LDA mode.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    if "zip5" not in out.columns:
        for col in ["zip", "district_id", "zipcode"]:
            if col in out.columns:
                out["zip5"] = out[col]
                break
    if "review_text" not in out.columns:
        for col in ["text", "review", "comment", "content"]:
            if col in out.columns:
                out["review_text"] = out[col]
                break
    if "shop_name" not in out.columns:
        out["shop_name"] = "Unknown Shop"
    if "rating" not in out.columns:
        out["rating"] = pd.NA

    out["zip5"] = out["zip5"].astype(str).str.extract(r"(\d{5})", expand=False)
    out["review_text"] = out["review_text"].fillna("").astype(str)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out[out["zip5"].notna() & (out["review_text"].str.len() > 10)].copy()
    return out.reset_index(drop=True)


def normalize_reference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    if "review_text" not in out.columns:
        for col in ["text", "review", "comment", "content"]:
            if col in out.columns:
                out["review_text"] = out[col]
                break
    out["review_text"] = out.get("review_text", "").fillna("").astype(str)
    out = out[out["review_text"].str.len() > 10].copy()
    return out.reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in STOP_WORDS]


def sentiment_score(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = (pos - neg) / max(len(toks), 1)
    return max(-1.0, min(1.0, score * 4.0))


def build_sentiment_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_df = df.copy()
    row_df["sentiment_score"] = row_df["review_text"].apply(sentiment_score)
    row_df["sentiment_label"] = row_df["sentiment_score"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )
    agg = (
        row_df.groupby("zip5", as_index=False)
        .agg(
            review_count=("review_text", "count"),
            mean_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment_label", lambda s: (s == "positive").mean()),
            negative_share=("sentiment_label", lambda s: (s == "negative").mean()),
            avg_rating=("rating", "mean"),
        )
    )
    agg["avg_rating"] = agg["avg_rating"].fillna(agg["avg_rating"].median())
    agg["text_signal_score"] = ((agg["mean_sentiment"] + 1.0) / 2.0) * 100.0
    return row_df, agg.sort_values("text_signal_score", ascending=False).reset_index(drop=True)


def _topics_with_lda(row_df: pd.DataFrame, ref_df: pd.DataFrame | None, max_topics: int) -> pd.DataFrame:
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as exc:
        raise RuntimeError(f"sklearn unavailable: {exc}") from exc

    docs_local = row_df["review_text"].tolist()
    docs_train = docs_local
    if ref_df is not None and not ref_df.empty:
        docs_train = ref_df["review_text"].tolist() + docs_local

    if len(docs_train) < 10:
        raise RuntimeError("Not enough reviews for stable LDA.")

    vec = CountVectorizer(stop_words="english", min_df=2, max_df=0.95)
    X = vec.fit_transform(docs_train)
    if X.shape[1] < 6:
        raise RuntimeError("Vocabulary too small for LDA.")

    n_topics = max(2, min(max_topics, X.shape[0] // 8, X.shape[1] // 5))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=12)
    lda.fit(X)

    X_local = vec.transform(docs_local)
    topic_prob = lda.transform(X_local)
    topic_id = topic_prob.argmax(axis=1)

    feature_names = vec.get_feature_names_out()
    topic_terms = {}
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:5]
        topic_terms[i] = ", ".join(feature_names[j] for j in top_idx)

    tmp = row_df.copy()
    tmp["topic_id"] = topic_id
    dist = (
        tmp.groupby(["zip5", "topic_id"], as_index=False)
        .agg(topic_count=("review_text", "count"))
    )
    total = dist.groupby("zip5", as_index=False)["topic_count"].sum().rename(columns={"topic_count": "zip_total"})
    dist = dist.merge(total, on="zip5", how="left")
    dist["topic_share"] = dist["topic_count"] / dist["zip_total"]
    top = dist.sort_values(["zip5", "topic_share"], ascending=[True, False]).groupby("zip5", as_index=False).head(1)
    top["topic_terms"] = top["topic_id"].map(topic_terms)
    return top[["zip5", "topic_id", "topic_terms", "topic_share", "topic_count"]].reset_index(drop=True)


def _topics_with_term_frequency(row_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zip5, g in row_df.groupby("zip5"):
        cnt = Counter()
        for txt in g["review_text"]:
            cnt.update(tokenize(txt))
        top_terms = ", ".join([w for w, _ in cnt.most_common(5)]) if cnt else "n/a"
        rows.append(
            {
                "zip5": zip5,
                "topic_id": -1,
                "topic_terms": top_terms,
                "topic_share": 1.0,
                "topic_count": int(g.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def build_topics_table(row_df: pd.DataFrame, max_topics: int, ref_df: pd.DataFrame | None) -> pd.DataFrame:
    try:
        return _topics_with_lda(row_df, ref_df, max_topics)
    except Exception as exc:
        print(f"[07] Warning: LDA mode failed ({exc}). Falling back to term-frequency topics.")
        return _topics_with_term_frequency(row_df)


def make_figure(sent_df: pd.DataFrame, topic_df: pd.DataFrame, fig_path: Path) -> None:
    ensure_parent(fig_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    s = sent_df.sort_values("mean_sentiment")
    axes[0].barh(s["zip5"], s["mean_sentiment"], color="#2a9d8f")
    axes[0].axvline(0.0, linestyle="--", linewidth=1)
    axes[0].set_title("Mean Sentiment by ZIP")
    axes[0].set_xlabel("Sentiment Score (-1 to 1)")
    axes[0].set_ylabel("ZIP")

    t = topic_df.copy()
    t["topic_label"] = t["topic_terms"].astype(str).str.slice(0, 36)
    axes[1].barh(t["zip5"], t["topic_share"], color="#457b9d")
    for y, (_, row) in enumerate(t.iterrows()):
        axes[1].text(
            min(float(row["topic_share"]) + 0.01, 0.98),
            y,
            row["topic_label"],
            va="center",
            fontsize=8,
        )
    axes[1].set_xlim(0, 1.0)
    axes[1].set_title("Top Topic Share by ZIP")
    axes[1].set_xlabel("Share")
    axes[1].set_ylabel("ZIP")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    root = project_root()
    inp = root / args.input
    ref_path = root / args.reference_input
    sent_out = root / args.sentiment_output
    topic_out = root / args.topics_output
    fig_out = root / args.figure_output

    if not inp.exists():
        raise FileNotFoundError(f"Review input not found: {inp}")

    raw = pd.read_csv(inp)
    norm = normalize_reviews(raw)
    if norm.empty:
        raise ValueError("No usable review rows after normalization.")

    ref_df = None
    if ref_path.exists():
        ref_raw = pd.read_csv(ref_path)
        ref_df = normalize_reference(ref_raw)
        if ref_df.empty:
            ref_df = None

    row_df, sent_df = build_sentiment_table(norm)
    topic_df = build_topics_table(row_df, args.max_topics, ref_df)

    ensure_parent(sent_out)
    ensure_parent(topic_out)
    sent_df.to_csv(sent_out, index=False)
    topic_df.to_csv(topic_out, index=False)
    make_figure(sent_df, topic_df, fig_out)

    print(
        "[07] Completed. "
        f"reviews={norm.shape[0]}, zips={sent_df['zip5'].nunique()}, "
        f"reference_reviews={0 if ref_df is None else ref_df.shape[0]}, "
        f"sentiment_out={sent_out}, topics_out={topic_out}"
    )
    print(f"[07] output figure: {fig_out}")


if __name__ == "__main__":
    main()
