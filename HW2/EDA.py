from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

from hw2_utils import (
    DEFAULT_SPLIT_PATH,
    add_text_statistics,
    compute_token_lengths,
    create_or_load_split,
    ensure_dir,
    load_and_prepare_dataset,
    materialize_split,
    resolve_in_project,
    write_json,
)


LABEL_NAMES = {0: "Human", 1: "AI"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW2 Part 1a: standalone EDA.")
    parser.add_argument("--data_path", default="./train_v2_drcat_02.csv", required=False, help="Dataset path inside HW2.")
    parser.add_argument("--text_col", default="text", help="Input text column name.")
    parser.add_argument("--label_col", default="label", help="Binary label column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the shared split.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument(
        "--split_path",
        default=str(DEFAULT_SPLIT_PATH.relative_to(Path(__file__).resolve().parent)),
        help="Shared split JSON path inside HW2.",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/baseline",
        help="Output directory for EDA artifacts.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-cased",
        help="Tokenizer used for truncation-rate analysis.",
    )
    parser.add_argument("--max_length", type=int, default=512, help="BERT sequence length for truncation stats.")
    return parser.parse_args()


def save_class_balance_plot(frame: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        return
    counts = frame["label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(
        [LABEL_NAMES[int(label)] for label in counts.index],
        counts.values,
        color=["#4E79A7", "#E15759"],
    )
    plt.title("Class Balance")
    plt.ylabel("Essay Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_histogram(frame: pd.DataFrame, column: str, title: str, xlabel: str, output_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    colors = {0: "#4E79A7", 1: "#E15759"}
    for label in sorted(frame["label"].unique()):
        subset = frame.loc[frame["label"] == label, column].to_numpy()
        plt.hist(
            subset,
            bins=40,
            alpha=0.55,
            label=LABEL_NAMES[int(label)],
            color=colors[int(label)],
            density=False,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Essay Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_boxplot(frame: pd.DataFrame, column: str, title: str, ylabel: str, output_path: Path) -> None:
    if plt is None:
        return
    grouped = [
        frame.loc[frame["label"] == label, column].to_numpy()
        for label in sorted(frame["label"].unique())
    ]
    plt.figure(figsize=(6, 5))
    plt.boxplot(grouped, labels=[LABEL_NAMES[int(label)] for label in sorted(frame["label"].unique())])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def summarize_stat(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for label, group in frame.groupby("label"):
        series = group[column]
        summary[LABEL_NAMES[int(label)]] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return summary


def sample_representative_text(frame: pd.DataFrame, label: int) -> pd.Series:
    subset = frame[frame["label"] == label].copy()
    target_word_count = float(subset["word_count"].median())
    subset["distance_to_median"] = (subset["word_count"] - target_word_count).abs()
    return subset.sort_values(["distance_to_median", "row_id"]).iloc[0]


def sentence_count(text: str) -> int:
    separators = [".", "!", "?"]
    total = sum(text.count(symbol) for symbol in separators)
    return max(total, 1)


def paragraph_count(text: str) -> int:
    chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    return max(len(chunks), 1)


def collect_qualitative_features(frame: pd.DataFrame) -> dict[str, dict[str, float | int | str]]:
    human_row = sample_representative_text(frame, 0)
    ai_row = sample_representative_text(frame, 1)

    human_sentence_avg = human_row["word_count"] / sentence_count(human_row["text"])
    ai_sentence_avg = ai_row["word_count"] / sentence_count(ai_row["text"])
    human_paragraphs = paragraph_count(human_row["text"])
    ai_paragraphs = paragraph_count(ai_row["text"])

    return {
        "human_representative_sample": {
            "row_id": int(human_row["row_id"]),
            "word_count": int(human_row["word_count"]),
            "paragraph_count": int(human_paragraphs),
            "avg_words_per_sentence": float(human_sentence_avg),
            "preview_text": human_row["text"][:700],
        },
        "ai_representative_sample": {
            "row_id": int(ai_row["row_id"]),
            "word_count": int(ai_row["word_count"]),
            "paragraph_count": int(ai_paragraphs),
            "avg_words_per_sentence": float(ai_sentence_avg),
            "preview_text": ai_row["text"][:700],
        },
    }


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    return float((2.0 * u_stat) / (n1 * n2) - 1.0)


def run_statistical_tests(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | str]] | None:
    if mannwhitneyu is None:
        return None

    human_frame = frame[frame["label"] == 0]
    ai_frame = frame[frame["label"] == 1]
    results: dict[str, dict[str, float | str]] = {}
    for column in columns:
        human_values = human_frame[column].to_numpy()
        ai_values = ai_frame[column].to_numpy()
        u_stat, p_value = mannwhitneyu(human_values, ai_values, alternative="two-sided")
        results[column] = {
            "test": "Mann-Whitney U",
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "rank_biserial_effect_size": rank_biserial_from_u(float(u_stat), len(human_values), len(ai_values)),
            "human_median": float(np.median(human_values)),
            "ai_median": float(np.median(ai_values)),
        }
    return results


def main() -> None:
    args = parse_args()
    if AutoTokenizer is None:
        raise ImportError("transformers is required for tokenizer-length analysis. Install it before running EDA.py.")

    output_root = ensure_dir(args.output_root)
    eda_dir = ensure_dir(output_root / "eda")

    data_path = resolve_in_project(args.data_path, must_exist=True)
    split_path = resolve_in_project(args.split_path)

    df = load_and_prepare_dataset(data_path, text_col=args.text_col, label_col=args.label_col)
    split_payload = create_or_load_split(df, split_path=split_path, seed=args.seed, val_ratio=args.val_ratio)
    train_df, val_df = materialize_split(df, split_payload)
    full_stats_df = add_text_statistics(df)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    token_lengths = compute_token_lengths(full_stats_df["text"].tolist(), tokenizer, batch_size=64)
    full_stats_df["token_length"] = token_lengths
    full_stats_df["is_truncated"] = full_stats_df["token_length"] > args.max_length

    save_class_balance_plot(full_stats_df, eda_dir / "class_balance.png")
    save_histogram(
        full_stats_df,
        column="word_count",
        title="Word Count Distribution by Label",
        xlabel="Word Count",
        output_path=eda_dir / "word_count_distribution.png",
    )
    save_boxplot(
        full_stats_df,
        column="word_count",
        title="Word Count by Label",
        ylabel="Word Count",
        output_path=eda_dir / "word_count_boxplot.png",
    )
    save_boxplot(
        full_stats_df,
        column="vocabulary_richness",
        title="Vocabulary Richness by Label",
        ylabel="Unique Tokens / Total Tokens",
        output_path=eda_dir / "vocabulary_richness_boxplot.png",
    )
    save_boxplot(
        full_stats_df,
        column="token_length",
        title="Tokenizer Length by Label",
        ylabel="Token Count",
        output_path=eda_dir / "token_length_boxplot.png",
    )

    qualitative_features = collect_qualitative_features(full_stats_df)
    statistical_tests = run_statistical_tests(
        full_stats_df,
        columns=["word_count", "vocabulary_richness", "token_length"],
    )

    eda_summary = {
        "data_path": str(data_path),
        "row_count": int(len(full_stats_df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "class_balance": {
            LABEL_NAMES[int(label)]: int(count)
            for label, count in full_stats_df["label"].value_counts().sort_index().items()
        },
        "word_count_stats": summarize_stat(full_stats_df, "word_count"),
        "vocabulary_richness_stats": summarize_stat(full_stats_df, "vocabulary_richness"),
        "token_length_stats": summarize_stat(full_stats_df, "token_length"),
        "truncation_rate_overall": float(full_stats_df["is_truncated"].mean()),
        "truncation_rate_by_label": {
            LABEL_NAMES[int(label)]: float(group["is_truncated"].mean())
            for label, group in full_stats_df.groupby("label")
        },
        "vocabulary_richness_formula": "unique_tokens / total_tokens",
        "qualitative_features": qualitative_features,
        "statistical_tests": statistical_tests,
    }
    write_json(eda_summary, output_root / "eda_summary.json")

    print(f"Prepared {len(df):,} cleaned rows from {data_path.name}")
    print(f"Saved shared split to {split_path}")
    if plt is None:
        print("Warning: matplotlib is not installed, so plot images were skipped.")
    print(f"Saved EDA artifacts to {eda_dir}")


if __name__ == "__main__":
    main()
