from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from hw2_utils import (
    DEFAULT_SPLIT_PATH,
    compute_binary_metrics,
    create_or_load_split,
    ensure_dir,
    load_and_prepare_dataset,
    materialize_split,
    resolve_in_project,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW2 Part 1b: TF-IDF baselines.")
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
        help="Output directory for baseline predictions and metrics.",
    )
    return parser.parse_args()


def train_tfidf_baseline(
    train_texts: pd.Series,
    train_labels: pd.Series,
    val_texts: pd.Series,
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
    min_df: int,
    seed: int,
) -> tuple[TfidfVectorizer, LogisticRegression, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)
    classifier.fit(train_features, train_labels)
    ai_probs = classifier.predict_proba(val_features)[:, 1]
    return vectorizer, classifier, ai_probs


def save_roc_curve(y_true: pd.Series, y_score: np.ndarray, model_name: str, output_path: Path) -> None:
    if plt is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4E79A7", linewidth=2, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion_matrix_plot(y_true: pd.Series, y_pred: np.ndarray, model_name: str, output_path: Path) -> None:
    if plt is None:
        return
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Human", "AI"])
    plt.yticks([0, 1], ["Human", "AI"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_classification_report_payload(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Human", "AI"],
        output_dict=True,
        zero_division=0,
    )


def main() -> None:
    args = parse_args()

    output_root = ensure_dir(args.output_root)
    model_dir = ensure_dir(output_root / "models")
    plot_dir = ensure_dir(output_root / "plots")

    data_path = resolve_in_project(args.data_path, must_exist=True)
    split_path = resolve_in_project(args.split_path)

    df = load_and_prepare_dataset(data_path, text_col=args.text_col, label_col=args.label_col)
    split_payload = create_or_load_split(df, split_path=split_path, seed=args.seed, val_ratio=args.val_ratio)
    train_df, val_df = materialize_split(df, split_payload)

    word_vectorizer, word_clf, word_probs = train_tfidf_baseline(
        train_df["text"],
        train_df["label"],
        val_df["text"],
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=2,
        seed=args.seed,
    )
    char_vectorizer, char_clf, char_probs = train_tfidf_baseline(
        train_df["text"],
        train_df["label"],
        val_df["text"],
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=80_000,
        min_df=2,
        seed=args.seed,
    )

    joblib.dump(
        {"vectorizer": word_vectorizer, "classifier": word_clf},
        model_dir / "word_tfidf_logreg.joblib",
    )
    joblib.dump(
        {"vectorizer": char_vectorizer, "classifier": char_clf},
        model_dir / "char_tfidf_logreg.joblib",
    )

    predictions_df = pd.DataFrame(
        {
            "row_id": val_df["row_id"],
            "label": val_df["label"],
            "word_tfidf_ai_prob": word_probs,
            "word_tfidf_pred_label": (word_probs >= 0.5).astype(int),
            "char_tfidf_ai_prob": char_probs,
            "char_tfidf_pred_label": (char_probs >= 0.5).astype(int),
        }
    )
    predictions_path = output_root / "validation_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    word_pred = (word_probs >= 0.5).astype(int)
    char_pred = (char_probs >= 0.5).astype(int)

    save_roc_curve(val_df["label"], word_probs, "Word TF-IDF + LR", plot_dir / "roc_curve_word_tfidf.png")
    save_roc_curve(val_df["label"], char_probs, "Char TF-IDF + LR", plot_dir / "roc_curve_char_tfidf.png")
    save_confusion_matrix_plot(
        val_df["label"],
        word_pred,
        "Word TF-IDF + LR",
        plot_dir / "confusion_matrix_word_tfidf.png",
    )
    save_confusion_matrix_plot(
        val_df["label"],
        char_pred,
        "Char TF-IDF + LR",
        plot_dir / "confusion_matrix_char_tfidf.png",
    )

    word_report_path = output_root / "classification_report_word_tfidf.json"
    char_report_path = output_root / "classification_report_char_tfidf.json"
    write_json(build_classification_report_payload(val_df["label"], word_pred), word_report_path)
    write_json(build_classification_report_payload(val_df["label"], char_pred), char_report_path)

    metrics_payload = {
        "word_tfidf_logreg": compute_binary_metrics(val_df["label"], word_probs),
        "char_tfidf_logreg": compute_binary_metrics(val_df["label"], char_probs),
        "split_path": str(split_path),
        "validation_predictions": str(predictions_path),
        "roc_curve_word_tfidf": str(plot_dir / "roc_curve_word_tfidf.png"),
        "roc_curve_char_tfidf": str(plot_dir / "roc_curve_char_tfidf.png"),
        "confusion_matrix_word_tfidf": str(plot_dir / "confusion_matrix_word_tfidf.png"),
        "confusion_matrix_char_tfidf": str(plot_dir / "confusion_matrix_char_tfidf.png"),
        "classification_report_word_tfidf": str(word_report_path),
        "classification_report_char_tfidf": str(char_report_path),
    }
    write_json(metrics_payload, output_root / "metrics.json")

    print(f"Prepared {len(df):,} cleaned rows from {data_path.name}")
    print(f"Saved shared split to {split_path}")
    print(f"Word TF-IDF ROC-AUC: {metrics_payload['word_tfidf_logreg']['roc_auc']:.4f}")
    print(f"Char TF-IDF ROC-AUC: {metrics_payload['char_tfidf_logreg']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
