from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SPLIT_PATH = DEFAULT_OUTPUT_ROOT / "splits" / "train_val_split_seed42.json"
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def configure_visible_gpu_from_cli(default_gpu: str = "1") -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_env", default=default_gpu)
    known_args, _ = parser.parse_known_args()
    gpu_env = str(known_args.gpu_env)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env
    return gpu_env


def resolve_in_project(pathlike: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(pathlike)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside {PROJECT_ROOT}: {resolved}") from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def ensure_dir(pathlike: str | Path) -> Path:
    path = resolve_in_project(pathlike)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(payload: dict[str, Any], pathlike: str | Path) -> Path:
    path = resolve_in_project(pathlike)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    return path


def read_json(pathlike: str | Path) -> dict[str, Any]:
    path = resolve_in_project(pathlike, must_exist=True)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_raw_dataframe(data_path: str | Path) -> pd.DataFrame:
    path = resolve_in_project(data_path, must_exist=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def normalize_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {
            "0": 0,
            "1": 1,
            "human": 0,
            "student": 0,
            "student-written": 0,
            "student_written": 0,
            "student-written essay": 0,
            "human-written": 0,
            "human_written": 0,
            "human-written essay": 0,
            "ai": 1,
            "llm": 1,
            "generated": 1,
            "machine": 1,
            "ai-generated": 1,
            "ai_generated": 1,
            "ai-generated essay": 1,
        }
        if lowered in mapping:
            return mapping[lowered]
        try:
            return int(float(lowered))
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_and_prepare_dataset(
    data_path: str | Path,
    *,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    raw_df = load_raw_dataframe(data_path)
    missing_columns = {text_col, label_col} - set(raw_df.columns)
    if missing_columns:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    prepared = raw_df[[text_col, label_col]].copy()
    prepared["source_index"] = raw_df.index.to_numpy()
    prepared = prepared.rename(columns={text_col: "text", label_col: "label"})
    prepared["text"] = prepared["text"].astype(str).str.strip()
    prepared["label"] = prepared["label"].apply(normalize_label)
    prepared = prepared.dropna(subset=["text", "label"])
    prepared = prepared[prepared["text"] != ""]
    prepared["label"] = prepared["label"].astype(int)
    prepared = prepared[prepared["label"].isin([0, 1])]
    prepared = prepared.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    prepared.insert(0, "row_id", np.arange(len(prepared), dtype=int))
    return prepared[["row_id", "source_index", "text", "label"]]


def create_or_load_split(
    df: pd.DataFrame,
    *,
    split_path: str | Path | None = None,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    target_path = resolve_in_project(split_path or DEFAULT_SPLIT_PATH)
    if target_path.exists():
        payload = read_json(target_path)
        validate_split_payload(payload, df)
        return payload

    train_ids, val_ids = train_test_split(
        df["row_id"].to_numpy(),
        test_size=val_ratio,
        random_state=seed,
        stratify=df["label"].to_numpy(),
    )
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "row_count": int(len(df)),
        "train_count": int(len(train_ids)),
        "val_count": int(len(val_ids)),
        "label_distribution": {
            "0": int((df["label"] == 0).sum()),
            "1": int((df["label"] == 1).sum()),
        },
        "train_row_ids": sorted(int(row_id) for row_id in train_ids.tolist()),
        "val_row_ids": sorted(int(row_id) for row_id in val_ids.tolist()),
    }
    write_json(payload, target_path)
    return payload


def validate_split_payload(payload: dict[str, Any], df: pd.DataFrame) -> None:
    row_ids = set(int(row_id) for row_id in df["row_id"].tolist())
    train_ids = {int(row_id) for row_id in payload.get("train_row_ids", [])}
    val_ids = {int(row_id) for row_id in payload.get("val_row_ids", [])}
    if not train_ids or not val_ids:
        raise ValueError("Split payload must contain non-empty train_row_ids and val_row_ids.")
    if train_ids & val_ids:
        raise ValueError("Split payload has overlapping train/val row IDs.")
    if (train_ids | val_ids) != row_ids:
        raise ValueError("Split payload does not match the cleaned dataset row IDs.")


def materialize_split(df: pd.DataFrame, split_payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = set(int(row_id) for row_id in split_payload["train_row_ids"])
    val_ids = set(int(row_id) for row_id in split_payload["val_row_ids"])
    train_df = df[df["row_id"].isin(train_ids)].sort_values("row_id").reset_index(drop=True)
    val_df = df[df["row_id"].isin(val_ids)].sort_values("row_id").reset_index(drop=True)
    return train_df, val_df


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def vocabulary_richness(text: str) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def add_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["word_count"] = enriched["text"].apply(word_count)
    enriched["vocabulary_richness"] = enriched["text"].apply(vocabulary_richness)
    return enriched


def batched(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_token_lengths(texts: Iterable[str], tokenizer: Any, *, batch_size: int = 64) -> list[int]:
    lengths: list[int] = []
    for batch in batched(texts, batch_size):
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    return lengths


def softmax_positive_class(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits[:, 1] / exp_logits.sum(axis=1)


def compute_binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_score_array = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score_array >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if len(np.unique(y_true_array)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics
