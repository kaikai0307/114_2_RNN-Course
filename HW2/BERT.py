from __future__ import annotations

import argparse
import gc
import inspect
import math
import os
from pathlib import Path

from hw2_utils import (
    DEFAULT_SPLIT_PATH,
    compute_binary_metrics,
    configure_visible_gpu_from_cli,
    create_or_load_split,
    ensure_dir,
    load_and_prepare_dataset,
    materialize_split,
    resolve_in_project,
    safe_model_name,
    softmax_positive_class,
    write_json,
)

GPU_ENV = configure_visible_gpu_from_cli("0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    DataCollatorWithPadding = None
    Trainer = None
    TrainingArguments = None
    set_seed = None


class EncodedTextDataset:
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW2 Part 2: BERT fine-tuning and scaling.")
    parser.add_argument("--data_path", default="./train_v2_drcat_02.csv", required=False, help="Dataset path inside HW2.")
    parser.add_argument(
        "--split_path",
        default=str(DEFAULT_SPLIT_PATH.relative_to(Path(__file__).resolve().parent)),
        help="Shared split JSON path inside HW2.",
    )
    parser.add_argument("--text_col", default="text", help="Input text column name.")
    parser.add_argument("--label_col", default="label", help="Binary label column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio used if split does not exist.")
    parser.add_argument("--model_name", default="bert-base-cased", help="HF checkpoint to fine-tune.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        default=32,
        help="Target effective batch size via gradient accumulation.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Initial per-device batch size. If omitted, the script picks a safe default and auto-retries on OOM.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging cadence.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--gpu_env", default=GPU_ENV, help="CUDA_VISIBLE_DEVICES target. Defaults to GPU 1.")
    parser.add_argument("--output_root", default="outputs/bert", help="Directory for checkpoints and metrics.")
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_true", help="Enable FP16 mixed precision.")
    fp16_group.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 mixed precision.")
    parser.set_defaults(fp16=True)
    return parser.parse_args()


def build_training_arguments(
    run_dir: Path,
    *,
    epochs: float,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    logging_steps: int,
    seed: int,
    fp16: bool,
    gradient_checkpointing: bool,
) -> TrainingArguments:
    if TrainingArguments is None:
        raise ImportError("transformers is required to build TrainingArguments.")
    kwargs = {
        "output_dir": str(run_dir / "trainer_output"),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "roc_auc",
        "greater_is_better": True,
        "seed": seed,
        "remove_unused_columns": True,
        "dataloader_num_workers": 0,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def candidate_batch_sizes(model_name: str, requested: int | None) -> list[int]:
    if requested is not None:
        sizes = [requested]
    elif "large" in model_name.lower():
        sizes = [4, 2, 1]
    else:
        sizes = [8, 4, 2, 1]

    expanded: list[int] = []
    for size in sizes:
        current = int(size)
        while current >= 1:
            expanded.append(current)
            current //= 2
    return sorted(set(expanded), reverse=True)


def tokenize_texts(tokenizer: AutoTokenizer, texts: list[str], max_length: int) -> dict[str, list[list[int]]]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def plot_training_history(history_frame: pd.DataFrame, output_path: Path) -> None:
    if history_frame.empty or plt is None:
        return

    plt.figure(figsize=(8, 5))
    if "loss" in history_frame.columns:
        train_points = history_frame.dropna(subset=["loss"])
        if not train_points.empty:
            plt.plot(train_points["step"], train_points["loss"], label="train_loss", color="#4E79A7")
    if "eval_loss" in history_frame.columns:
        eval_points = history_frame.dropna(subset=["eval_loss"])
        if not eval_points.empty:
            plt.plot(eval_points["step"], eval_points["eval_loss"], label="eval_loss", color="#E15759")
    if "eval_roc_auc" in history_frame.columns:
        eval_auc_points = history_frame.dropna(subset=["eval_roc_auc"])
        if not eval_auc_points.empty:
            plt.plot(
                eval_auc_points["step"],
                eval_auc_points["eval_roc_auc"],
                label="eval_roc_auc",
                color="#59A14F",
            )
    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if torch is None or Trainer is None or AutoTokenizer is None:
        raise ImportError("torch and transformers are required before running BERT.py.")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    data_path = resolve_in_project(args.data_path, must_exist=True)
    split_path = resolve_in_project(args.split_path)
    output_root = ensure_dir(args.output_root)
    model_slug = safe_model_name(args.model_name)
    model_root = ensure_dir(output_root / model_slug / f"seed{args.seed}")

    df = load_and_prepare_dataset(data_path, text_col=args.text_col, label_col=args.label_col)
    split_payload = create_or_load_split(df, split_path=split_path, seed=args.seed, val_ratio=args.val_ratio)
    train_df, val_df = materialize_split(df, split_payload)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_encodings = tokenize_texts(tokenizer, train_df["text"].tolist(), args.max_length)
    val_encodings = tokenize_texts(tokenizer, val_df["text"].tolist(), args.max_length)
    train_dataset = EncodedTextDataset(train_encodings, train_df["label"].astype(int).tolist())
    val_dataset = EncodedTextDataset(val_encodings, val_df["label"].astype(int).tolist())
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    gradient_checkpointing = "large" in args.model_name.lower()
    device_is_cuda = torch.cuda.is_available()
    if not device_is_cuda:
        print("Warning: CUDA is not available. Training will fall back to CPU.")

    trial_errors: list[str] = []
    successful_run: dict[str, object] | None = None

    for train_batch_size in candidate_batch_sizes(args.model_name, args.per_device_batch_size):
        gradient_accumulation_steps = max(1, math.ceil(args.effective_batch_size / train_batch_size))
        eval_batch_size = max(1, min(args.eval_batch_size, max(train_batch_size, 1) * 2))
        run_dir = ensure_dir(model_root / f"bs{train_batch_size}_ga{gradient_accumulation_steps}")

        if device_is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        training_args = build_training_arguments(
            run_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            seed=args.seed,
            fp16=bool(args.fp16 and device_is_cuda),
            gradient_checkpointing=gradient_checkpointing,
        )

        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            logits, labels = eval_pred
            positive_scores = softmax_positive_class(np.asarray(logits))
            return compute_binary_metrics(labels, positive_scores)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        try:
            train_result = trainer.train()
            eval_metrics = trainer.evaluate()
            prediction_output = trainer.predict(val_dataset)
            positive_scores = softmax_positive_class(np.asarray(prediction_output.predictions))
            final_metrics = compute_binary_metrics(val_df["label"], positive_scores)
            peak_vram_mb = (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                if device_is_cuda
                else 0.0
            )

            history_frame = pd.DataFrame(trainer.state.log_history)
            history_frame.to_csv(run_dir / "training_history.csv", index=False)
            plot_training_history(history_frame, run_dir / "training_history.png")

            predictions_df = pd.DataFrame(
                {
                    "row_id": val_df["row_id"],
                    "label": val_df["label"],
                    "ai_prob": positive_scores,
                    "pred_label": (positive_scores >= 0.5).astype(int),
                }
            )
            predictions_df.to_csv(run_dir / "validation_predictions.csv", index=False)

            best_model_dir = ensure_dir(run_dir / "best_model")
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            metrics_payload = {
                "model_name": args.model_name,
                "data_path": str(data_path),
                "split_path": str(split_path),
                "train_count": int(len(train_df)),
                "val_count": int(len(val_df)),
                "seed": args.seed,
                "requested_effective_batch_size": args.effective_batch_size,
                "per_device_train_batch_size": train_batch_size,
                "per_device_eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "fp16": bool(args.fp16 and device_is_cuda),
                "gradient_checkpointing": gradient_checkpointing,
                "gpu_env": str(args.gpu_env),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_name": torch.cuda.get_device_name(0) ,#if device_is_cuda else "cpu",
                "peak_vram_mb": peak_vram_mb,
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "validation_metrics": final_metrics,
                "best_model_dir": str(best_model_dir),
                "history_csv": str(run_dir / "training_history.csv"),
                "history_plot": str(run_dir / "training_history.png"),
                "validation_predictions": str(run_dir / "validation_predictions.csv"),
            }
            write_json(metrics_payload, run_dir / "metrics.json")

            successful_run = {
                "run_dir": run_dir,
                "metrics": final_metrics,
                "batch_size": train_batch_size,
                "accumulation": gradient_accumulation_steps,
            }
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            trial_errors.append(
                f"OOM at per_device_train_batch_size={train_batch_size}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}: {exc}"
            )
            del trainer
            del model
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()
            continue

    if successful_run is None:
        raise RuntimeError(
            "All BERT training trials ran out of memory.\n" + "\n".join(trial_errors)
        )

    print(f"Model: {args.model_name}")
    print(f"Visible GPU selection: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(
        "Successful configuration: "
        f"per-device batch size={successful_run['batch_size']}, "
        f"gradient accumulation={successful_run['accumulation']}"
    )
    if plt is None:
        print("Warning: matplotlib is not installed, so the training-history plot was skipped.")
    print(f"Validation ROC-AUC: {successful_run['metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
