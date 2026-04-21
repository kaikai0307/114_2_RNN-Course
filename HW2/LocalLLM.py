from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from hw2_utils import (
    DEFAULT_SPLIT_PATH,
    add_text_statistics,
    configure_visible_gpu_from_cli,
    create_or_load_split,
    ensure_dir,
    load_and_prepare_dataset,
    materialize_split,
    resolve_in_project,
    safe_model_name,
    write_json,
)

GPU_ENV = configure_visible_gpu_from_cli("1")

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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


PROMPT_TEMPLATES = {
    "student_voice": (
        "Rewrite the essay so it sounds like it was written by a thoughtful high school student. "
        "Keep the original meaning, but use a natural voice with varied sentence lengths and no AI disclaimers.\n\n"
        "Essay:\n{essay}"
    ),
    "natural_imperfect": (
        "Rewrite the essay to feel human and slightly imperfect while preserving the argument. "
        "Use natural transitions, a few uneven sentence lengths, and realistic phrasing. "
        "Do not mention that you rewrote anything.\n\nEssay:\n{essay}"
    ),
    "minimal_edit": (
        "Revise the essay with the lightest possible touch. Preserve most wording, keep the same structure, "
        "and do not make it sound polished or assistant-like. Keep a few rough edges if they already exist.\n\n"
        "Essay:\n{essay}"
    ),
    "messy_student": (
        "Rewrite the essay so it sounds like a real student draft written under time pressure. "
        "Keep the argument and most examples, but allow a few awkward transitions, uneven sentences, and ordinary phrasing. "
        "Avoid polished rhetoric, AI disclaimers, or list-like structure.\n\nEssay:\n{essay}"
    ),
    "lighter_revision": (
        "Make the essay read like a lightly edited school submission. Preserve the original claims and evidence, "
        "avoid over-explaining, and keep the tone plain, concrete, and slightly imperfect.\n\nEssay:\n{essay}"
    ),
}

PROMPT_SETS = {
    "default": ["student_voice", "natural_imperfect"],
    "stealth": ["minimal_edit", "messy_student", "lighter_revision"],
    "all": list(PROMPT_TEMPLATES.keys()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW2 Part 3: Local LLM adversarial attack.")
    parser.add_argument("--detector_dir", required=True, help="Path to the best BERT detector directory inside HW2.")
    parser.add_argument("--data_path", default="./train_v2_drcat_02.csv", required=False, help="Dataset path inside HW2.")
    parser.add_argument(
        "--split_path",
        default=str(DEFAULT_SPLIT_PATH.relative_to(Path(__file__).resolve().parent)),
        help="Shared split JSON path inside HW2.",
    )
    parser.add_argument("--text_col", default="text", help="Input text column name.")
    parser.add_argument("--label_col", default="label", help="Binary label column name.")
    parser.add_argument(
        "--gen_model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Local generation model for adversarial rewriting.",
    )
    parser.add_argument("--num_source_essays", type=int, default=10, help="Number of human source essays to attack.")
    parser.add_argument(
        "--prompt_set",
        choices=sorted(PROMPT_SETS.keys()),
        default="default",
        help="Prompt template subset used for rewriting.",
    )
    parser.add_argument("--gpu_env", default=GPU_ENV, help="CUDA_VISIBLE_DEVICES target. Defaults to GPU 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source essay selection.")
    parser.add_argument("--max_length", type=int, default=512, help="Detector tokenizer max length.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated rewrite length.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty for generation.")
    parser.add_argument("--detector_batch_size", type=int, default=8, help="Batch size for detector scoring.")
    parser.add_argument("--stage", choices=["all", "generate", "evaluate"], default="all", help="Run full pipeline or a single stage.")
    parser.add_argument("--output_tag", default="", help="Optional suffix to separate attack runs.")
    parser.add_argument("--output_root", default="outputs/attacks", help="Directory for attack generations and summaries.")
    return parser.parse_args()


def detector_device() -> torch.device:
    if torch is None:
        raise ImportError("torch is required before running LocalLLM.py.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)[:, 1]
        scores.append(probabilities.detach().cpu().numpy())
    if not scores:
        return np.array([], dtype=float)
    return np.concatenate(scores)


def bucket_targets(num_source_essays: int) -> dict[str, int]:
    weights = [("short", 0.3), ("medium", 0.4), ("long", 0.3)]
    raw_counts = {name: num_source_essays * weight for name, weight in weights}
    counts = {name: int(np.floor(value)) for name, value in raw_counts.items()}
    remainder = num_source_essays - sum(counts.values())
    ranked = sorted(weights, key=lambda item: raw_counts[item[0]] - counts[item[0]], reverse=True)
    for bucket_name, _ in ranked[:remainder]:
        counts[bucket_name] += 1
    return counts


def assign_length_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    lower_cut = float(enriched["word_count"].quantile(1 / 3))
    upper_cut = float(enriched["word_count"].quantile(2 / 3))

    def _bucket(word_count: int) -> str:
        if word_count <= lower_cut:
            return "short"
        if word_count <= upper_cut:
            return "medium"
        return "long"

    enriched["length_bucket"] = enriched["word_count"].apply(_bucket)
    return enriched


def select_source_essays(candidate_frame: pd.DataFrame, num_source_essays: int, seed: int) -> pd.DataFrame:
    candidates = assign_length_bucket(candidate_frame)
    targets = bucket_targets(num_source_essays)
    selections: list[pd.DataFrame] = []
    selected_ids: set[int] = set()

    for bucket, target_count in targets.items():
        bucket_frame = candidates[candidates["length_bucket"] == bucket]
        take_count = min(target_count, len(bucket_frame))
        if take_count == 0:
            continue
        sampled = bucket_frame.sample(n=take_count, random_state=seed)
        selections.append(sampled)
        selected_ids.update(sampled["row_id"].astype(int).tolist())

    combined = pd.concat(selections, ignore_index=True) if selections else candidates.iloc[0:0].copy()
    remaining = num_source_essays - len(combined)
    if remaining > 0:
        remainder_pool = candidates[~candidates["row_id"].isin(selected_ids)]
        if not remainder_pool.empty:
            filler = remainder_pool.sample(
                n=min(remaining, len(remainder_pool)),
                random_state=seed,
            )
            combined = pd.concat([combined, filler], ignore_index=True)

    return combined.sort_values(["length_bucket", "word_count", "row_id"]).head(num_source_essays).reset_index(drop=True)


def load_detector(detector_dir: Path) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(detector_dir)
    model = AutoModelForSequenceClassification.from_pretrained(detector_dir)
    model.to(detector_device())
    model.eval()
    return model, tokenizer


def clear_cuda() -> None:
    if torch is None:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_generator(model_name: str):
    try:
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
    except Exception as exc:
        message = (
            f"Unable to load generation model '{model_name}'. "
            "If this is a gated Llama checkpoint, authenticate with Hugging Face first. "
            "If you need a fallback, explicitly switch --gen_model to mistralai/Mistral-7B-Instruct-v0.3."
        )
        raise RuntimeError(message) from exc


def build_prompt(tokenizer: AutoTokenizer, prompt_name: str, essay: str) -> str:
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt template: {prompt_name}")
    messages = [
        {"role": "system", "content": "You rewrite essays while preserving the original meaning."},
        {"role": "user", "content": PROMPT_TEMPLATES[prompt_name].format(essay=essay)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "System: You rewrite essays while preserving the original meaning.\n"
        f"User: {PROMPT_TEMPLATES[prompt_name].format(essay=essay)}\nAssistant:"
    )


def active_prompt_names(prompt_set: str) -> list[str]:
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"Unsupported prompt set: {prompt_set}")
    return PROMPT_SETS[prompt_set]


def plot_attack_summary(attack_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or attack_df.empty:
        return

    prompt_order = sorted(attack_df["prompt_name"].unique())

    success_rates = (
        attack_df.groupby("prompt_name")["attack_success"]
        .mean()
        .reindex(prompt_order)
    )
    plt.figure(figsize=(8, 4.5))
    plt.bar(success_rates.index, success_rates.values, color="#4E79A7")
    plt.ylim(0, 1)
    plt.ylabel("Attack Success Rate")
    plt.title("Attack Success Rate by Prompt")
    plt.tight_layout()
    plt.savefig(output_dir / "attack_success_by_prompt.png", dpi=160)
    plt.close()

    grouped_probs = [
        attack_df.loc[attack_df["prompt_name"] == prompt_name, "rewritten_ai_prob"].to_numpy()
        for prompt_name in prompt_order
    ]
    plt.figure(figsize=(8, 4.5))
    plt.boxplot(grouped_probs, labels=prompt_order)
    plt.ylabel("Rewritten AI Probability")
    plt.title("Rewritten AI Probability by Prompt")
    plt.tight_layout()
    plt.savefig(output_dir / "rewritten_ai_prob_by_prompt.png", dpi=160)
    plt.close()

    grouped_delta = [
        attack_df.loc[attack_df["prompt_name"] == prompt_name, "ai_prob_delta"].to_numpy()
        for prompt_name in prompt_order
    ]
    plt.figure(figsize=(8, 4.5))
    plt.boxplot(grouped_delta, labels=prompt_order)
    plt.ylabel("AI Probability Delta")
    plt.title("AI Probability Change by Prompt")
    plt.tight_layout()
    plt.savefig(output_dir / "ai_prob_delta_by_prompt.png", dpi=160)
    plt.close()


def write_case_tables(attack_df: pd.DataFrame, output_dir: Path) -> None:
    if attack_df.empty:
        return

    display_columns = [
        "source_row_id",
        "prompt_name",
        "original_text_ai_prob",
        "rewritten_ai_prob",
        "ai_prob_delta",
        "attack_success",
        "rewritten_text",
    ]
    attack_df.sort_values("rewritten_ai_prob").head(5)[display_columns].to_csv(
        output_dir / "best_attack_cases.csv",
        index=False,
    )
    attack_df.sort_values("rewritten_ai_prob", ascending=False).head(5)[display_columns].to_csv(
        output_dir / "worst_attack_cases.csv",
        index=False,
    )


def run_generation(source_frame: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    generator = load_generator(args.gen_model)
    attack_rows: list[dict[str, object]] = []
    prompt_names = active_prompt_names(args.prompt_set)

    for _, row in source_frame.iterrows():
        essay = row["text"]
        for prompt_name in prompt_names:
            prompt = build_prompt(generator.tokenizer, prompt_name, essay)
            outputs = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                return_full_text=False,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
            generated_text = outputs[0]["generated_text"].strip()
            attack_rows.append(
                {
                    "source_row_id": int(row["row_id"]),
                    "source_label": int(row["label"]),
                    "source_word_count": int(row["word_count"]),
                    "length_bucket": row["length_bucket"],
                    "prompt_name": prompt_name,
                    "original_text": essay,
                    "original_text_ai_prob": float(row["original_text_ai_prob"]),
                    "rewritten_text": generated_text,
                    "prompt_set": args.prompt_set,
                }
            )

    attack_df = pd.DataFrame(attack_rows)
    attack_df.to_csv(output_dir / "attack_generations.csv", index=False)

    del generator
    clear_cuda()
    return attack_df


def run_evaluation(
    attack_df: pd.DataFrame,
    detector_dir: Path,
    *,
    max_length: int,
    batch_size: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    detector_model, detector_tokenizer = load_detector(detector_dir)
    rewritten_scores = score_texts(
        detector_model,
        detector_tokenizer,
        attack_df["rewritten_text"].tolist(),
        max_length=max_length,
        batch_size=batch_size,
    )
    attack_df = attack_df.copy()
    attack_df["rewritten_ai_prob"] = rewritten_scores
    attack_df["rewritten_pred_label"] = (rewritten_scores >= 0.5).astype(int)
    attack_df["attack_success"] = (attack_df["rewritten_pred_label"] == 0).astype(int)
    attack_df["ai_prob_delta"] = attack_df["rewritten_ai_prob"] - attack_df["original_text_ai_prob"]
    attack_df.to_csv(output_dir / "attack_results.csv", index=False)

    summary = {
        "detector_dir": str(detector_dir),
        "generation_model": attack_df.get("generation_model", pd.Series(dtype=str)).iloc[0]
        if "generation_model" in attack_df.columns and not attack_df.empty
        else None,
        "prompt_set": attack_df.get("prompt_set", pd.Series(dtype=str)).iloc[0]
        if "prompt_set" in attack_df.columns and not attack_df.empty
        else None,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "num_source_essays": int(attack_df["source_row_id"].nunique()),
        "num_attack_samples": int(len(attack_df)),
        "overall_attack_success_rate": float(attack_df["attack_success"].mean()) if len(attack_df) else 0.0,
        "mean_original_ai_prob": float(attack_df["original_text_ai_prob"].mean()) if len(attack_df) else 0.0,
        "mean_rewritten_ai_prob": float(attack_df["rewritten_ai_prob"].mean()) if len(attack_df) else 0.0,
        "success_rate_by_prompt": {
            prompt_name: float(group["attack_success"].mean())
            for prompt_name, group in attack_df.groupby("prompt_name")
        },
        "mean_rewritten_ai_prob_by_prompt": {
            prompt_name: float(group["rewritten_ai_prob"].mean())
            for prompt_name, group in attack_df.groupby("prompt_name")
        },
    }
    write_json(summary, output_dir / "attack_summary.json")
    plot_attack_summary(attack_df, output_dir)
    write_case_tables(attack_df, output_dir)

    del detector_model
    clear_cuda()
    return attack_df


def main() -> None:
    args = parse_args()
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None or pipeline is None:
        raise ImportError("torch and transformers are required before running LocalLLM.py.")

    data_path = resolve_in_project(args.data_path, must_exist=True)
    split_path = resolve_in_project(args.split_path)
    detector_dir = resolve_in_project(args.detector_dir, must_exist=True)
    output_root = ensure_dir(args.output_root)
    detector_slug = safe_model_name(str(detector_dir.relative_to(Path(__file__).resolve().parent)))
    output_tag = safe_model_name(args.output_tag) if args.output_tag else "default"
    generation_config_slug = safe_model_name(
        f"{args.prompt_set}_temp{args.temperature}_top_p{args.top_p}_max{args.max_new_tokens}_rep{args.repetition_penalty}"
    )
    output_dir = ensure_dir(
        output_root
        / detector_slug
        / safe_model_name(args.gen_model)
        / generation_config_slug
        / output_tag
        / f"seed{args.seed}"
    )

    df = load_and_prepare_dataset(data_path, text_col=args.text_col, label_col=args.label_col)
    split_payload = create_or_load_split(df, split_path=split_path, seed=args.seed, val_ratio=0.2)
    _, val_df = materialize_split(df, split_payload)
    val_stats_df = add_text_statistics(val_df)

    detector_model, detector_tokenizer = load_detector(detector_dir)
    human_val_df = val_stats_df[val_stats_df["label"] == 0].copy()
    if human_val_df.empty:
        raise ValueError("Validation split contains no human essays to attack.")

    original_scores = score_texts(
        detector_model,
        detector_tokenizer,
        human_val_df["text"].tolist(),
        max_length=args.max_length,
        batch_size=args.detector_batch_size,
    )
    human_val_df["original_text_ai_prob"] = original_scores
    human_val_df["original_pred_label"] = (original_scores >= 0.5).astype(int)
    candidate_sources = human_val_df[human_val_df["original_pred_label"] == 0].copy()

    del detector_model
    clear_cuda()

    if candidate_sources.empty:
        raise RuntimeError("Detector did not correctly classify any human validation essays for source selection.")

    selected_sources = select_source_essays(candidate_sources, args.num_source_essays, args.seed)
    selected_sources.to_csv(output_dir / "selected_source_essays.csv", index=False)

    generation_path = output_dir / "attack_generations.csv"
    if args.stage in {"all", "generate"}:
        generation_df = run_generation(selected_sources, args, output_dir)
        generation_df["generation_model"] = args.gen_model
        generation_df.to_csv(generation_path, index=False)
    else:
        if not generation_path.exists():
            raise FileNotFoundError(f"Missing generation file for evaluation stage: {generation_path}")
        generation_df = pd.read_csv(generation_path)

    if args.stage in {"all", "evaluate"}:
        result_df = run_evaluation(
            generation_df,
            detector_dir,
            max_length=args.max_length,
            batch_size=args.detector_batch_size,
            output_dir=output_dir,
            args=args,
        )
        print(f"Attack samples evaluated: {len(result_df)}")
        print(f"Overall attack success rate: {result_df['attack_success'].mean():.4f}")
    else:
        print(f"Generated attack samples: {len(generation_df)}")

    print(f"Visible GPU selection: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Selected human source essays: {selected_sources['row_id'].nunique()}")


if __name__ == "__main__":
    main()
