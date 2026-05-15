import csv
import json
import re
from pathlib import Path
from typing import Any

import torch


CONCISE_MAX_NEW_TOKENS = 8
NATURAL_MAX_NEW_TOKENS = 64
RELAXED_NUMERIC_TOLERANCE = 0.05


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return str(answer).strip()


def normalize_text(text: Any) -> str:
    text = normalize_answer(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def postprocess_prediction(text: Any) -> str:
    text = normalize_answer(text)
    for separator in ("\n", "USER:", "ASSISTANT:"):
        if separator in text:
            text = text.split(separator, 1)[0]
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return ""

    lowered = text.lower()
    for prefix in ("answer:", "the answer is", "it is", "it's"):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip(" .,:;")
            lowered = text.lower()
    if not text:
        return ""

    tokens = text.split()
    first_token = tokens[0].strip(" .,:;")

    if first_token.lower() in {"yes", "no"}:
        return first_token.capitalize()

    if try_parse_number(first_token) is not None:
        return first_token

    number_match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", text)
    if number_match is not None:
        return number_match.group(0)

    deduplicated_tokens = [tokens[0]]
    for token in tokens[1:]:
        if token != deduplicated_tokens[-1]:
            deduplicated_tokens.append(token)

    return " ".join(deduplicated_tokens[:3]).strip(" .")


def try_parse_number(text: Any) -> float | None:
    normalized = normalize_text(text).replace(",", "")
    match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?", normalized)
    if match is None:
        return None
    if normalized.endswith("%"):
        normalized = normalized[:-1]
    try:
        return float(normalized)
    except ValueError:
        return None


def exact_match(prediction: Any, ground_truth: Any) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def relaxed_match(
    prediction: Any,
    ground_truth: Any,
    tolerance: float = RELAXED_NUMERIC_TOLERANCE,
) -> bool:
    pred_number = try_parse_number(prediction)
    gt_number = try_parse_number(ground_truth)

    if pred_number is not None and gt_number is not None:
        if gt_number == 0:
            return abs(pred_number - gt_number) <= tolerance
        return abs(pred_number - gt_number) <= tolerance * abs(gt_number)

    return exact_match(prediction, ground_truth)


def format_concise_question(question: str) -> str:
    return (
        "Answer the question based on the chart. "
        "Give only the final answer, without explanation.\n"
        f"Question: {str(question).strip()}"
    )


def build_prompt(processor, question: str, concise: bool = False) -> str:
    prompt_text = format_concise_question(question) if concise else str(question).strip()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(
    model,
    processor,
    image,
    question: str,
    max_new_tokens: int | None = None,
    concise: bool = False,
) -> str:
    if max_new_tokens is None:
        max_new_tokens = CONCISE_MAX_NEW_TOKENS if concise else NATURAL_MAX_NEW_TOKENS

    inputs = processor(
        text=build_prompt(processor, question, concise=concise),
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    answer_ids = generate_ids[:, prompt_length:]
    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if concise:
        return postprocess_prediction(answer)
    return re.sub(r"\s+", " ", answer).strip()


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(exact_match(row["prediction"], row["ground_truth"]) for row in rows)
    relaxed = sum(relaxed_match(row["prediction"], row["ground_truth"]) for row in rows)

    return {
        "num_samples": total,
        "exact_match": exact / total if total else 0.0,
        "relaxed_accuracy": relaxed / total if total else 0.0,
    }


def save_prediction_artifacts(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_prediction_rows(rows)

    metrics_path = output_path / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    csv_path = output_path / f"{prefix}_predictions.csv"
    fieldnames = ["sample", "question", "ground_truth", "prediction"]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "sample": row.get("sample"),
                "question": row.get("question"),
                "ground_truth": row.get("ground_truth"),
                "prediction": row.get("prediction"),
            }
            for row in rows
        )

    return metrics
