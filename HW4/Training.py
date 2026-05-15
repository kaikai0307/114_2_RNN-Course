import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from Config import prepare_model
from Eval import (
    format_concise_question,
    generate_answer,
    normalize_answer,
    save_prediction_artifacts,
)


DATASET_NAME = "HuggingFaceM4/ChartQA"
TRAIN_SAMPLE_SIZE = 10000 #1000
VAL_SAMPLE_SIZE = 1000     #200
RANDOM_SEED = 42
NUM_TRAIN_EPOCHS = 3
MAX_GENERATIVE_EVAL_SAMPLES = 100
PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
LOGGING_STEPS = 1
EVAL_STEPS = 20
SAVE_STEPS = 20
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01


def build_conversation(
    sample: dict[str, Any],
    answer_eos_token: str | None = None,
) -> list[dict[str, Any]]:
    question = str(sample["query"]).strip()
    answer = normalize_answer(sample["label"])
    if answer_eos_token is not None and not answer.endswith(answer_eos_token):
        answer = f"{answer}{answer_eos_token}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": format_concise_question(question)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


@dataclass
class LlavaDataCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        images = [sample["image"].convert("RGB") for sample in features]
        eos_token = self.processor.tokenizer.eos_token
        conversations = [
            build_conversation(sample, answer_eos_token=eos_token)
            for sample in features
        ]

        full_prompts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in conversations
        ]
        user_prompts = [
            self.processor.apply_chat_template(
                [conversation[0]],
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]

        batch = self.processor(
            text=full_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        user_batch = self.processor(
            text=user_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        labels[batch["input_ids"] == self.processor.image_token_id] = -100

        full_lengths = batch["attention_mask"].sum(dim=1).tolist()
        user_lengths = user_batch["attention_mask"].sum(dim=1).tolist()
        sequence_length = labels.size(1)
        eos_token_id = self.processor.tokenizer.eos_token_id

        for index, (full_length, user_length) in enumerate(zip(full_lengths, user_lengths)):
            prompt_end = sequence_length - full_length + user_length
            labels[index, :prompt_end] = -100
            if eos_token_id is not None:
                supervised_eos_positions = (
                    labels[index, prompt_end:] == eos_token_id
                ).nonzero()
                if supervised_eos_positions.numel() > 0:
                    eos_position = prompt_end + supervised_eos_positions[0].item()
                    labels[index, eos_position + 1:] = -100

        batch["labels"] = labels
        return batch


def preview_supervised_labels(processor: Any, sample: dict[str, Any]) -> str:
    batch = LlavaDataCollator(processor)([sample])
    supervised_ids = batch["labels"][0][batch["labels"][0] != -100]
    return processor.tokenizer.decode(
        supervised_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def load_training_datasets(dataset_name: str = DATASET_NAME):
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(
        lambda sample: sample["image"] is not None
        and sample["query"] is not None
        and sample["label"] is not None
    )
    dataset = dataset.shuffle(seed=RANDOM_SEED)

    total_size = TRAIN_SAMPLE_SIZE + VAL_SAMPLE_SIZE
    if len(dataset) < total_size:
        raise ValueError(
            f"Need at least {total_size} valid samples, but found {len(dataset)}."
        )

    train_dataset = dataset.select(range(TRAIN_SAMPLE_SIZE))
    eval_dataset = dataset.select(range(TRAIN_SAMPLE_SIZE, total_size))
    return train_dataset, eval_dataset


def save_training_artifacts(trainer: SFTTrainer, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer.state.save_to_json(str(output_path / "trainer_state.json"))

    log_history = trainer.state.log_history
    fieldnames = sorted({key for row in log_history for key in row.keys()})
    with (output_path / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_history)



    train_steps = [row["step"] for row in log_history if "loss" in row]
    train_losses = [row["loss"] for row in log_history if "loss" in row]
    eval_steps = [row["step"] for row in log_history if "eval_loss" in row]
    eval_losses = [row["eval_loss"] for row in log_history if "eval_loss" in row]

    if not train_losses and not eval_losses:
        return

    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train_loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, marker="o", label="val_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png")
    plt.close()


def run_generative_eval(
    model,
    processor,
    dataset,
    output_dir: str,
    prefix: str = "val_generation",
) -> dict[str, Any]:
    eval_size = min(MAX_GENERATIVE_EVAL_SAMPLES, len(dataset))
    eval_dataset = dataset.select(range(eval_size))

    rows = []
    for index, sample in enumerate(eval_dataset, start=1):
        question = str(sample["query"]).strip()
        ground_truth = normalize_answer(sample["label"])
        prediction = generate_answer(
            model,
            processor,
            sample["image"],
            question,
            concise=True,
        )
        rows.append(
            {
                "sample": index,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
        )

    return save_prediction_artifacts(rows, output_dir, prefix)


training_args = SFTConfig(
    output_dir="./llava-finetuned",
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    fp16=False,
    logging_steps=LOGGING_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    seed=RANDOM_SEED,
    report_to=[],
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=0.3,
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=None,
)


if __name__ == "__main__":
    train_dataset, eval_dataset = load_training_datasets()
    print(f"train samples: {len(train_dataset)}")
    print(f"eval samples : {len(eval_dataset)}")

    model, processor = prepare_model()
    model.config.use_cache = False
    print(
        "Supervised target preview:",
        repr(preview_supervised_labels(processor, train_dataset[0])),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LlavaDataCollator(processor),
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    save_training_artifacts(trainer, training_args.output_dir)
    metrics = run_generative_eval(
        trainer.model,
        processor,
        eval_dataset,
        training_args.output_dir,
    )
    print(metrics)
