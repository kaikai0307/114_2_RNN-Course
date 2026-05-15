# HW4

Homework 4 for the course `RNN and Transformer`.

This folder contains a multimodal VQA fine-tuning pipeline based on `llava-hf/llava-1.5-7b-hf` and the `HuggingFaceM4/ChartQA` dataset. The main goal is to compare zero-shot baseline behavior with QLoRA fine-tuned adapters, including a `1000`-sample version and a `10000`-sample version.

## Main Files

- `Training.py`: QLoRA training script for ChartQA
- `Inference.ipynb`: baseline vs `llava-finetuned` (`10000` samples) comparison
- `Inference_1000.ipynb`: baseline vs `llava-finetuned_1000` (`1000` samples) comparison
- `Load.py`: model and adapter loading utilities
- `Eval.py`: answer generation and evaluation helpers
- `Config.py`: LoRA / model preparation settings
- `requirements.txt`: required Python packages

## Model Output Folders

- `llava-finetuned/`: adapter and evaluation artifacts for the `10000`-sample run
- `llava-finetuned_1000/`: adapter and evaluation artifacts for the `1000`-sample run
- `report_assets/`: images used in the written report

## Report Files

- `report.md`: markdown report draft


## Notes

- `llava-finetuned` was trained with `10000` samples.
- `llava-finetuned_1000` was trained with `1000` samples.
- The `10000`-sample model gives better validation performance than the `1000`-sample model in the current experiments.
