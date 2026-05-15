# Homework 4 Report

## Overview

This report summarizes the multimodal visual question answering experiments for Homework 4.
The goal of this homework is to fine-tune a pre-trained Vision-Language Model with QLoRA so that it can answer chart-based questions more reliably than the zero-shot baseline.

This implementation uses ChartQA as the target domain and `llava-hf/llava-1.5-7b-hf` as the base VLM.
Two fine-tuned adapters are compared in this report:

- `llava-finetuned`: trained with **10,000** ChartQA samples
- `llava-finetuned_1000`: trained with **1,000** ChartQA samples

GitHub repository:

- <https://github.com/kaikai0307/114_2_RNN-Course>

Relevant source files in this homework folder:

- `Training.py`: QLoRA fine-tuning pipeline
- `Inference.ipynb`: baseline vs 10k-adapter qualitative comparison
- `Inference_1000.ipynb`: baseline vs 1k-adapter qualitative comparison
- `Load.py`: model / adapter loading utilities
- `Eval.py`: generation and evaluation helpers
- `requirements.txt`: dependencies

## Experimental Setup

### Model And Training Strategy

- Base model: `llava-hf/llava-1.5-7b-hf`
- Quantization: 4-bit QLoRA with `bitsandbytes`
- PEFT method: LoRA
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Training library: `trl.SFTTrainer`
- Vision tower: frozen
- Multimodal projector: frozen in this implementation

The implementation follows the homework requirement to fine-tune a VLM with PEFT under limited VRAM.
The base model is loaded in 4-bit mode, while only LoRA adapters on the language model attention projections are trained.

### Dataset Choice

The selected dataset is `HuggingFaceM4/ChartQA`.

Why ChartQA was chosen:

- It is a clear domain-specific VQA benchmark rather than a generic photo-caption task.
- Many questions require exact number reading, comparison, counting, or simple chart reasoning.
- This makes hallucination and vague baseline answers easy to observe.
- Improvement after fine-tuning can be judged not only by fluency, but also by answer precision and format consistency.

### Data Formatting

Each training sample is converted into a standard multimodal conversation:

`Image + concise user question -> assistant answer`

The prompt format used during training is:

```text
Answer the question based on the chart. Give only the final answer, without explanation.
Question: <question text>
```

The custom collator builds the full chat template, tokenizes image-text pairs, and masks the user-side tokens so that the supervised loss is applied only to the assistant answer.
This part is important for multimodal SFT because text-only collators do not correctly handle image tokens and answer-only supervision.

### Training Variants

| Adapter | Train Samples | Epochs | Observed Train Steps | Final Train Loss | Best Eval Loss | Eval Exact Match | Eval Relaxed Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llava-finetuned_1000` | 1,000 | 3 | 189 | 1.3512 | 1.3390 | 0.06 | 0.11 |
| `llava-finetuned` | 10,000 | 3 | 1,875 | 1.1632 | 1.1417 | 0.12 | 0.25 |

Notes:

- The metrics above come from `val_generation_metrics.json` and `metrics.csv` stored in each adapter directory.
- The generative evaluation artifact in both folders uses 100 validation samples.
- Relaxed accuracy allows small tolerance for numeric answers, which is more appropriate than strict exact match for chart reading.

## Part 1: Baseline Inference

Before fine-tuning, the zero-shot base LLaVA model was tested on ChartQA examples.
The main failure patterns were:

- answering with a general chart description instead of a short final answer
- producing a plausible but incorrect number
- failing simple comparison/counting questions

From the notebook outputs, the baseline behavior is clearly weak for chart-domain VQA.
For example:

- For "How many food item is shown in the bar graph?", the base model described the chart instead of giving a count.
- For "What percent who think of President Donald Trump as Dangerous?", the base model answered `75%` while the ground truth is `62`.
- Even when the base model was semantically close, it often returned verbose sentences rather than the concise format requested by the homework.

## Part 2: Visual Instruction Tuning With QLoRA

### Training Hyperparameters

- Learning rate: `2e-5`
- Batch size: `16`
- Gradient accumulation: `1`
- Epochs: `3`
- Warmup ratio: `0.03`
- Weight decay: `0.01`
- Logging steps: `1`
- Eval / save steps: `20`

### Loss Curves

`llava-finetuned_1000`:

![Loss Curve 1000](llava-finetuned_1000/loss_curve.png)

`llava-finetuned`:

![Loss Curve 10000](llava-finetuned/loss_curve.png)

### Training Interpretation

Both runs show that the model can be optimized stably with LoRA adapters only, but the 10k-sample run converges to a noticeably better validation loss.

- `1,000` samples: best eval loss `1.3390`
- `10,000` samples: best eval loss `1.1417`

This gap is consistent with the final generative evaluation: the larger-data run doubles exact match and more than doubles relaxed accuracy.

### Why The Training Loss Fluctuates So Much

The per-step training loss looks noisy, especially for the `10,000`-sample run, but the logs suggest this is normal minibatch-level variance rather than instability.

Main reasons:

- `logging_steps=1`, so the curve records every single training step instead of a smoothed average.
- The effective batch size is only `16` (`per_device_train_batch_size=16`, `gradient_accumulation_steps=1`), so each logged point is sensitive to the exact batch content.
- ChartQA is heterogeneous: some samples are easy counting / yes-no questions, while others require harder numeric reading or comparison, so batch difficulty changes a lot from step to step.
- This is multimodal SFT on a quantized 7B VLM with LoRA adapters only, so optimization noise is naturally higher than in a large fully supervised text-only setup.

Why I do not treat it as divergence:

- The validation loss is much smoother than the training loss.
- For `llava-finetuned`, eval loss decreases from `1.5856` at step `20` to about `1.1423` at the end.
- For `llava-finetuned_1000`, eval loss decreases from `1.5550` at step `20` to `1.3390` at the end.
- The model still improves on downstream generation metrics, so the noisy training curve is not just random oscillation without learning.

So the correct interpretation is:

- the step-level train loss is noisy
- the overall optimization trend is still downward
- the eval curve is the better indicator of actual training stability

## Part 3: Evaluation And Analysis

### Quantitative Comparison

The 10k-data adapter is consistently stronger than the 1k-data adapter on the stored validation generation benchmark.

| Adapter | Exact Match | Relaxed Accuracy | Relative Observation |
|---|---:|---:|---|
| `llava-finetuned_1000` | 0.06 | 0.11 | Learns some answer formatting, but numeric QA remains weak |
| `llava-finetuned_10000` | 0.12 | 0.25 | Better answer precision and more stable chart reading |

Interpretation:

- The larger training set improves both optimization and generalization.
- The biggest gain is on numeric tolerance-based scoring, which suggests the 10k model more often lands near the correct chart value.
- Even so, absolute accuracy is still limited, so the adapter improves the base model but does not solve ChartQA completely.

### Case Studies

The homework asks for side-by-side comparisons on the same images.
Below, the base model answer, the 1k adapter answer, and the 10k adapter answer are shown together.

#### Case 1: Clear Improvement From More Training Data

<img src="report_assets/sample_7.png" alt="Sample 7" width="320">

| Field | Result |
|---|---|
| Question | What percent who think of President Donald Trump as Dangerous? |
| Ground Truth | `62` |
| Base Model | `According to the image, 75% of people think of President Donald Trump as dangerous.` |
| Fine-tuned 1k | `75` |
| Fine-tuned 10k | `62` |

Analysis:

- The base model produced a confident but incorrect numeric answer.
- The 1k adapter made the output more concise, but it kept the same wrong value.
- The 10k adapter corrected the value completely, which is the clearest qualitative evidence that more chart-domain supervision helps.

#### Case 2: Format Consistency Improves After Fine-tuning

<img src="report_assets/sample_5.png" alt="Sample 5" width="320">

| Field | Result |
|---|---|
| Question | What's the value of the lowest bar? |
| Ground Truth | `23` |
| Base Model | `The lowest bar represents 23%.` |
| Fine-tuned 1k | `23` |
| Fine-tuned 10k | `23` |

Analysis:

- The base model was close, but it answered with a descriptive sentence and percentage notation.
- Both fine-tuned models returned the exact concise answer format requested in the prompt.
- This example shows that fine-tuning helps not only accuracy, but also answer style consistency.

#### Case 3: Yes/No Chart Reasoning Becomes More Controlled

<img src="report_assets/sample_8.png" alt="Sample 8" width="320">

| Field | Result |
|---|---|
| Question | Is the sum of the percents of Charismatic and Well-qualified to be president more than A strong leader ? |
| Ground Truth | `Yes` |
| Base Model | `Yes, the sum of the percents of Charismatic and Well-qualified to be president is more than A strong leader. The Charismatic percentage is 22%, while the Well-qualified percentage is 28%. In contrast, the A strong leader percentage is only 12%.` |
| Fine-tuned 1k | `Yes` |
| Fine-tuned 10k | `Yes` |

Analysis:

- The base model reached the correct conclusion, but it returned a long explanation.
- Both adapters obeyed the concise-answer instruction much better.
- This is useful for evaluation because ChartQA labels are short answers, not paragraph explanations.

### Remaining Failure Modes

Although the tuned adapters are better than the baseline, there are still many failures:

- counting bars incorrectly
- confusing nearby numeric values
- failing subtraction / comparison questions
- sometimes outputting a number that is formatted cleanly but still wrong

So the improvement is real, but the current training setup is still far from saturated performance on ChartQA.

## Discussion

### Why The 10k Model Works Better

The comparison between `llava-finetuned` and `llava-finetuned_1000` suggests that data scale matters a lot for chart-domain adaptation.

- With only 1,000 training samples, the model learns the response format, but often does not actually read chart values correctly.
- With 10,000 samples, the model is still imperfect, but it more frequently maps visual chart elements to the correct numeric answer.
- The gap between relaxed accuracy `0.11 -> 0.25` is large enough to conclude that the extra data is worthwhile.

### Challenges Faced

The main engineering and modeling challenges were:

- fitting a 7B VLM into limited VRAM, which required 4-bit loading and LoRA adapters
- handling multimodal batch collation correctly so that image tokens and text labels are aligned
- keeping supervision only on the assistant answer tokens
- dealing with ChartQA answers that are numeric, short, and sensitive to small formatting differences

Another challenge is that chart understanding is stricter than generic image description.
A model may sound fluent while still being numerically wrong, so qualitative inspection and numeric evaluation are both necessary.

## Conclusion

This homework successfully implemented a QLoRA fine-tuning pipeline for a multimodal VLM on ChartQA.
The base LLaVA model performs poorly in zero-shot chart VQA, often hallucinating or giving verbose descriptions instead of final answers.
After visual instruction tuning, the model becomes more format-consistent, and the 10k-data adapter shows clearly better results than the 1k-data adapter.

In short:

- the training pipeline works
- ChartQA data formatting is handled correctly for multimodal SFT
- fine-tuning improves chart-domain behavior
- using **10,000** samples is meaningfully better than using **1,000** samples
