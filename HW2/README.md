# HW2

Homework 2 for the course `RNN and Transformer`.

This folder contains the full workflow for AI-generated text detection on `DAIGT V2`, including:

- standalone EDA
- TF-IDF baselines
- BERT fine-tuning
- local LLM adversarial rewriting analysis

## Main Files

- `EDA.py`
  - standalone exploratory data analysis script
- `Baseline.py`
  - TF-IDF + LogisticRegression baselines
- `BERT.py`
  - BERT fine-tuning and evaluation
- `LocalLLM.py`
  - local LLM rewriting attack pipeline
- `hw2_utils.py`
  - shared data loading, split, metrics, and helper utilities

## Report Files

- `REPORT_DRAFT.md`
  - markdown report draft

## Figures

- `report_figures/`
  - all images used by the report

## Data

- `train_v2_drcat_02.csv`
  - local copy of the homework dataset

## Notes

- The root repository is `/ssd6/jiakai/114_2_RNN`.
- This folder is intended to be tracked from the root repo, not as a separate git repository.
