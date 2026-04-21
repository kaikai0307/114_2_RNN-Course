# RNN Homework 1

This repository contains the final submission package for Homework 1 of the course `RNN and Transformer`.

## Submission Files

- `final_version_report.ipynb`: consolidated homework notebook with Phase 1, Phase 2, and Phase 3 analysis.
- `final_version_v14_lb67_predict.ipynb`: main `2408.TW` experiment notebook used in the homework workflow.
- `Homework1_report.md`: final written report in Markdown.
- `final_submission_artifacts/`: generated tables, plots, and CSV outputs used by the report.

## Supporting Materials

- `Stock_predict.ipynb`: the instructor-provided baseline notebook and the official starting point for comparison.
- `RNN HW 1.xlsx`: submitted trading record used in Phase 3 analysis.
- `VERSION_LOG.md`: experiment and revision history.

## Directory Structure

- `2330/`
  contains the `2330.TW` baseline notebook, runners, and experiment folders used for Phase 1 and Phase 2.
- `2408/`
  contains the `2408.TW` baseline notebook, runners, and experiment folders kept for traceability and comparison.
- repository root
  contains the submission notebooks, report files, shared utilities, and final generated artifacts.

## Utility Scripts

- `build_final_submission.py`: assembles the final notebook/report outputs.
- `final_submission_utils.py`: shared helpers for report generation and plotting.
- stock-specific runners now live inside `2330/` and `2408/`.
