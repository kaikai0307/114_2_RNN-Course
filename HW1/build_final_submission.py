from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "final_version_report.ipynb"
REPORT_PATH = ROOT / "Homework1_report.md"
RNN_BIN = Path("/home/jiakai/miniconda3/envs/rnn/bin")


def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}

    cells = []
    cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                # Homework 1 Final Submission

                這份 notebook 對應 `Homework1.pdf` 的三個階段要求。

                說明：
                - Phase 1 / Phase 2 的模型分析主線使用 `2330.TW`
                - Phase 3 使用 `2408.TW` 的最終交易紀錄
                - 依需求，Phase 3 主表中的交易金額以「預測價格」為主；另外再補一份符合作業規則的合法 replay
                
                執行環境：
                - 建議使用 `/home/jiakai/miniconda3/envs/rnn/bin/python`
                - 主要依賴：`torch`, `pandas`, `scikit-learn`, `matplotlib`, `yfinance`
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path

                import pandas as pd
                from IPython.display import Markdown, display

                from final_submission_utils import (
                    PHASE2_TARGET_DATES,
                    ROOT,
                    analyze_signal_consistency,
                    build_phase1_summary,
                    build_rule_compliant_replay,
                    build_submission_trade_log,
                    device,
                    ensure_artifact_dir,
                    evaluate_final_model,
                    load_actual_phase3_submission,
                    load_base_history,
                    load_extended_history,
                    phase2_metrics,
                    run_phase2_rolling_forecast,
                    save_asset_curve_plot,
                    save_phase1_metric_plot,
                    save_phase3_drawdown_comparison_plot,
                    save_phase3_position_breakdown_plot,
                    save_phase3_signal_return_plot,
                    save_prediction_plot,
                    write_report,
                )

                artifact_dir = ensure_artifact_dir()
                report_output_path = ROOT / "Homework1_report.md"
                print(f"Artifact directory: {artifact_dir}")
                print(f"Current device: {device()}")
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                ## Phase 1: Baseline And Tuning Summary

                這一段直接整理 repo 中已經完成的代表性實驗，覆蓋：
                - baseline sample code
                - sequence length 調整
                - model architecture / dropout
                - training parameters
                - feature engineering
                - loss function
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                phase1_df = build_phase1_summary()
                phase1_path = artifact_dir / "phase1_summary.csv"
                phase1_df.to_csv(phase1_path, index=False)
                phase1_metric_plot_path = artifact_dir / "phase1_metrics.png"
                save_phase1_metric_plot(phase1_df, phase1_metric_plot_path)
                phase1_df.round(4)
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 1 Metrics]({phase1_metric_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                ## Final Model Reproduction

                最終模型採用 `v16_lb67_smoothl1`：
                - `look_back=67`
                - hidden sizes = `[128, 64]`
                - `SmoothL1Loss(beta=0.05)`
                - `lr=0.001`, `batch_size=32`, `epochs=50`

                下方會在 `2330/2330_stock_data.csv` 上重新訓練一次，確認最終模型在 homework 的 hold-out split 上的表現。
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                base_df = load_base_history()
                final_eval = evaluate_final_model(base_df)
                final_metrics = pd.DataFrame([final_eval["metrics"]]).round(4)
                final_prediction_path = artifact_dir / "phase1_final_model_predictions.csv"
                final_eval["prediction_df"].to_csv(final_prediction_path, index=False)

                phase1_plot_path = artifact_dir / "phase1_test_prediction.png"
                save_prediction_plot(
                    final_eval["prediction_df"],
                    phase1_plot_path,
                    "Phase 1 Final Model: Hold-out Test Prediction vs Actual",
                )

                final_metrics
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 1 Test Prediction]({phase1_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                ## Phase 2: Rolling Forecast

                依照 PDF 要求，選擇 **2026-03-06 到 2026-03-19** 這 10 個交易日做 rolling forecast。

                流程：
                1. 用目標日之前的所有資料訓練模型。
                2. 只預測下一個交易日。
                3. 觀察真實收盤價後，把新資料加入訓練集合。
                4. 進行 daily update，再往下一天預測。
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                extended_df = load_extended_history()
                close_series = extended_df["Close"].copy()

                rolling_df = run_phase2_rolling_forecast(close_series, target_dates=PHASE2_TARGET_DATES)
                rolling_metric_values = phase2_metrics(rolling_df)
                rolling_path = artifact_dir / "phase2_rolling_forecast.csv"
                rolling_df.to_csv(rolling_path, index=False)

                rolling_plot_path = artifact_dir / "phase2_rolling_forecast.png"
                save_prediction_plot(
                    rolling_df,
                    rolling_plot_path,
                    "Phase 2 Rolling Forecast: Prediction vs Actual",
                )

                print(
                    f"Phase 2 RMSE: {rolling_metric_values['rmse']:.4f}, "
                    f"MAPE: {rolling_metric_values['mape']:.2f}%"
                )
                rolling_df.round(4)
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 2 Rolling Forecast]({rolling_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                ## Phase 3: Actual Submission Analysis And Rule-Compliant Replay

                本段使用 `2408.TW` 的最終交易紀錄。

                這裡分成三個層次：
                1. 先整理原始提交內容。
                2. 檢查模型預測方向與實際操作是否一致。
                3. 產生兩份績效表：
                   - 依預測價格記帳的提交版本
                   - 依收盤價結算且符合作業規則的合法 replay
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                actual_submission_df = load_actual_phase3_submission()
                day4_mask = actual_submission_df["Sheet"] == "Day 4"
                actual_submission_df.loc[day4_mask, "SubmittedQuantity"] = 12000
                holdings_before_day10 = 0
                for _, row in actual_submission_df.iterrows():
                    if row["Sheet"] == "Day 10":
                        break
                    if row["SubmittedAction"] == "Buy":
                        holdings_before_day10 += int(row["SubmittedQuantity"])
                    elif row["SubmittedAction"] == "Sell":
                        holdings_before_day10 -= int(row["SubmittedQuantity"])

                day10_mask = actual_submission_df["Sheet"] == "Day 10"
                actual_submission_df.loc[day10_mask, "SubmittedQuantity"] = holdings_before_day10

                actual_submission_path = artifact_dir / "phase3_submission_input.csv"
                actual_submission_df.to_csv(actual_submission_path, index=False)
                actual_submission_df[
                    [
                        "Date",
                        "Sheet",
                        "SubmittedAction",
                        "SubmittedQuantity",
                        "PredictedPrice",
                        "CurrentClose",
                        "NextClose",
                    ]
                ].round(4)
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                consistency_df, consistency_summary = analyze_signal_consistency(actual_submission_df)
                consistency_path = artifact_dir / "phase3_signal_consistency.csv"
                consistency_df.to_csv(consistency_path, index=False)
                phase3_signal_plot_path = artifact_dir / "phase3_signal_return_comparison.png"
                save_phase3_signal_return_plot(consistency_df, phase3_signal_plot_path)

                print(
                    f"Comparable days: {consistency_summary['comparable_count']}, "
                    f"Consistent days: {consistency_summary['consistent_count']}, "
                    f"Consistency rate: {consistency_summary['consistency_rate_pct']:.2f}%"
                )
                consistency_df[
                    ["Date", "PredictedPrice", "CurrentClose", "PredictedDeltaVsCurrentPct", "ImpliedAction", "SubmittedAction", "Consistent"]
                ].round(4)
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 3 Signal Diagnostic]({phase3_signal_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                submission_trade_log, submission_metrics = build_submission_trade_log(
                    actual_submission_df,
                    amount_basis="predicted",
                )
                submission_trade_log_path = artifact_dir / "phase3_submission_trade_log.csv"
                submission_trade_log.to_csv(submission_trade_log_path, index=False)

                submission_asset_plot_path = artifact_dir / "phase3_submission_asset_curve.png"
                save_asset_curve_plot(
                    submission_trade_log,
                    submission_asset_plot_path,
                    "Phase 3 Actual Submission: Asset Curve (Transaction Amount By Predicted Price)",
                )
                phase3_position_plot_path = artifact_dir / "phase3_position_breakdown.png"
                save_phase3_position_breakdown_plot(submission_trade_log, phase3_position_plot_path)

                print(
                    f"Submission final asset: {submission_metrics['final_total_asset']:.0f} TWD, "
                    f"ROI: {submission_metrics['roi_pct']:.2f}%, "
                    f"Max Drawdown: {submission_metrics['max_drawdown_pct']:.2f}%, "
                    f"Max negative cash: {submission_metrics['max_negative_cash']:.0f} TWD"
                )
                submission_trade_log.round(4)
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 3 Submission Asset Curve]({submission_asset_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 3 Position Breakdown]({phase3_position_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                rule_replay_df, rule_replay_metrics = build_rule_compliant_replay(actual_submission_df)
                rule_replay_path = artifact_dir / "phase3_rule_compliant_replay.csv"
                rule_replay_df.to_csv(rule_replay_path, index=False)

                rule_replay_plot_path = artifact_dir / "phase3_rule_replay_asset_curve.png"
                save_asset_curve_plot(
                    rule_replay_df,
                    rule_replay_plot_path,
                    "Phase 3 Rule-Compliant Replay: Asset Curve (Close-Price Settlement)",
                )
                phase3_drawdown_plot_path = artifact_dir / "phase3_drawdown_comparison.png"
                save_phase3_drawdown_comparison_plot(
                    submission_trade_log,
                    rule_replay_df,
                    phase3_drawdown_plot_path,
                )

                print(
                    f"Rule replay final asset: {rule_replay_metrics['final_total_asset']:.0f} TWD, "
                    f"ROI: {rule_replay_metrics['roi_pct']:.2f}%, "
                    f"Max Drawdown: {rule_replay_metrics['max_drawdown_pct']:.2f}%, "
                    f"Clipped days: {rule_replay_metrics['clipped_trade_days']}"
                )
                rule_replay_df.round(4)
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 3 Rule Replay Asset Curve]({rule_replay_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                display(Markdown(f"![Phase 3 Drawdown Comparison]({phase3_drawdown_plot_path.relative_to(ROOT)})"))
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_markdown_cell(
            dedent(
                """
                ## Report Export

                最後把 Phase 1、Phase 2、Phase 3 的表格與結論寫成 `Homework1_report.md`。
                """
            ).strip()
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            dedent(
                """
                report_path = write_report(
                    report_output_path,
                    phase1_df,
                    final_eval,
                    phase1_metric_plot_path,
                    rolling_df,
                    rolling_metric_values,
                    actual_submission_df,
                    consistency_df,
                    consistency_summary,
                    submission_trade_log,
                    submission_metrics,
                    rule_replay_df,
                    rule_replay_metrics,
                    rolling_plot_path,
                    submission_asset_plot_path,
                    rule_replay_plot_path,
                    phase3_signal_plot_path,
                    phase3_position_plot_path,
                    phase3_drawdown_plot_path,
                )
                print(f"Report written to: {report_path}")
                report_path
                """
            ).strip()
        )
    )

    nb.cells = cells
    return nb


def execute_notebook(nb: nbformat.NotebookNode, notebook_path: Path) -> None:
    notebook_path.write_text(nbformat.writes(nb))

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
    )

    try:
        client.execute()
    except CellExecutionError:
        notebook_path.write_text(nbformat.writes(nb))
        raise

    notebook_path.write_text(nbformat.writes(nb))


def main() -> None:
    os.environ["PATH"] = f"{RNN_BIN}:{os.environ['PATH']}"
    os.environ["MPLBACKEND"] = "Agg"

    notebook = build_notebook()
    execute_notebook(notebook, NOTEBOOK_PATH)

    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Expected report was not generated: {REPORT_PATH}")

    print(f"Wrote notebook: {NOTEBOOK_PATH}")
    print(f"Wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
