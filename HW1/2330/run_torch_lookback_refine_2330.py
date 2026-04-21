import json
import os
from pathlib import Path

import nbformat

from run_torch_experiments_2330 import ORIGINAL_NOTEBOOK, execute_notebook, parse_summary, prepare_notebook

ROOT = Path(__file__).resolve().parent
REFERENCE_RESULTS = ROOT / "experiments_2330" / "results" / "all_experiments.json"
SWEEP_RESULTS = ROOT / "experiments_2330_lb_sweep" / "results" / "all_experiments.json"
REFINE_ROOT = ROOT / "experiments_2330_lb_refine"
NOTEBOOK_DIR = REFINE_ROOT / "notebooks"
RESULTS_DIR = REFINE_ROOT / "results"

REFINE_EXPERIMENTS = [
    {
        "version": "v12_lb62",
        "description": "Fine look-back search on 2330 with look_back=62.",
        "look_back": 62,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v13_lb63",
        "description": "Fine look-back search on 2330 with look_back=63.",
        "look_back": 63,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v14_lb67",
        "description": "Fine look-back search on 2330 with look_back=67.",
        "look_back": 67,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v15_lb68",
        "description": "Fine look-back search on 2330 with look_back=68.",
        "look_back": 68,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
]

BASELINE_CONFIG = {
    "version": "v1_baseline_torch",
    "description": "Corrected 2330 Torch baseline.",
    "look_back": 100,
    "hidden_sizes": [128, 64],
    "dropout": 0.0,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "use_indicators": False,
}


def delta_metrics(metrics, reference_metrics):
    return {
        key: round(metrics[key] - reference_metrics[key], 4)
        for key in reference_metrics
        if key in metrics
    }


def build_reference_entries(reference_data, sweep_data):
    baseline = {
        "version": "v1_baseline_torch",
        "description": "Corrected 2330 Torch baseline.",
        "config": BASELINE_CONFIG,
        "notebook": reference_data["baseline"]["notebook"],
        "summary": reference_data["baseline"]["summary"],
        "source": "experiments_2330/results/all_experiments.json",
    }

    v2_entry = next(exp for exp in reference_data["experiments"] if exp["version"] == "v2_lb60")
    v2 = {
        "version": v2_entry["version"],
        "description": v2_entry["description"],
        "config": v2_entry["config"],
        "notebook": v2_entry["notebook"],
        "summary": v2_entry["summary"],
        "delta_vs_baseline": v2_entry["delta_vs_baseline"],
        "source": "experiments_2330/results/all_experiments.json",
    }

    v7_entry = next(exp for exp in sweep_data["sweep_experiments"] if exp["version"] == "v7_lb50")
    v7 = {
        "version": v7_entry["version"],
        "description": v7_entry["description"],
        "config": v7_entry["config"],
        "notebook": v7_entry["notebook"],
        "summary": v7_entry["summary"],
        "delta_vs_baseline": v7_entry["delta_vs_baseline"],
        "delta_vs_v2_lb60": v7_entry["delta_vs_v2_lb60"],
        "source": "experiments_2330_lb_sweep/results/all_experiments.json",
    }

    v9_entry = next(exp for exp in sweep_data["sweep_experiments"] if exp["version"] == "v9_lb65")
    v9 = {
        "version": v9_entry["version"],
        "description": v9_entry["description"],
        "config": v9_entry["config"],
        "notebook": v9_entry["notebook"],
        "summary": v9_entry["summary"],
        "delta_vs_baseline": v9_entry["delta_vs_baseline"],
        "delta_vs_v2_lb60": v9_entry["delta_vs_v2_lb60"],
        "source": "experiments_2330_lb_sweep/results/all_experiments.json",
    }

    return [baseline, v2, v7, v9]


def build_ranking(reference_entries, refine_entries):
    ranking = []
    for entry in reference_entries + refine_entries:
        metrics = entry["summary"]["metrics"]
        ranking.append(
            {
                "version": entry["version"],
                "source": entry.get("source", "experiments_2330_lb_refine"),
                "notebook": entry["notebook"],
                "look_back": entry["config"]["look_back"],
                "test_rmse": metrics["test_rmse"],
                "test_mae": metrics["test_mae"],
                "test_mape": metrics["test_mape"],
            }
        )
    ranking.sort(key=lambda item: item["test_rmse"])
    return ranking


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["MPLBACKEND"] = "Agg"

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reference_data = json.loads(REFERENCE_RESULTS.read_text())
    sweep_data = json.loads(SWEEP_RESULTS.read_text())
    reference_entries = build_reference_entries(reference_data, sweep_data)

    baseline_metrics = reference_entries[0]["summary"]["metrics"]
    v2_metrics = reference_entries[1]["summary"]["metrics"]
    v7_metrics = reference_entries[2]["summary"]["metrics"]
    v9_metrics = reference_entries[3]["summary"]["metrics"]

    original_nb = nbformat.read(ORIGINAL_NOTEBOOK, as_version=4)

    refine_entries = []
    for config in REFINE_EXPERIMENTS:
        version = config["version"]
        notebook_path = NOTEBOOK_DIR / f"{version}.ipynb"
        prepared_nb = prepare_notebook(original_nb, config)
        executed_nb = execute_notebook(prepared_nb, notebook_path)
        summary = parse_summary(executed_nb)

        result = {
            "version": version,
            "description": config["description"],
            "config": config,
            "notebook": str(notebook_path.relative_to(ROOT)),
            "summary": summary,
            "delta_vs_baseline": delta_metrics(summary["metrics"], baseline_metrics),
            "delta_vs_v2_lb60": delta_metrics(summary["metrics"], v2_metrics),
            "delta_vs_v7_lb50": delta_metrics(summary["metrics"], v7_metrics),
            "delta_vs_v9_lb65": delta_metrics(summary["metrics"], v9_metrics),
        }
        refine_entries.append(result)
        (RESULTS_DIR / f"{version}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    ranking = build_ranking(reference_entries, refine_entries)
    best_entry = ranking[0]

    results = {
        "suite": "2330_lookback_refine",
        "reference_versions": reference_entries,
        "refine_experiments": refine_entries,
        "ranking_by_test_rmse": ranking,
        "best_overall_by_test_rmse": best_entry,
        "best_new_experiment": min(
            (
                {
                    "version": entry["version"],
                    "source": "experiments_2330_lb_refine",
                    "notebook": entry["notebook"],
                    "look_back": entry["config"]["look_back"],
                    "test_rmse": entry["summary"]["metrics"]["test_rmse"],
                    "test_mae": entry["summary"]["metrics"]["test_mae"],
                    "test_mape": entry["summary"]["metrics"]["test_mape"],
                }
                for entry in refine_entries
            ),
            key=lambda item: item["test_rmse"],
        ),
        "beats_v9_lb65": best_entry["version"] != "v9_lb65",
    }

    combined_path = RESULTS_DIR / "all_experiments.json"
    combined_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
