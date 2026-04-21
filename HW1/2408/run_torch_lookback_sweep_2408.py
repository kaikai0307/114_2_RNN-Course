import json
import os
from pathlib import Path

import nbformat

from run_torch_experiments import ORIGINAL_NOTEBOOK, execute_notebook, parse_summary, prepare_notebook

ROOT = Path(__file__).resolve().parent
REFERENCE_RESULTS = ROOT / "experiments" / "results" / "all_experiments.json"
SWEEP_ROOT = ROOT / "experiments_2408_lb_sweep"
NOTEBOOK_DIR = SWEEP_ROOT / "notebooks"
RESULTS_DIR = SWEEP_ROOT / "results"

SWEEP_EXPERIMENTS = [
    {
        "version": "v6_lb40",
        "description": "Look-back sweep around the current 2408 winner with look_back=40.",
        "look_back": 40,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v7_lb50",
        "description": "Look-back sweep around the current 2408 winner with look_back=50.",
        "look_back": 50,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v8_lb55",
        "description": "Look-back sweep around the current 2408 winner with look_back=55.",
        "look_back": 55,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v9_lb65",
        "description": "Look-back sweep around the current 2408 winner with look_back=65.",
        "look_back": 65,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v10_lb70",
        "description": "Look-back sweep around the current 2408 winner with look_back=70.",
        "look_back": 70,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v11_lb80",
        "description": "Look-back sweep around the current 2408 winner with look_back=80.",
        "look_back": 80,
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
    "description": "Corrected 2408 Torch baseline.",
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


def build_reference_entries(reference_data):
    baseline = {
        "version": "v1_baseline_torch",
        "description": "Corrected 2408 Torch baseline.",
        "config": BASELINE_CONFIG,
        "notebook": reference_data["baseline"]["notebook"],
        "summary": reference_data["baseline"]["summary"],
        "source": "experiments/results/all_experiments.json",
    }

    v2_entry = next(exp for exp in reference_data["experiments"] if exp["version"] == "v2_lb60")
    v2 = {
        "version": v2_entry["version"],
        "description": v2_entry["description"],
        "config": v2_entry["config"],
        "notebook": v2_entry["notebook"],
        "summary": v2_entry["summary"],
        "delta_vs_baseline": v2_entry["delta_vs_baseline"],
        "source": "experiments/results/all_experiments.json",
    }

    return baseline, v2


def build_ranking(reference_entries, sweep_entries):
    ranking = []
    for entry in reference_entries + sweep_entries:
        metrics = entry["summary"]["metrics"]
        ranking.append(
            {
                "version": entry["version"],
                "source": entry.get("source", "experiments_2408_lb_sweep"),
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
    baseline_ref, v2_ref = build_reference_entries(reference_data)
    baseline_metrics = baseline_ref["summary"]["metrics"]
    v2_metrics = v2_ref["summary"]["metrics"]

    original_nb = nbformat.read(ORIGINAL_NOTEBOOK, as_version=4)

    sweep_entries = []
    for config in SWEEP_EXPERIMENTS:
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
        }
        sweep_entries.append(result)
        (RESULTS_DIR / f"{version}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    ranking = build_ranking([baseline_ref, v2_ref], sweep_entries)
    best_entry = ranking[0]
    results = {
        "suite": "2408_lookback_sweep",
        "reference_versions": [baseline_ref, v2_ref],
        "sweep_experiments": sweep_entries,
        "ranking_by_test_rmse": ranking,
        "best_overall_by_test_rmse": best_entry,
        "beats_v2_lb60": best_entry["version"] != "v2_lb60",
    }

    combined_path = RESULTS_DIR / "all_experiments.json"
    combined_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
