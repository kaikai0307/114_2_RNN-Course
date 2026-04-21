import copy
import json
import os
from pathlib import Path

import nbformat

from run_torch_experiments_2330 import (
    ORIGINAL_NOTEBOOK,
    build_dataset_cell,
    build_inverse_transform_cell,
    build_model_cell,
    build_predict_cell,
    execute_notebook,
    parse_summary,
)

ROOT = Path(__file__).resolve().parent
BASE_RESULTS = ROOT / "experiments_2330" / "results" / "all_experiments.json"
SWEEP_RESULTS = ROOT / "experiments_2330_lb_sweep" / "results" / "all_experiments.json"
REFINE_RESULTS = ROOT / "experiments_2330_lb_refine" / "results" / "all_experiments.json"
ADVANCED_ROOT = ROOT / "experiments_2330_feature_loss"
NOTEBOOK_DIR = ADVANCED_ROOT / "notebooks"
RESULTS_DIR = ADVANCED_ROOT / "results"

ADVANCED_EXPERIMENTS = [
    {
        "version": "v16_lb67_smoothl1",
        "description": "Keep the best look_back=67 and swap MSE for SmoothL1Loss to reduce the impact of tail errors.",
        "look_back": 67,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "feature_mode": "close_only",
        "loss_name": "smooth_l1",
        "loss_beta": 0.05,
    },
    {
        "version": "v17_lb67_ratio_rsi",
        "description": "Use ratio-style MA features and centered RSI around the current best look_back=67 setup.",
        "look_back": 67,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "feature_mode": "ratio_rsi",
        "loss_name": "mse",
        "loss_beta": 0.05,
    },
    {
        "version": "v18_lb67_ema_macd",
        "description": "Use EMA-gap and MACD family features around the current best look_back=67 setup.",
        "look_back": 67,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "feature_mode": "ema_macd",
        "loss_name": "mse",
        "loss_beta": 0.05,
    },
    {
        "version": "v19_lb67_ratio_rsi_smoothl1",
        "description": "Combine ratio-style MA and centered RSI features with SmoothL1Loss on the best look_back=67 setup.",
        "look_back": 67,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "feature_mode": "ratio_rsi",
        "loss_name": "smooth_l1",
        "loss_beta": 0.05,
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
    "feature_mode": "close_only",
    "loss_name": "mse",
    "loss_beta": 0.05,
}


def delta_metrics(metrics, reference_metrics):
    return {
        key: round(metrics[key] - reference_metrics[key], 4)
        for key in reference_metrics
        if key in metrics
    }


def build_reference_entries(base_data, sweep_data, refine_data):
    baseline = {
        "version": "v1_baseline_torch",
        "description": "Corrected 2330 Torch baseline.",
        "config": BASELINE_CONFIG,
        "notebook": base_data["baseline"]["notebook"],
        "summary": base_data["baseline"]["summary"],
        "source": "experiments_2330/results/all_experiments.json",
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

    v14_entry = next(exp for exp in refine_data["refine_experiments"] if exp["version"] == "v14_lb67")
    v14 = {
        "version": v14_entry["version"],
        "description": v14_entry["description"],
        "config": v14_entry["config"],
        "notebook": v14_entry["notebook"],
        "summary": v14_entry["summary"],
        "delta_vs_baseline": v14_entry["delta_vs_baseline"],
        "delta_vs_v2_lb60": v14_entry["delta_vs_v2_lb60"],
        "delta_vs_v7_lb50": v14_entry["delta_vs_v7_lb50"],
        "delta_vs_v9_lb65": v14_entry["delta_vs_v9_lb65"],
        "source": "experiments_2330_lb_refine/results/all_experiments.json",
    }

    return [baseline, v9, v14]


def build_preprocess_cell(config):
    feature_mode = config["feature_mode"]
    return f"""import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

file_path = filename
stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
stock_df = stock_df.copy()
print(f"Data loaded from {{file_path}}. First 5 rows:\\n{{stock_df.head()}}\\n")

feature_mode = '{feature_mode}'
feature_columns = ['Close']

if feature_mode == 'ratio_rsi':
    stock_df['MA5'] = stock_df['Close'].rolling(5).mean()
    stock_df['MA20'] = stock_df['Close'].rolling(20).mean()
    delta = stock_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    stock_df['RSI14'] = 100 - (100 / (1 + rs))
    stock_df['CLOSE_MA5_RATIO'] = stock_df['Close'] / stock_df['MA5']
    stock_df['CLOSE_MA20_RATIO'] = stock_df['Close'] / stock_df['MA20']
    stock_df['RSI14_CENTERED'] = stock_df['RSI14'] - 50
    feature_columns = ['Close', 'CLOSE_MA5_RATIO', 'CLOSE_MA20_RATIO', 'RSI14_CENTERED']
elif feature_mode == 'ema_macd':
    stock_df['EMA10'] = stock_df['Close'].ewm(span=10, adjust=False).mean()
    stock_df['EMA20'] = stock_df['Close'].ewm(span=20, adjust=False).mean()
    stock_df['MACD'] = stock_df['EMA10'] - stock_df['EMA20']
    stock_df['MACD_SIGNAL'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
    stock_df['MACD_HIST'] = stock_df['MACD'] - stock_df['MACD_SIGNAL']
    stock_df['EMA10_GAP'] = (stock_df['Close'] - stock_df['EMA10']) / stock_df['EMA10']
    stock_df['EMA20_GAP'] = (stock_df['Close'] - stock_df['EMA20']) / stock_df['EMA20']
    feature_columns = ['Close', 'EMA10_GAP', 'EMA20_GAP', 'MACD', 'MACD_SIGNAL', 'MACD_HIST']
elif feature_mode != 'close_only':
    raise ValueError(f"Unsupported feature_mode: {{feature_mode}}")

stock_df = stock_df.dropna(subset=feature_columns).copy()

print(f"Feature mode: {{feature_mode}}")
print(f"Feature columns: {{feature_columns}}")

close_prices = stock_df[['Close']].values
feature_values = stock_df[feature_columns].values
print(f"'Close' prices extracted. Shape: {{close_prices.shape}}\\n")

feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(feature_values)
scaled_close_prices = target_scaler.fit_transform(close_prices)

print("Feature scaling completed.")
print(f"Scaled feature shape: {{scaled_features.shape}}\\n")

look_back = {config["look_back"]}
print(f"look_back period defined as: {{look_back}}")
"""


def build_optimizer_cell(config):
    loss_name = config["loss_name"]
    loss_beta = config.get("loss_beta", 0.05)
    return f"""import torch.optim as optim

learning_rate = {config["learning_rate"]}
batch_size = {config["batch_size"]}
epochs = {config["epochs"]}
loss_name = "{loss_name}"
loss_beta = {loss_beta}
scheduler_name = "none"

if loss_name == "smooth_l1":
    criterion = nn.SmoothL1Loss(beta=loss_beta)
elif loss_name == "mse":
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss_name: {{loss_name}}")

optimizer = optim.Adam(model_with_attention.parameters(), lr=learning_rate)
scheduler = None

print("PyTorch optimizer and loss are ready.")
print(f"Training config ready on device: {{device}}")
print(f"learning_rate: {{learning_rate}}")
print(f"batch_size: {{batch_size}}")
print(f"epochs: {{epochs}}")
print(f"loss_name: {{loss_name}}")
print(f"loss_beta: {{loss_beta}}")
"""


def build_train_cell():
    return """from torch.utils.data import DataLoader, TensorDataset

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

history_attention = []

print("Starting PyTorch Attention-LSTM model training...")
for epoch in range(epochs):
    model_with_attention.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model_with_attention(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    history_attention.append(epoch_loss)
    if scheduler is not None and scheduler_name == "plateau":
        scheduler.step(epoch_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f} - lr: {current_lr:.6f}")

print("PyTorch Attention-LSTM model training completed.")
"""


def prepare_advanced_notebook(original_nb, config):
    nb = copy.deepcopy(original_nb)
    version = config["version"]
    feature_mode = config["feature_mode"]
    loss_name = config["loss_name"]

    nb.cells[3].source = (
        "import importlib.util\n"
        "\n"
        "if importlib.util.find_spec('yfinance') is None:\n"
        "    %pip install yfinance\n"
        "else:\n"
        "    print('yfinance is already installed.')\n"
    )
    nb.cells[12].source = (
        f"# Experiment {version}\n"
        "ticker_symbol = '2330.TW'\n"
        f"print('本次實驗版本: {version}')\n"
        "print(f'本次實驗使用股票代碼: {ticker_symbol}')\n"
    )
    nb.cells[13].source = """from pathlib import Path

filename = f"{ticker_symbol.replace('.TW', '')}_stock_data.csv"

if Path(filename).exists():
    print(f"Using cached stock data from {filename}")
else:
    stock_data = fetch_stock_data(ticker_symbol)
    if not stock_data.empty:
        stock_data.to_csv(filename, index=True)
        print(f"股票資料已成功儲存為 {filename}")
    else:
        raise RuntimeError(f"無法為 {ticker_symbol} 抓取到股票資料，因此沒有檔案儲存。")
"""
    nb.cells[14].source = build_preprocess_cell(config)
    nb.cells[15].source = build_dataset_cell()
    nb.cells[20].source = f"## Experiment {version}: PyTorch Attention-LSTM"
    nb.cells[21].source = f"**Reasoning**:\n{config['description']}"
    nb.cells[22].source = build_model_cell(config)
    nb.cells[23].source = f"## Experiment {version}: Optimizer and Loss"
    nb.cells[24].source = (
        "**Reasoning**:\n"
        f"Fix `look_back={config['look_back']}` on 2330, then test `feature_mode={feature_mode}` with `loss_name={loss_name}`."
    )
    nb.cells[25].source = build_optimizer_cell(config)
    nb.cells[26].source = f"## Experiment {version}: Training"
    nb.cells[27].source = (
        "**Reasoning**:\n"
        "Keep the corrected Torch Attention-LSTM training flow unchanged apart from the configured feature set and loss."
    )
    nb.cells[28].source = build_train_cell()
    nb.cells[29].source = f"## Experiment {version}: Prediction"
    nb.cells[30].source = (
        "**Reasoning**:\n"
        "Generate predictions and compare them with the same evaluation flow used by the existing 2330 runs."
    )
    nb.cells[31].source = build_predict_cell()
    nb.cells[34].source = build_inverse_transform_cell()

    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb


def build_ranking(reference_entries, advanced_entries):
    ranking = []
    for entry in reference_entries + advanced_entries:
        metrics = entry["summary"]["metrics"]
        ranking.append(
            {
                "version": entry["version"],
                "source": entry.get("source", "experiments_2330_feature_loss"),
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

    base_data = json.loads(BASE_RESULTS.read_text())
    sweep_data = json.loads(SWEEP_RESULTS.read_text())
    refine_data = json.loads(REFINE_RESULTS.read_text())
    reference_entries = build_reference_entries(base_data, sweep_data, refine_data)

    baseline_metrics = reference_entries[0]["summary"]["metrics"]
    v9_metrics = reference_entries[1]["summary"]["metrics"]
    v14_metrics = reference_entries[2]["summary"]["metrics"]

    original_nb = nbformat.read(ORIGINAL_NOTEBOOK, as_version=4)

    advanced_entries = []
    for config in ADVANCED_EXPERIMENTS:
        version = config["version"]
        notebook_path = NOTEBOOK_DIR / f"{version}.ipynb"
        prepared_nb = prepare_advanced_notebook(original_nb, config)
        executed_nb = execute_notebook(prepared_nb, notebook_path)
        summary = parse_summary(executed_nb)

        result = {
            "version": version,
            "description": config["description"],
            "config": config,
            "notebook": str(notebook_path.relative_to(ROOT)),
            "summary": summary,
            "delta_vs_baseline": delta_metrics(summary["metrics"], baseline_metrics),
            "delta_vs_v9_lb65": delta_metrics(summary["metrics"], v9_metrics),
            "delta_vs_v14_lb67": delta_metrics(summary["metrics"], v14_metrics),
        }
        advanced_entries.append(result)
        (RESULTS_DIR / f"{version}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    ranking = build_ranking(reference_entries, advanced_entries)
    best_entry = ranking[0]
    best_new_experiment = min(
        (
            {
                "version": entry["version"],
                "source": "experiments_2330_feature_loss",
                "notebook": entry["notebook"],
                "look_back": entry["config"]["look_back"],
                "test_rmse": entry["summary"]["metrics"]["test_rmse"],
                "test_mae": entry["summary"]["metrics"]["test_mae"],
                "test_mape": entry["summary"]["metrics"]["test_mape"],
            }
            for entry in advanced_entries
        ),
        key=lambda item: item["test_rmse"],
    )

    results = {
        "suite": "2330_feature_loss",
        "reference_versions": reference_entries,
        "advanced_experiments": advanced_entries,
        "ranking_by_test_rmse": ranking,
        "best_overall_by_test_rmse": best_entry,
        "best_new_experiment": best_new_experiment,
        "beats_v14_lb67": best_entry["version"] != "v14_lb67",
    }

    combined_path = RESULTS_DIR / "all_experiments.json"
    combined_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
