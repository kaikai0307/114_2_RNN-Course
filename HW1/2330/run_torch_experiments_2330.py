import copy
import json
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ORIGINAL_NOTEBOOK = REPO_ROOT / "Stock_predict.ipynb"
BASELINE_NOTEBOOK = ROOT / "Stock_predict_v1_2330TW_torch.ipynb"
EXPERIMENT_ROOT = ROOT / "experiments_2330"
NOTEBOOK_DIR = EXPERIMENT_ROOT / "notebooks"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
TARGET_TICKER = "2330.TW"

EXPERIMENTS = [
    {
        "version": "v2_lb60",
        "description": "Sequence length test with look_back reduced to 60.",
        "look_back": 60,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v3_deep_dropout",
        "description": "Deeper architecture with larger hidden sizes and dropout.",
        "look_back": 100,
        "hidden_sizes": [256, 128, 64],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "use_indicators": False,
    },
    {
        "version": "v4_train_tune",
        "description": "Training-parameter test with lower learning rate, larger batch size, and more epochs.",
        "look_back": 100,
        "hidden_sizes": [128, 64],
        "dropout": 0.0,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 80,
        "use_indicators": False,
    },
    {
        "version": "v5_indicators",
        "description": "Optional feature-engineering test with MA5, MA20, and RSI14.",
        "look_back": 90,
        "hidden_sizes": [128, 64],
        "dropout": 0.2,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "epochs": 60,
        "use_indicators": True,
    },
]


def collect_text_output(cell):
    parts = []
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            text = output.get("text", "")
            parts.append(text if isinstance(text, str) else "".join(text))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/plain" in data:
                text = data["text/plain"]
                parts.append(text if isinstance(text, str) else "".join(text))
        elif output_type == "error":
            parts.append("\n".join(output.get("traceback", [])))
    return "\n".join(parts)


def parse_metrics(text):
    pattern = re.compile(
        r"Train RMSE: (?P<train_rmse>[0-9.]+)\s+"
        r"Train MAE: (?P<train_mae>[0-9.]+)\s+"
        r"Train MAPE: (?P<train_mape>[0-9.]+)%\s+"
        r"Test RMSE: (?P<test_rmse>[0-9.]+)\s+"
        r"Test MAE: (?P<test_mae>[0-9.]+)\s+"
        r"Test MAPE: (?P<test_mape>[0-9.]+)%",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return {}
    return {key: float(value) for key, value in match.groupdict().items()}


def parse_first_match(text, pattern):
    match = re.search(pattern, text)
    return match.group(1) if match else None


def parse_summary(notebook):
    cell13 = collect_text_output(notebook.cells[13])
    cell14 = collect_text_output(notebook.cells[14])
    cell18 = collect_text_output(notebook.cells[18])
    cell19 = collect_text_output(notebook.cells[19])
    cell22 = collect_text_output(notebook.cells[22])
    cell25 = collect_text_output(notebook.cells[25])
    cell42 = collect_text_output(notebook.cells[42])

    return {
        "csv_file": parse_first_match(cell13, r"(2330_stock_data\.csv)"),
        "feature_columns": parse_first_match(cell14, r"Feature columns: ([^\n]+)"),
        "close_rows": int(parse_first_match(cell14, r"Shape: \((\d+), 1\)") or 0),
        "look_back": int(parse_first_match(cell14, r"look_back period defined as: (\d+)") or 0),
        "latest_date": parse_first_match(cell19, r"The latest date in the data is: ([^\n]+)"),
        "x_train_shape": parse_first_match(cell18, r"X_train shape: ([^\n]+)"),
        "x_test_shape": parse_first_match(cell18, r"X_test shape: ([^\n]+)"),
        "y_train_shape": parse_first_match(cell18, r"y_train shape: ([^\n]+)"),
        "y_test_shape": parse_first_match(cell18, r"y_test shape: ([^\n]+)"),
        "cuda_visible_devices": parse_first_match(cell22, r"CUDA_VISIBLE_DEVICES: ([^\n]+)"),
        "torch_cuda_available": parse_first_match(cell22, r"torch.cuda.is_available\(\): ([^\n]+)"),
        "device": parse_first_match(cell22, r"device: ([^\n]+)") or parse_first_match(cell25, r"device: ([^\n]+)"),
        "device_name": parse_first_match(cell22, r"device_name: ([^\n]+)"),
        "metrics": parse_metrics(cell42),
    }


def build_preprocess_cell(config):
    use_indicators = config["use_indicators"]
    return f"""import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

file_path = filename
stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
print(f"Data loaded from {{file_path}}. First 5 rows:\\n{{stock_df.head()}}\\n")

feature_columns = ['Close']

if {use_indicators}:
    stock_df['MA5'] = stock_df['Close'].rolling(5).mean()
    stock_df['MA20'] = stock_df['Close'].rolling(20).mean()
    delta = stock_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    stock_df['RSI14'] = 100 - (100 / (1 + rs))
    stock_df = stock_df.dropna().copy()
    feature_columns = ['Close', 'MA5', 'MA20', 'RSI14']

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


def build_dataset_cell():
    return """def create_dataset(feature_dataset, target_dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(feature_dataset) - look_back):
        dataX.append(feature_dataset[i:(i + look_back)])
        dataY.append(target_dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

X, y = create_dataset(scaled_features, scaled_close_prices, look_back)
print(f"X dataset shape: {X.shape}")
print(f"y dataset shape: {y.shape}")
"""


def build_model_cell(config):
    hidden_sizes = json.dumps(config["hidden_sizes"])
    dropout = config["dropout"]
    return f"""import os
import random
import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_sizes = {hidden_sizes}
dropout_rate = {dropout}

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.lstms = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.lstms.append(nn.LSTM(input_size=prev_size, hidden_size=hidden_size, batch_first=True))
            prev_size = hidden_size

        self.attn_weight = nn.Parameter(torch.empty(prev_size, prev_size))
        self.attn_bias = nn.Parameter(torch.zeros(prev_size))
        self.output = nn.Linear(prev_size, 1)
        nn.init.normal_(self.attn_weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.attn_bias)

    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
            x = self.dropout(x)
        ui = torch.tanh(torch.matmul(x, self.attn_weight) + self.attn_bias)
        alpha = torch.softmax(torch.sum(ui, dim=2), dim=1).unsqueeze(-1)
        context = torch.sum(x * alpha, dim=1)
        context = self.dropout(context)
        return self.output(context)

model_with_attention = AttentionLSTM(
    input_size=X_train.shape[2],
    hidden_sizes=hidden_sizes,
    dropout_rate=dropout_rate,
).to(device)

print("PyTorch Attention-LSTM model defined.")
print(f"CUDA_VISIBLE_DEVICES: {{os.environ.get('CUDA_VISIBLE_DEVICES')}}")
print(f"torch.cuda.is_available(): {{torch.cuda.is_available()}}")
print(f"device: {{device}}")
print(f"hidden_sizes: {{hidden_sizes}}")
print(f"dropout_rate: {{dropout_rate}}")
print(f"input_features: {{X_train.shape[2]}}")
if torch.cuda.is_available():
    print(f"device_name: {{torch.cuda.get_device_name(0)}}")
print(model_with_attention)
"""


def build_optimizer_cell(config):
    return f"""import torch.optim as optim

learning_rate = {config["learning_rate"]}
batch_size = {config["batch_size"]}
epochs = {config["epochs"]}

criterion = nn.MSELoss()
optimizer = optim.Adam(model_with_attention.parameters(), lr=learning_rate)

print("PyTorch optimizer and loss are ready.")
print(f"Training config ready on device: {{device}}")
print(f"learning_rate: {{learning_rate}}")
print(f"batch_size: {{batch_size}}")
print(f"epochs: {{epochs}}")
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
    print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

print("PyTorch Attention-LSTM model training completed.")
"""


def build_predict_cell():
    return """print("Generating predictions using PyTorch Attention-LSTM model...")

model_with_attention.eval()
with torch.no_grad():
    train_predict_attention = model_with_attention(X_train_tensor.to(device)).cpu().numpy()
    test_predict_attention = model_with_attention(X_test_tensor.to(device)).cpu().numpy()

print(f"Shape of train_predict_attention: {train_predict_attention.shape}")
print(f"Shape of test_predict_attention: {test_predict_attention.shape}")
print("Predictions generated successfully.")
"""


def build_inverse_transform_cell():
    return """print("Inverse transforming predictions and actual values...")

train_predict_attention = target_scaler.inverse_transform(train_predict_attention)
test_predict_attention = target_scaler.inverse_transform(test_predict_attention)
y_train_inverse = target_scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inverse = target_scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"Shape of inverse transformed train_predict_attention: {train_predict_attention.shape}")
print(f"Shape of inverse transformed test_predict_attention: {test_predict_attention.shape}")
print(f"Shape of inverse transformed y_train_inverse: {y_train_inverse.shape}")
print(f"Shape of inverse transformed y_test_inverse: {y_test_inverse.shape}")

print("Inverse transformation completed.")
"""


def prepare_notebook(original_nb, config):
    nb = copy.deepcopy(original_nb)
    version = config["version"]

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
        f"ticker_symbol = '{TARGET_TICKER}'\n"
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
    nb.cells[21].source = (
        "**Reasoning**:\n"
        f"{config['description']}"
    )
    nb.cells[22].source = build_model_cell(config)
    nb.cells[23].source = f"## Experiment {version}: Optimizer and Loss"
    nb.cells[24].source = (
        "**Reasoning**:\n"
        "Keep the regression loss fixed to MSE and tune the requested hyperparameters around the Torch baseline."
    )
    nb.cells[25].source = build_optimizer_cell(config)
    nb.cells[26].source = f"## Experiment {version}: Training"
    nb.cells[27].source = (
        "**Reasoning**:\n"
        "Train the configured Torch model and keep the rest of the evaluation flow unchanged so the versions remain comparable."
    )
    nb.cells[28].source = build_train_cell()
    nb.cells[29].source = f"## Experiment {version}: Prediction"
    nb.cells[30].source = (
        "**Reasoning**:\n"
        "Generate training and testing predictions with the configured experiment model."
    )
    nb.cells[31].source = build_predict_cell()
    nb.cells[34].source = build_inverse_transform_cell()

    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return nb


def execute_notebook(notebook, notebook_path):
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, notebook_path)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
    )

    try:
        client.execute()
    except CellExecutionError:
        nbformat.write(notebook, notebook_path)
        raise

    nbformat.write(notebook, notebook_path)
    return notebook


def compare_to_baseline(metrics, baseline_metrics):
    return {
        key: round(metrics[key] - baseline_metrics[key], 4)
        for key in baseline_metrics
        if key in metrics
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["MPLBACKEND"] = "Agg"

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_nb = nbformat.read(ORIGINAL_NOTEBOOK, as_version=4)
    baseline_nb = nbformat.read(BASELINE_NOTEBOOK, as_version=4)
    baseline_summary = parse_summary(baseline_nb)
    baseline_metrics = baseline_summary["metrics"]

    results = {
        "baseline": {
            "version": "v1_baseline_torch",
            "notebook": str(BASELINE_NOTEBOOK.relative_to(ROOT)),
            "summary": baseline_summary,
        },
        "experiments": [],
    }

    for config in EXPERIMENTS:
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
            "delta_vs_baseline": compare_to_baseline(summary["metrics"], baseline_metrics),
        }
        results["experiments"].append(result)

        result_path = RESULTS_DIR / f"{version}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    combined_path = RESULTS_DIR / "all_experiments.json"
    combined_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
