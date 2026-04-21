import copy
import json
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
ORIGINAL_NOTEBOOK = ROOT.parent / "Stock_predict.ipynb"
OUTPUT_NOTEBOOK = ROOT / "Stock_predict_v1_2330TW_torch.ipynb"
TARGET_TICKER = "2330.TW"


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


def parse_summary(nb):
    cell12 = collect_text_output(nb.cells[12])
    cell13 = collect_text_output(nb.cells[13])
    cell14 = collect_text_output(nb.cells[14])
    cell18 = collect_text_output(nb.cells[18])
    cell19 = collect_text_output(nb.cells[19])
    cell22 = collect_text_output(nb.cells[22])
    cell25 = collect_text_output(nb.cells[25])
    cell42 = collect_text_output(nb.cells[42])

    return {
        "ticker": parse_first_match(cell12, r"([0-9]{4}\.TW)"),
        "csv_file": parse_first_match(cell13, r"儲存為 ([^\n]+)"),
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


def prepare_notebook(original_nb):
    new_nb = copy.deepcopy(original_nb)

    new_nb.cells[3].source = (
        "import importlib.util\n"
        "\n"
        "if importlib.util.find_spec('yfinance') is None:\n"
        "    %pip install yfinance\n"
        "else:\n"
        "    print('yfinance is already installed.')\n"
    )

    new_nb.cells[12].source = (
        "# Version 1 baseline ticker\n"
        f"ticker_symbol = '{TARGET_TICKER}'\n"
        "print(f'本次 baseline 使用股票代碼: {ticker_symbol}')\n"
    )

    new_nb.cells[20].source = "## Define PyTorch LSTM Model with Attention Mechanism"
    new_nb.cells[21].source = (
        "**Reasoning**:\n"
        "The TensorFlow model is replaced by a PyTorch Attention-LSTM so the rest of the notebook can keep the same preprocessing, prediction, plotting, and metric flow."
    )
    new_nb.cells[22].source = '''import os
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

class AttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size_1=128, hidden_size_2=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2, batch_first=True)
        self.attn_weight = nn.Parameter(torch.empty(hidden_size_2, hidden_size_2))
        self.attn_bias = nn.Parameter(torch.zeros(hidden_size_2))
        self.output = nn.Linear(hidden_size_2, 1)
        nn.init.normal_(self.attn_weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.attn_bias)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        ui = torch.tanh(torch.matmul(x, self.attn_weight) + self.attn_bias)
        alpha = torch.softmax(torch.sum(ui, dim=2), dim=1).unsqueeze(-1)
        context = torch.sum(x * alpha, dim=1)
        return self.output(context)

model_with_attention = AttentionLSTM().to(device)
print("PyTorch Attention-LSTM model defined.")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"device: {device}")
if torch.cuda.is_available():
    print(f"device_name: {torch.cuda.get_device_name(0)}")
print(model_with_attention)
'''

    new_nb.cells[23].source = "## Configure PyTorch Optimizer and Loss"
    new_nb.cells[24].source = (
        "**Reasoning**:\n"
        "PyTorch separates model definition from optimizer and loss setup, so the baseline keeps Adam and mean squared error to stay comparable to the original notebook."
    )
    new_nb.cells[25].source = '''import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model_with_attention.parameters(), lr=0.001)
print("PyTorch optimizer and loss are ready.")
print(f"Training config ready on device: {device}")
'''

    new_nb.cells[26].source = "## Train PyTorch Attention-LSTM Model"
    new_nb.cells[27].source = (
        "**Reasoning**:\n"
        "The training loop iterates over mini-batches from a DataLoader and keeps the same 50-epoch / batch-size-32 baseline setting used in the TensorFlow notebook."
    )
    new_nb.cells[28].source = '''from torch.utils.data import DataLoader, TensorDataset

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=32,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

num_epochs = 50
history_attention = []

print("Starting PyTorch Attention-LSTM model training...")
for epoch in range(num_epochs):
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
    print(f"Epoch {epoch + 1}/{num_epochs} - loss: {epoch_loss:.6f}")

print("PyTorch Attention-LSTM model training completed.")
'''

    new_nb.cells[29].source = "## Make Predictions with PyTorch Attention-LSTM Model"
    new_nb.cells[30].source = (
        "**Reasoning**:\n"
        "The trained PyTorch model predicts on the full training and testing tensors so downstream inverse scaling and metric cells can stay unchanged."
    )
    new_nb.cells[31].source = '''print("Generating predictions using PyTorch Attention-LSTM model...")

model_with_attention.eval()
with torch.no_grad():
    train_predict_attention = model_with_attention(X_train_tensor.to(device)).cpu().numpy()
    test_predict_attention = model_with_attention(X_test_tensor.to(device)).cpu().numpy()

print(f"Shape of train_predict_attention: {train_predict_attention.shape}")
print(f"Shape of test_predict_attention: {test_predict_attention.shape}")
print("Predictions generated successfully.")
'''

    for cell in new_nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return new_nb


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["MPLBACKEND"] = "Agg"

    original_nb = nbformat.read(ORIGINAL_NOTEBOOK, as_version=4)
    original_summary = parse_summary(original_nb)

    output_nb = prepare_notebook(original_nb)
    nbformat.write(output_nb, OUTPUT_NOTEBOOK)

    client = NotebookClient(
        output_nb,
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
    )

    try:
        client.execute()
    except CellExecutionError:
        nbformat.write(output_nb, OUTPUT_NOTEBOOK)
        raise

    nbformat.write(output_nb, OUTPUT_NOTEBOOK)
    new_summary = parse_summary(output_nb)

    print(
        json.dumps(
            {
                "original_saved_output": original_summary,
                "v1_torch_executed_output": new_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
