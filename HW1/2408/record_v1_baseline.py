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
OUTPUT_NOTEBOOK = ROOT / "Stock_predict_v1_2408TW.ipynb"
TARGET_TICKER = "2408.TW"


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
        "metrics": parse_metrics(cell42),
    }


def prepare_notebook(original_nb):
    new_nb = copy.deepcopy(original_nb)

    if len(new_nb.cells) <= 12:
        raise RuntimeError("Notebook structure is shorter than expected.")

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
                "v1_executed_output": new_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
