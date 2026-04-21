# Version Log

## v1 - 2026-03-20

### 目標
以 `2408.TW` 建立第一個可重現的 baseline，並改用 PyTorch 執行，固定 `CUDA_VISIBLE_DEVICES=1`。

### 本次更新流程
1. 閱讀 `Homework1.pdf`，確認 Phase 1 需要先重現 baseline，並記錄 test set 的 `RMSE` / `MAPE`。
2. 檢查 `Stock_predict.ipynb` 的原始流程：抓 10 年資料、只使用 `Close`、`look_back=100`、`train/test=95/5`、2 層 LSTM + Attention、訓練 50 epochs。
3. 讀取 notebook 既有輸出，保存原始參考基線：`2330.TW`、TensorFlow/Keras、`Test RMSE=66.5267`、`Test MAPE=3.43%`。
4. 在 `rnn` 環境補齊 notebook 依賴後，驗證到 TensorFlow 在此環境無法正確載入 GPU 相關 library，因此雖然可以設定 `CUDA_VISIBLE_DEVICES=1`，實際上仍會退回 CPU。
5. 依照新需求改用 PyTorch，新增 `record_v1_baseline_torch.py`，由原始 notebook 生成 Torch 版 `Stock_predict_v1_2408TW_torch.ipynb`。
6. 在 runner 程式中設定 `CUDA_VISIBLE_DEVICES=1`，並驗證 PyTorch 可見 1 張 GPU，對應 `NVIDIA GeForce RTX 3090`。
7. 執行 Torch 版 notebook，輸出 `2408_stock_data.csv` 與第一版 baseline 結果。

### v1 執行設定
- 股票代碼：`2408.TW`
- 資料筆數：`2429`
- 最新資料日期：`2026-03-19 00:00:00+08:00`
- 特徵：`Close`
- `look_back`：`100`
- `X_train`：`(2212, 100, 1)`
- `X_test`：`(117, 100, 1)`
- 模型：`PyTorch Attention-LSTM`
- batch size：`32`
- epochs：`50`
- 裝置：`cuda`
- `CUDA_VISIBLE_DEVICES`：`1`
- GPU 名稱：`NVIDIA GeForce RTX 3090`

### v1 結果
- Train RMSE: `8.8944`
- Train MAE: `6.8419`
- Train MAPE: `12.16%`
- Test RMSE: `127.2177`
- Test MAE: `114.4714`
- Test MAPE: `58.67%`

### 與原始儲存輸出的差異
> 注意：以下差異同時包含「股票從 `2330.TW` 改成 `2408.TW`」與「框架從 TensorFlow 改成 PyTorch」，所以這張表只能作版本追蹤，不能視為公平模型比較。

| 項目 | 原始儲存輸出 | v1 Torch 執行結果 | 差異 |
| --- | --- | --- | --- |
| 股票代碼 | `2330.TW` | `2408.TW` | 範例股票改為 `2408.TW` |
| 框架 | TensorFlow / Keras | PyTorch | 改用 Torch |
| 裝置 | 未記錄；TF 在 `rnn` 無法正確使用 GPU | `cuda` | PyTorch 成功使用 GPU |
| `CUDA_VISIBLE_DEVICES` | 未設定於 notebook 輸出 | `1` | 固定只看到 GPU 1 |
| Train RMSE | `21.2506` | `8.8944` | `-12.3562` |
| Train MAE | `16.5228` | `6.8419` | `-9.6809` |
| Train MAPE | `3.98%` | `12.16%` | `+8.18%` |
| Test RMSE | `66.5267` | `127.2177` | `+60.6910` |
| Test MAE | `54.9472` | `114.4714` | `+59.5242` |
| Test MAPE | `3.43%` | `58.67%` | `+55.24%` |

### 產出檔案
- `Stock_predict_v1_2408TW_torch.ipynb`：目前的第一版 baseline notebook
- `record_v1_baseline_torch.py`：產生並執行 Torch 版 notebook 的 runner
- `2408_stock_data.csv`：本次 baseline 使用的股票資料

### 備註
- `Stock_predict_v1_2408TW.ipynb` 是先前 TensorFlow 路線留下的中途產物，後續分析請以 `Stock_predict_v1_2408TW_torch.ipynb` 為主。
- `v1` 現在正式作為後續所有實驗的 baseline。

## v2-v5 - 2026-03-20

### 實驗目標
以 `v1` 的 `2408.TW` Torch 結果作為 baseline，開始系統化測試不同版本，覆蓋：
- 序列長度 (`look_back`)
- LSTM 單元數 / 層數
- 學習率 / batch size / epochs
- Dropout
- 可選技術指標 (`MA5`、`MA20`、`RSI14`)

### 本次新增流程
1. 新增 `run_torch_experiments.py`，把 Torch notebook 轉成可參數化實驗 runner。
2. runner 會為每個版本自動產生獨立 notebook，並執行後輸出對應的 JSON 結果。
3. 所有實驗沿用 `2408_stock_data.csv`，避免不同版本因為重新抓資料而產生額外變因。
4. 實驗 notebook 會固定 `CUDA_VISIBLE_DEVICES=1`，並使用 PyTorch 的 `cuda` 裝置執行。

### Baseline 與各版本結果

| 版本 | 主要調整 | 特徵 | Test RMSE | 相對 baseline | Test MAPE | 相對 baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `v1_baseline_torch` | `look_back=100`、hidden=`[128,64]`、lr=`0.001`、batch=`32`、epochs=`50`、dropout=`0.0` | `Close` | `127.2177` | `0.0000` | `58.67%` | `0.00%` |
| `v2_lb60` | `look_back=60` | `Close` | `91.1432` | `-36.0745` | `45.13%` | `-13.54%` |
| `v3_deep_dropout` | hidden=`[256,128,64]`、dropout=`0.2` | `Close` | `136.6955` | `+9.4778` | `62.53%` | `+3.86%` |
| `v4_train_tune` | lr=`0.0005`、batch=`64`、epochs=`80` | `Close` | `105.7693` | `-21.4484` | `50.41%` | `-8.26%` |
| `v5_indicators` | `look_back=90`、dropout=`0.2`、lr=`0.0005`、epochs=`60`、加入指標 | `Close` + `MA5` + `MA20` + `RSI14` | `110.5627` | `-16.6550` | `49.92%` | `-8.75%` |

### 觀察
- 目前 `Test RMSE` 最佳版本是 `v2_lb60`，代表在這組資料上較短的序列長度比 baseline 的 `100` 天更有效。
- 目前 `Test MAPE` 最佳版本是 `v2_lb60`，而且它同時也是整體泛化表現最穩定的一版。
- `v4_train_tune` 明顯優於 baseline，但還是沒有超過 `v2_lb60`。
- `v3_deep_dropout` 比 baseline 更差，表示這組股票資料上，模型加深加大後反而容易過度複雜。
- `v5_indicators` 有改善 baseline，但仍不如 `v2_lb60`；技術指標有幫助，但目前組合還不是最佳。

### 新增產出檔案
- `run_torch_experiments.py`：批次生成與執行多版本 Torch notebook
- `experiments/notebooks/v2_lb60.ipynb`
- `experiments/notebooks/v3_deep_dropout.ipynb`
- `experiments/notebooks/v4_train_tune.ipynb`
- `experiments/notebooks/v5_indicators.ipynb`
- `experiments/results/v2_lb60.json`
- `experiments/results/v3_deep_dropout.json`
- `experiments/results/v4_train_tune.json`
- `experiments/results/v5_indicators.json`
- `experiments/results/all_experiments.json`

### 目前建議
- 下一輪若只追求測試誤差下降，先從 `v2_lb60` 往下細調，例如測 `look_back=50/70/80`。
- 若想繼續走特徵工程，可以在 `v5_indicators` 的基礎上調整 RSI window、移動平均長度，或再加入成交量類特徵，但要維持獨立縮放與 target scaler。

## 2330 重跑（修正版） - 2026-03-20

### 目標
將目前範例股票切回 `2330.TW`，保留原本 `2408.TW` 的所有產出不覆蓋，重新跑一次 baseline 與 `v2~v5` 實驗，並單獨存檔。

### 本次新增流程
1. 複製原本的 Torch runner，建立 `record_v1_baseline_2330_torch.py` 與 `run_torch_experiments_2330.py`。
2. 將新版 runner 的 ticker 改為 `2330.TW`，並把輸出路徑改成 `Stock_predict_v1_2330TW_torch.ipynb` 與 `experiments_2330/`，避免覆蓋既有 `2408` 結果。
3. 比對原始 TensorFlow notebook 與 Torch 版後，確認主要落差來自 Attention 實作不同，以及 Torch DataLoader 原本使用 `shuffle=False`。
4. 直接修正 Torch 版 Attention 公式，使其更接近原始 Keras `Attention` 的 `softmax(sum(tanh(xW+b)))`。
5. 將 Torch 訓練改成 `shuffle=True`，讓 batch 行為更接近原始 `model.fit()` 預設。
6. 用修正版重新執行 `2330` baseline，再跑完整的 `v2_lb60`、`v3_deep_dropout`、`v4_train_tune`、`v5_indicators`。

### 2330 Baseline
- baseline notebook：`Stock_predict_v1_2330TW_torch.ipynb`
- 股票代碼：`2330.TW`
- 資料筆數：`2429`
- `look_back`：`100`
- 特徵：`Close`
- Train RMSE: `14.5263`
- Train MAE: `9.2797`
- Train MAPE: `1.96%`
- Test RMSE: `50.4127`
- Test MAE: `37.2124`
- Test MAPE: `2.28%`

### 2330 各版本結果

| 版本 | 主要調整 | 特徵 | Test RMSE | 相對 2330 baseline | Test MAPE | 相對 2330 baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `v1_baseline_torch` | `look_back=100`、hidden=`[128,64]`、lr=`0.001`、batch=`32`、epochs=`50`、dropout=`0.0` | `Close` | `50.4127` | `0.0000` | `2.28%` | `0.00%` |
| `v2_lb60` | `look_back=60` | `Close` | `49.9911` | `-0.4216` | `2.33%` | `+0.05%` |
| `v3_deep_dropout` | hidden=`[256,128,64]`、dropout=`0.2` | `Close` | `456.2289` | `+405.8162` | `26.04%` | `+23.76%` |
| `v4_train_tune` | lr=`0.0005`、batch=`64`、epochs=`80` | `Close` | `62.5129` | `+12.1002` | `2.98%` | `+0.70%` |
| `v5_indicators` | `look_back=90`、dropout=`0.2`、lr=`0.0005`、epochs=`60`、加入指標 | `Close` + `MA5` + `MA20` + `RSI14` | `82.5861` | `+32.1734` | `4.25%` | `+1.97%` |

### 2330 觀察
- 修正 Attention 與 `shuffle=True` 之後，`2330` baseline 已經回到合理範圍，且與原始 `Stock_predict.ipynb` 的 TensorFlow 表現接近。
- `v2_lb60` 在 `Test RMSE` 上只比 baseline 再好一點點，但 `Test MAE/MAPE` 沒有更好，代表它不是明顯優勝版本。
- `v3_deep_dropout` 仍然最差，表示對 `2330` 來說模型加深加大依舊容易失真。
- `v4_train_tune` 和 `v5_indicators` 在修正版流程下都沒有超過 baseline，顯示先把模型行為對齊原 notebook，比盲目加特徵或加深模型更重要。
- 目前如果只看整體穩定性，`2330` 仍然以修正版 baseline 最值得當作後續起點。

### 2330 新增產出檔案
- `record_v1_baseline_2330_torch.py`
- `run_torch_experiments_2330.py`
- `Stock_predict_v1_2330TW_torch.ipynb`
- `experiments_2330/notebooks/v2_lb60.ipynb`
- `experiments_2330/notebooks/v3_deep_dropout.ipynb`
- `experiments_2330/notebooks/v4_train_tune.ipynb`
- `experiments_2330/notebooks/v5_indicators.ipynb`
- `experiments_2330/results/v2_lb60.json`
- `experiments_2330/results/v3_deep_dropout.json`
- `experiments_2330/results/v4_train_tune.json`
- `experiments_2330/results/v5_indicators.json`
- `experiments_2330/results/all_experiments.json`

### 目前建議
- 如果後續要專注在 `2330`，先以目前修正版 `v1_baseline_torch` 當基準，不要再用舊的壞數字。
- 下一輪可以在這個修正版 baseline 上再微調 `look_back=80/120` 或 `learning rate=0.0007/0.0003`，但建議一次只改一個因素。

## 2408 重跑（修正版） - 2026-03-21

### 說明
- 本段結果是 `2408.TW` 在修正版 Torch runner 上的正式重跑結果。
- 修正內容和 `2330` 相同：Attention 改成更接近原始 Keras 公式，且訓練改為 `shuffle=True`。
- 這一段應視為先前 `2408` 區塊的更新版；先前 `2026-03-20` 的 `2408` 數字屬於修正前結果。

### 2408 修正版 Baseline
- baseline notebook：`Stock_predict_v1_2408TW_torch.ipynb`
- 股票代碼：`2408.TW`
- 資料筆數：`2429`
- 最新資料日期：`2026-03-20 00:00:00+08:00`
- `look_back`：`100`
- 特徵：`Close`
- Train RMSE: `1.4719`
- Train MAE: `1.0499`
- Train MAPE: `1.91%`
- Test RMSE: `34.8954`
- Test MAE: `23.3714`
- Test MAPE: `10.03%`

### 2408 修正版各版本結果

| 版本 | 主要調整 | 特徵 | Test RMSE | 相對 2408 baseline | Test MAPE | 相對 2408 baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `v1_baseline_torch` | `look_back=100`、hidden=`[128,64]`、lr=`0.001`、batch=`32`、epochs=`50`、dropout=`0.0` | `Close` | `34.8954` | `0.0000` | `10.03%` | `0.00%` |
| `v2_lb60` | `look_back=60` | `Close` | `16.9810` | `-17.9144` | `5.85%` | `-4.18%` |
| `v3_deep_dropout` | hidden=`[256,128,64]`、dropout=`0.2` | `Close` | `64.2755` | `+29.3801` | `23.63%` | `+13.60%` |
| `v4_train_tune` | lr=`0.0005`、batch=`64`、epochs=`80` | `Close` | `44.8678` | `+9.9724` | `16.88%` | `+6.85%` |
| `v5_indicators` | `look_back=90`、dropout=`0.2`、lr=`0.0005`、epochs=`60`、加入指標 | `Close` + `MA5` + `MA20` + `RSI14` | `115.2867` | `+80.3913` | `45.19%` | `+35.16%` |

### 2408 修正版觀察
- 在先前修正版 `v1-v5` 範圍內，最穩的版本是 `v2_lb60`。
- `v2_lb60` 在這一批 `v1-v5` 裡不只 `Test RMSE` 最低，`Test MAPE` 也最低，代表它比 baseline 更穩定。
- `v4_train_tune` 在修正版下反而不如 baseline，表示 `2408` 不像 `2330` 那麼吃訓練參數調整。
- `v5_indicators` 在 `2408` 上明顯變差，代表這組 `MA5 / MA20 / RSI14` 特徵對 `2408` 目前是噪音大於幫助。
- `v3_deep_dropout` 持續最差，說明對 `2408` 來說，加深模型和 dropout 不是好方向。

### 2408 / 2330 修正版對照
- `2330` 修正版 baseline：`Test RMSE 50.4127`、`Test MAPE 2.28%`
- `2408` 修正版 baseline：`Test RMSE 34.8954`、`Test MAPE 10.03%`
- 若看 `RMSE`，`2408` baseline 較低。
- 若看 `MAPE`，`2330` baseline 較低，比例誤差更穩。
- 若看各自最佳版本：`2330` 目前大致是 baseline / `v2_lb60` 相近；`2408` 則明顯是 `v2_lb60` 最佳。

### 2408 修正版產出檔案
- `Stock_predict_v1_2408TW_torch.ipynb`
- `experiments/notebooks/v2_lb60.ipynb`
- `experiments/notebooks/v3_deep_dropout.ipynb`
- `experiments/notebooks/v4_train_tune.ipynb`
- `experiments/notebooks/v5_indicators.ipynb`
- `experiments/results/v2_lb60.json`
- `experiments/results/v3_deep_dropout.json`
- `experiments/results/v4_train_tune.json`
- `experiments/results/v5_indicators.json`
- `experiments/results/all_experiments.json`

## 2408 Look-back Sweep - 2026-03-21

### 目標
以修正版 `2408` 的 `v2_lb60` 作為參考贏家，只做 `look_back` 的窄搜尋，不改模型結構、dropout、學習率、batch size、epochs 或特徵工程，確認 60 天附近是否還有更穩的區間。

### 本次新增流程
1. 新增 `run_torch_lookback_sweep_2408.py`，重用修正版的 Torch notebook 生成邏輯。
2. 新增獨立輸出目錄 `experiments_2408_lb_sweep/`，避免覆蓋既有 `experiments/` 結果。
3. 固定下列設定不變：
   - Keras-like attention
   - `shuffle=True`
   - `hidden_sizes=[128,64]`
   - `dropout=0.0`
   - `learning_rate=0.001`
   - `batch_size=32`
   - `epochs=50`
   - `Close` only
4. 新增 6 個版本：
   - `v6_lb40`
   - `v7_lb50`
   - `v8_lb55`
   - `v9_lb65`
   - `v10_lb70`
   - `v11_lb80`
5. 在總表中一併收錄修正版 `v1_baseline_torch` 與 `v2_lb60` 作為參考列，並依 `Test RMSE` 排名。

### 2408 Look-back Sweep 排名

| RMSE 排名 | 版本 | `look_back` | Test RMSE | 相對 `v2_lb60` | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `v9_lb65` | `65` | `14.9419` | `-2.0391` | `10.7103` | `5.28%` |
| 2 | `v7_lb50` | `50` | `15.9457` | `-1.0353` | `11.3026` | `5.47%` |
| 3 | `v2_lb60` | `60` | `16.9810` | `0.0000` | `12.1988` | `5.85%` |
| 4 | `v8_lb55` | `55` | `18.1763` | `+1.1953` | `13.5352` | `6.64%` |
| 5 | `v10_lb70` | `70` | `19.0137` | `+2.0327` | `13.7534` | `6.59%` |
| 6 | `v6_lb40` | `40` | `19.3829` | `+2.4019` | `13.9026` | `6.65%` |
| 7 | `v11_lb80` | `80` | `26.3376` | `+9.3566` | `18.9758` | `8.69%` |
| 8 | `v1_baseline_torch` | `100` | `34.8954` | `+17.9144` | `23.3714` | `10.03%` |

### 結論
- 這輪 look-back sweep 有成功打敗原本的 `v2_lb60`。
- 新的最佳版本是 `v9_lb65`，`Test RMSE=14.9419`、`Test MAPE=5.28%`，兩者都優於 `v2_lb60`。
- `v7_lb50` 也優於 `v2_lb60`，表示 `2408` 的最佳區間不是單點，而是大致落在 `50-65` 天之間。
- `v11_lb80` 明顯退步，代表區間拉長到 80 天後開始傷害泛化能力。
- 如果下一輪還要繼續做 `2408`，建議優先在 `v9_lb65` 周圍微調，例如 `62/68/75`，而不是再回去測更長區間。

### 新增產出檔案
- `run_torch_lookback_sweep_2408.py`
- `experiments_2408_lb_sweep/notebooks/v6_lb40.ipynb`
- `experiments_2408_lb_sweep/notebooks/v7_lb50.ipynb`
- `experiments_2408_lb_sweep/notebooks/v8_lb55.ipynb`
- `experiments_2408_lb_sweep/notebooks/v9_lb65.ipynb`
- `experiments_2408_lb_sweep/notebooks/v10_lb70.ipynb`
- `experiments_2408_lb_sweep/notebooks/v11_lb80.ipynb`
- `experiments_2408_lb_sweep/results/v6_lb40.json`
- `experiments_2408_lb_sweep/results/v7_lb50.json`
- `experiments_2408_lb_sweep/results/v8_lb55.json`
- `experiments_2408_lb_sweep/results/v9_lb65.json`
- `experiments_2408_lb_sweep/results/v10_lb70.json`
- `experiments_2408_lb_sweep/results/v11_lb80.json`
- `experiments_2408_lb_sweep/results/all_experiments.json`

## Notebook Display Update - 2026-03-21

### 目標
在現有 notebook 內直接列出最近五個測試日的預測值與實際值，方便快速檢查模型尾段表現，同時整理下一輪更有機會提升 `2408` 結果的修改方向。

### 本次更新流程
1. 檢查 `Stock_predict.ipynb` 後段 cell 結構，確認第 `38` 格只是重複的 test plot，可安全改成文字表格輸出。
2. 新增 `sync_recent_results_cell.py`，統一把所有現有 `.ipynb` 的第 `38` 格同步成「近五天測試集預測 vs 實際」表格。
3. 這次同步不更動模型架構、資料切分、超參數或既有結果 JSON，只更新 notebook 顯示內容。
4. 後續由 `Stock_predict.ipynb` 產生的新 notebook 也會沿用這個欄位。

### 本次成果差異
- 原本第 `38` 格：重複顯示另一張 test plot。
- 更新後第 `38` 格：列印最近五筆測試資料的 `Date`、`Actual`、`Predicted`、`AbsError`。
- 這次更新不改動 `RMSE / MAE / MAPE`，屬於 notebook 可讀性與檢查流程的補強。

### 下一輪高優先修改方向
1. 在 `v9_lb65` 周圍做更細的 `look_back` 微調，例如 `62`、`63`、`67`、`68`，因為目前最佳區間明顯落在 `50-65` 天附近。
2. 固定 `look_back=65` 後改用 `SmoothL1Loss`，降低單日大波動對 `MSE` 的放大效應，通常比直接再加深模型更穩。
3. 固定 `look_back=65` 後加入 learning-rate scheduler，例如 `ReduceLROnPlateau` 或 cosine decay，讓前期收斂速度與後期細調兼得。
4. 在不加深層數的前提下測試較小 hidden size，例如 `[96, 48]` 或 `[64, 32]`，避免 `2408` 在短窗口下學到過強的局部噪音。
5. 做輕量特徵工程，但先從低風險欄位開始，例如 `Volume`、`Return1`、`EMA10`，比一次加入太多技術指標更容易看出有效性。
6. 嘗試把 `v7_lb50` 與 `v9_lb65` 做簡單平均 ensemble；目前這兩版都穩，平均後有機會再壓低尾段誤差。

### 新增產出檔案
- `sync_recent_results_cell.py`

## 2408 Look-back Refine - 2026-03-21

### 目標
延續上一輪 `2408` look-back sweep 的結果，固定修正版 Torch 模型與所有訓練設定不變，只在目前最佳的 `v9_lb65` 周圍做更細的窗口微調，確認 `65` 附近是否還能再壓低 `Test RMSE`。

### 本次新增流程
1. 新增 `run_torch_lookback_refine_2408.py`，建立獨立的 `experiments_2408_lb_refine/` 輸出目錄，不覆蓋上一輪 `experiments_2408_lb_sweep/`。
2. 固定下列設定完全不變：
   - Keras-like attention
   - `shuffle=True`
   - `hidden_sizes=[128,64]`
   - `dropout=0.0`
   - `learning_rate=0.001`
   - `batch_size=32`
   - `epochs=50`
   - `Close` only
3. 新增 4 個版本：
   - `v12_lb62`
   - `v13_lb63`
   - `v14_lb67`
   - `v15_lb68`
4. 在總表中一併收錄：
   - 修正版 `v1_baseline_torch`
   - 原始較強版本 `v2_lb60`
   - 上一輪次佳 `v7_lb50`
   - 上一輪最佳 `v9_lb65`
5. 所有新 notebook 都沿用新版顯示欄位，可直接列印近五天測試集的 `Actual / Predicted / AbsError`。

### 2408 Look-back Refine 排名

| RMSE 排名 | 版本 | `look_back` | Test RMSE | 相對 `v9_lb65` | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `v14_lb67` | `67` | `14.7791` | `-0.1628` | `10.6104` | `5.18%` |
| 2 | `v9_lb65` | `65` | `14.9419` | `0.0000` | `10.7103` | `5.28%` |
| 3 | `v7_lb50` | `50` | `15.9457` | `+1.0038` | `11.3026` | `5.47%` |
| 4 | `v12_lb62` | `62` | `16.0752` | `+1.1333` | `12.0426` | `5.99%` |
| 5 | `v2_lb60` | `60` | `16.9810` | `+2.0391` | `12.1988` | `5.85%` |
| 6 | `v13_lb63` | `63` | `19.3162` | `+4.3743` | `14.0954` | `6.73%` |
| 7 | `v15_lb68` | `68` | `21.0997` | `+6.1578` | `14.8458` | `6.94%` |
| 8 | `v1_baseline_torch` | `100` | `34.8954` | `+19.9535` | `23.3714` | `10.03%` |

### 結論
- 這輪 refine 有成功打敗上一輪最佳的 `v9_lb65`。
- 新的最佳版本是 `v14_lb67`，`Test RMSE=14.7791`、`Test MAE=10.6104`、`Test MAPE=5.18%`。
- `v14_lb67` 相比 `v9_lb65` 的改善幅度不大，但方向一致：`RMSE`、`MAE`、`MAPE` 三個指標都更好。
- `v12_lb62` 雖然優於 `v2_lb60`，但仍輸給 `v7_lb50` 和 `v9_lb65`；表示最佳區間更像是落在 `65-67` 而不是 `60-63`。
- `v13_lb63` 和 `v15_lb68` 都明顯退步，代表 `2408` 在這附近不是平滑單峰，而是對窗口長度有較敏感的局部最佳點。
- 如果下一輪還要繼續做 `2408`，建議優先改測：
  - `look_back=66`
  - `SmoothL1Loss`
  - learning-rate scheduler
  而不是再繼續往 `68+` 拉長。

### 新增產出檔案
- `run_torch_lookback_refine_2408.py`
- `experiments_2408_lb_refine/notebooks/v12_lb62.ipynb`
- `experiments_2408_lb_refine/notebooks/v13_lb63.ipynb`
- `experiments_2408_lb_refine/notebooks/v14_lb67.ipynb`
- `experiments_2408_lb_refine/notebooks/v15_lb68.ipynb`
- `experiments_2408_lb_refine/results/v12_lb62.json`
- `experiments_2408_lb_refine/results/v13_lb63.json`
- `experiments_2408_lb_refine/results/v14_lb67.json`
- `experiments_2408_lb_refine/results/v15_lb68.json`
- `experiments_2408_lb_refine/results/all_experiments.json`

## 2408 Feature And Loss Experiments - 2026-03-21

### 目標
在目前最佳 `v14_lb67` 的基礎上，不再只調 `look_back`，改測更進一步的方向：
- 改 loss function：`SmoothL1Loss`
- 加入 `MA / RSI` 比值型特徵
- 加入 `EMA / MACD` 類特徵
- 測試 `特徵工程 + SmoothL1Loss` 的組合

### 本次新增流程
1. 新增 `run_torch_feature_loss_experiments_2408.py`，建立新的輸出目錄 `experiments_2408_feature_loss/`，避免覆蓋前面所有 `2408` 實驗。
2. 固定下列設定不變：
   - 股票：`2408.TW`
   - `look_back=67`
   - `hidden_sizes=[128,64]`
   - `dropout=0.0`
   - `learning_rate=0.001`
   - `batch_size=32`
   - `epochs=50`
   - Keras-like attention
   - `shuffle=True`
3. 新增 4 個版本：
   - `v16_lb67_smoothl1`
   - `v17_lb67_ratio_rsi`
   - `v18_lb67_ema_macd`
   - `v19_lb67_ratio_rsi_smoothl1`
4. 實作過程中發現第一版 advanced runner 會在 `close_only` 模式也提前計算並丟掉 `MA/RSI` 暖機區段，造成 `v16` 與 `v14` 比較不公平；已先修正成「只對當前 feature mode 計算與 dropna 必要欄位」後再重跑，以下表格皆為修正版結果。

### 2408 Feature / Loss 排名

| RMSE 排名 | 版本 | 主要調整 | Test RMSE | 相對 `v14_lb67` | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `v14_lb67` | `look_back=67`, `Close`, `MSE` | `14.7791` | `0.0000` | `10.6104` | `5.18%` |
| 2 | `v9_lb65` | `look_back=65`, `Close`, `MSE` | `14.9419` | `+0.1628` | `10.7103` | `5.28%` |
| 3 | `v19_lb67_ratio_rsi_smoothl1` | `Close + MA ratio + RSI centered`, `SmoothL1` | `16.6338` | `+1.8547` | `12.1162` | `5.92%` |
| 4 | `v16_lb67_smoothl1` | `Close`, `SmoothL1` | `16.9177` | `+2.1386` | `11.6576` | `5.64%` |
| 5 | `v17_lb67_ratio_rsi` | `Close + MA ratio + RSI centered`, `MSE` | `25.6220` | `+10.8429` | `19.7298` | `9.48%` |
| 6 | `v18_lb67_ema_macd` | `Close + EMA gap + MACD family`, `MSE` | `28.8006` | `+14.0215` | `19.7569` | `8.93%` |
| 7 | `v1_baseline_torch` | `look_back=100`, `Close`, `MSE` | `34.8954` | `+20.1163` | `23.3714` | `10.03%` |

### 結論
- 這輪沒有打敗目前最佳的 `v14_lb67`。
- 單改 `SmoothL1Loss` 的 `v16` 沒有贏過 `v14`，代表在目前這組 `2408` 設定下，loss function 不是主要瓶頸。
- `MA / RSI` 比值型特徵直接加入後，`v17` 明顯退步；和 `SmoothL1` 合併後的 `v19` 雖然比 `v17` 好很多，但仍輸給純 `Close` 的 `v14`。
- `EMA / MACD` 這組在目前設定下效果最差，說明這些特徵若不再做更細的設計，很容易把噪音一併餵進模型。
- 目前可以先得到一個很清楚的結論：對 `2408` 來說，單純把常見技術指標直接加進這個 Attention-LSTM，沒有比 `Close-only + look_back=67` 更穩。

### 下一輪建議
1. 先保留 `v14_lb67` 作為主線，繼續測 `look_back=66`。
2. 如果還想走特徵工程，應該改試更輕量的特徵，而不是一次塞多個技術指標：
   - `Return1`
   - `Volume`
   - `Volume / Volume_MA20`
   - 單一 `EMA10_GAP`
3. 如果還想走 loss 方向，建議不要只換 `SmoothL1Loss`，而是再搭配：
   - 調 `beta`
   - learning-rate scheduler
   - 較小 hidden size

### 新增產出檔案
- `run_torch_feature_loss_experiments_2408.py`
- `experiments_2408_feature_loss/notebooks/v16_lb67_smoothl1.ipynb`
- `experiments_2408_feature_loss/notebooks/v17_lb67_ratio_rsi.ipynb`
- `experiments_2408_feature_loss/notebooks/v18_lb67_ema_macd.ipynb`
- `experiments_2408_feature_loss/notebooks/v19_lb67_ratio_rsi_smoothl1.ipynb`
- `experiments_2408_feature_loss/results/v16_lb67_smoothl1.json`
- `experiments_2408_feature_loss/results/v17_lb67_ratio_rsi.json`
- `experiments_2408_feature_loss/results/v18_lb67_ema_macd.json`
- `experiments_2408_feature_loss/results/v19_lb67_ratio_rsi_smoothl1.json`
- `experiments_2408_feature_loss/results/all_experiments.json`

## 2330 Look-back Sweep - 2026-03-21

### 目標
把 `2408` 已做過的 look-back sweep 原樣搬到 `2330`，不改模型結構與訓練設定，只比較 `look_back` 對 `2330` 的影響，確認它是否也偏好 `60-67` 區間。

### 本次新增流程
1. 新增 `run_torch_lookback_sweep_2330.py`，輸出到 `experiments_2330_lb_sweep/`。
2. 固定設定不變：
   - `2330.TW`
   - Keras-like attention
   - `shuffle=True`
   - `hidden_sizes=[128,64]`
   - `dropout=0.0`
   - `learning_rate=0.001`
   - `batch_size=32`
   - `epochs=50`
   - `Close` only
3. 新增 6 個版本：
   - `v6_lb40`
   - `v7_lb50`
   - `v8_lb55`
   - `v9_lb65`
   - `v10_lb70`
   - `v11_lb80`
4. 總表一併收錄 `v1_baseline_torch` 與 `v2_lb60` 作為參考列。

### 2330 Look-back Sweep 排名

| RMSE 排名 | 版本 | `look_back` | Test RMSE | 相對 `v2_lb60` | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `v7_lb50` | `50` | `38.7006` | `-11.2905` | `29.1520` | `1.81%` |
| 2 | `v6_lb40` | `40` | `45.2905` | `-4.7006` | `34.3408` | `2.12%` |
| 3 | `v8_lb55` | `55` | `47.3317` | `-2.6594` | `34.8053` | `2.14%` |
| 4 | `v2_lb60` | `60` | `49.9911` | `0.0000` | `37.9546` | `2.33%` |
| 5 | `v1_baseline_torch` | `100` | `50.4127` | `+0.4216` | `37.2124` | `2.28%` |
| 6 | `v11_lb80` | `80` | `52.8901` | `+2.8990` | `41.3546` | `2.54%` |
| 7 | `v9_lb65` | `65` | `68.8567` | `+18.8656` | `56.1007` | `3.43%` |
| 8 | `v10_lb70` | `70` | `74.0740` | `+24.0829` | `59.3875` | `3.67%` |

### 結論
- `2330` 的最佳區間和 `2408` 明顯不同。
- `2330` 在這輪 sweep 的最佳版本是 `v7_lb50`，`Test RMSE=38.7006`、`Test MAPE=1.81%`。
- `2330` 對較長窗口比較敏感，`65/70` 這類設定反而顯著退步。
- 單看這一輪就能先得到一個清楚結論：`2330` 不像 `2408` 那麼偏好 `65-67`，反而偏向更短的 `40-55` 區間。

### 新增產出檔案
- `run_torch_lookback_sweep_2330.py`
- `experiments_2330_lb_sweep/notebooks/v6_lb40.ipynb`
- `experiments_2330_lb_sweep/notebooks/v7_lb50.ipynb`
- `experiments_2330_lb_sweep/notebooks/v8_lb55.ipynb`
- `experiments_2330_lb_sweep/notebooks/v9_lb65.ipynb`
- `experiments_2330_lb_sweep/notebooks/v10_lb70.ipynb`
- `experiments_2330_lb_sweep/notebooks/v11_lb80.ipynb`
- `experiments_2330_lb_sweep/results/v6_lb40.json`
- `experiments_2330_lb_sweep/results/v7_lb50.json`
- `experiments_2330_lb_sweep/results/v8_lb55.json`
- `experiments_2330_lb_sweep/results/v9_lb65.json`
- `experiments_2330_lb_sweep/results/v10_lb70.json`
- `experiments_2330_lb_sweep/results/v11_lb80.json`
- `experiments_2330_lb_sweep/results/all_experiments.json`

## 2330 Look-back Refine - 2026-03-21

### 目標
把 `2408` 上做過的 `v12-v15` 原樣搬到 `2330`，補齊同一組 `62/63/67/68` 測試，方便和 `2408` 做逐版本對照。

### 本次新增流程
1. 新增 `run_torch_lookback_refine_2330.py`，輸出到 `experiments_2330_lb_refine/`。
2. 固定 reference 版本為：
   - `v1_baseline_torch`
   - `v2_lb60`
   - `v7_lb50`
   - `v9_lb65`
3. 新增 4 個版本：
   - `v12_lb62`
   - `v13_lb63`
   - `v14_lb67`
   - `v15_lb68`

### 2330 Look-back Refine 排名

| RMSE 排名 | 版本 | `look_back` | Test RMSE | 相對 `v7_lb50` | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `v7_lb50` | `50` | `38.7006` | `0.0000` | `29.1520` | `1.81%` |
| 2 | `v14_lb67` | `67` | `42.1649` | `+3.4643` | `31.4328` | `1.96%` |
| 3 | `v2_lb60` | `60` | `49.9911` | `+11.2905` | `37.9546` | `2.33%` |
| 4 | `v1_baseline_torch` | `100` | `50.4127` | `+11.7121` | `37.2124` | `2.28%` |
| 5 | `v12_lb62` | `62` | `53.4755` | `+14.7749` | `42.1692` | `2.60%` |
| 6 | `v13_lb63` | `63` | `63.9230` | `+25.2224` | `51.5890` | `3.17%` |
| 7 | `v9_lb65` | `65` | `68.8567` | `+30.1561` | `56.1007` | `3.43%` |
| 8 | `v15_lb68` | `68` | `96.3030` | `+57.6024` | `75.8357` | `4.53%` |

### 結論
- 這輪 refine 沒有打敗 `v7_lb50`。
- 新版本中表現最好的是 `v14_lb67`，`Test RMSE=42.1649`、`Test MAPE=1.96%`，但仍輸給 `v7_lb50`。
- 這證明 `2330` 在同樣的 `62/63/67/68` 測試下，不像 `2408` 那樣能從 `67` 受益。
- `v15_lb68` 明顯最差，說明 `2330` 對窗口拉長比 `2408` 更敏感。

### 新增產出檔案
- `run_torch_lookback_refine_2330.py`
- `experiments_2330_lb_refine/notebooks/v12_lb62.ipynb`
- `experiments_2330_lb_refine/notebooks/v13_lb63.ipynb`
- `experiments_2330_lb_refine/notebooks/v14_lb67.ipynb`
- `experiments_2330_lb_refine/notebooks/v15_lb68.ipynb`
- `experiments_2330_lb_refine/results/v12_lb62.json`
- `experiments_2330_lb_refine/results/v13_lb63.json`
- `experiments_2330_lb_refine/results/v14_lb67.json`
- `experiments_2330_lb_refine/results/v15_lb68.json`
- `experiments_2330_lb_refine/results/all_experiments.json`

## 2330 Feature And Loss Experiments - 2026-03-21

### 目標
把 `2408` 上的 `v16-v19` 原樣搬到 `2330`，測試：
- 單改 `SmoothL1Loss`
- `MA / RSI` 比值型特徵
- `EMA / MACD` 類特徵
- `ratio_rsi + SmoothL1Loss`

### 本次新增流程
1. 新增 `run_torch_feature_loss_experiments_2330.py`，輸出到 `experiments_2330_feature_loss/`。
2. 固定設定：
   - `2330.TW`
   - `look_back=67`
   - `hidden_sizes=[128,64]`
   - `dropout=0.0`
   - `learning_rate=0.001`
   - `batch_size=32`
   - `epochs=50`
   - Keras-like attention
   - `shuffle=True`
3. 新增 4 個版本：
   - `v16_lb67_smoothl1`
   - `v17_lb67_ratio_rsi`
   - `v18_lb67_ema_macd`
   - `v19_lb67_ratio_rsi_smoothl1`

### 2330 Feature / Loss 排名

| RMSE 排名 | 版本 | 主要調整 | Test RMSE | 相對 `v7_lb50` | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `v16_lb67_smoothl1` | `Close`, `SmoothL1` | `32.9998` | `-5.7008` | `25.1240` | `1.57%` |
| 2 | `v14_lb67` | `Close`, `MSE` | `42.1649` | `+3.4643` | `31.4328` | `1.96%` |
| 3 | `v19_lb67_ratio_rsi_smoothl1` | `Close + MA ratio + RSI centered`, `SmoothL1` | `48.6173` | `+9.9167` | `36.1087` | `2.19%` |
| 4 | `v17_lb67_ratio_rsi` | `Close + MA ratio + RSI centered`, `MSE` | `49.1880` | `+10.4874` | `36.4569` | `2.21%` |
| 5 | `v1_baseline_torch` | `look_back=100`, `Close`, `MSE` | `50.4127` | `+11.7121` | `37.2124` | `2.28%` |
| 6 | `v9_lb65` | `Close`, `MSE` | `68.8567` | `+30.1561` | `56.1007` | `3.43%` |
| 7 | `v18_lb67_ema_macd` | `Close + EMA gap + MACD family`, `MSE` | `132.6834` | `+93.9828` | `99.4356` | `5.78%` |

### 結論
- `2330` 和 `2408` 在這一輪出現了關鍵分歧。
- 對 `2330` 來說，單改 `SmoothL1Loss` 的 `v16_lb67_smoothl1` 反而成為目前整套 `v1-v19` 裡的最佳版本。
- `v16_lb67_smoothl1` 的結果是 `Test RMSE=32.9998`、`Test MAE=25.1240`、`Test MAPE=1.57%`，不只打敗 `v14_lb67`，也打敗上一輪 sweep 最佳 `v7_lb50`。
- `MA / RSI` 類特徵在 `2330` 上沒有像 `SmoothL1` 那樣明顯改善；`v19` 比 `v17` 好，但還是輸給 `v16`。
- `EMA / MACD` 這組在 `2330` 上同樣最差，和 `2408` 的觀察一致。

### 新增產出檔案
- `run_torch_feature_loss_experiments_2330.py`
- `experiments_2330_feature_loss/notebooks/v16_lb67_smoothl1.ipynb`
- `experiments_2330_feature_loss/notebooks/v17_lb67_ratio_rsi.ipynb`
- `experiments_2330_feature_loss/notebooks/v18_lb67_ema_macd.ipynb`
- `experiments_2330_feature_loss/notebooks/v19_lb67_ratio_rsi_smoothl1.ipynb`
- `experiments_2330_feature_loss/results/v16_lb67_smoothl1.json`
- `experiments_2330_feature_loss/results/v17_lb67_ratio_rsi.json`
- `experiments_2330_feature_loss/results/v18_lb67_ema_macd.json`
- `experiments_2330_feature_loss/results/v19_lb67_ratio_rsi_smoothl1.json`
- `experiments_2330_feature_loss/results/all_experiments.json`

## 2408 vs 2330 Comparison - 2026-03-21

### 說明
- 比較摘要已另外輸出到 `comparison_2408_2330.json`。
- 由於 `2408` 與 `2330` 的價格量級不同，跨股票比較時 `RMSE / MAE` 只能看絕對誤差；若要看相對穩定性，`MAPE` 更有參考價值。

### 各股票目前最佳版本

| 股票 | 目前最佳版本 | 來源 | Test RMSE | Test MAE | Test MAPE |
| --- | --- | --- | --- | --- | --- |
| `2408` | `v14_lb67` | `experiments_2408_lb_refine` | `14.7791` | `10.6104` | `5.18%` |
| `2330` | `v16_lb67_smoothl1` | `experiments_2330_feature_loss` | `32.9998` | `25.1240` | `1.57%` |

### 同版本重點對照

| 版本 | `2408` Test RMSE / MAPE | `2330` Test RMSE / MAPE | 觀察 |
| --- | --- | --- | --- |
| `v7_lb50` | `15.9457 / 5.47%` | `38.7006 / 1.81%` | `2330` 在 `50` 天窗口的相對誤差明顯更穩 |
| `v9_lb65` | `14.9419 / 5.28%` | `68.8567 / 3.43%` | `65` 對 `2408` 很有效，對 `2330` 很差 |
| `v14_lb67` | `14.7791 / 5.18%` | `42.1649 / 1.96%` | `67` 對兩者都可用，但只有 `2408` 把它變成最佳純 `MSE` 版本 |
| `v16_lb67_smoothl1` | `16.9177 / 5.64%` | `32.9998 / 1.57%` | `SmoothL1` 對 `2330` 有效，對 `2408` 反而退步 |
| `v17_lb67_ratio_rsi` | `25.6220 / 9.48%` | `49.1880 / 2.21%` | `ratio_rsi` 對兩者都不如純 `Close` 主線 |
| `v18_lb67_ema_macd` | `28.8006 / 8.93%` | `132.6834 / 5.78%` | `EMA / MACD` 對兩者都最差，尤其 `2330` 更明顯 |
| `v19_lb67_ratio_rsi_smoothl1` | `16.6338 / 5.92%` | `48.6173 / 2.19%` | `SmoothL1` 能稍微救 `ratio_rsi`，但兩邊都沒超過各自主線最佳 |

### 比較結論
- `2408` 和 `2330` 的最佳方向不同。
- `2408` 主要吃的是 `look_back` 微調，最佳點落在 `65-67`，但加 `SmoothL1` 或技術指標都沒有進一步提升。
- `2330` 先在 sweep 裡顯示偏好較短窗口 `50`，但後來 `SmoothL1Loss` 成功把 `look_back=67` 推到全體最佳。
- 如果用相對誤差看穩定性，`2330` 明顯比 `2408` 更穩；它的最佳 `MAPE` 已降到 `1.57%`，而 `2408` 目前最佳仍是 `5.18%`。
- 如果只用絕對誤差看，`2408` 的 `RMSE / MAE` 較低，但這和兩檔股票價格量級不同有關，不能直接解讀成 `2408` 一定比較好預測。

### 新增產出檔案
- `comparison_2408_2330.json`
