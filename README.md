# VNSP — Vietnam Stock Prediction System

> Tái hiện và mở rộng: **Li et al., *Engineering Applications of Artificial Intelligence*, 165 (2026) 113390**  
> BiLSTM + Stationary Wavelet Transform (SWT db4) · VCB & VIC · HOSE · 2012–2024

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt](#3-cài-đặt)
4. [Pipeline triển khai (5 bước)](#4-pipeline-triển-khai-5-bước)
5. [Chạy Backend & Frontend](#5-chạy-backend--frontend)
6. [API Reference nhanh](#6-api-reference-nhanh)
7. [Cấu hình quan trọng](#7-cấu-hình-quan-trọng)
8. [Troubleshooting](#8-troubleshooting)
9. [Tổng quan thực nghiệm](#9-tổng-quan-thực-nghiệm)

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu | Ghi chú |
|-----------|---------|---------|
| **OS** | macOS (Apple Silicon M1/M2/M3) | Linux/Windows cũng OK nhưng chưa kiểm thử |
| **RAM** | ≥ 16 GB | 32 GB khuyến nghị (training toàn bộ ~3h) |
| **Python** | **3.12.9** (pinned) | Dùng `uv` để quản lý — tự cài đúng version |
| **Package manager** | [`uv`](https://github.com/astral-sh/uv) | **Không dùng `pip` trực tiếp** |
| **GPU** | **Không cần / Không dùng** | CPU-only training (MPS bị disabled do bug PyTorch BiLSTM) |
| **Internet** | Cần khi tải dữ liệu lần đầu | Sau đó offline hoàn toàn |

> ⚠️ **QUAN TRỌNG — Không dùng MPS (Apple Metal):**  
> PyTorch có bug với `nn.LSTM(bidirectional=True)` trên MPS backend, gây kết quả sai và NaN loss.  
> Dự án buộc `DEVICE = "cpu"` trong `app/config.py`. **Không thay đổi setting này.**

---

## 2. Cấu trúc thư mục

```
vnsp/
├── app/                          # FastAPI application
│   ├── main.py                   # Entry point — server + routes
│   ├── config.py                 # Tất cả constants (KHÔNG hardcode nơi khác)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── data.py               # GET /api/data/*
│   │   ├── experiments.py        # GET/POST /api/experiments/*
│   │   ├── results.py            # GET /api/results/*
│   │   └── viz.py                # GET /api/viz/* (21 endpoints)
│   ├── models/
│   │   ├── base.py               # BaseModel + build_model factory
│   │   ├── dnn.py                # Dense Neural Network
│   │   ├── rnn.py                # Recurrent Neural Network
│   │   ├── gru.py                # Gated Recurrent Unit
│   │   ├── lstm.py               # Long Short-Term Memory
│   │   └── bilstm.py             # BiLSTM dual-branch (main model)
│   ├── services/
│   │   ├── data_service.py       # Load/save VCB VIC CSV
│   │   ├── preprocessing.py      # Deviation, FeatureScaler, correlation filter
│   │   ├── wavelet_service.py    # SWT db4 level-1 decomposition
│   │   ├── dataset_builder.py    # StockDataset, sliding window, weekly target
│   │   ├── training_service.py   # train_model, EarlyStopping, LR scheduler
│   │   ├── hpo_service.py        # Optuna TPE HPO (regression + classification)
│   │   ├── experiment_runner.py  # run_single_experiment, run_all_experiments
│   │   ├── evaluation_service.py # load_all_results, comparison tables
│   │   ├── trading_service.py    # simulate_trading_weekly, Sharpe/MDD
│   │   └── viz_service.py        # matplotlib figures (Fig. 1–11 + extended)
│   └── utils/
│       ├── seeds.py              # set_all_seeds(SEED=42)
│       └── metrics.py            # compute_regression/classification_metrics
│
├── data/
│   ├── raw/                      # [tạo tự động] VCB_raw.csv, VIC_raw.csv
│   └── processed/                # [tạo tự động] *.pkl (4 files)
│
├── experiments/                  # [tạo tự động] kết quả 240 experiments
│   └── {exp_id}/fold_{k}/
│       ├── metrics.json          # metrics + train_history
│       ├── predictions.npz       # y_pred, y_true, (y_prob), dates
│       └── best_params.json      # Optuna HPO best params
│
├── frontend/
│   ├── index.html                # Single Page App (4 sections)
│   ├── css/style.css             # Dark theme
│   └── js/
│       ├── api.js                # Fetch wrappers cho tất cả API endpoints
│       ├── charts.js             # Chart.js 4.4.2 + custom zoom/pan
│       └── main.js               # UI logic
│
├── scripts/                      # CLI scripts — chạy lần lượt theo thứ tự
│   ├── download_data.py          # Bước 1: tải VCB/VIC CSV từ vnstock
│   ├── preprocess.py             # Bước 2: wavelet + scaling → .pkl
│   ├── run_hpo.py                # Bước 3: Optuna HPO cho BiLSTM
│   └── run_experiments.py        # Bước 4: chạy 240 experiments
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
│
├── pyproject.toml                # uv project config + dependencies
├── README.md                     # File này
└── final.md                      # Báo cáo toàn diện dự án
```

---

## 3. Cài đặt

### Bước 3.1 — Cài `uv` (nếu chưa có)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Kiểm tra
uv --version
# uv 0.4.x hoặc mới hơn
```

### Bước 3.2 — Clone / mở project

```bash
# Di chuyển vào thư mục project
cd vnsp/

# Kiểm tra cấu trúc
ls -la
# Phải thấy: pyproject.toml, app/, scripts/, frontend/
```

### Bước 3.3 — Tạo virtual environment và cài dependencies

```bash
# uv tự tạo .venv và cài đúng Python 3.12.9
uv sync

# Cài thêm dev dependencies (pytest, jupyter)
uv sync --group dev
```

> `uv sync` đọc `pyproject.toml`, tự tải Python 3.12.9 nếu chưa có, tạo `.venv/`, cài đúng versions đã pin.

### Bước 3.4 — Kiểm tra cài đặt

```bash
# Kiểm tra Python version
uv run python --version
# Python 3.12.9

# Kiểm tra PyTorch (phải là CPU-only)
uv run python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
# 2.x.x+cpu
# MPS: False  ← bình thường, không cần MPS

# Kiểm tra pywt
uv run python -c "import pywt; print('db4 OK:', 'db4' in pywt.wavelist())"
# db4 OK: True
```

---

## 4. Pipeline triển khai (5 bước)

Chạy theo **đúng thứ tự** bên dưới. Mỗi bước tạo output cho bước tiếp theo.

```
[Bước 1] Download data
    ↓
[Bước 2] Preprocessing  (≈ 2 phút)
    ↓
[Bước 3] HPO — BiLSTM   (≈ 30–60 phút/combo × 4 combos ≈ 2–4 giờ)
    ↓
[Bước 4] Run experiments (≈ 3–4 giờ)
    ↓
[Bước 5] Start server   → truy cập Dashboard
```

---

### Bước 4.1 — Tải dữ liệu thô

```bash
# Tải VCB_raw.csv và VIC_raw.csv từ vnstock (source=VCI)
# Lưu vào data/raw/ — chỉ cần chạy 1 lần
uv run python scripts/download_data.py
```

**Output:**
```
data/raw/VCB_raw.csv   (~3,246 hàng, 2012-01-02 → 2024-12-31)
data/raw/VIC_raw.csv   (~3,246 hàng, 2012-01-02 → 2024-12-31)
```

> 💡 **Nếu VCI bị lỗi:** Mở `app/config.py`, đổi `VNSTOCK_SOURCE = "KBS"` rồi chạy lại.  
> Lưu ý: VCI trả về giá đơn vị **nghìn VND** (ví dụ 62.5 = 62,500 VND) — `data_service.py` tự nhân ×1000.

---

### Bước 4.2 — Preprocessing (Wavelet + Scaling)

```bash
# Xử lý 4 combinations: VCB/VIC × Wavelet/NoWavelet
# Tạo .pkl files trong data/processed/
uv run python scripts/preprocess.py
```

**Output:**
```
data/processed/VCB_VND_wavelet.pkl    (~3,246 hàng, ~8 features)
data/processed/VCB_VND_nowave.pkl     (~3,246 hàng, 5 features)
data/processed/VIC_VND_wavelet.pkl    (~3,246 hàng, ~8 features)
data/processed/VIC_VND_nowave.pkl     (~3,246 hàng, 5 features)
```

**Thời gian:** ~2 phút (SWT decomposition + correlation filter).

> ⚠️ **Cấu trúc PKL:** Mỗi file là dict `{"df": DataFrame, "feature_cols": list, "target_col": "Close"}`.  
> Phải unpack: `data = pickle.load(f); df = data["df"]` — KHÔNG dùng `data` trực tiếp như DataFrame.

---

### Bước 4.3 — Hyperparameter Optimization (Optuna)

HPO chạy **riêng cho từng combination**. Chạy song song trong 4 terminal để tiết kiệm thời gian.

```bash
# Terminal 1 — VCB Wavelet (Regression)
uv run python scripts/run_hpo.py --ticker VCB --wavelet true

# Terminal 2 — VCB No-Wavelet (Regression)
uv run python scripts/run_hpo.py --ticker VCB --wavelet false

# Terminal 3 — VIC Wavelet (Regression)
uv run python scripts/run_hpo.py --ticker VIC --wavelet true

# Terminal 4 — VIC No-Wavelet (Regression)
uv run python scripts/run_hpo.py --ticker VIC --wavelet false
```

**Nếu cần HPO riêng cho Classification** (cải thiện Task B):

```bash
uv run python scripts/run_hpo.py --ticker VCB --wavelet true  --task classification
uv run python scripts/run_hpo.py --ticker VCB --wavelet false --task classification
uv run python scripts/run_hpo.py --ticker VIC --wavelet true  --task classification
uv run python scripts/run_hpo.py --ticker VIC --wavelet false --task classification
```

**Output cho mỗi combination:**
```
experiments/VCB_VND_wavelet/fold_1/best_params.json
experiments/VCB_VND_wavelet/fold_2/best_params.json
experiments/VCB_VND_wavelet/fold_3/best_params.json
```

**Thời gian ước tính:** ~30–60 phút / combination × 3 folds × 30 trials = ~2–4 giờ/combination.

> 💡 **Resume HPO:** Nếu bị ngắt giữa chừng, chạy lại lệnh trên — HPO tự skip các fold đã có `best_params.json` hợp lệ (`best_val_loss != inf`).  
> Nếu muốn chạy lại từ đầu: thêm flag `--force-rerun`.

---

### Bước 4.4 — Chạy tất cả experiments

```bash
# Chạy đồng thời Regression và Classification
# 240 experiments = 2 tickers × 1 currency × 2 conditions × 5 models × 2 tasks × 3 folds
uv run python scripts/run_experiments.py

# Chỉ chạy một task cụ thể
uv run python scripts/run_experiments.py --task regression
uv run python scripts/run_experiments.py --task classification

# Force re-run tất cả (ghi đè kết quả cũ)
uv run python scripts/run_experiments.py --force-rerun
```

**Output:**
```
experiments/
└── VCB_VND_wavelet_BiLSTM_regression/
│   ├── fold_1/
│   │   ├── metrics.json        ← MSE, MAE, MAPE, RMSE, R²
│   │   ├── predictions.npz     ← y_pred, y_true, dates
│   │   └── best_params.json    ← Optuna HPO params
│   ├── fold_2/ ...
│   └── fold_3/ ...
├── VCB_VND_wavelet_BiLSTM_classification/ ...
├── VCB_VND_nowave_DNN_regression/ ...
└── ... (240 experiment directories)
```

**Thời gian ước tính:** ~3–4 giờ trên CPU (M1 Pro 32GB đã test: 3h 9m, 0 errors).

> 💡 **Resume experiments:** Script tự **skip** nếu `metrics.json` đã tồn tại.  
> Log tiến độ được in ra console theo format: `[x/240] VCB_VND_wavelet_BiLSTM_regression | fold 1 → ✅`.

---

### Bước 4.5 — Khởi động server và truy cập Dashboard

```bash
# Development mode (hot-reload, verbose logs)
uv run fastapi dev app/main.py

# Production mode (ổn định hơn, ít logs)
uv run fastapi run app/main.py
```

Sau khi server khởi động:

| URL | Nội dung |
|-----|---------|
| `http://127.0.0.1:8000` | **Dashboard chính** (SPA với 4 sections) |
| `http://127.0.0.1:8000/docs` | **Swagger UI** — thử nghiệm API trực tiếp |
| `http://127.0.0.1:8000/redoc` | ReDoc — API documentation |
| `http://127.0.0.1:8000/health` | Health check: `{"status": "ok"}` |

---

## 5. Chạy Backend & Frontend

### Cấu trúc URL

```
http://127.0.0.1:8000/
├── /                        ← Frontend: frontend/index.html
├── /static/*                ← Static files: CSS, JS từ frontend/
├── /health                  ← Health check
├── /docs                    ← Swagger UI
│
├── /api/data/
│   ├── /raw?ticker=VCB                        ← OHLCV table
│   └── /processed?ticker=VCB&wavelet=true     ← Processed features
│
├── /api/experiments/
│   ├── /list                                  ← Danh sách experiments
│   ├── /{exp_id}                              ← Kết quả 3 folds
│   └── /{exp_id}/params?fold_idx=1            ← HPO best params
│
├── /api/results/
│   ├── /comparison-table?ticker=VCB&currency=VND&wavelet=true&fold=3
│   ├── /predicted-vs-actual?exp_id=...&fold=3
│   ├── /loss-curves?exp_id=...&fold=3
│   ├── /classification-metrics?ticker=VCB&currency=VND&wavelet=true&fold=3
│   ├── /roc-curves?ticker=VCB&currency=VND&wavelet=true&fold=3
│   ├── /confusion-matrix?exp_id=...&fold=3
│   ├── /trading-metrics?ticker=VCB&currency=VND&wavelet=true&fold=3
│   └── /trading-timeseries?exp_id=...&fold=3
│
└── /api/viz/
    ├── /fig1                                  ← Pipeline framework diagram
    ├── /fig2?ticker=VCB&currency=VND          ← Deviation scatter plot
    ├── /fig3?ticker=VCB&currency=VND          ← Feature distributions
    ├── /fig4                                  ← Scaling flowchart
    ├── /fig5?ticker=VCB&currency=VND          ← Wavelet decomposition
    ├── /fig6                                  ← db4 wavelet functions
    ├── /fig7?ticker=VIC&currency=VND          ← Approx coefficients (VIC)
    ├── /fig8?ticker=VCB&currency=VND          ← Detail coefficients (VCB)
    ├── /fig9                                  ← Level-1 decomposition diagram
    ├── /fig10?ticker=VCB&currency=VND&wavelet=true ← Correlation matrix
    └── /fig11?ticker=VCB&currency=VND         ← MSE comparison bar chart
```

### Dashboard — 4 sections

| Section | Truy cập | Nội dung chính |
|---------|---------|----------------|
| 📊 **Data** | Sidebar → Data | Raw OHLCV, Wavelet decomposition, Correlation matrix |
| 📈 **Regression** | Sidebar → Regression | Predicted vs Actual, Loss Curves, MSE comparison table |
| 🎯 **Classification** | Sidebar → Classification | Confusion Matrix, ROC Curves, Accuracy/F1/AUC table |
| 💹 **Trading** | Sidebar → Trading | Cumulative Returns vs Buy&Hold, Sharpe/MDD/Win Rate table |

### Controls Dashboard

```
Ticker:  [VCB ▼]   [VIC ▼]
Fold:    [Fold 1 ▼] [Fold 2 ▼] [Fold 3 ▼]
Wavelet: [Wavelet]  [No Wavelet]   ← toggle buttons
         [Load Results]             ← trigger load
```

---

## 6. API Reference nhanh

### Lấy metrics regression (JSON)

```bash
# VCB, VND, Wavelet, Fold 3 — tất cả 5 models
curl "http://127.0.0.1:8000/api/results/comparison-table?ticker=VCB&currency=VND&wavelet=true&fold=3"
```

Response:
```json
{
  "ticker": "VCB", "currency": "VND", "wavelet": true, "fold": 3,
  "models": [
    {"model": "BiLSTM", "MSE": 1332338.58, "MAE": 801.37, "MAPE": 2.28, "RMSE": 1154.27, "R2": 0.9189},
    ...
  ]
}
```

### Lấy figure dưới dạng base64 PNG

```bash
# Fig. 10 — Correlation matrix heatmap
curl "http://127.0.0.1:8000/api/viz/fig10?ticker=VCB&currency=VND&wavelet=true"
```

Response:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

### Kiểm tra trạng thái một experiment

```bash
# Xem 3 folds của VCB wavelet BiLSTM regression
curl "http://127.0.0.1:8000/api/experiments/VCB_VND_wavelet_BiLSTM_regression"
```

---

## 7. Cấu hình quan trọng

### `app/config.py` — File constants tập trung

```python
# ── Thay đổi nguồn dữ liệu nếu VCI bị lỗi ────────────────────────────────
VNSTOCK_SOURCE = "VCI"   # Đổi thành "KBS" nếu VCI lỗi trên Colab/Kaggle

# ── Thiết bị training ─────────────────────────────────────────────────────
DEVICE = "cpu"           # KHÔNG đổi thành "mps" hoặc "cuda" cho BiLSTM

# ── Random seed (reproducibility) ─────────────────────────────────────────
SEED = 42                # Thay đổi nếu muốn ensemble từ nhiều seeds

# ── Optuna HPO ────────────────────────────────────────────────────────────
OPTUNA_TRIALS     = 30   # Tăng lên 50-100 để tìm params tốt hơn
OPTUNA_HPO_EPOCHS = 50   # Số epoch tối đa trong mỗi HPO trial
MAX_TRAIN_EPOCHS  = 200  # Số epoch tối đa khi train thật

# ── Early stopping ─────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 20   # Dừng sau 20 epoch val_loss không cải thiện
```

### Experiment ID format

```
{ticker}_{currency}_{condition}_{model}_{task}

Ví dụ:
  VCB_VND_wavelet_BiLSTM_regression
  VCB_VND_nowave_GRU_classification
  VIC_VND_wavelet_LSTM_regression
```

### Cấu trúc file PKL (quan trọng!)

```python
import pickle

# ✅ ĐÚNG — luôn unpack ["df"]
with open("data/processed/VCB_VND_wavelet.pkl", "rb") as f:
    data = pickle.load(f)
df           = data["df"]           # DataFrame, index=DatetimeIndex
feature_cols = data["feature_cols"] # list[str]
target_col   = data["target_col"]   # "Close"

# ❌ SAI — dict không phải DataFrame
df = pickle.load(f)  # AttributeError: 'dict' object has no attribute 'iloc'
```

---

## 8. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'app'`

```bash
# Nguyên nhân: chạy script không từ project root
# Giải pháp: luôn chạy từ thư mục gốc vnsp/
cd vnsp/
uv run python scripts/download_data.py   # ✅
```

### ❌ `FileNotFoundError: best_params.json không tồn tại`

```bash
# Nguyên nhân: chưa chạy HPO cho combination này
# Giải pháp:
uv run python scripts/run_hpo.py --ticker VCB --wavelet true

# Kiểm tra xem file đã tồn tại chưa:
ls experiments/VCB_VND_wavelet/fold_*/best_params.json
```

### ❌ `vnstock API error` / Không tải được dữ liệu

```bash
# Thử đổi source trong app/config.py:
VNSTOCK_SOURCE = "KBS"   # thay vì "VCI"

# Hoặc kiểm tra kết nối:
uv run python -c "
from vnstock import Vnstock
s = Vnstock().stock('VCB', source='KBS')
print(s.quote.history(start='2024-01-01', end='2024-01-10', interval='1D').head())
"
```

### ❌ `ValueError: odd-length input to SWT`

```bash
# Nguyên nhân: SWT yêu cầu input có độ dài chẵn
# Giải pháp: đã xử lý tự động trong wavelet_service.py bằng np.pad(mode='wrap')
# Nếu vẫn lỗi, kiểm tra version PyWavelets:
uv run python -c "import pywt; print(pywt.__version__)"
# Phải là 1.6.0+
```

### ❌ `best_val_loss = inf` trong best_params.json

```bash
# Nguyên nhân: HPO run cũ bị lỗi, tạo file với inf
# Giải pháp: xóa file cũ và chạy lại với --force-rerun
uv run python scripts/run_hpo.py --ticker VCB --wavelet true --force-rerun
```

### ❌ `RuntimeError: Expected all tensors to be on the same device`

```bash
# Nguyên nhân: model hoặc tensor vô tình lên device khác
# Giải pháp: kiểm tra app/config.py
grep "DEVICE" app/config.py
# DEVICE = "cpu"   ← phải là "cpu"
```

### ❌ Frontend trống / API không response

```bash
# Kiểm tra server đang chạy:
curl http://127.0.0.1:8000/health
# {"status": "ok", "service": "VNSP API", "version": "1.0.0"}

# Kiểm tra frontend directory tồn tại:
ls frontend/
# index.html  css/  js/

# Xem server logs trong terminal — tìm dòng:
# "Static files mounted: /static → .../frontend"
```

### ❌ `BatchNorm` lỗi khi batch size = 1

```bash
# Nguyên nhân: DataLoader có drop_last=False và tập cuối chỉ có 1 sample
# Giải pháp: đã fix trong training_service.py với drop_last=True
# Kiểm tra training_service.py:
grep "drop_last" app/services/training_service.py
# drop_last=True   ← phải có dòng này
```

---

## 9. Tổng quan thực nghiệm

### Ma trận experiments

```
Tổng = Tickers × Currencies × Conditions × Models × Tasks × Folds
     = 2 × 1 × 2 × 5 × 2 × 3 = 240 experiments

Tickers:    VCB, VIC
Currencies: VND (USD đã loại bỏ theo yêu cầu GV)
Conditions: wavelet (SWT db4 lv-1), nowave (raw features)
Models:     DNN, RNN, GRU, LSTM, BiLSTM
Tasks:      regression (giá Close t+1), classification (hướng tuần kế tiếp)
Folds:      3 (Walk-Forward Validation, expanding window)
```

### Walk-Forward Validation folds

| Fold | Train | Test | Đặc điểm thị trường |
|------|-------|------|---------------------|
| Fold 1 | 2012–2017 | 2018 | Tăng trưởng bình thường |
| Fold 2 | 2012–2019 | 2020 | COVID-19 — biến động cực mạnh |
| Fold 3 | 2012–2021 | 2022–2024 | Phục hồi + thắt chặt tiền tệ |

### HPO search space

| Hyperparameter | Giá trị thử | Tìm cho |
|----------------|-------------|---------|
| `num_layers` | 1, 2, 3 | BiLSTM (dùng chung cho các model baseline) |
| `hidden_units` | 32, 64, 128, 256 | BiLSTM |
| `dropout_rate` | 0.1, 0.2, 0.3, 0.4, 0.5 | BiLSTM |
| `learning_rate` | 1e-4, 5e-4, 1e-3 | BiLSTM |
| `batch_size` | 32, 64, 128 | BiLSTM |
| `sequence_length` | 15, 20, 30, 50 | BiLSTM |

> DNN, RNN, GRU, LSTM dùng chung `best_params` của BiLSTM trong cùng fold → đảm bảo so sánh công bằng.

### Thời gian chạy tham khảo (M1 Pro 32GB, CPU-only)

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| Download data | ~2 phút | Tùy tốc độ mạng |
| Preprocessing (4 combos) | ~2 phút | SWT + correlation filter |
| HPO regression (4 combos) | ~2–4 giờ | 30 trials × 3 folds × 4 combos |
| Run experiments (240) | ~3–4 giờ | 0 errors trên test run |
| **Tổng** | **~6–9 giờ** | Có thể song song HPO |

---

## Thông tin bổ sung

### Tài liệu tham khảo chính

- **Li et al. (2026):** https://doi.org/10.1016/j.engappai.2025.113390
- **Optuna docs:** https://optuna.readthedocs.io
- **PyTorch docs:** https://pytorch.org/docs
- **PyWavelets docs:** https://pywavelets.readthedocs.io
- **FastAPI docs:** https://fastapi.tiangolo.com
- **vnstock GitHub:** https://github.com/thinh-vu/vnstock

### Liên hệ & báo lỗi

Xem `final.md` để biết đầy đủ về kiến trúc, thuật toán, và kết quả thực nghiệm.

---

*VNSP v1.0.0 · 2026-06-15 · Python 3.12.9 · PyTorch CPU-only · Optuna TPE · SWT db4 lv-1*
