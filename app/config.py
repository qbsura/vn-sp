"""
app/config.py
=============
Tập trung toàn bộ constants của dự án VNSP.
Không hardcode bất kỳ giá trị nào ở nơi khác — luôn import từ đây.
"""

# ── Experiment matrix ────────────────────────────────────────────────────────
TICKERS = ["VCB", "VIC"]          # Vietcombank, Vingroup
CURRENCIES = ["VND", "USD"]
WAVELET_CONDITIONS = [True, False] # True = dùng SWT, False = raw features
MODELS = ["DNN", "RNN", "GRU", "LSTM", "BiLSTM"]
TASKS = ["regression", "classification"]

# ── Date range ────────────────────────────────────────────────────────────────
DATE_START = "2012-01-01"
DATE_END   = "2024-12-31"

# ── Walk-Forward Validation folds ─────────────────────────────────────────────
FOLDS = [
    {
        "fold_id":    1,
        "train_end":  "2017-12-31",
        "test_start": "2018-01-01",
        "test_end":   "2018-12-31",
    },
    {
        "fold_id":    2,
        "train_end":  "2019-12-31",
        "test_start": "2020-01-01",
        "test_end":   "2020-12-31",
    },
    {
        "fold_id":    3,
        "train_end":  "2021-12-31",
        "test_start": "2022-01-01",
        "test_end":   "2024-12-31",
    },
]

# ── Wavelet config ────────────────────────────────────────────────────────────
# Dùng SWT (Stationary Wavelet Transform) — shift-invariant, giữ nguyên length
# Bài báo section 3.3.3 đề xuất SWT; db4 level-1 theo section 3.3.1
WAVELET_CONFIG = {
    "wavelet": "db4",         # Daubechies-4: orthogonal, 4 vanishing moments
    "level":   1,             # Level-1 để tránh over-smoothing
    "mode":    "periodization",  # pywt mode: tránh boundary artifacts, giữ length
}

# ── Feature scaling assignment ────────────────────────────────────────────────
# Theo bài báo section 3.2.2
SCALER_CONFIG = {
    # Standard scaler cho features phân phối gần chuẩn
    "standard": [
        "Open_Detail", "High_Detail", "Low_Detail",
        "Deviation_Approx", "Deviation_Detail",
        # No-wavelet case:
        "High", "Low", "Close", "Deviation",
    ],
    # Robust scaler cho features có outlier / phân phối lệch
    "robust": [
        "High_Approx", "Low_Approx",
        "Volume_Approx", "Volume_Detail",
        # No-wavelet case:
        "Volume",
    ],
    # Không scale (giữ trend information)
    "none": [
        "Open_Approx",   # wavelet case
        "Open",          # no-wavelet case
    ],
}

# Target column (không đưa vào input features)
TARGET_COL = "Close"

# ── Feature selection ─────────────────────────────────────────────────────────
# Loại features có |Pearson correlation| với nhau > threshold
CORRELATION_THRESHOLD = 0.95

# ── Sequence (sliding window) ─────────────────────────────────────────────────
# Giá trị mặc định; Optuna sẽ tìm giá trị tối ưu
DEFAULT_SEQUENCE_LENGTH = 20

# ── Hyperparameter search space (Optuna) ─────────────────────────────────────
HPO_SEARCH_SPACE = {
    "num_layers":      [1, 2, 3],
    "hidden_units":    [32, 64, 128, 256],
    "dropout_rate":    [0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate":   [1e-4, 5e-4, 1e-3],
    "batch_size":      [32, 64, 128],
    "sequence_length": [15, 20, 30, 50],
}
OPTUNA_TRIALS     = 30    # trials per fold
OPTUNA_HPO_EPOCHS = 50    # max epochs trong HPO (để nhanh)
MAX_TRAIN_EPOCHS  = 200   # max epochs khi train thật

# ── Training callbacks ────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE  = 20
LR_SCHEDULER_PATIENCE    = 10
LR_SCHEDULER_FACTOR      = 0.1   # reduce LR ×0.1 nếu val_loss không giảm
GRADIENT_CLIP_NORM       = 1.0   # bài báo section 4.5 recommend

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
# LUÔN CPU — TUYỆT ĐỐI không dùng "mps": bug BiLSTM bidirectional trên PyTorch MPS
# Xem: https://github.com/pytorch/pytorch/issues/94691
DEVICE = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    "raw":         "data/raw",
    "processed":   "data/processed",
    "experiments": "experiments",
    "figures":     "data/figures",
}

# ── vnstock data source ───────────────────────────────────────────────────────
# vnstock v4 (28-04-2026): TCBS bị loại bỏ.
# Source hợp lệ: vci, kbs, msn, dnse, binance, fmp, fmarket
# VCI: dữ liệu đầy đủ nhất cho HOSE, khuyến nghị cho local environment
# KBS: ổn định hơn trên Google Colab/Kaggle (không bị block)
# Lưu ý đơn vị VCI: giá trả về đơn vị nghìn VND → data_service.py tự ×1000
VNSTOCK_SOURCE   = "VCI"  # đổi thành "KBS" nếu VCI bị lỗi
VNSTOCK_INTERVAL = "1D"   # daily OHLCV