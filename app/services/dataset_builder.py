"""
app/services/dataset_builder.py
=================================
Sequence building và PyTorch Dataset class cho VNSP pipeline.

Task 2.5 trong Phase 2 — Preprocessing.

Pipeline position (sau wavelet/scaling/feature-selection):
  preprocessed df  →  build_sequences()  →  StockDataset  →  DataLoader

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.1: Sliding window sequences cho time-series modeling.

Hai tasks — KHÁC sequence builder, khác target:
  Task A (Regression)      : build_sequences()        — daily, y = Close(t+1) scaled value
  Task B (Classification)  : build_weekly_sequences()  — weekly (T2→T6, Phương án D)
                             y_W = 1 nếu Close(F_{W+1}) > Close(F_W), else 0
                             F_W = phiên giao dịch cuối cùng của tuần W (thường là T6).
                             1 sample / tuần (không phải 1 sample / ngày).
                             → dựa trên GIÁ GỐC (unscaled) để tránh bias từ scaling.

  Lý do thay đổi (feedback giảng viên, 2026-06): Task B chuyển từ dự đoán
  hướng đi ngày kế tiếp (t+1) sang dự đoán xu thế TUẦN kế tiếp, neo theo
  lịch tuần Thứ 2 → Thứ 6 (không phải rolling 5-phiên). Task A không đổi.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.config import DEFAULT_SEQUENCE_LENGTH, TARGET_COL
from app.services.preprocessing import FeatureScaler

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# =============================================================================
# SEQUENCE BUILDER
# =============================================================================

def build_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    target_col: str = TARGET_COL,
    task: str = "regression",
    original_close: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Xây dựng sliding-window sequences từ DataFrame đã scale.

    Cơ chế:
      X[i] = features[i : i+seq_len]   → shape (seq_len, n_features)
      y[i] = target[i+seq_len]          → regression: scaled Close(t+1)
           = direction[i+seq_len]       → classification: 0/1

    Lưu ý quan trọng về input features:
      - X KHÔNG bao gồm target_col ('Close') — chỉ input features.
      - Columns của df: [feature_1, ..., feature_k, Close]
        → X dùng feature_1..k, y dùng Close.

    Lưu ý (2026-06): prepare_fold_data() chỉ gọi build_sequences() cho
    task='regression' (Task A, daily t+1, không đổi). Task B (classification)
    nay dùng build_weekly_sequences() — xem hàm đó để biết target mới
    (weekly T2→T6). Nhánh task='classification' ở đây được giữ lại để
    tương thích/tham khảo, không còn nằm trong pipeline chính.

    Classification target (Task B) — LEGACY, daily t+1:
      y[i] = 1 nếu Close[i+seq_len] > Close[i+seq_len-1], else 0
      - Dùng original_close (unscaled) để tính direction, tránh distortion.
      - Nếu original_close=None: dùng scaled Close làm fallback
        (ít chính xác hơn nhưng direction thường vẫn đúng).
      - Close[i+seq_len-1] là ngày CUỐI trong window (ngày t hiện tại).
      - Close[i+seq_len] là ngày CẦN DỰ ĐOÁN (ngày t+1).

    Args:
        df:              Scaled DataFrame, index=DatetimeIndex.
                         Phải có target_col và ít nhất 1 feature column khác.
        sequence_length: Số timesteps trong mỗi window (≥ 1).
        target_col:      Tên cột target (default: 'Close').
        task:            'regression' hoặc 'classification'.
        original_close:  Array 1-D với giá Close GỐC (unscaled, đơn vị VND/USD).
                         Cần thiết cho classification để tính direction đúng.
                         Phải có len == len(df). Nếu None → dùng scaled Close.

    Returns:
        (X, y)
        - X: np.ndarray shape (N, sequence_length, n_features), dtype float32
             N = len(df) - sequence_length
        - y: np.ndarray shape (N,), dtype float32
             regression:     scaled Close values (cần inverse_transform để ra VND/USD)
             classification: 0.0 (DOWN) hoặc 1.0 (UP)

    Raises:
        ValueError: Nếu df quá ngắn, không có feature columns, hoặc task không hợp lệ.
        ValueError: Nếu original_close (nếu truyền) có length khác len(df).
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    n = len(df)
    if sequence_length < 1:
        raise ValueError(f"sequence_length={sequence_length} phải >= 1.")
    if n <= sequence_length:
        raise ValueError(
            f"build_sequences: len(df)={n} <= sequence_length={sequence_length}. "
            "Cần ít nhất seq_len+1 rows để tạo ít nhất 1 sequence."
        )
    if task not in ("regression", "classification"):
        raise ValueError(f"task='{task}' không hợp lệ. Dùng 'regression' hoặc 'classification'.")
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' không có trong df.columns={list(df.columns)}")

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError(
            f"build_sequences: không có feature columns nào "
            f"(chỉ có target_col='{target_col}')."
        )

    if original_close is not None:
        original_close = np.asarray(original_close, dtype=np.float64)
        if len(original_close) != n:
            raise ValueError(
                f"len(original_close)={len(original_close)} != len(df)={n}. "
                "original_close phải có cùng length với df."
            )

    # ── Data extraction ───────────────────────────────────────────────────────
    n_features = len(feature_cols)
    n_sequences = n - sequence_length

    features = df[feature_cols].values.astype(np.float32)   # (n, n_features)
    scaled_close = df[target_col].values.astype(np.float64) # (n,)

    # Giá Close để tính direction: ưu tiên giá gốc, fallback về scaled
    close_for_direction = (
        original_close if original_close is not None else scaled_close
    )

    # ── Pre-allocate ──────────────────────────────────────────────────────────
    X = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)
    y = np.empty(n_sequences, dtype=np.float32)

    # ── Build sequences (sliding window) ──────────────────────────────────────
    for i in range(n_sequences):
        # Input: window [i, i+seq_len)
        X[i] = features[i : i + sequence_length]

        # Target: ngày ngay sau window (= t+1)
        t_next = i + sequence_length      # index của ngày cần dự đoán
        t_curr = i + sequence_length - 1  # index của ngày cuối trong window (= t)

        if task == "regression":
            # Scaled Close tại t+1 (inverse_transform sau khi model predict)
            y[i] = float(scaled_close[t_next])
        else:
            # Direction: UP=1 nếu Close(t+1) > Close(t), DOWN=0 ngược lại
            y[i] = float(close_for_direction[t_next] > close_for_direction[t_curr])

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info(
        f"[build_sequences] task={task} | seq_len={sequence_length} | "
        f"n_features={n_features} | N={n_sequences} | "
        f"X={X.shape} | y={y.shape}"
    )

    if task == "classification":
        n_up = int(y.sum())
        pct_up = 100.0 * n_up / n_sequences
        logger.info(
            f"[build_sequences] Direction balance: "
            f"UP={n_up}/{n_sequences} ({pct_up:.1f}%) | "
            f"DOWN={n_sequences-n_up}/{n_sequences} ({100-pct_up:.1f}%)"
        )

    return X, y


# =============================================================================
# WEEKLY SEQUENCE BUILDER (Phương án D — Task B mới, thay thế daily t+1)
# =============================================================================

def build_weekly_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    target_col: str = TARGET_COL,
    original_close: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Xây dựng sequences cho Task B — Weekly Direction Classification (T2→T6).

    Bối cảnh (feedback giảng viên, 2026-06):
      Thay vì dự đoán hướng đi của NGÀY kế tiếp (t+1), Task B mới dự đoán
      hướng đi của TUẦN kế tiếp (T2→T6 theo lịch dương), 1 sample / tuần.

    Định nghĩa "tuần" (F_W):
      - Mỗi tuần W = nhóm các phiên giao dịch trong cùng 1 tuần ISO (T2-T6),
        dùng pandas Period 'W-FRI' (tuần kết thúc vào Thứ 6) để group —
        tự động xử lý các tuần thiếu phiên do nghỉ lễ (F_W = phiên cuối
        cùng có dữ liệu trong tuần, không nhất thiết là Thứ 6).
      - F_W = vị trí (index) của phiên giao dịch CUỐI CÙNG trong tuần W.

    Sequence & target:
      X_W = features[F_W - seq_len + 1 : F_W + 1]   → N ngày daily, kết thúc tại F_W
      y_W = 1 nếu Close(F_{W+1}) > Close(F_W), else 0

      - Close(F_W)      : đã biết tại thời điểm dự đoán (mốc tham chiếu).
      - Close(F_{W+1})  : tương lai — đây là mục tiêu cần dự đoán.
      - → KHÔNG leakage: input X_W và mốc tham chiếu Close(F_W) chỉ dùng
        dữ liệu ≤ F_W; nhãn y_W hoàn toàn nằm ở tương lai so với F_W.
      - Dùng original_close (giá GỐC, unscaled) để tính direction, giống
        quy ước của build_sequences() — tránh distortion do scaling.

    Tần suất: 1 sample / tuần (không phải 1 sample / ngày). Tuần cuối cùng
    trong df bị bỏ qua (không có F_{W+1} để tính nhãn). Các tuần đầu không
    đủ `sequence_length` ngày lịch sử trước F_W cũng bị bỏ qua.

    Args:
        df:              Scaled DataFrame, index=DatetimeIndex (daily).
                         Phải có target_col và ít nhất 1 feature column khác.
        sequence_length: Số ngày daily trong mỗi window (≥ 1).
        target_col:      Tên cột target (default: 'Close').
        original_close:  Array 1-D giá Close GỐC (unscaled), len == len(df).
                         Nếu None → dùng scaled Close làm fallback.

    Returns:
        (X, y, dates)
        - X: np.ndarray shape (N, sequence_length, n_features), dtype float32
        - y: np.ndarray shape (N,), dtype float32 — 0.0 (DOWN) / 1.0 (UP)
        - dates: pd.DatetimeIndex shape (N,) — ngày F_W (cuối tuần W,
          thời điểm dự đoán được tạo ra cho tuần W+1)

    Raises:
        ValueError: Nếu df quá ngắn, không có feature columns, original_close
                     sai length, ít hơn 2 tuần, hoặc không có sample hợp lệ
                     nào (sequence_length quá lớn so với dữ liệu).
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    n = len(df)
    if sequence_length < 1:
        raise ValueError(f"sequence_length={sequence_length} phải >= 1.")
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' không có trong df.columns={list(df.columns)}")

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError(
            f"build_weekly_sequences: không có feature columns nào "
            f"(chỉ có target_col='{target_col}')."
        )

    if original_close is not None:
        original_close = np.asarray(original_close, dtype=np.float64)
        if len(original_close) != n:
            raise ValueError(
                f"len(original_close)={len(original_close)} != len(df)={n}. "
                "original_close phải có cùng length với df."
            )

    # ── Data extraction ───────────────────────────────────────────────────────
    features = df[feature_cols].values.astype(np.float32)    # (n, n_features)
    scaled_close = df[target_col].values.astype(np.float64)  # (n,)

    close_for_direction = (
        original_close if original_close is not None else scaled_close
    )

    # ── Xác định F_W: phiên giao dịch CUỐI CÙNG của mỗi tuần (T2-T6) ───────────
    # 'W-FRI' = tuần kết thúc vào Thứ 6 → group đúng các phiên T2..T6 vào
    # cùng 1 tuần (dữ liệu không có T7/CN nên không bị lệch anchor).
    week_periods = df.index.to_period("W-FRI")
    last_pos_per_week = (
        pd.Series(np.arange(n), index=week_periods)
        .groupby(level=0)
        .max()
        .sort_index()
    )
    f_positions = last_pos_per_week.to_numpy()
    n_weeks = len(f_positions)

    if n_weeks < 2:
        raise ValueError(
            f"build_weekly_sequences: chỉ có {n_weeks} tuần trong df, "
            "cần ít nhất 2 tuần (tuần W và tuần W+1) để tạo 1 sample."
        )

    # ── Build sequences (1 sample / tuần) ──────────────────────────────────────
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    date_list: list = []

    for w in range(n_weeks - 1):  # tuần cuối không có W+1 → bỏ
        f_w = int(f_positions[w])
        f_w_next = int(f_positions[w + 1])

        start = f_w - sequence_length + 1
        if start < 0:
            # Không đủ sequence_length ngày lịch sử trước F_W → bỏ qua tuần này
            continue

        X_list.append(features[start : f_w + 1])  # N ngày, kết thúc tại F_W
        y_list.append(
            float(close_for_direction[f_w_next] > close_for_direction[f_w])
        )
        date_list.append(df.index[f_w])

    if not X_list:
        raise ValueError(
            f"build_weekly_sequences: không có sample hợp lệ nào "
            f"(n_weeks={n_weeks}, sequence_length={sequence_length}). "
            "df quá ngắn so với sequence_length."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates = pd.DatetimeIndex(date_list)

    # ── Logging ───────────────────────────────────────────────────────────────
    n_samples = len(y)
    logger.info(
        f"[build_weekly_sequences] seq_len={sequence_length} | "
        f"n_features={features.shape[1]} | n_weeks_total={n_weeks} | "
        f"N_samples={n_samples} | X={X.shape} | y={y.shape}"
    )

    n_up = int(y.sum())
    pct_up = 100.0 * n_up / n_samples
    logger.info(
        f"[build_weekly_sequences] Weekly direction balance: "
        f"UP={n_up}/{n_samples} ({pct_up:.1f}%) | "
        f"DOWN={n_samples - n_up}/{n_samples} ({100 - pct_up:.1f}%)"
    )

    return X, y, dates


# =============================================================================
# STOCK DATASET
# =============================================================================

class StockDataset(Dataset):
    """
    PyTorch Dataset cho stock price sequences.

    Wrap (X, y) arrays từ build_sequences() thành Dataset tương thích
    với torch.utils.data.DataLoader.

    Regression  : y shape = (N, 1)  → BCELoss không dùng, chỉ dùng MSELoss
    Classification: y shape = (N,)   → BCELoss / BCEWithLogitsLoss

    Args:
        X:    Input sequences, float32 array shape (N, seq_len, n_features).
        y:    Target array, float32 shape (N,).
        task: 'regression' (y được unsqueeze → (N,1)) hoặc 'classification' (y flat).

    Properties:
        n_features (int): Số input features.
        seq_len (int):    Window size.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = "regression") -> None:
        if task not in ("regression", "classification"):
            raise ValueError(f"task='{task}' không hợp lệ.")

        self.task = task
        self.X = torch.FloatTensor(X)   # (N, seq_len, n_features)

        if task == "regression":
            # Unsqueeze để khớp với output shape của model (N, 1)
            self.y = torch.FloatTensor(y).unsqueeze(-1)  # (N, 1)
        else:
            # Classification: flat (N,) cho BCELoss
            self.y = torch.FloatTensor(y)   # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        """Số input features (chiều cuối của X)."""
        return self.X.shape[-1]

    @property
    def seq_len(self) -> int:
        """Độ dài sequence window."""
        return self.X.shape[1]

    def __repr__(self) -> str:
        return (
            f"StockDataset(task={self.task}, "
            f"N={len(self)}, "
            f"seq_len={self.seq_len}, "
            f"n_features={self.n_features})"
        )


# =============================================================================
# FOLD DATA PREPARATION
# =============================================================================

def prepare_fold_data(
    df: pd.DataFrame,
    fold: dict,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    task: str = "regression",
    scaler: Optional[FeatureScaler] = None,
    target_col: str = TARGET_COL,
) -> dict:
    """
    Chuẩn bị train/test Dataset cho một Walk-Forward fold.

    Pipeline đầy đủ (không data leakage):
      1. Split df thành train/test theo date range trong fold.
      2. Lưu giá Close GỐC trước khi scale (dùng cho classification direction).
      3. Fit FeatureScaler trên TRAIN ONLY → transform cả train và test.
      4. Build sequences cho train và test riêng biệt.
      5. Trả về StockDataset + metadata.

    Walk-Forward Validation folds (từ config.py):
      Fold 1: Train[2012→2017] | Test[2018]
      Fold 2: Train[2012→2019] | Test[2020]
      Fold 3: Train[2012→2021] | Test[2022→2024]

    Tái sử dụng scaler:
      Nếu `scaler` được truyền vào (đã fit), skip bước fit và chỉ transform.
      Hữu ích khi dùng cùng scaler cho Task A và Task B của cùng một fold
      (đảm bảo features được scale nhất quán).

    Args:
        df:              Full preprocessed DataFrame, index=DatetimeIndex.
                         Chưa scale — scaling được thực hiện trong hàm này.
                         Columns: features + target_col (sau wavelet hoặc raw OHLCV).
        fold:            Dict định nghĩa fold, cần có keys:
                           'fold_id'    : int
                           'train_end'  : str (YYYY-MM-DD), ngày cuối train
                           'test_start' : str (YYYY-MM-DD), ngày đầu test
                           'test_end'   : str (YYYY-MM-DD), ngày cuối test
        sequence_length: Window size (timesteps). Default từ config.
        task:            'regression' hoặc 'classification'.
        scaler:          Pre-fitted FeatureScaler (optional).
                         - None → tạo và fit FeatureScaler mới trên train data.
                         - Provided → dùng ngay, skip fit (chỉ transform).
        target_col:      Tên cột target, default 'Close'.

    Returns:
        dict với các keys:
          "train_dataset" : StockDataset — sequences cho training
          "test_dataset"  : StockDataset — sequences cho evaluation
          "scaler"        : FeatureScaler đã fit (dùng cho inverse_transform_target)
          "train_dates"   : DatetimeIndex — ngày ứng với mỗi y_train
                            - Task A (regression): ngày được predict (t+1)
                            - Task B (classification): ngày F_W — phiên cuối
                              tuần W (thời điểm dự đoán được tạo ra cho tuần W+1)
          "test_dates"    : DatetimeIndex — tương tự train_dates, cho test set
          "n_features"    : int — số input features
          "fold_id"       : int — fold ID (từ fold dict)

    Raises:
        KeyError:  Nếu thiếu keys trong fold dict.
        ValueError: Nếu train hoặc test split rỗng / quá ngắn cho sequences.
    """
    # ── Parse fold definition ─────────────────────────────────────────────────
    fold_id    = fold["fold_id"]
    train_end  = pd.Timestamp(fold["train_end"])
    test_start = pd.Timestamp(fold["test_start"])
    test_end   = pd.Timestamp(fold["test_end"])

    # ── Date split ────────────────────────────────────────────────────────────
    df_train = df[df.index <= train_end].copy()
    df_test  = df[(df.index >= test_start) & (df.index <= test_end)].copy()

    if df_train.empty:
        raise ValueError(
            f"[prepare_fold_data] Fold {fold_id}: train split rỗng. "
            f"Kiểm tra fold['train_end']={fold['train_end']} và date range của df."
        )
    if df_test.empty:
        raise ValueError(
            f"[prepare_fold_data] Fold {fold_id}: test split rỗng. "
            f"Kiểm tra fold['test_start/end'] và date range của df."
        )

    logger.info(
        f"[Fold {fold_id}] task={task} | seq_len={sequence_length} | "
        f"Train: {df_train.index[0].date()} → {df_train.index[-1].date()} "
        f"({len(df_train):,} rows) | "
        f"Test:  {df_test.index[0].date()} → {df_test.index[-1].date()} "
        f"({len(df_test):,} rows)"
    )

    # ── Lưu Close GỐC (trước scaling) để tính direction cho classification ────
    # Task B: y = 1 nếu Close(t+1) > Close(t) — phải dùng giá thực, không phải scaled
    orig_close_train = df_train[target_col].values.copy()
    orig_close_test  = df_test[target_col].values.copy()

    # ── Scaling: fit CHỈ trên train, transform cả train và test ───────────────
    if scaler is None:
        scaler = FeatureScaler()
        df_train_scaled = scaler.fit_transform(df_train)
        logger.info(f"[Fold {fold_id}] FeatureScaler fitted on train data.")
    else:
        # Dùng scaler đã fit bên ngoài (tránh re-fit, đảm bảo consistency)
        df_train_scaled = scaler.transform(df_train)
        logger.info(f"[Fold {fold_id}] Using pre-fitted FeatureScaler.")

    df_test_scaled = scaler.transform(df_test)

    # ── Build sequences ───────────────────────────────────────────────────────
    if task == "classification":
        # Phương án D — Task B mới: Weekly Direction Classification (T2→T6).
        #   X_W = N ngày daily kết thúc tại F_W (phiên cuối tuần W)
        #   y_W = 1 nếu Close(F_{W+1}) > Close(F_W), dùng giá GỐC (unscaled)
        #   1 sample / tuần — KHÔNG phải 1 sample / ngày như Task A.
        #   dates trả về = F_W (ngày dự đoán được tạo ra, cho tuần W+1).
        X_train, y_train, train_dates = build_weekly_sequences(
            df_train_scaled, sequence_length, target_col, orig_close_train
        )
        X_test, y_test, test_dates = build_weekly_sequences(
            df_test_scaled, sequence_length, target_col, orig_close_test
        )
    else:
        # Task A (Regression): daily t+1 — không đổi.
        X_train, y_train = build_sequences(
            df_train_scaled, sequence_length, target_col, task, None
        )
        X_test, y_test = build_sequences(
            df_test_scaled, sequence_length, target_col, task, None
        )

        # Sequence i dự đoán ngày df.index[i + sequence_length]
        # → dates[0] = ngày đầu tiên được predict (sau window đầu tiên)
        train_dates = df_train.index[sequence_length:]
        test_dates  = df_test.index[sequence_length:]

    # ── Tạo PyTorch Datasets ──────────────────────────────────────────────────
    train_dataset = StockDataset(X_train, y_train, task=task)
    test_dataset  = StockDataset(X_test,  y_test,  task=task)

    n_features = train_dataset.n_features

    logger.info(
        f"[Fold {fold_id}] Xong | "
        f"n_features={n_features} | "
        f"Train: {train_dataset} | "
        f"Test:  {test_dataset}"
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset":  test_dataset,
        "scaler":        scaler,
        "train_dates":   train_dates,
        "test_dates":    test_dates,
        "n_features":    n_features,
        "fold_id":       fold_id,
    }