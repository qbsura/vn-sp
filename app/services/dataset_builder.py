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

Hai tasks song song — cùng sequences, khác target:
  Task A (Regression)      : y = Close(t+1) scaled value
  Task B (Classification)  : y = 1 nếu Close(t+1) > Close(t), else 0
                             → dựa trên GIÁ GỐC (unscaled) để tránh bias từ scaling
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

    Classification target (Task B):
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
          "train_dates"   : DatetimeIndex — dates ứng với y_train (ngày được predict)
          "test_dates"    : DatetimeIndex — dates ứng với y_test (ngày được predict)
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
    # Classification: truyền original_close để tính direction từ giá thực
    orig_train = orig_close_train if task == "classification" else None
    orig_test  = orig_close_test  if task == "classification" else None

    X_train, y_train = build_sequences(
        df_train_scaled, sequence_length, target_col, task, orig_train
    )
    X_test, y_test = build_sequences(
        df_test_scaled, sequence_length, target_col, task, orig_test
    )

    # ── Prediction dates ──────────────────────────────────────────────────────
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