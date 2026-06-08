"""
app/utils/metrics.py
=====================
Utility functions tính evaluation metrics cho VNSP — Phase 5.

Cung cấp 3 hàm công khai:
  - compute_regression_metrics(y_true, y_pred)           → Task A metrics
  - compute_classification_metrics(y_true, y_pred_prob)  → Task B metrics
  - aggregate_fold_metrics(metrics_list)                  → Mean ± std / 3 folds

Key note về key naming convention:
  - Module này dùng PascalCase / UPPERCASE keys ("MSE", "MAE", "AUC_ROC", ...)
    để nhất quán với Table 1 trong bài báo và output của visualization_service.
  - Khác với compute_metrics() trong experiment_runner.py (dùng lowercase keys
    "mse", "mae", ... để lưu JSON nội bộ trong experiments/).

Công thức theo bài báo (Li et al., EAAI 2026, equations 4–8):
  MAE  = (1/n) Σ |y_i - ŷ_i|
  RMSE = sqrt((1/n) Σ (y_i - ŷ_i)²)
  MSE  = (1/n) Σ (y_i - ŷ_i)²
  MAPE = (100/n) Σ |y_i - ŷ_i| / |y_i|
  R²   = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

Tham chiếu:
  Li et al., Engineering Applications of Artificial Intelligence,
  165 (2026) 113390. Sections 4.1–4.2.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# =============================================================================
# REGRESSION METRICS  (Task A)
# =============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Tính đầy đủ 5 regression metrics theo bài báo (equations 4–8).

    Tất cả metrics được tính trên giá thực đã inverse-transform (VND hoặc USD),
    không phải trên giá đã scaled.

    Args:
        y_true: Actual close prices, shape (N,). Numpy array hoặc list.
        y_pred: Predicted close prices, shape (N,). Numpy array hoặc list.

    Returns:
        dict với 5 keys:
          "MSE"  (float): Mean Squared Error — đơn vị price².
          "MAE"  (float): Mean Absolute Error — đơn vị price.
          "MAPE" (float): Mean Absolute Percentage Error × 100 → ra %.
                          Ví dụ: 2.5 nghĩa là 2.5%.
          "RMSE" (float): Root Mean Squared Error — đơn vị price.
          "R2"   (float): Coefficient of Determination, range (-∞, 1].
                          1.0 = perfect fit, 0 = predict mean, <0 = worse than mean.

    Notes:
        MAPE: dùng epsilon=1e-8 khi y_true ≈ 0 để tránh division by zero.
              Thực tế giá cổ phiếu luôn > 0 nên điều này chỉ là safeguard.

    Examples:
        >>> y_true = np.array([100.0, 200.0, 150.0])
        >>> y_pred = np.array([105.0, 195.0, 148.0])
        >>> m = compute_regression_metrics(y_true, y_pred)
        >>> round(m["MAE"], 4)
        4.0
    """
    # Chuyển về 1-D float64
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"compute_regression_metrics: y_true length ({len(y_true)}) "
            f"!= y_pred length ({len(y_pred)})"
        )
    if len(y_true) == 0:
        raise ValueError("compute_regression_metrics: input arrays rỗng.")

    # MSE — equation 6
    mse = float(mean_squared_error(y_true, y_pred))

    # MAE — equation 4
    mae = float(mean_absolute_error(y_true, y_pred))

    # RMSE — equation 5
    rmse = float(np.sqrt(mse))

    # MAPE — equation 7: ×100 để ra %
    # Dùng |y_true| + eps thay vì y_true để handle negative/zero edge cases
    eps  = 1e-8
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
    )

    # R² — equation 8
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MSE"  : mse,
        "MAE"  : mae,
        "MAPE" : mape,
        "RMSE" : rmse,
        "R2"   : r2,
    }


# =============================================================================
# CLASSIFICATION METRICS  (Task B)
# =============================================================================

def compute_classification_metrics(
    y_true     : np.ndarray,
    y_pred_prob: np.ndarray,
    threshold  : float = 0.5,
) -> dict:
    """
    Tính đầy đủ 5 classification metrics (Task B — direction prediction).

    Nhận xác suất P(UP) từ sigmoid output, áp threshold để ra binary labels,
    rồi tính metrics. AUC-ROC dùng xác suất gốc (không threshold).

    Args:
        y_true:      True binary labels {0=DOWN, 1=UP}, shape (N,).
        y_pred_prob: Predicted probabilities P(UP) ∈ [0, 1], shape (N,).
                     Là raw sigmoid output từ model.
        threshold:   Ngưỡng phân loại, mặc định 0.5.
                     y_pred = (y_pred_prob >= threshold).astype(int)

    Returns:
        dict với 5 keys:
          "Accuracy"  (float): Tỉ lệ dự đoán đúng, range [0, 1].
          "Precision" (float): Precision cho class UP (=1), range [0, 1].
          "Recall"    (float): Recall cho class UP (=1), range [0, 1].
          "F1"        (float): F1-score binary, range [0, 1].
          "AUC_ROC"   (float): Area Under ROC Curve, range [0, 1].
                                0.5 = random, 1.0 = perfect.

    Notes:
        Precision/Recall/F1: dùng zero_division=0 (trả về 0 thay vì warning
        khi một class không có prediction — có thể xảy ra ở early training).
        AUC-ROC: fallback về 0.5 nếu y_true chỉ chứa 1 class (edge case).

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_prob = np.array([0.8, 0.3, 0.6, 0.9, 0.4])
        >>> m = compute_classification_metrics(y_true, y_prob)
        >>> m["Accuracy"]
        1.0
    """
    # Chuyển về numpy 1-D
    y_true      = np.asarray(y_true,      dtype=np.float64).flatten()
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float64).flatten()

    if len(y_true) != len(y_pred_prob):
        raise ValueError(
            f"compute_classification_metrics: y_true length ({len(y_true)}) "
            f"!= y_pred_prob length ({len(y_pred_prob)})"
        )
    if len(y_true) == 0:
        raise ValueError("compute_classification_metrics: input arrays rỗng.")

    # Áp threshold để ra binary labels
    y_pred_int = (y_pred_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    # Accuracy
    accuracy = float(accuracy_score(y_true_int, y_pred_int))

    # Precision, Recall, F1 — binary classification (positive = UP = 1)
    precision = float(precision_score(
        y_true_int, y_pred_int,
        zero_division=0,
        average="binary",
    ))
    recall = float(recall_score(
        y_true_int, y_pred_int,
        zero_division=0,
        average="binary",
    ))
    f1 = float(f1_score(
        y_true_int, y_pred_int,
        zero_division=0,
        average="binary",
    ))

    # AUC-ROC: dùng xác suất gốc (không threshold)
    try:
        auc_val = roc_auc_score(y_true_int, y_pred_prob)
        # Một số sklearn versions trả về nan thay vì raise ValueError
        # khi y_true chỉ có 1 class
        if np.isnan(auc_val) or np.isinf(auc_val):
            raise ValueError("AUC undefined (single class)")
        auc_roc = float(auc_val)
    except (ValueError, TypeError):
        # Xảy ra khi y_true chỉ chứa 1 class (e.g. tất cả UP hoặc tất cả DOWN)
        logger.warning(
            "compute_classification_metrics: y_true chỉ có 1 class "
            "→ AUC-ROC = 0.5 (fallback)."
        )
        auc_roc = 0.5

    return {
        "Accuracy"  : accuracy,
        "Precision" : precision,
        "Recall"    : recall,
        "F1"        : f1,
        "AUC_ROC"   : auc_roc,
    }


# =============================================================================
# FOLD AGGREGATION
# =============================================================================

def aggregate_fold_metrics(metrics_list: list[dict]) -> dict:
    """
    Tổng hợp metrics từ nhiều folds thành mean ± std.

    Dùng để tóm tắt Walk-Forward Validation (3 folds) thành 1 row
    trong bảng so sánh (replica Table 1 của bài báo).

    Args:
        metrics_list: List các dict metrics, mỗi dict là kết quả 1 fold.
                      Thường là 3 folds, nhưng hàm xử lý bất kỳ N ≥ 1 fold.
                      Tất cả dict phải có cùng set of keys.

                      Regression example:
                        [{"MSE": 0.01, "MAE": 0.08, ...},   # fold 1
                         {"MSE": 0.02, "MAE": 0.09, ...},   # fold 2
                         {"MSE": 0.015, "MAE": 0.085, ...}] # fold 3

                      Classification example:
                        [{"Accuracy": 0.55, "F1": 0.52, ...}, ...]

    Returns:
        Flat dict với format "{key}_mean" và "{key}_std" cho mỗi metric.

        Regression example output:
          {
            "MSE_mean":  0.015,
            "MSE_std":   0.005,
            "MAE_mean":  0.085,
            "MAE_std":   0.005,
            "MAPE_mean": ...,
            "MAPE_std":  ...,
            "RMSE_mean": ...,
            "RMSE_std":  ...,
            "R2_mean":   ...,
            "R2_std":    ...,
          }

        Classification example output:
          {
            "Accuracy_mean":  ..., "Accuracy_std":  ...,
            "Precision_mean": ..., "Precision_std": ...,
            "Recall_mean":    ..., "Recall_std":    ...,
            "F1_mean":        ..., "F1_std":        ...,
            "AUC_ROC_mean":   ..., "AUC_ROC_std":   ...,
          }

    Notes:
        - std dùng ddof=0 (population std) vì số folds nhỏ (3).
          Điều này nhất quán với cách bài báo report results.
        - Nếu chỉ có 1 fold, std = 0.0 cho tất cả metrics.

    Raises:
        ValueError: Nếu metrics_list rỗng.
        ValueError: Nếu các dict không có cùng keys.

    Examples:
        >>> fold_metrics = [
        ...     {"MSE": 0.01, "MAE": 0.08},
        ...     {"MSE": 0.02, "MAE": 0.10},
        ...     {"MSE": 0.015,"MAE": 0.09},
        ... ]
        >>> agg = aggregate_fold_metrics(fold_metrics)
        >>> round(agg["MSE_mean"], 4)
        0.015
        >>> round(agg["MAE_std"], 6)
        0.008165
    """
    if not metrics_list:
        raise ValueError("aggregate_fold_metrics: metrics_list rỗng.")

    # Lấy keys từ fold đầu tiên, validate consistency
    keys = list(metrics_list[0].keys())
    for i, fold_m in enumerate(metrics_list[1:], start=2):
        missing = set(keys) - set(fold_m.keys())
        extra   = set(fold_m.keys()) - set(keys)
        if missing or extra:
            raise ValueError(
                f"aggregate_fold_metrics: fold {i} có keys không khớp. "
                f"Missing: {missing}, Extra: {extra}"
            )

    result: dict = {}
    for key in keys:
        # Thu thập giá trị của key này qua tất cả folds
        values = np.array([float(m[key]) for m in metrics_list], dtype=np.float64)

        # ddof=0: population std (nhất quán với N=3 folds nhỏ)
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"]  = float(np.std(values, ddof=0))

    return result