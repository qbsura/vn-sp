"""
app/services/evaluation_service.py
=====================================
Tổng hợp và so sánh kết quả từ toàn bộ 240 experiments — Phase 5.

Cung cấp 4 hàm công khai:
  - load_all_results()              → pd.DataFrame  (raw flat table)
  - create_comparison_table(...)    → pd.DataFrame  (replica Table 1 bài báo)
  - get_best_model_per_condition()  → pd.DataFrame  (best model per condition)

Cấu trúc metrics.json (đọc từ experiments/):
  {
    "metrics":    {"mse": float, "mae": float, ...}  # lowercase keys từ experiment_runner
    "exp_id":     str,       # e.g. "VCB_VND_wavelet_BiLSTM_regression"
    "fold_idx":   int,       # 1, 2, 3
    "model_name": str,       # "DNN" | "RNN" | "GRU" | "LSTM" | "BiLSTM"
    "task":       str,       # "regression" | "classification"
    "ticker":     str,       # "VCB" | "VIC"
    "currency":   str,       # "VND" | "USD"
    "use_wavelet": bool,
    ...
  }

Convention:
  - DataFrame dùng PascalCase/UPPERCASE columns: MSE, MAE, MAPE, RMSE, R2,
    Accuracy, Precision, Recall, F1, AUC_ROC — nhất quán với app/utils/metrics.py.
  - metrics.json lưu lowercase keys ("mse", "mae"...) → map khi load.
  - Các metric không liên quan đến task được điền NaN (e.g., MSE=NaN cho classification rows).

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Table 1: Before/After wavelet × 5 models × MSE/MAE/MAPE (regression).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.config import CURRENCIES, MODELS, PATHS, TASKS, TICKERS

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(PATHS["experiments"])

# ── Key mapping: lowercase JSON → PascalCase DataFrame column ─────────────────
_REGRESSION_KEY_MAP = {
    "mse"  : "MSE",
    "mae"  : "MAE",
    "mape" : "MAPE",
    "rmse" : "RMSE",
    "r2"   : "R2",
}
_CLASSIFICATION_KEY_MAP = {
    "accuracy"  : "Accuracy",
    "precision" : "Precision",
    "recall"    : "Recall",
    "f1"        : "F1",
    "auc_roc"   : "AUC_ROC",
}

# Tất cả metric columns trong DataFrame
_ALL_METRIC_COLS = list(_REGRESSION_KEY_MAP.values()) + list(_CLASSIFICATION_KEY_MAP.values())

# Columns định danh thí nghiệm
_ID_COLS = ["ticker", "currency", "use_wavelet", "model", "task", "fold"]


# =============================================================================
# 1. LOAD ALL RESULTS
# =============================================================================

def load_all_results() -> pd.DataFrame:
    """
    Đọc tất cả metrics.json từ experiments/ và trả về DataFrame phẳng.

    Quét toàn bộ cấu trúc:
      experiments/{exp_id}/fold_{i}/metrics.json

    Mỗi row = 1 experiment (1 ticker × 1 currency × 1 wavelet ×
               1 model × 1 task × 1 fold).

    Returns:
        pd.DataFrame với columns:
          ID columns:      ticker, currency, use_wavelet, model, task, fold
          Regression:      MSE, MAE, MAPE, RMSE, R2         (NaN nếu task=classification)
          Classification:  Accuracy, Precision, Recall, F1, AUC_ROC  (NaN nếu task=regression)
          Extra:           n_test_samples, train_elapsed_s, best_epoch,
                           best_val_loss, stopped_early, n_epochs_run, exp_id

        Nếu không tìm thấy file nào → trả về DataFrame rỗng với đúng columns.

    Notes:
        - Chỉ đọc các file metrics.json tồn tại; bỏ qua fold bị thiếu (resume scenario).
        - Cảnh báo qua logger nếu file bị corrupt (JSONDecodeError).
    """
    rows = []

    # Glob tất cả metrics.json trong experiments/*/fold_*/
    json_files = sorted(EXPERIMENTS_DIR.glob("*/fold_*/metrics.json"))

    if not json_files:
        logger.warning(
            "[load_all_results] Không tìm thấy metrics.json nào trong '%s'. "
            "Đã chạy run_experiments.py chưa?",
            EXPERIMENTS_DIR,
        )
        return _empty_results_df()

    loaded = 0
    errors = 0
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[load_all_results] Lỗi đọc '%s': %s", json_path, exc)
            errors += 1
            continue

        row = _parse_metrics_json(data)
        rows.append(row)
        loaded += 1

    logger.info(
        "[load_all_results] Loaded %d experiments | %d errors | path='%s'",
        loaded, errors, EXPERIMENTS_DIR,
    )

    if not rows:
        return _empty_results_df()

    df = pd.DataFrame(rows)

    # Sắp xếp để dễ đọc
    df = df.sort_values(_ID_COLS).reset_index(drop=True)

    # Chắc chắn các metric columns là float
    for col in _ALL_METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _parse_metrics_json(data: dict) -> dict:
    """Parse một metrics.json dict thành 1 flat row dict."""
    task    = data.get("task", "")
    metrics = data.get("metrics", {})
    history = data.get("train_history", {})

    # Khởi tạo row với NaN cho tất cả metrics
    row = {col: np.nan for col in _ALL_METRIC_COLS}

    # Điền metrics theo task
    if task == "regression":
        for json_key, col in _REGRESSION_KEY_MAP.items():
            row[col] = metrics.get(json_key, np.nan)
    elif task == "classification":
        for json_key, col in _CLASSIFICATION_KEY_MAP.items():
            row[col] = metrics.get(json_key, np.nan)

    # ID + metadata
    row.update({
        "ticker"          : data.get("ticker", ""),
        "currency"        : data.get("currency", ""),
        "use_wavelet"     : data.get("use_wavelet", False),
        "model"           : data.get("model_name", ""),
        "task"            : task,
        "fold"            : data.get("fold_idx", -1),
        "n_test_samples"  : data.get("n_test_samples", np.nan),
        "train_elapsed_s" : data.get("train_elapsed_s", np.nan),
        "best_epoch"      : history.get("best_epoch", np.nan),
        "best_val_loss"   : history.get("best_val_loss", np.nan),
        "stopped_early"   : history.get("stopped_early", None),
        "n_epochs_run"    : history.get("n_epochs_run", np.nan),
        "exp_id"          : data.get("exp_id", ""),
    })
    return row


def _empty_results_df() -> pd.DataFrame:
    """Trả về DataFrame rỗng với schema đúng."""
    cols = _ID_COLS + _ALL_METRIC_COLS + [
        "n_test_samples", "train_elapsed_s", "best_epoch",
        "best_val_loss", "stopped_early", "n_epochs_run", "exp_id",
    ]
    return pd.DataFrame(columns=cols)


# =============================================================================
# 2. COMPARISON TABLE (replica Table 1 bài báo)
# =============================================================================

def create_comparison_table(
    df_results : pd.DataFrame,
    ticker     : str = "VCB",
    currency   : str = "VND",
    task       : str = "regression",
) -> pd.DataFrame:
    """
    Tạo bảng so sánh replica Table 1 trong bài báo.

    Bảng hiển thị performance của 5 models × Before/After wavelet,
    với giá trị là **mean across 3 folds**.

    Args:
        df_results: DataFrame từ load_all_results().
        ticker:     "VCB" hoặc "VIC" (mặc định: "VCB").
        currency:   "VND" hoặc "USD" (mặc định: "VND").
        task:       "regression" (mặc định) → MSE/MAE/MAPE
                    "classification" → Accuracy/F1/AUC_ROC

    Returns:
        pd.DataFrame với MultiIndex columns:
          Level 0: "Before Wavelet" | "After Wavelet"
          Level 1: metric names (e.g., MSE, MAE, MAPE)
          Index: model names theo thứ tự chuẩn [DNN, RNN, GRU, LSTM, BiLSTM]

        Ví dụ (regression):
                     Before Wavelet              After Wavelet
                     MSE    MAE    MAPE          MSE    MAE    MAPE
          DNN        ...    ...    ...           ...    ...    ...
          RNN        ...    ...    ...           ...    ...    ...
          ...

    Notes:
        - Giá trị là mean của 3 folds. Nếu thiếu fold → mean của folds có sẵn.
        - Nếu không có data cho condition này → trả về DataFrame rỗng + warning.
        - Thứ tự models: DNN, RNN, GRU, LSTM, BiLSTM (nhất quán với bài báo).
    """
    # Chọn metrics phù hợp với task
    if task == "regression":
        metric_cols = ["MSE", "MAE", "MAPE"]
    elif task == "classification":
        metric_cols = ["Accuracy", "F1", "AUC_ROC"]
    else:
        raise ValueError(f"create_comparison_table: task='{task}' không hợp lệ.")

    # Filter theo ticker, currency, task
    mask = (
        (df_results["ticker"]   == ticker) &
        (df_results["currency"] == currency) &
        (df_results["task"]     == task)
    )
    df_filtered = df_results[mask].copy()

    if df_filtered.empty:
        logger.warning(
            "[create_comparison_table] Không có data cho "
            "ticker=%s currency=%s task=%s", ticker, currency, task,
        )
        return pd.DataFrame()

    # Aggregate: mean across folds, group by (model, use_wavelet)
    df_agg = (
        df_filtered
        .groupby(["model", "use_wavelet"], observed=True)[metric_cols]
        .mean()
        .reset_index()
    )

    # Pivot: rows=model, columns=use_wavelet (True=After, False=Before)
    pivot_frames = {}
    for wavelet_flag, label in [(False, "Before Wavelet"), (True, "After Wavelet")]:
        sub = df_agg[df_agg["use_wavelet"] == wavelet_flag].set_index("model")[metric_cols]
        pivot_frames[label] = sub

    # Concat columns với MultiIndex
    df_table = pd.concat(pivot_frames, axis=1)

    # Reindex để đảm bảo thứ tự models chuẩn (DNN→BiLSTM)
    model_order = [m for m in MODELS if m in df_table.index]
    df_table = df_table.reindex(model_order)
    df_table.index.name = "Model"

    return df_table


# =============================================================================
# 3. BEST MODEL PER CONDITION
# =============================================================================

def get_best_model_per_condition(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Tìm best model cho mỗi (ticker, currency, use_wavelet, task).

    Tiêu chí best:
      - Regression:      MSE thấp nhất (mean across 3 folds)
      - Classification:  F1 cao nhất (mean across 3 folds)

    Args:
        df_results: DataFrame từ load_all_results().

    Returns:
        pd.DataFrame, mỗi row = 1 condition với:
          [ticker, currency, use_wavelet, task,
           best_model, best_MSE, best_MAE, best_MAPE, best_R2]   ← regression
          hoặc
          [ticker, currency, use_wavelet, task,
           best_model, best_Accuracy, best_Precision, best_Recall, best_F1, best_AUC_ROC]

        Cả 2 tasks được gộp vào cùng 1 DataFrame; metric NaN tương ứng task còn lại.

    Notes:
        Nếu có nhiều models cho cùng score (rất hiếm) → chọn model đầu tiên theo MODELS order.
    """
    if df_results.empty:
        return pd.DataFrame()

    # Tính mean metrics across folds
    group_cols_reg  = ["ticker", "currency", "use_wavelet", "model", "task"]
    reg_metrics     = ["MSE", "MAE", "MAPE", "RMSE", "R2"]
    cls_metrics     = ["Accuracy", "Precision", "Recall", "F1", "AUC_ROC"]
    all_m           = reg_metrics + cls_metrics

    df_mean = (
        df_results
        .groupby(group_cols_reg, observed=True)[all_m]
        .mean()
        .reset_index()
    )

    rows = []
    for (ticker, currency, wavelet, task), group in df_mean.groupby(
        ["ticker", "currency", "use_wavelet", "task"], observed=True
    ):
        if group.empty:
            continue

        # Sắp xếp models theo thứ tự chuẩn để tie-breaking nhất quán
        group = group.copy()
        model_rank = {m: i for i, m in enumerate(MODELS)}
        group["_rank"] = group["model"].map(model_rank).fillna(999)
        group = group.sort_values("_rank")

        # Chọn best model theo tiêu chí của task
        if task == "regression":
            best_row = group.loc[group["MSE"].idxmin()]
        else:
            best_row = group.loc[group["F1"].idxmax()]

        row = {
            "ticker"     : ticker,
            "currency"   : currency,
            "use_wavelet": wavelet,
            "task"       : task,
            "best_model" : best_row["model"],
        }
        # Đính kèm metrics của best model
        for col in all_m:
            val = best_row.get(col, np.nan)
            row[f"best_{col}"] = float(val) if pd.notna(val) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_best = pd.DataFrame(rows).sort_values(
        ["ticker", "currency", "use_wavelet", "task"]
    ).reset_index(drop=True)

    return df_best
    """
    So sánh metrics của VND vs USD cho cùng ticker/model/wavelet/task.

    Giá trị là **mean across 3 folds** của mỗi metric.

    Args:
        df_results: DataFrame từ load_all_results().

    Returns:
        pd.DataFrame với MultiIndex columns:
          Level 0: "VND" | "USD"
          Level 1: metric names phù hợp với task
          Index: (ticker, use_wavelet, model, task)

        Ví dụ:
                                                      VND              USD
                                                      MSE    MAE       MSE    MAE
          (VCB, True, BiLSTM, regression)             ...    ...       ...    ...
          (VCB, True, LSTM,   regression)             ...    ...       ...    ...
          ...

        Thêm cột delta columns:
          "Δ MSE (USD-VND)": USD_MSE - VND_MSE (negative = USD tốt hơn về MSE)
          tương tự cho MAE, MAPE (regression) hoặc Δ F1, Δ Accuracy (classification)

    Notes:
        - Cả regression và classification đều được include trong cùng 1 DataFrame.
        - Rows bị thiếu data (chỉ có VND hoặc chỉ có USD) vẫn được giữ với NaN.
    """
    if df_results.empty:
        return pd.DataFrame()

    # Aggregate mean across folds
    group_cols = ["ticker", "currency", "use_wavelet", "model", "task"]
    df_mean = (
        df_results
        .groupby(group_cols, observed=True)[_ALL_METRIC_COLS]
        .mean()
        .reset_index()
    )

    # Tách VND và USD
    df_vnd = df_mean[df_mean["currency"] == "VND"].drop(columns="currency")
    df_usd = df_mean[df_mean["currency"] == "USD"].drop(columns="currency")

    merge_cols = ["ticker", "use_wavelet", "model", "task"]

    # Outer merge để giữ tất cả combinations
    df_merged = pd.merge(
        df_vnd, df_usd,
        on    = merge_cols,
        how   = "outer",
        suffixes = ("_VND", "_USD"),
    )

    # Tạo MultiIndex columns
    vnd_cols = {f"{c}_VND": ("VND", c) for c in _ALL_METRIC_COLS}
    usd_cols = {f"{c}_USD": ("USD", c) for c in _ALL_METRIC_COLS}

    # Rename flat → MultiIndex
    rename_map = {**vnd_cols, **usd_cols}
    df_merged = df_merged.rename(columns=rename_map)

    # Tách các cột id và metric
    id_cols_flat  = merge_cols
    metric_tuples = list(vnd_cols.values()) + list(usd_cols.values())

    # Tính delta columns (USD - VND) cho metrics chính
    delta_pairs_reg = [("MSE", True), ("MAE", True), ("MAPE", True), ("R2", False)]
    delta_pairs_cls = [("F1", False), ("Accuracy", False), ("AUC_ROC", False)]
    all_delta_pairs = delta_pairs_reg + delta_pairs_cls

    delta_data = {}
    for metric, lower_is_better in all_delta_pairs:
        vnd_col = ("VND", metric)
        usd_col = ("USD", metric)
        # Kiểm tra cột tồn tại trong MultiIndex
        if vnd_col in df_merged.columns and usd_col in df_merged.columns:
            delta_key = ("Δ (USD−VND)", metric)
            delta_data[delta_key] = df_merged[usd_col].values - df_merged[vnd_col].values

    # Xây dựng DataFrame với MultiIndex columns
    # Cột id trước (single-level), metric cols sau (multi-level)
    df_id      = df_merged[id_cols_flat].copy()
    df_metrics = df_merged[metric_tuples].copy()

    # Nối delta
    if delta_data:
        df_delta = pd.DataFrame(
            delta_data,
            index = df_merged.index,
        )
        df_final_metrics = pd.concat([df_metrics, df_delta], axis=1)
    else:
        df_final_metrics = df_metrics

    # MultiIndex cho id columns (single level → ("", col))
    df_id.columns = pd.MultiIndex.from_tuples([("", c) for c in id_cols_flat])

    df_comparison = pd.concat([df_id, df_final_metrics], axis=1)

    # Sort để dễ đọc
    df_comparison = df_comparison.sort_values(
        [("", "ticker"), ("", "task"), ("", "use_wavelet"), ("", "model")]
    ).reset_index(drop=True)

    return df_comparison