"""
app/api/results.py
===================
API routes cho Results & Metrics — /api/results/*

Endpoints:
  GET /api/results/summary                      → tất cả kết quả (DataFrame → JSON)
  GET /api/results/comparison-table             → Table 1 bài báo (5 models × before/after)
  GET /api/results/best-models                  → best model per condition
  GET /api/results/vnd-vs-usd                   → VND vs USD comparison
  GET /api/results/{exp_id}/predictions         → y_true và y_pred cho visualization
  GET /api/results/{exp_id}/classification-report → Accuracy, F1, AUC per fold  ← Task 7.4
  GET /api/results/{exp_id}/confusion-matrix    → raw confusion matrix data       ← Task 7.4
  GET /api/results/{exp_id}/roc-curve           → FPR, TPR, thresholds            ← Task 7.4
  GET /api/results/trading/{ticker}/{currency}            → trading simulation results (summary)
  GET /api/results/trading/{ticker}/{currency}/timeseries → time-series data cho Chart.js

Routing order: FastAPI ưu tiên literal paths trước parameterized paths,
nên /summary, /comparison-table, /best-models, /vnd-vs-usd không bị
capture bởi /{exp_id}. Tương tự /trading/{t}/{c} có 3 segments nên
không conflict với /{exp_id}/{sub} (2 segments).

Tham chiếu:
  Li et al., EAAI 2026, Table 1: MSE/MAE/MAPE before/after wavelet × 5 models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.config import CURRENCIES, FOLDS, PATHS, TICKERS

logger = logging.getLogger(__name__)

router = APIRouter()

_EXPERIMENTS_DIR = Path(PATHS["experiments"])


# =============================================================================
# SUMMARY — tất cả kết quả
# =============================================================================

@router.get("/summary", summary="Tất cả kết quả experiments")
def get_summary(
    ticker  : Optional[str]  = Query(None),
    currency: Optional[str]  = Query(None),
    task    : Optional[str]  = Query(None),
    wavelet : Optional[bool] = Query(None),
) -> dict:
    """
    Đọc tất cả metrics.json và trả về bảng tổng hợp.

    Args:
        ticker:   Filter theo ticker (None = tất cả).
        currency: Filter theo currency (None = tất cả).
        task:     Filter theo task: "regression" hoặc "classification".
        wavelet:  Filter theo wavelet condition (None = cả hai).

    Returns:
        {"n_rows": int, "columns": list[str], "data": list[dict]}
    """
    from app.services.evaluation_service import load_all_results

    df = load_all_results()

    if df.empty:
        return {"n_rows": 0, "columns": [], "data": []}

    # Apply filters
    if ticker:
        df = df[df["ticker"] == ticker]
    if currency:
        df = df[df["currency"] == currency]
    if task:
        df = df[df["task"] == task]
    if wavelet is not None:
        df = df[df["use_wavelet"] == wavelet]

    # Convert NaN → None để JSON-serializable
    records = df.where(df.notna(), None).to_dict(orient="records")

    return {
        "n_rows" : len(records),
        "columns": list(df.columns),
        "data"   : records,
    }


# =============================================================================
# COMPARISON TABLE (Table 1 bài báo)
# =============================================================================

@router.get("/comparison-table", summary="Table 1: Before/After wavelet × 5 models")
def get_comparison_table(
    ticker  : str = Query("VCB", description="VCB hoặc VIC"),
    currency: str = Query("VND", description="VND hoặc USD"),
    task    : str = Query("regression", description="regression hoặc classification"),
) -> dict:
    """
    Tạo bảng so sánh replica Table 1 của bài báo.

    Returns:
        {"ticker": str, "currency": str, "task": str,
         "models": list[str],
         "before_wavelet": {"MSE": [...], "MAE": [...], "MAPE": [...]},
         "after_wavelet":  {"MSE": [...], "MAE": [...], "MAPE": [...]}}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.evaluation_service import load_all_results, create_comparison_table

    df = load_all_results()
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có kết quả nào. Chạy experiments trước.")

    tbl = create_comparison_table(df, ticker=ticker, currency=currency, task=task)
    if tbl.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Không có data cho ticker={ticker}, currency={currency}, task={task}.",
        )

    # Convert MultiIndex DataFrame → JSON-friendly dict
    models = list(tbl.index)
    result: dict = {
        "ticker"  : ticker,
        "currency": currency,
        "task"    : task,
        "models"  : models,
    }

    for top_label in tbl.columns.get_level_values(0).unique():
        key = top_label.lower().replace(" ", "_")   # "before_wavelet" / "after_wavelet"
        sub = tbl[top_label]
        result[key] = {
            col: [
                round(v, 6) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                for v in sub[col].tolist()
            ]
            for col in sub.columns
        }

    return result


# =============================================================================
# BEST MODELS
# =============================================================================

@router.get("/best-models", summary="Best model per condition")
def get_best_models(
    ticker  : Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
) -> dict:
    """
    Tìm best model cho mỗi (ticker, currency, wavelet, task).

    Best = lowest MSE (regression) hoặc highest F1 (classification).

    Returns:
        {"n_rows": int, "data": list[dict]}
    """
    from app.services.evaluation_service import load_all_results, get_best_model_per_condition

    df = load_all_results()
    if df.empty:
        return {"n_rows": 0, "data": []}

    if ticker:
        df = df[df["ticker"] == ticker]
    if currency:
        df = df[df["currency"] == currency]

    best = get_best_model_per_condition(df)
    if best.empty:
        return {"n_rows": 0, "data": []}

    records = best.where(best.notna(), None).to_dict(orient="records")
    return {"n_rows": len(records), "data": records}


# =============================================================================
# VND vs USD COMPARISON
# =============================================================================

@router.get("/vnd-vs-usd", summary="VND vs USD comparison table")
def get_vnd_vs_usd(
    ticker: Optional[str] = Query(None),
    task  : Optional[str] = Query(None),
) -> dict:
    """
    So sánh metrics của VND vs USD cho cùng ticker/model/condition.

    Returns:
        {"n_rows": int, "columns": list, "data": list[dict]}
    """
    from app.services.evaluation_service import load_all_results, compare_vnd_vs_usd

    df = load_all_results()
    if df.empty:
        return {"n_rows": 0, "columns": [], "data": []}

    if ticker:
        df = df[df["ticker"] == ticker]
    if task:
        df = df[df["task"] == task]

    comp = compare_vnd_vs_usd(df)
    if comp.empty:
        return {"n_rows": 0, "columns": [], "data": []}

    # Flatten MultiIndex columns → "VND_MSE", "USD_MSE", "Delta_MSE" etc.
    flat_cols = []
    for col in comp.columns:
        if isinstance(col, tuple):
            top, sub = col
            if top == "":
                flat_cols.append(sub)
            else:
                flat_cols.append(
                    f"{top}_{sub}".replace(" ", "_").replace("(", "").replace(")", "")
                )
        else:
            flat_cols.append(str(col))
    comp.columns = flat_cols

    records = comp.where(comp.notna(), None).to_dict(orient="records")
    return {"n_rows": len(records), "columns": flat_cols, "data": records}


# =============================================================================
# PREDICTIONS (cho visualization)
# =============================================================================

@router.get("/{exp_id}/predictions", summary="y_true và y_pred của test set")
def get_predictions(
    exp_id  : str,
    fold_idx: int = Query(3, ge=1, le=3, description="Fold index (default=3, tốt nhất)"),
) -> dict:
    """
    Trả về y_pred, y_true (và y_prob nếu classification) từ predictions.npz.

    Dùng cho:
      - Regression: Predicted vs Actual price chart
      - Classification: Confusion Matrix, ROC Curve

    Returns:
        {"exp_id": str, "fold_idx": int, "task": str,
         "y_true": list[float], "y_pred": list[float],
         "y_prob": list[float] | None}
    """
    npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"predictions.npz không tồn tại: {exp_id}/fold_{fold_idx}",
        )

    data   = np.load(str(npz_path), allow_pickle=False)
    y_pred = data["y_pred"].flatten().tolist()
    y_true = data["y_true"].flatten().tolist()
    y_prob = data["y_prob"].flatten().tolist() if "y_prob" in data else None

    # Xác định task từ exp_id (phần cuối sau dấu gạch dưới cuối cùng)
    parts = exp_id.split("_")
    task  = parts[-1] if parts else "unknown"

    return {
        "exp_id"   : exp_id,
        "fold_idx" : fold_idx,
        "task"     : task,
        "n_samples": len(y_pred),
        "y_true"   : y_true,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
    }


# =============================================================================
# CLASSIFICATION REPORT  — Task 7.4
# =============================================================================

@router.get("/{exp_id}/classification-report", summary="Classification metrics per fold (Weekly, T2-T6)")
def get_classification_report(
    exp_id   : str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Probability threshold"),
) -> dict:
    """
    Tính và trả về Accuracy, Precision, Recall, F1, AUC-ROC cho từng fold.

    GHI CHÚ (2026-06, Phương án D): Task B hiện dự đoán hướng đi TUẦN kế tiếp
    (T2→T6), 1 sample/tuần — không phải hướng đi ngày kế tiếp như trước.
    Cấu trúc response và cách tính metrics không đổi.

    Đọc từ predictions.npz (y_true, y_prob) và gọi compute_classification_metrics.

    Args:
        exp_id:    Classification experiment ID (phải chứa "classification").
        threshold: Ngưỡng phân loại P(UP) ≥ threshold → UP. Default: 0.5.

    Returns:
        {"exp_id": str, "threshold": float,
         "folds": [{"fold_idx": int, "status": str, "metrics": dict}],
         "aggregated": {"Accuracy_mean": ..., "F1_mean": ..., ...}}

    Raises:
        400: Nếu exp_id không phải classification experiment.
    """
    if "classification" not in exp_id:
        raise HTTPException(
            status_code = 400,
            detail      = "Endpoint này chỉ dành cho classification experiments. "
                          f"exp_id='{exp_id}' không chứa 'classification'.",
        )

    from app.utils.metrics import aggregate_fold_metrics, compute_classification_metrics

    fold_results  = []
    metrics_list  = []   # dùng để aggregate

    for fold in FOLDS:
        fold_idx = fold["fold_id"]
        npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"

        if not npz_path.exists():
            fold_results.append({"fold_idx": fold_idx, "status": "pending", "metrics": None})
            continue

        data   = np.load(str(npz_path), allow_pickle=False)
        y_true = data["y_true"].flatten()

        # Ưu tiên dùng y_prob (sigmoid output); fallback về y_pred nếu thiếu
        if "y_prob" in data:
            y_prob = data["y_prob"].flatten()
        else:
            # y_pred là binary (0/1) → không có xác suất → AUC-ROC sẽ = 0.5
            y_prob = data["y_pred"].flatten().astype(float)

        metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)

        fold_results.append({
            "fold_idx": fold_idx,
            "status"  : "done",
            "metrics" : {k: round(v, 6) for k, v in metrics.items()},
        })
        metrics_list.append(metrics)

    # Aggregate nếu có ít nhất 1 fold xong
    aggregated = None
    if metrics_list:
        agg_raw    = aggregate_fold_metrics(metrics_list)
        aggregated = {k: round(v, 6) for k, v in agg_raw.items()}

    return {
        "exp_id"    : exp_id,
        "threshold" : threshold,
        "folds"     : fold_results,
        "aggregated": aggregated,
    }


# =============================================================================
# CONFUSION MATRIX DATA  — Task 7.4
# =============================================================================

@router.get("/{exp_id}/confusion-matrix", summary="Confusion matrix data (raw numbers)")
def get_confusion_matrix(
    exp_id  : str,
    fold_idx: int   = Query(3, ge=1, le=3),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
) -> dict:
    """
    Trả về raw confusion matrix data (không phải image).

    Cho frontend tự render (heatmap via Chart.js hoặc D3).
    Image version nằm tại GET /api/viz/confusion-matrix.

    Args:
        exp_id:    Classification experiment ID.
        fold_idx:  Fold index (default=3).
        threshold: Probability threshold. Default: 0.5.

    Returns:
        {"exp_id": str, "fold_idx": int,
         "matrix": [[TN, FP], [FN, TP]],
         "labels": ["DOWN (0)", "UP (1)"],
         "stats": {"TN": int, "FP": int, "FN": int, "TP": int,
                   "total": int, "accuracy": float}}

    Raises:
        400: Nếu exp_id không phải classification experiment.
        404: Nếu predictions.npz không tồn tại.
    """
    if "classification" not in exp_id:
        raise HTTPException(
            status_code = 400,
            detail      = "Endpoint này chỉ dành cho classification experiments.",
        )

    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"predictions.npz không tồn tại: {exp_id}/fold_{fold_idx}",
        )

    data   = np.load(str(npz_path), allow_pickle=False)
    y_true = data["y_true"].flatten().astype(int)

    # Dùng y_prob nếu có, else y_pred
    if "y_prob" in data:
        y_pred_binary = (data["y_prob"].flatten() >= threshold).astype(int)
    else:
        y_pred_binary = data["y_pred"].flatten().astype(int)

    cm = sk_confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    return {
        "exp_id"  : exp_id,
        "fold_idx": fold_idx,
        "threshold": threshold,
        "matrix"  : cm.tolist(),
        "labels"  : ["DOWN (0)", "UP (1)"],
        "stats"   : {
            "TN"      : tn,
            "FP"      : fp,
            "FN"      : fn,
            "TP"      : tp,
            "total"   : total,
            "accuracy": round((tn + tp) / total, 6) if total > 0 else 0.0,
        },
    }


# =============================================================================
# ROC CURVE DATA  — Task 7.4
# =============================================================================

@router.get("/{exp_id}/roc-curve", summary="ROC curve data: FPR, TPR, thresholds")
def get_roc_curve(
    exp_id  : str,
    fold_idx: int = Query(3, ge=1, le=3, description="Fold index (default=3)"),
) -> dict:
    """
    Tính và trả về dữ liệu ROC curve (FPR, TPR, thresholds) để frontend vẽ.

    Dùng sklearn.metrics.roc_curve — yêu cầu y_prob (xác suất), không phải binary.

    Args:
        exp_id:   Classification experiment ID.
        fold_idx: Fold index (default=3).

    Returns:
        {"exp_id": str, "fold_idx": int, "auc_roc": float,
         "fpr": list[float], "tpr": list[float], "thresholds": list[float],
         "n_points": int}

    Raises:
        400: Nếu exp_id không phải classification, hoặc y_prob không có trong npz.
        404: Nếu predictions.npz không tồn tại.
    """
    if "classification" not in exp_id:
        raise HTTPException(
            status_code = 400,
            detail      = "Endpoint này chỉ dành cho classification experiments.",
        )

    from sklearn.metrics import roc_auc_score, roc_curve

    npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"predictions.npz không tồn tại: {exp_id}/fold_{fold_idx}",
        )

    data = np.load(str(npz_path), allow_pickle=False)

    if "y_prob" not in data:
        raise HTTPException(
            status_code = 400,
            detail      = "y_prob không có trong predictions.npz. "
                          "ROC curve cần xác suất (sigmoid output), không phải binary.",
        )

    y_true = data["y_true"].flatten().astype(int)
    y_prob = data["y_prob"].flatten().astype(float)

    # Kiểm tra tính hợp lệ (cần cả 2 classes)
    if len(np.unique(y_true)) < 2:
        raise HTTPException(
            status_code = 422,
            detail      = "y_true chỉ có 1 class — ROC curve không thể tính.",
        )

    # Tính ROC curve và AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    auc_roc = float(roc_auc_score(y_true, y_prob))

    # Downsample nếu quá nhiều điểm (>500) để giảm kích thước response
    if len(fpr) > 500:
        step   = len(fpr) // 500
        fpr    = fpr[::step]
        tpr    = tpr[::step]
        thresholds = thresholds[::step]

    return {
        "exp_id"     : exp_id,
        "fold_idx"   : fold_idx,
        "auc_roc"    : round(auc_roc, 6),
        "fpr"        : [round(float(x), 6) for x in fpr],
        "tpr"        : [round(float(x), 6) for x in tpr],
        "thresholds" : [round(float(x), 6) for x in thresholds],
        "n_points"   : len(fpr),
    }


# =============================================================================
# TRADING SIMULATION RESULTS
# =============================================================================

@router.get("/trading/{ticker}/{currency}", summary="Trading simulation results (Weekly)")
def get_trading_results(
    ticker  : str,
    currency: str,
    fold_idx: int = Query(3, ge=1, le=3),
) -> dict:
    """
    Chạy trading simulation cho tất cả 5 models × 2 wavelet conditions.
    Dùng predictions từ Task B (classification, Weekly T2-T6).

    GHI CHÚ (2026-06, Phương án D): mỗi sample = 1 tuần. Mua nếu model dự
    đoán tuần kế tiếp UP (return = (Close(F_W+1)-Close(F_W))/Close(F_W)),
    đứng ngoài nếu DOWN (no shorting). Sharpe Ratio annualize ×√52.
    Cấu trúc response (field names) không đổi so với bản daily cũ.

    Returns:
        {"ticker": str, "currency": str, "fold_idx": int,
         "n_models": int, "data": list[dict]}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.trading_service import run_trading_simulation_all_models

    df_result = run_trading_simulation_all_models(ticker, currency, fold_idx)

    if df_result.empty:
        return {"ticker": ticker, "currency": currency, "fold_idx": fold_idx,
                "n_models": 0, "data": []}

    records = df_result.where(df_result.notna(), None).to_dict(orient="records")
    return {
        "ticker"  : ticker,
        "currency": currency,
        "fold_idx": fold_idx,
        "n_models": len(records),
        "data"    : records,
    }


# =============================================================================
# TRADING TIME-SERIES — Chart.js interactive cumulative return chart
# =============================================================================

@router.get("/trading/{ticker}/{currency}/timeseries",
            summary="Trading cumulative return time-series for Chart.js")
def get_trading_timeseries(
    ticker  : str,
    currency: str,
    fold_idx: int = Query(3, ge=1, le=3),
) -> dict:
    """
    Trả về chuỗi thời gian Strategy_Cumulative + BuyHold_Cumulative
    cho tất cả 5 models × 2 wavelet conditions.

    Khác với /trading/{t}/{c} chỉ trả summary metrics, endpoint này trả
    toàn bộ chuỗi thời gian để frontend vẽ Chart.js line chart tương tác.

    Returns:
        {"ticker", "currency", "fold_idx",
         "data": [
           {"model": str, "use_wavelet": bool,
            "dates":    list[str],   ← F_W dates (ISO "YYYY-MM-DD")
            "strategy": list[float], ← Strategy_Cumulative (1.0 = initial capital)
            "buyhold":  list[float], ← BuyHold_Cumulative
           }, ...
         ]}
    """
    import numpy as np
    import pandas as pd

    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.config import MODELS, FOLDS
    from app.services.trading_service import _load_processed_df, simulate_trading_weekly

    # Validate fold
    fold_map = {f["fold_id"]: f for f in FOLDS}
    if fold_idx not in fold_map:
        raise HTTPException(
            status_code=400,
            detail=f"fold_idx={fold_idx} không hợp lệ. Chọn: {list(fold_map.keys())}",
        )

    results = []

    for use_wavelet in [True, False]:
        cond_str = "wavelet" if use_wavelet else "nowave"
        pkl_path = Path(PATHS["processed"]) / f"{ticker}_{currency}_{cond_str}.pkl"

        # Load processed DataFrame to get Close price series
        try:
            df_full      = _load_processed_df(pkl_path)
            close_series = df_full["Close"].astype(np.float64)
        except Exception as exc:
            logger.warning("timeseries: cannot load pkl %s — %s", pkl_path.name, exc)
            continue

        for model_name in MODELS:
            exp_id   = f"{ticker}_{currency}_{cond_str}_{model_name}_classification"
            npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"

            if not npz_path.exists():
                logger.debug("timeseries: npz missing %s", npz_path)
                continue

            try:
                npz_data = np.load(str(npz_path), allow_pickle=False)

                # Weekly Task B predictions must have "dates" key
                if "dates" not in npz_data.files:
                    logger.warning("timeseries: no 'dates' key in %s", npz_path)
                    continue

                y_pred    = npz_data["y_pred"].astype(np.int32).flatten()
                f_w_dates = pd.DatetimeIndex(npz_data["dates"])

                # Run trading simulation (weekly, Phương án D)
                trade_df = simulate_trading_weekly(
                    y_pred          = y_pred,
                    f_w_dates       = f_w_dates.values,
                    close_series    = close_series,
                    initial_capital = 1.0,
                )

                results.append({
                    "model"      : model_name,
                    "use_wavelet": use_wavelet,
                    # F_W dates as ISO strings for Chart.js X axis
                    "dates"   : [str(d.date()) for d in trade_df["Date"]],
                    # Cumulative factor (1.0 = no gain/loss)
                    "strategy": [round(float(v), 6) for v in trade_df["Strategy_Cumulative"]],
                    "buyhold" : [round(float(v), 6) for v in trade_df["BuyHold_Cumulative"]],
                })

            except Exception as exc:
                logger.warning("timeseries: error %s %s fold%d — %s",
                               model_name, cond_str, fold_idx, exc)

    return {
        "ticker"  : ticker,
        "currency": currency,
        "fold_idx": fold_idx,
        "data"    : results,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ticker(ticker: str) -> None:
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"ticker='{ticker}' không hợp lệ. Chọn từ: {TICKERS}",
        )

def _validate_currency(currency: str) -> None:
    if currency not in CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"currency='{currency}' không hợp lệ. Chọn từ: {CURRENCIES}",
        )