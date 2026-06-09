"""
app/api/viz.py
===============
API routes cho Visualization — /api/viz/*

Tất cả endpoints trả về base64-encoded PNG:
  {"image": "data:image/png;base64,..."}

Paper figure routes (gọi viz_service.py):
  GET /api/viz/fig1     → Pipeline framework diagram (static)
  GET /api/viz/fig2     → Deviation scatter plot
  GET /api/viz/fig3     → Feature distribution histograms
  GET /api/viz/fig4     → Scaling flowchart (static)
  GET /api/viz/fig5     → Wavelet decomposition time series
  GET /api/viz/fig6     → db4 wavelet & scaling functions (static)
  GET /api/viz/fig7     → Approximation coefficients multi-panel
  GET /api/viz/fig8     → Detail coefficients multi-panel
  GET /api/viz/fig9     → Level-1 decomposition diagram (static)
  GET /api/viz/fig10    → Correlation matrix heatmap
  GET /api/viz/fig11    → MSE comparison bar chart (replica bài báo)

Extended routes:
  GET /api/viz/predicted-vs-actual → Predicted vs Actual price chart
  GET /api/viz/loss-curves         → Train/val loss curves
  GET /api/viz/confusion-matrix    → Confusion matrix heatmap (seaborn style)
  GET /api/viz/trading-returns     → Cumulative return multi-line chart
  GET /api/viz/walkforward         → Walk-forward stability chart
  GET /api/viz/roc-curves          → Multi-model ROC curves overlay

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.config import CURRENCIES, FOLDS, MODELS, PATHS, TICKERS

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Paths ─────────────────────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(PATHS["experiments"])
_PROCESSED_DIR   = Path(PATHS["processed"])


# =============================================================================
# HELPERS
# =============================================================================

def _fig_to_base64(fig) -> str:
    """Convert matplotlib Figure → base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


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


def _load_pkl(ticker: str, currency: str, use_wavelet: bool) -> tuple:
    """
    Load processed pkl file → (DataFrame, feature_cols list).

    PKL structure: {"df": DataFrame, "feature_cols": list, "target_col": str}

    Args:
        ticker, currency: Xác định file.
        use_wavelet:      True → wavelet pkl, False → nowave pkl.

    Returns:
        (df, feature_cols)

    Raises:
        HTTPException 404: Nếu pkl không tồn tại.
    """
    suffix   = "wavelet" if use_wavelet else "nowave"
    pkl_path = _PROCESSED_DIR / f"{ticker}_{currency}_{suffix}.pkl"

    if not pkl_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Processed pkl không tồn tại: {pkl_path.name}. "
                "Chạy scripts/preprocess.py trước."
            ),
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # PKL là dict: {"df": DataFrame, "feature_cols": [...], "target_col": "Close"}
    if isinstance(data, dict):
        df           = data["df"]
        feature_cols = data.get("feature_cols", [c for c in df.columns if c != "Close"])
    else:
        # Backward compat nếu pkl là DataFrame trực tiếp
        df           = data
        feature_cols = [c for c in df.columns if c != "Close"]

    return df, feature_cols


def _load_npz(exp_id: str, fold_idx: int) -> dict:
    """
    Load predictions.npz cho một experiment fold.

    Returns:
        Dict với keys: y_true, y_pred, y_prob (nếu có).

    Raises:
        HTTPException 404: Nếu file không tồn tại.
    """
    npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"predictions.npz không tồn tại: {exp_id}/fold_{fold_idx}",
        )
    raw = np.load(str(npz_path), allow_pickle=False)
    result = {k: raw[k] for k in raw.files}
    return result


def _load_metrics_json(exp_id: str, fold_idx: int) -> dict:
    """
    Load metrics.json cho một experiment fold.

    Raises:
        HTTPException 404: Nếu file không tồn tại.
    """
    json_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "metrics.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"metrics.json không tồn tại: {exp_id}/fold_{fold_idx}",
        )
    with open(json_path, "r") as f:
        return json.load(f)


# =============================================================================
# PAPER FIGURE ROUTES — FIG. 1 (static)
# =============================================================================

@router.get("/fig1", summary="Fig. 1: Pipeline framework diagram")
def get_fig1() -> dict:
    """
    Sơ đồ kiến trúc pipeline đầy đủ (static diagram, không cần params).

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    from app.services.viz_service import fig1_pipeline_framework
    try:
        fig = fig1_pipeline_framework()
    except Exception as exc:
        logger.error("fig1 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig1: {exc}")
    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 2 — Deviation Scatter Plot
# =============================================================================

@router.get("/fig2", summary="Fig. 2: Deviation scatter plot")
def get_fig2(
    ticker  : str = Query("VCB", description="VCB hoặc VIC"),
    currency: str = Query("VND", description="VND hoặc USD"),
) -> dict:
    """
    Scatter: Close Price (X) vs Deviation (Y).
    Load từ nowave pkl (có cột Close và Deviation gốc).

    Returns:
        {"image": "data:image/png;base64,...", "ticker": str, "currency": str}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig2_deviation_scatter

    # Dùng nowave pkl — giữ raw features bao gồm Close và Deviation
    df, _ = _load_pkl(ticker, currency, use_wavelet=False)

    try:
        fig = fig2_deviation_scatter(df, ticker)
    except Exception as exc:
        logger.error("fig2 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig2: {exc}")

    return {"image": _fig_to_base64(fig), "ticker": ticker, "currency": currency}


# =============================================================================
# FIG. 3 — Feature Distributions
# =============================================================================

@router.get("/fig3", summary="Fig. 3: Wavelet feature distribution histograms")
def get_fig3(
    ticker  : str = Query("VCB"),
    currency: str = Query("VND"),
) -> dict:
    """
    2×n grid histograms + KDE cho Approx và Detail wavelet features.
    Bimodal distribution do tăng trưởng giá theo thời gian.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig3_feature_distributions

    df, _ = _load_pkl(ticker, currency, use_wavelet=True)

    try:
        fig = fig3_feature_distributions(df, ticker)
    except Exception as exc:
        logger.error("fig3 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig3: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 4 — Scaling Flowchart (static)
# =============================================================================

@router.get("/fig4", summary="Fig. 4: Feature scaling flowchart")
def get_fig4() -> dict:
    """
    Flowchart phân loại features theo scaler:
    Normal → Standard Scaler; Skewed/Outliers → Robust Scaler; Trend → No Scale.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    from app.services.viz_service import fig4_scaling_flowchart
    try:
        fig = fig4_scaling_flowchart()
    except Exception as exc:
        logger.error("fig4 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig4: {exc}")
    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 5 — Wavelet Decomposition Time Series
# =============================================================================

@router.get("/fig5", summary="Fig. 5: Wavelet decomposition visualization")
def get_fig5(
    ticker  : str = Query("VCB"),
    currency: str = Query("VND"),
) -> dict:
    """
    Time series của Deviation (và A1/D1 nếu có).
    Minh họa SWT decomposition theo Fig. 5 bài báo.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig5_wavelet_decomposition

    # Thử wavelet pkl trước (có Deviation_Approx, Deviation_Detail)
    try:
        df, _ = _load_pkl(ticker, currency, use_wavelet=True)
        # Lấy thêm Deviation gốc từ nowave pkl
        df_raw, _ = _load_pkl(ticker, currency, use_wavelet=False)
        if "Deviation" in df_raw.columns:
            df["Deviation"] = df_raw["Deviation"].reindex(df.index)
    except HTTPException:
        # Fallback: dùng nowave pkl
        df, _ = _load_pkl(ticker, currency, use_wavelet=False)

    try:
        fig = fig5_wavelet_decomposition(df, ticker)
    except Exception as exc:
        logger.error("fig5 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig5: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 6 — Wavelet & Scaling Functions (static, pywt)
# =============================================================================

@router.get("/fig6", summary="Fig. 6: db4 wavelet and scaling functions")
def get_fig6() -> dict:
    """
    db4 wavelet function ψ (left) và scaling function φ (right).
    Computed với pywt.Wavelet('db4').wavefun(level=10).

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    from app.services.viz_service import fig6_wavelet_functions
    try:
        fig = fig6_wavelet_functions()
    except Exception as exc:
        logger.error("fig6 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig6: {exc}")
    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 7 — Approximation Coefficients
# =============================================================================

@router.get("/fig7", summary="Fig. 7: Approximation coefficients (A1) — VIC analog")
def get_fig7(
    ticker  : str = Query("VIC", description="VIC (Tesla analog trong bài báo)"),
    currency: str = Query("VND"),
) -> dict:
    """
    Multi-panel time series của tất cả *_Approx features cho ticker.
    VIC ↔ Tesla trong bài báo (Fig. 7).

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig7_approx_coefficients

    df, _ = _load_pkl(ticker, currency, use_wavelet=True)

    try:
        fig = fig7_approx_coefficients(df, ticker)
    except Exception as exc:
        logger.error("fig7 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig7: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 8 — Detail Coefficients
# =============================================================================

@router.get("/fig8", summary="Fig. 8: Detail coefficients (D1) — VCB analog")
def get_fig8(
    ticker  : str = Query("VCB", description="VCB (Apple analog trong bài báo)"),
    currency: str = Query("VND"),
) -> dict:
    """
    Multi-panel time series của tất cả *_Detail features cho ticker.
    VCB ↔ Apple trong bài báo (Fig. 8).

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig8_detail_coefficients

    df, _ = _load_pkl(ticker, currency, use_wavelet=True)

    try:
        fig = fig8_detail_coefficients(df, ticker)
    except Exception as exc:
        logger.error("fig8 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig8: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 9 — Level-1 Decomposition Diagram (static)
# =============================================================================

@router.get("/fig9", summary="Fig. 9: Level-1 SWT decomposition diagram")
def get_fig9() -> dict:
    """
    Sơ đồ SWT level-1: s(t) → [H] → A1 (Approx), s(t) → [G] → D1 (Detail).

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    from app.services.viz_service import fig9_level1_decomposition
    try:
        fig = fig9_level1_decomposition()
    except Exception as exc:
        logger.error("fig9 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig9: {exc}")
    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 10 — Correlation Matrix Heatmap
# =============================================================================

@router.get("/fig10", summary="Fig. 10: Feature correlation matrix heatmap")
def get_fig10(
    ticker  : str  = Query("VCB"),
    currency: str  = Query("VND"),
    wavelet : bool = Query(True, description="True = wavelet features, False = raw features"),
) -> dict:
    """
    Heatmap correlation matrix sau wavelet transformation + feature selection.
    Annotate giá trị; cho thấy features đã loại có corr > 0.95.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig10_correlation_matrix

    df, _ = _load_pkl(ticker, currency, use_wavelet=wavelet)

    try:
        fig = fig10_correlation_matrix(df, ticker)
    except Exception as exc:
        logger.error("fig10 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig10: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# FIG. 11 — MSE Comparison Bar Chart (replica bài báo)
# =============================================================================

@router.get("/fig11", summary="Fig. 11: MSE comparison bar chart (5 models × Before/After wavelet)")
def get_fig11(
    ticker  : str = Query("VCB", description="VCB hoặc VIC"),
    currency: str = Query("VND", description="VND hoặc USD"),
) -> dict:
    """
    Tái hiện Fig. 11: Bar chart MSE của 5 models × Before Wavelet vs After Wavelet.

    Dữ liệu: mean MSE across 3 folds từ experiment results.
    Auto log scale nếu MSE range > 100×.

    Returns:
        {"image": "data:image/png;base64,...", "ticker": str, "currency": str}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig11_mse_comparison
    try:
        fig = fig11_mse_comparison(ticker, currency)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("fig11 error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate fig11: {exc}")

    return {"image": _fig_to_base64(fig), "ticker": ticker, "currency": currency}


# =============================================================================
# EXTENDED: PREDICTED vs ACTUAL (Regression)
# =============================================================================

@router.get("/predicted-vs-actual", summary="Predicted vs Actual price chart")
def get_predicted_vs_actual(
    exp_id  : str = Query(..., description="Experiment ID (regression task)"),
    fold_idx: int = Query(3, ge=1, le=3),
) -> dict:
    """
    Line chart: Actual Close price vs Predicted Close price trên test set.
    Chỉ hợp lệ cho regression experiments.

    Returns:
        {"image": "data:image/png;base64,...", "exp_id": str, "fold_idx": int}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "classification" in exp_id:
        raise HTTPException(
            status_code=400,
            detail="Endpoint này chỉ dành cho regression experiments.",
        )

    npz_data = _load_npz(exp_id, fold_idx)
    y_true   = npz_data["y_true"].flatten()
    y_pred   = npz_data["y_pred"].flatten()

    currency_label = "USD" if "_USD_" in exp_id else "VND"
    x = np.arange(len(y_true))

    from app.services.viz_service import DARK_RC
    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(x, y_true, label="Actual",    color="#64b5f6", linewidth=1.2, alpha=0.9)
        ax.plot(x, y_pred, label="Predicted", color="#e94560", linewidth=1.0, alpha=0.85,
                linestyle="--")
        ax.set_title(f"Predicted vs Actual — {exp_id} | Fold {fold_idx}", fontsize=13)
        ax.set_xlabel("Test Day Index", fontsize=11)
        ax.set_ylabel(f"Close Price ({currency_label})", fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return {
        "image"   : _fig_to_base64(fig),
        "exp_id"  : exp_id,
        "fold_idx": fold_idx,
    }


# =============================================================================
# EXTENDED: LOSS CURVES
# =============================================================================

@router.get("/loss-curves", summary="Train/val loss curves per epoch")
def get_loss_curves_viz(
    exp_id  : str = Query(..., description="Experiment ID"),
    fold_idx: int = Query(1, ge=1, le=3),
) -> dict:
    """
    Line chart: Train loss và Val loss per epoch.
    Vertical dashed line tại best epoch (early stopping).

    Returns:
        {"image": "data:image/png;base64,...", "exp_id": str, "fold_idx": int,
         "best_epoch": int | None}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_data = _load_metrics_json(exp_id, fold_idx)
    history      = metrics_data.get("train_history", {})
    train_losses = history.get("train_losses", [])
    val_losses   = history.get("val_losses",   [])
    best_epoch   = history.get("best_epoch",   None)

    if not train_losses:
        raise HTTPException(
            status_code=404,
            detail=f"Loss history rỗng trong metrics.json ({exp_id}/fold_{fold_idx})",
        )

    epochs = list(range(1, len(train_losses) + 1))

    from app.services.viz_service import DARK_RC
    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_losses, label="Train Loss",
                color="#64b5f6", linewidth=1.8)
        ax.plot(epochs, val_losses,   label="Val Loss",
                color="#e94560", linewidth=1.8, linestyle="--")

        if best_epoch:
            ax.axvline(x=best_epoch, color="#4caf50", linestyle=":",
                       linewidth=2.0, label=f"Best Epoch ({best_epoch})")
            # Annotate best val loss
            if best_epoch - 1 < len(val_losses):
                bv = val_losses[best_epoch - 1]
                ax.annotate(f"val={bv:.4f}",
                            xy=(best_epoch, bv),
                            xytext=(best_epoch + len(epochs) * 0.03, bv),
                            fontsize=8, color="#4caf50")

        ax.set_title(f"Loss Curves — {exp_id} | Fold {fold_idx}", fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return {
        "image"      : _fig_to_base64(fig),
        "exp_id"     : exp_id,
        "fold_idx"   : fold_idx,
        "best_epoch" : best_epoch,
    }


# =============================================================================
# EXTENDED: CONFUSION MATRIX (seaborn-style heatmap)
# =============================================================================

@router.get("/confusion-matrix", summary="Confusion matrix heatmap (classification)")
def get_confusion_matrix_viz(
    exp_id  : str = Query(..., description="Classification experiment ID"),
    fold_idx: int = Query(3, ge=1, le=3),
) -> dict:
    """
    2×2 confusion matrix heatmap.
    Annotate với count + percentage.

    Returns:
        {"image": "data:image/png;base64,...",
         "matrix": [[TN,FP],[FN,TP]],
         "exp_id": str, "fold_idx": int}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if "classification" not in exp_id:
        raise HTTPException(
            status_code=400,
            detail="Endpoint này chỉ dành cho classification experiments.",
        )

    npz_data = _load_npz(exp_id, fold_idx)
    y_true   = npz_data["y_true"].flatten().astype(int)
    y_pred   = npz_data["y_pred"].flatten().astype(int)
    cm       = confusion_matrix(y_true, y_pred)

    # Lấy model name từ exp_id để hiển thị
    parts      = exp_id.split("_")
    model_name = parts[3] if len(parts) > 3 else exp_id

    from app.services.viz_service import DARK_RC, C_BG, C_FG
    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(6, 5.5))

        # Custom 2-color imshow
        cmap = plt.cm.Blues
        im   = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                         vmin=0, vmax=cm.max() * 1.2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels = ["DOWN (0)", "UP (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — {model_name}\nFold {fold_idx}",
            fontsize=13,
        )

        # Annotate cells với count + percentage
        thresh = cm.max() / 2.0
        total  = cm.sum()
        for i in range(2):
            for j in range(2):
                val      = cm[i, j]
                pct      = 100 * val / total if total > 0 else 0
                txt_col  = "white" if val > thresh else C_FG
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=12,
                        color=txt_col, fontweight="bold")

        plt.tight_layout()

    return {
        "image"   : _fig_to_base64(fig),
        "matrix"  : cm.tolist(),
        "exp_id"  : exp_id,
        "fold_idx": fold_idx,
    }


# =============================================================================
# EXTENDED: CUMULATIVE RETURNS (Trading Simulation)
# =============================================================================

@router.get("/trading-returns", summary="Cumulative return chart — 5 models vs Buy & Hold")
def get_trading_returns_viz(
    ticker  : str  = Query("VCB"),
    currency: str  = Query("VND"),
    fold_idx: int  = Query(3, ge=1, le=3),
    wavelet : bool = Query(True),
) -> dict:
    """
    Multi-line cumulative return chart: 5 models + Buy & Hold baseline.

    Load actual prices từ processed pkl (đã fix: pkl là dict, bắt buộc ["df"]).
    Load predictions từ experiments/{exp_id}/fold_{n}/predictions.npz.

    Returns:
        {"image": "data:image/png;base64,...",
         "ticker": str, "currency": str, "fold_idx": int, "wavelet": bool}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.trading_service import simulate_trading
    from app.services.viz_service import DARK_RC, MODEL_PALETTE

    cond     = "wavelet" if wavelet else "nowave"
    fold_def = next((f for f in FOLDS if f["fold_id"] == fold_idx), None)
    if fold_def is None:
        raise HTTPException(status_code=400,
                            detail=f"fold_idx={fold_idx} không hợp lệ.")

    test_start = pd.Timestamp(fold_def["test_start"])
    test_end   = pd.Timestamp(fold_def["test_end"])

    # Load pkl — FIXED: pkl là dict {"df": ..., ...}
    df_full, _ = _load_pkl(ticker, currency, use_wavelet=wavelet)

    df_test   = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
    close_arr = df_test["Close"].values.astype(float)

    if len(close_arr) < 5:
        raise HTTPException(status_code=404,
                            detail="Không đủ dữ liệu giá để tạo chart.")

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(14, 6))
        buyhold_plotted = False

        for i, model_name in enumerate(MODELS):
            exp_id   = f"{ticker}_{currency}_{cond}_{model_name}_classification"
            npz_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"
            if not npz_path.exists():
                continue

            try:
                raw_npz = np.load(str(npz_path), allow_pickle=False)
                y_pred  = raw_npz["y_pred"].flatten().astype(int)
            except Exception as exc:
                logger.warning("Không load được %s: %s", npz_path, exc)
                continue

            n_pred = len(y_pred)
            T      = len(close_arr)
            if n_pred >= T:
                # seq_len suy ra từ độ dài
                continue

            seq_len    = T - n_pred
            act_prices = close_arr[seq_len - 1:]          # shape (N+1,)
            dates      = df_test.index[seq_len - 1: seq_len - 1 + n_pred]

            trade_df = simulate_trading(y_pred, act_prices, dates=dates.values)
            color    = MODEL_PALETTE[i % len(MODEL_PALETTE)]

            ax.plot(
                range(len(trade_df)),
                (trade_df["Strategy_Cumulative"] - 1) * 100,
                label=model_name, color=color, linewidth=1.8,
            )

            # Buy & Hold — vẽ một lần dựa trên model đầu tiên thành công
            if not buyhold_plotted:
                ax.plot(
                    range(len(trade_df)),
                    (trade_df["BuyHold_Cumulative"] - 1) * 100,
                    label="Buy & Hold", color="#78909c", linewidth=2.0,
                    linestyle="--", alpha=0.85,
                )
                buyhold_plotted = True

        ax.axhline(y=0, color="#444466", linewidth=0.8, linestyle=":")
        ax.set_title(
            f"Cumulative Returns — {ticker} ({currency}) | {cond} | Fold {fold_idx}",
            fontsize=13,
        )
        ax.set_xlabel("Trading Day Index", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return {
        "image"   : _fig_to_base64(fig),
        "ticker"  : ticker,
        "currency": currency,
        "fold_idx": fold_idx,
        "wavelet" : wavelet,
    }


# =============================================================================
# EXTENDED: WALK-FORWARD STABILITY
# =============================================================================

@router.get("/walkforward", summary="Walk-forward stability chart — metric per fold")
def get_walkforward_stability(
    ticker  : str  = Query("VCB"),
    currency: str  = Query("VND"),
    task    : str  = Query("regression",
                           description="regression hoặc classification"),
    metric  : str  = Query("MSE",
                           description="MSE, MAE, MAPE, R2, Accuracy, F1, AUC_ROC"),
    wavelet : bool = Query(True),
) -> dict:
    """
    Line chart: metric per fold cho từng model.
    Cho thấy consistency và ổn định qua 3 folds.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import ext_walkforward_stability
    try:
        fig = ext_walkforward_stability(ticker, currency, task, metric, wavelet)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("walkforward error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate walkforward: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# EXTENDED: MULTI-MODEL ROC CURVES
# =============================================================================

@router.get("/roc-curves", summary="Multi-model ROC curves overlay")
def get_roc_curves_viz(
    ticker  : str  = Query("VCB"),
    currency: str  = Query("VND"),
    fold_idx: int  = Query(3, ge=1, le=3),
    wavelet : bool = Query(True),
) -> dict:
    """
    Overlay ROC curves của 5 models + diagonal reference.
    AUC value per model hiển thị trong legend.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import ext_roc_curves
    try:
        fig = ext_roc_curves(ticker, currency, fold_idx, wavelet)
    except Exception as exc:
        logger.error("roc_curves error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate roc-curves: {exc}")

    return {"image": _fig_to_base64(fig)}


# =============================================================================
# TASK 9.2 ROUTES — Classification Table & VND vs USD Table
# =============================================================================

@router.get("/fig-classification-table",
            summary="Classification metrics table (Accuracy/F1/AUC per model per fold)")
def get_fig_classification_table(
    ticker  : str  = Query("VCB"),
    currency: str  = Query("VND"),
    wavelet : bool = Query(True, description="True = After Wavelet, False = Before Wavelet"),
) -> dict:
    """
    Matplotlib table figure: Accuracy / F1 / AUC_ROC cho 5 models × 3 folds + Mean.
    Best model per metric được highlight xanh lá.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig_classification_table
    try:
        fig = fig_classification_table(ticker, currency, wavelet)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("fig_classification_table error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate table: {exc}")

    return {
        "image"   : _fig_to_base64(fig),
        "ticker"  : ticker,
        "currency": currency,
        "wavelet" : wavelet,
    }


@router.get("/fig-vnd-vs-usd",
            summary="VND vs USD side-by-side metric comparison table")
def get_fig_vnd_vs_usd_table(
    model_name: str = Query("BiLSTM",
                            description="DNN, RNN, GRU, LSTM, hoặc BiLSTM"),
) -> dict:
    """
    Matplotlib table figure: so sánh VND và USD side-by-side cho model chỉ định.

    Rows: (ticker, wavelet_condition, task)
    Columns: metric × (VND, USD, Δ USD−VND)
    Delta column: xanh = USD tốt hơn; đỏ = VND tốt hơn.

    Returns:
        {"image": "data:image/png;base64,...", "model_name": str}
    """
    from app.config import MODELS
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model_name='{model_name}' không hợp lệ. Chọn từ: {MODELS}",
        )

    from app.services.viz_service import fig_vnd_vs_usd_table
    try:
        fig = fig_vnd_vs_usd_table(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("fig_vnd_vs_usd_table error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate table: {exc}")

    return {"image": _fig_to_base64(fig), "model_name": model_name}


@router.get("/fig-walkforward",
            summary="Walk-forward stability chart cho một experiment cụ thể")
def get_fig_walkforward_stability(
    exp_id: str = Query(..., description="Experiment ID đầy đủ"),
) -> dict:
    """
    Bar chart metrics per fold cho một experiment_id cụ thể.
    Detect task (regression/classification) từ exp_id để chọn đúng metrics.

    Returns:
        {"image": "data:image/png;base64,...", "exp_id": str}
    """
    from app.services.viz_service import fig_walkforward_stability
    try:
        fig = fig_walkforward_stability(exp_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("fig_walkforward_stability error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate chart: {exc}")

    return {"image": _fig_to_base64(fig), "exp_id": exp_id}


@router.get("/fig-cumulative-return",
            summary="Cumulative return chart với Sharpe/MaxDD annotations (service-based)")
def get_fig_cumulative_return(
    ticker  : str  = Query("VCB"),
    currency: str  = Query("VND"),
    fold_idx: int  = Query(3, ge=1, le=3),
    wavelet : bool = Query(True),
) -> dict:
    """
    Multi-line cumulative return chart từ viz_service.py.
    X-axis: date; Y-axis: return (%); panel phụ: Sharpe/MaxDD table.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.viz_service import fig_cumulative_return
    try:
        fig = fig_cumulative_return(ticker, currency, fold_idx, wavelet)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("fig_cumulative_return error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi generate chart: {exc}")

    return {
        "image"   : _fig_to_base64(fig),
        "ticker"  : ticker,
        "currency": currency,
        "fold_idx": fold_idx,
        "wavelet" : wavelet,
    }