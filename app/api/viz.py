"""
app/api/viz.py
===============
API routes cho Visualization — /api/viz/*

Tất cả endpoints trả về base64-encoded PNG image:
  {"image": "data:image/png;base64,..."}

Các endpoints (sẽ được hoàn thiện ở Task 9.1/9.2):
  GET /api/viz/fig11           → Fig. 11: MSE comparison bar chart
  GET /api/viz/predicted-vs-actual → Predicted vs Actual price
  GET /api/viz/loss-curves     → Train/val loss curves
  GET /api/viz/confusion-matrix → Confusion matrix heatmap
  GET /api/viz/roc-curves      → ROC curves (5 models)
  GET /api/viz/walkforward     → Walk-forward stability chart
  GET /api/viz/trading-returns → Cumulative return chart

Tham chiếu:
  Li et al., EAAI 2026, Fig. 11: MSE comparison of different models.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import CURRENCIES, TICKERS

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# HELPERS
# =============================================================================

def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure → base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
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


# =============================================================================
# FIG. 11 — MSE Comparison Bar Chart
# =============================================================================

@router.get("/fig11", summary="Fig. 11: MSE comparison bar chart")
def get_fig11(
    ticker  : str = Query("VCB", description="VCB hoặc VIC"),
    currency: str = Query("VND", description="VND hoặc USD"),
) -> dict:
    """
    Tái hiện Fig. 11 từ bài báo: MSE comparison của 5 models × Before/After wavelet.

    Returns:
        {"image": "data:image/png;base64,...", "ticker": str, "currency": str}
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (không mở cửa sổ GUI)
    import matplotlib.pyplot as plt
    import numpy as np

    _validate_ticker(ticker)
    _validate_currency(currency)

    from app.services.evaluation_service import load_all_results, create_comparison_table

    df = load_all_results()
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có kết quả. Chạy experiments trước.")

    tbl = create_comparison_table(df, ticker=ticker, currency=currency, task="regression")
    if tbl.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Không có data cho {ticker}/{currency} regression.",
        )

    models  = list(tbl.index)
    x       = np.arange(len(models))
    width   = 0.35

    # Lấy MSE values
    before_mse = tbl[("Before Wavelet", "MSE")].fillna(0).tolist()
    after_mse  = tbl[("After Wavelet",  "MSE")].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before_mse, width, label="Before Wavelet",
                   color="#4472C4", alpha=0.85)
    bars2 = ax.bar(x + width/2, after_mse,  width, label="After Wavelet",
                   color="#70AD47", alpha=0.85)

    ax.set_title(f"MSE Comparison — {ticker} ({currency})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {"image": image_data, "ticker": ticker, "currency": currency}


# =============================================================================
# PREDICTED vs ACTUAL (Regression)
# =============================================================================

@router.get("/predicted-vs-actual", summary="Predicted vs Actual price chart")
def get_predicted_vs_actual(
    exp_id  : str = Query(..., description="Experiment ID (regression task)"),
    fold_idx: int = Query(3, ge=1, le=3),
) -> dict:
    """
    Line chart: Actual Close price vs Predicted Close price trên test set.

    Returns:
        {"image": "data:image/png;base64,...", "exp_id": str, "fold_idx": int}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    npz_path = Path(TICKERS[0]).parent / ".." / "experiments" / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    # Dùng PATHS để lấy đúng path
    from app.config import PATHS
    npz_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "predictions.npz"

    if not npz_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"predictions.npz không tồn tại: {exp_id}/fold_{fold_idx}",
        )

    data   = np.load(str(npz_path), allow_pickle=False)
    y_true = data["y_true"].flatten()
    y_pred = data["y_pred"].flatten()

    # Xác định task từ exp_id
    if "classification" in exp_id:
        raise HTTPException(
            status_code=400,
            detail="Endpoint này chỉ dành cho regression experiments.",
        )

    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y_true, label="Actual",    color="#2196F3", linewidth=1.2, alpha=0.9)
    ax.plot(x, y_pred, label="Predicted", color="#F44336", linewidth=1.0, alpha=0.85,
            linestyle="--")

    # Xác định unit từ exp_id
    currency_label = "USD" if "_USD_" in exp_id else "VND"
    ax.set_title(f"Predicted vs Actual — {exp_id} | Fold {fold_idx}", fontsize=13)
    ax.set_xlabel("Test Day Index", fontsize=11)
    ax.set_ylabel(f"Close Price ({currency_label})", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {"image": image_data, "exp_id": exp_id, "fold_idx": fold_idx}


# =============================================================================
# LOSS CURVES
# =============================================================================

@router.get("/loss-curves", summary="Train/val loss curves")
def get_loss_curves_viz(
    exp_id  : str = Query(...),
    fold_idx: int = Query(1, ge=1, le=3),
) -> dict:
    """
    Line chart: Train loss vs Val loss per epoch, với vertical line tại best epoch.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path
    from app.config import PATHS

    json_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "metrics.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"metrics.json không tồn tại.")

    with open(json_path, "r") as f:
        data = json.load(f)

    history      = data.get("train_history", {})
    train_losses = history.get("train_losses", [])
    val_losses   = history.get("val_losses",   [])
    best_epoch   = history.get("best_epoch",   None)

    if not train_losses:
        raise HTTPException(status_code=404, detail="Loss history rỗng.")

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val Loss",   color="#F44336", linewidth=1.5,
            linestyle="--")

    if best_epoch:
        ax.axvline(x=best_epoch, color="#4CAF50", linestyle=":", linewidth=2,
                   label=f"Best Epoch ({best_epoch})")

    ax.set_title(f"Loss Curves — {exp_id} | Fold {fold_idx}", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {"image": image_data, "exp_id": exp_id, "fold_idx": fold_idx}


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

@router.get("/confusion-matrix", summary="Confusion matrix heatmap")
def get_confusion_matrix_viz(
    exp_id  : str = Query(..., description="Classification experiment ID"),
    fold_idx: int = Query(3, ge=1, le=3),
) -> dict:
    """
    2×2 confusion matrix heatmap cho classification experiment.

    Returns:
        {"image": "data:image/png;base64,...", "matrix": [[TN,FP],[FN,TP]]}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import confusion_matrix
    from app.config import PATHS

    if "classification" not in exp_id:
        raise HTTPException(
            status_code=400,
            detail="Endpoint này chỉ dành cho classification experiments.",
        )

    npz_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="predictions.npz không tồn tại.")

    data   = np.load(str(npz_path), allow_pickle=False)
    y_true = data["y_true"].flatten().astype(int)
    y_pred = data["y_pred"].flatten().astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["DOWN (0)", "UP (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {exp_id.split('_')[3]}\nFold {fold_idx}", fontsize=13)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({100*cm[i,j]/cm.sum():.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {"image": image_data, "exp_id": exp_id, "fold_idx": fold_idx,
            "matrix": cm.tolist()}


# =============================================================================
# CUMULATIVE RETURNS (Trading)
# =============================================================================

@router.get("/trading-returns", summary="Cumulative return chart — all models vs Buy&Hold")
def get_trading_returns_viz(
    ticker  : str = Query("VCB"),
    currency: str = Query("VND"),
    fold_idx: int = Query(3, ge=1, le=3),
    wavelet : bool = Query(True),
) -> dict:
    """
    Multi-line cumulative return chart: 5 models + Buy & Hold baseline.

    Returns:
        {"image": "data:image/png;base64,..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from pathlib import Path
    from app.config import MODELS, PATHS, FOLDS
    from app.services.trading_service import simulate_trading

    _validate_ticker(ticker)
    _validate_currency(currency)

    cond     = "wavelet" if wavelet else "nowave"
    fold_def = next((f for f in FOLDS if f["fold_id"] == fold_idx), None)
    if fold_def is None:
        raise HTTPException(status_code=400, detail=f"fold_idx={fold_idx} không hợp lệ.")

    import pandas as pd
    test_start = pd.Timestamp(fold_def["test_start"])
    test_end   = pd.Timestamp(fold_def["test_end"])

    pkl_path = Path(PATHS["processed"]) / f"{ticker}_{currency}_{cond}.pkl"
    if not pkl_path.exists():
        raise HTTPException(status_code=404, detail="Processed pkl không tồn tại.")

    with open(pkl_path, "rb") as f:
        df_full = pickle.load(f)

    df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
    close_arr = df_test["Close"].values.astype(float)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    buyhold_plotted = False

    for idx, model_name in enumerate(MODELS):
        exp_id   = f"{ticker}_{currency}_{cond}_{model_name}_classification"
        npz_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "predictions.npz"
        if not npz_path.exists():
            continue

        data   = np.load(str(npz_path), allow_pickle=False)
        y_pred = data["y_pred"].flatten().astype(int)
        n_pred = len(y_pred)
        T      = len(close_arr)
        if n_pred >= T:
            continue

        seq_len  = T - n_pred
        act_prices = close_arr[seq_len - 1:]   # (N+1,)
        dates      = df_test.index[seq_len - 1 : seq_len - 1 + n_pred]

        trade_df = simulate_trading(y_pred, act_prices, dates=dates.values)

        ax.plot(
            range(len(trade_df)),
            (trade_df["Strategy_Cumulative"] - 1) * 100,
            label=model_name, color=colors[idx], linewidth=1.5,
        )

        # Buy & Hold — plot một lần
        if not buyhold_plotted:
            ax.plot(
                range(len(trade_df)),
                (trade_df["BuyHold_Cumulative"] - 1) * 100,
                label="Buy & Hold", color="#607D8B", linewidth=2,
                linestyle="--", alpha=0.8,
            )
            buyhold_plotted = True

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(
        f"Cumulative Returns — {ticker} ({currency}) | {cond} | Fold {fold_idx}",
        fontsize=13,
    )
    ax.set_xlabel("Trading Day Index", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    image_data = _fig_to_base64(fig)
    plt.close(fig)

    return {"image": image_data, "ticker": ticker, "currency": currency,
            "fold_idx": fold_idx, "wavelet": wavelet}