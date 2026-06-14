"""
app/api/data.py
================
API routes cho Data Management — /api/data/*

Endpoints:
  GET  /api/data/status                  → kiểm tra CSV/PKL files tồn tại chưa
  GET  /api/data/{ticker}/raw            → xem raw OHLCV data (JSON)
  GET  /api/data/{ticker}/features       → xem features sau preprocessing
  GET  /api/data/{ticker}/deviation-plot → Fig. 2: Deviation scatter plot (base64 PNG)
  POST /api/data/preprocess              → trigger preprocessing pipeline

Tham chiếu:
  Li et al., EAAI 2026 — Section 3.2: Data preparation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import CURRENCIES, PATHS, TICKERS

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Paths ──────────────────────────────────────────────────────────────────────
_RAW_DIR       = Path(PATHS["raw"])
_PROCESSED_DIR = Path(PATHS["processed"])


# =============================================================================
# STATUS — kiểm tra files tồn tại
# =============================================================================

@router.get("/status", summary="Kiểm tra trạng thái data files")
def get_data_status() -> dict:
    """
    Kiểm tra sự tồn tại của tất cả raw CSV và processed PKL files.

    Returns:
        dict:
          "raw":       {"VCB": bool, "VIC": bool}
          "processed": {"VCB_VND_wavelet": bool, ..., "VIC_VND_nowave": bool}
          "all_raw_ready":       bool — tất cả raw CSVs tồn tại
          "all_processed_ready": bool — tất cả 4 PKLs tồn tại
    """
    raw_status = {
        ticker: (_RAW_DIR / f"{ticker}_raw.csv").exists()
        for ticker in TICKERS
    }

    processed_status = {}
    for ticker in TICKERS:
        for currency in CURRENCIES:  # ["VND"] only
            for cond in ["wavelet", "nowave"]:
                key = f"{ticker}_{currency}_{cond}"
                processed_status[key] = (_PROCESSED_DIR / f"{key}.pkl").exists()

    return {
        "raw"                : raw_status,
        "processed"          : processed_status,
        "all_raw_ready"      : all(raw_status.values()),
        "all_processed_ready": all(processed_status.values()),
    }


# =============================================================================
# RAW DATA PREVIEW
# =============================================================================

@router.get("/{ticker}/raw", summary="Xem raw OHLCV data")
def get_raw_data(
    ticker    : str,
    currency  : str = Query("VND", description="VND"),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date  : Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit     : int = Query(100, ge=1, le=5000, description="Số rows trả về"),
) -> dict:
    """
    Trả về raw OHLCV data dạng JSON, có filter theo ngày và limit.

    Args:
        ticker:     "VCB" hoặc "VIC".
        currency:   "VND" (duy nhất hỗ trợ).
        start_date: Ngày bắt đầu (YYYY-MM-DD), optional.
        end_date:   Ngày kết thúc (YYYY-MM-DD), optional.
        limit:      Số rows tối đa trả về (1–5000, default 100).

    Returns:
        {"ticker": str, "currency": str, "n_rows": int, "data": [...]}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    import pandas as pd

    csv_path = _RAW_DIR / f"{ticker}_raw.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"{ticker}_raw.csv không tồn tại. Chạy scripts/download_data.py trước.",
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Filter theo ngày
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # Limit + convert về JSON-serializable
    df = df.tail(limit)
    records = df.reset_index().rename(columns={"index": "Date", "time": "Date"})
    records["Date"] = records["Date"].astype(str)

    return {
        "ticker"  : ticker,
        "currency": currency,
        "n_rows"  : len(records),
        "data"    : records.to_dict(orient="records"),
    }


# =============================================================================
# FEATURES PREVIEW
# =============================================================================

@router.get("/{ticker}/features", summary="Xem features sau preprocessing")
def get_features(
    ticker     : str,
    currency   : str  = Query("VND", description="VND"),
    use_wavelet: bool = Query(True,  description="True = wavelet features"),
    limit      : int  = Query(50,   ge=1, le=1000),
) -> dict:
    """
    Trả về processed features từ data/processed/ PKL file.

    Returns:
        {"ticker": str, "currency": str, "use_wavelet": bool,
         "feature_names": list[str], "n_rows": int, "sample": [...]}
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    import pickle

    cond     = "wavelet" if use_wavelet else "nowave"
    pkl_path = _PROCESSED_DIR / f"{ticker}_{currency}_{cond}.pkl"

    if not pkl_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"{pkl_path.name} không tồn tại. Chạy scripts/preprocess.py trước.",
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # PKL stores dict {"df": DataFrame, "feature_cols": list}
    df = data["df"] if isinstance(data, dict) else data

    sample = df.tail(limit).reset_index()
    sample.columns = [str(c) for c in sample.columns]

    # Convert Timestamp → string cho JSON
    for col in sample.select_dtypes(include=["datetime64"]).columns:
        sample[col] = sample[col].astype(str)

    return {
        "ticker"        : ticker,
        "currency"      : currency,
        "use_wavelet"   : use_wavelet,
        "feature_names" : list(df.columns),
        "n_rows"        : len(df),
        "date_range"    : {
            "start": str(df.index[0].date()),
            "end"  : str(df.index[-1].date()),
        },
        "sample"        : sample.to_dict(orient="records"),
    }


# =============================================================================
# DEVIATION SCATTER PLOT  — Fig. 2 bài báo
# =============================================================================

@router.get("/{ticker}/deviation-plot", summary="Fig. 2: Deviation vs Close scatter plot")
def get_deviation_plot(
    ticker  : str,
    currency: str = Query("VND", description="VND"),
) -> dict:
    """
    Tạo scatter plot Deviation (Close − Open) vs Close Price — replica Fig. 2 bài báo.

    Dữ liệu lấy từ raw CSV (giá gốc chưa scale) để đảm bảo tỉ lệ chính xác.

    Args:
        ticker:   "VCB" hoặc "VIC".
        currency: "VND" (duy nhất hỗ trợ).

    Returns:
        {"image": "data:image/png;base64,...", "ticker": str, "currency": str}

    Raises:
        404: Nếu raw CSV chưa tồn tại.
    """
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — không mở cửa sổ GUI
    import matplotlib.pyplot as plt
    import pandas as pd

    _validate_ticker(ticker)
    _validate_currency(currency)

    # ── Load raw data ─────────────────────────────────────────────────────────
    csv_path = _RAW_DIR / f"{ticker}_raw.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"{ticker}_raw.csv không tồn tại. Chạy scripts/download_data.py trước.",
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # ── Tính Deviation ────────────────────────────────────────────────────────
    # Deviation = Close - Open: đo áp lực mua/bán trong phiên
    if "Close" not in df.columns or "Open" not in df.columns:
        raise HTTPException(
            status_code = 422,
            detail      = "CSV thiếu cột 'Close' hoặc 'Open'.",
        )
    df = df.dropna(subset=["Close", "Open"])
    df["Deviation"] = df["Close"] - df["Open"]

    # ── Plot scatter ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df["Close"], df["Deviation"],
        alpha      = 0.35,
        s          = 10,
        color      = "#1565C0",
        edgecolors = "none",
    )

    # Đường zero (deviation = 0) để dễ đọc
    ax.axhline(y=0, color="#E53935", linewidth=1.0, linestyle="--", alpha=0.7,
               label="Deviation = 0")

    ax.set_title(
        f"{ticker} — Deviation Changes as Stock Price Increases (VND)",
        fontsize   = 13,
        fontweight = "bold",
    )
    ax.set_xlabel("Close Price (VND)", fontsize=11)
    ax.set_ylabel("Deviation (Close − Open)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    # ── Encode → base64 ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return {
        "image"   : f"data:image/png;base64,{b64}",
        "ticker"  : ticker,
        "currency": currency,
    }


# =============================================================================
# PREPROCESS TRIGGER
# =============================================================================

@router.post("/preprocess", summary="Trigger preprocessing pipeline")
def trigger_preprocess(
    ticker     : str,
    currency   : str  = "VND",
    use_wavelet: bool = True,
) -> dict:
    """
    Chạy preprocessing pipeline cho một (ticker, wavelet) combination.

    Gọi trực tiếp scripts/preprocess.py pipeline qua subprocess.
    Lưu kết quả ra data/processed/{ticker}_{currency}_{cond}.pkl.

    Returns:
        {"status": "done", "ticker": str, "currency": str,
         "use_wavelet": bool, "n_features": int, "n_rows": int}

    Note:
        Chạy đồng bộ (synchronous) — với dataset lớn có thể mất 1–2 giây.
    """
    _validate_ticker(ticker)
    _validate_currency(currency)

    import pickle
    import subprocess
    import sys

    cond_arg = "true" if use_wavelet else "false"

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.preprocess",
                "--ticker",   ticker,
                "--currency", currency,
                "--wavelet",  cond_arg,
            ],
            capture_output = True,
            text           = True,
            timeout        = 120,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code = 500,
                detail      = f"Preprocessing thất bại: {result.stderr[:500]}",
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Verify output tồn tại
    cond     = "wavelet" if use_wavelet else "nowave"
    pkl_path = _PROCESSED_DIR / f"{ticker}_{currency}_{cond}.pkl"
    if not pkl_path.exists():
        raise HTTPException(status_code=500, detail="PKL file không được tạo.")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df = data["df"] if isinstance(data, dict) else data

    return {
        "status"     : "done",
        "ticker"     : ticker,
        "currency"   : currency,
        "use_wavelet": use_wavelet,
        "n_features" : len(df.columns),
        "n_rows"     : len(df),
    }


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ticker(ticker: str) -> None:
    """Raise 400 nếu ticker không hợp lệ."""
    if ticker not in TICKERS:
        raise HTTPException(
            status_code = 400,
            detail      = f"ticker='{ticker}' không hợp lệ. Chọn từ: {TICKERS}",
        )

def _validate_currency(currency: str) -> None:
    """Raise 400 nếu currency không hợp lệ."""
    if currency not in CURRENCIES:
        raise HTTPException(
            status_code = 400,
            detail      = f"currency='{currency}' không hợp lệ. Chọn từ: {CURRENCIES}",
        )