"""
app/services/trading_service.py
==================================
Task C — Trading Simulation cho VNSP.

=== CẬP NHẬT 2026-06 (Phương án D — Weekly, theo feedback giảng viên) ===
Task B (classification) đã chuyển sang dự đoán hướng đi TUẦN kế tiếp
(T2→T6, anchor 'W-FRI'), 1 sample / tuần. Task C được viết lại tương ứng:

  - Nguồn dữ liệu: predictions.npz của Task B mới, có thêm key "dates"
    = F_W (phiên giao dịch CUỐI CÙNG của tuần W — mốc dự đoán cho tuần W+1).
  - Với mỗi sample i:
      F_W      = dates[i]            (đã biết tại thời điểm dự đoán)
      F_{W+1}  = dates[i+1]           (tuần kế tiếp — LẤY TỪ chính dates,
                                        vì build_weekly_sequences() tạo
                                        các tuần liên tục)
      Riêng sample CUỐI (i = N-1): F_{W+1} không có trong `dates` (tuần đó
      bị build_weekly_sequences() bỏ vì không có F_{W+2} để tính label) —
      ta tự tìm F_{W+1} bằng cách lấy phiên cuối cùng của tuần kế tiếp
      trong df_full (processed pkl, có dữ liệu vượt ra ngoài test period).
      Nếu không còn dữ liệu nào sau F_W (cuối dataset) → bỏ sample cuối.
  - Strategy: predict UP (y_pred=1) cho tuần W+1
        → return_W = (Close(F_{W+1}) - Close(F_W)) / Close(F_W)
      predict DOWN (y_pred=0) → return_W = 0 (no shorting, đứng ngoài).
  - Sharpe Ratio: annualize ×√52 (TRADING_WEEKS_PER_YEAR), ddof=1.
  - Max Drawdown: trên equity curve weekly (công thức giữ nguyên).
  - exp_id GIỮ NGUYÊN — kết quả trading mới OVERWRITE kết quả daily cũ.

Hàm `simulate_trading()` (daily, generic — dùng bởi viz_service.py /
api_viz.py cho các chart cũ) được GIỮ NGUYÊN để không phá vỡ các module
khác (sẽ cập nhật ở Phase 4 — API/Frontend). Hàm mới cho weekly là
`simulate_trading_weekly()`.

Bug fix: load processed pkl qua `_load_processed_df()` — unpack dict
{"df": DataFrame, ...} an toàn (tránh dùng nhầm cả dict làm DataFrame).

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 4.3: Trading simulation methodology.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.config import FOLDS, MODELS, PATHS

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(PATHS["experiments"])
PROCESSED_DIR   = Path(PATHS["processed"])

# Annualization factors cho Sharpe Ratio
TRADING_DAYS_PER_YEAR  = 252  # daily  (simulate_trading() — giữ cho các chart cũ)
TRADING_WEEKS_PER_YEAR = 52   # weekly (simulate_trading_weekly() — Phương án D)


# =============================================================================
# 0. HELPERS
# =============================================================================

def _load_processed_df(pkl_path: Path) -> pd.DataFrame:
    """
    Load processed pkl và unpack DataFrame an toàn.

    Cấu trúc pkl chuẩn: {"df": pd.DataFrame, "feature_cols": [...], "target_col": "Close"}
    (xem preprocess.py / scripts/preprocess.py).

    Bug fix (2026-06): bản cũ có nơi dùng trực tiếp `pickle.load(f)` làm
    DataFrame mà thiếu unpack `["df"]` → lỗi vì object là dict, không phải
    DataFrame. Hàm này luôn unpack đúng, kèm fallback nếu pkl là DataFrame
    "trần" (không bọc dict).

    Returns:
        pd.DataFrame với index=DatetimeIndex, đã sort tăng dần.

    Raises:
        TypeError: Nếu sau khi unpack vẫn không phải DataFrame.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df = data["df"] if isinstance(data, dict) else data

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"_load_processed_df: '{pkl_path}' không chứa DataFrame hợp lệ "
            f"(type={type(df)})."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)

    return df.sort_index()


# =============================================================================
# 1. SIMULATE TRADING — DAILY (generic, giữ cho viz_service.py / api_viz.py)
# =============================================================================

def simulate_trading(
    predictions    : np.ndarray,
    actual_prices  : np.ndarray,
    initial_capital: float = 1.0,
    dates          : Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Mô phỏng chiến lược giao dịch theo bài báo (alignment kiểu daily t→t+1).

    GHI CHÚ: Hàm này được GIỮ NGUYÊN (không đổi logic) để các chart cũ trong
    viz_service.py / api_viz.py (Predicted-vs-Actual, Trading Returns chart...)
    không bị crash trước khi Phase 4 cập nhật chúng sang weekly. Đối với
    Task C (kết quả trading chính thức, weekly), dùng `simulate_trading_weekly()`.

    Strategy:
      - predictions[i] == 1 (UP): mua tại actual_prices[i], bán tại actual_prices[i+1]
        → period_return[i] = (actual_prices[i+1] - actual_prices[i]) / actual_prices[i]
      - predictions[i] == 0 (DOWN): không giao dịch → period_return[i] = 0.0
      - cumulative[i] = initial_capital × ∏(1 + period_return[0..i])
      - buy_hold[i]   = initial_capital × actual_prices[i+1] / actual_prices[0]

    Args:
        predictions:     Binary labels {0=DOWN, 1=UP}, shape (N,).
        actual_prices:   Close prices liên tiếp, shape (N+1,). Giá thực (VND/USD).
        initial_capital: Vốn ban đầu, mặc định 1.0 (normalize).
        dates:           Optional array of dates, shape (N,), cho cột Date.

    Returns:
        pd.DataFrame với N rows, columns:
          Date, Close, Prediction, Daily_Return,
          Strategy_Cumulative, BuyHold_Cumulative

    Raises:
        ValueError: Nếu len(actual_prices) != len(predictions) + 1,
                    hoặc predictions không phải binary {0, 1}, hoặc rỗng.
    """
    predictions   = np.asarray(predictions,   dtype=np.int32  ).flatten()
    actual_prices = np.asarray(actual_prices, dtype=np.float64).flatten()

    n = len(predictions)

    if len(actual_prices) != n + 1:
        raise ValueError(
            f"simulate_trading: len(actual_prices)={len(actual_prices)} "
            f"phải bằng len(predictions)+1={n+1}."
        )
    unique_preds = set(predictions.tolist())
    if not unique_preds.issubset({0, 1}):
        raise ValueError(
            f"simulate_trading: predictions chứa giá trị ngoài {{0, 1}}: {unique_preds}"
        )
    if n == 0:
        raise ValueError("simulate_trading: predictions rỗng.")

    eps = 1e-8
    price_changes = (actual_prices[1:] - actual_prices[:-1]) / (
        np.abs(actual_prices[:-1]) + eps
    )  # shape (N,)

    daily_returns = np.where(predictions == 1, price_changes, 0.0)

    cumulative_factors = np.cumprod(1.0 + daily_returns)        # shape (N,)
    strategy_cum = initial_capital * cumulative_factors

    buyhold_cum = initial_capital * (actual_prices[1:] / (actual_prices[0] + eps))

    if dates is not None:
        date_col = np.asarray(dates).flatten()[:n]
    else:
        date_col = np.arange(n)

    df = pd.DataFrame({
        "Date"                : date_col,
        "Close"               : actual_prices[:n],
        "Prediction"          : predictions,
        "Daily_Return"        : daily_returns,
        "Strategy_Cumulative" : strategy_cum,
        "BuyHold_Cumulative"  : buyhold_cum,
    })

    return df


# =============================================================================
# 2. SIMULATE TRADING — WEEKLY (Phương án D, Task B/C mới)
# =============================================================================

def simulate_trading_weekly(
    y_pred         : np.ndarray,
    f_w_dates      : np.ndarray,
    close_series   : pd.Series,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Mô phỏng chiến lược trading WEEKLY (Phương án D).

    Mỗi sample i ứng với F_W = f_w_dates[i] (phiên cuối tuần W — mốc dự đoán
    cho tuần W+1). Nếu y_pred[i] == 1 (predict UP cho tuần W+1):
        return_i = (Close(F_{W+1}) - Close(F_W)) / Close(F_W)
    Ngược lại (predict DOWN): return_i = 0.0 (no shorting, đứng ngoài).

    Xác định F_{W+1}:
      - Với i < N-1: F_{W+1} = f_w_dates[i+1] (build_weekly_sequences() tạo
        các tuần LIÊN TỤC nên tuần kế tiếp trong `dates` chính là F_{W+1}
        của sample i).
      - Với i == N-1 (sample cuối): F_{W+1} KHÔNG có trong f_w_dates (tuần
        đó bị build_weekly_sequences() bỏ vì không có nhãn). Tự tìm bằng
        cách lấy phiên giao dịch CUỐI CÙNG của tuần kế tiếp F_W trong
        `close_series` (close_series là Close của TOÀN BỘ dataset, có thể
        vượt ra ngoài test period). Nếu không còn dữ liệu nào sau F_W
        (F_W là phiên cuối cùng của toàn dataset) → bỏ sample này.

    Args:
        y_pred:          Binary labels {0=DOWN, 1=UP}, shape (N,). y_pred từ
                         predictions.npz của Task B (weekly).
        f_w_dates:       Array datetime64, shape (N,) — F_W cho mỗi sample
                         (key "dates" trong predictions.npz). Phải SORTED tăng
                         dần (đảm bảo bởi build_weekly_sequences()).
        close_series:    pd.Series Close GỐC (unscaled), index=DatetimeIndex,
                         lấy từ TOÀN BỘ df_full (processed pkl) — không chỉ
                         test period — để có thể tìm F_{W+1} cho sample cuối.
        initial_capital: Vốn ban đầu, mặc định 1.0 (normalize).

    Returns:
        pd.DataFrame, mỗi row = 1 tuần W, columns:
          Date                : F_W (ngày quyết định, datetime64)
          Date_Next           : F_{W+1} (ngày thực hiện return)
          Close               : Close(F_W)
          Close_Next          : Close(F_{W+1})
          Prediction          : 0 (DOWN/no-trade) hoặc 1 (UP/trade)
          Daily_Return        : return của tuần nếu trade, 0.0 nếu không
                                 (tên giữ "Daily_Return" để compute_trading_metrics()
                                  tái sử dụng được — ở đây là return THEO TUẦN)
          Strategy_Cumulative : vốn tích lũy của strategy
          BuyHold_Cumulative  : vốn tích lũy của Buy & Hold (mua tại F_W đầu tiên)

    Raises:
        ValueError: Nếu y_pred/f_w_dates rỗng, length không khớp,
                     predictions không phải binary {0,1}, hoặc không có
                     sample hợp lệ nào sau alignment (ví dụ close_series
                     thiếu hoàn toàn các ngày F_W).
    """
    y_pred = np.asarray(y_pred, dtype=np.int32).flatten()
    f_w_dates = pd.DatetimeIndex(np.asarray(f_w_dates).flatten())
    n = len(y_pred)

    if len(f_w_dates) != n:
        raise ValueError(
            f"simulate_trading_weekly: len(f_w_dates)={len(f_w_dates)} "
            f"!= len(y_pred)={n}."
        )
    unique_preds = set(y_pred.tolist())
    if not unique_preds.issubset({0, 1}):
        raise ValueError(
            f"simulate_trading_weekly: y_pred chứa giá trị ngoài {{0,1}}: {unique_preds}"
        )
    if n == 0:
        raise ValueError("simulate_trading_weekly: y_pred rỗng.")

    close_series = close_series.sort_index()
    idx = close_series.index
    eps = 1e-8

    def _snap_to_index(d: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Tìm ngày <= d gần nhất có trong close_series.index (phòng lệch index)."""
        if d in idx:
            return d
        pos = idx.searchsorted(d, side="right") - 1
        if pos < 0:
            return None
        return idx[pos]

    rows: list[dict] = []

    for i in range(n):
        f_w = _snap_to_index(f_w_dates[i])
        if f_w is None:
            logger.debug(
                "[simulate_trading_weekly] F_W=%s không tìm thấy trong close_series, bỏ sample.",
                f_w_dates[i],
            )
            continue

        if i < n - 1:
            # F_{W+1} = F_W của sample kế tiếp (dates liên tục)
            f_w_next = _snap_to_index(f_w_dates[i + 1])
            if f_w_next is None or f_w_next <= f_w:
                logger.debug(
                    "[simulate_trading_weekly] F_{W+1}=%s không hợp lệ (sample %d), bỏ.",
                    f_w_dates[i + 1], i,
                )
                continue
        else:
            # Sample cuối: tự tìm F_{W+1} = phiên cuối tuần kế tiếp trong close_series
            future = idx[idx > f_w]
            if len(future) == 0:
                logger.debug(
                    "[simulate_trading_weekly] Sample cuối (F_W=%s) không còn dữ liệu "
                    "tuần kế tiếp → bỏ.",
                    f_w,
                )
                break
            next_period = future[0].to_period("W-FRI")
            week_mask = future.to_period("W-FRI") == next_period
            f_w_next = future[week_mask].max()

        c_w      = float(close_series.loc[f_w])
        c_w_next = float(close_series.loc[f_w_next])

        ret = (c_w_next - c_w) / (abs(c_w) + eps)

        rows.append({
            "Date"      : f_w,
            "Date_Next" : f_w_next,
            "Close"     : c_w,
            "Close_Next": c_w_next,
            "Prediction": int(y_pred[i]),
            "Daily_Return": ret if y_pred[i] == 1 else 0.0,   # weekly return
        })

    if not rows:
        raise ValueError(
            "simulate_trading_weekly: không có sample hợp lệ nào sau alignment "
            "(kiểm tra close_series có cover đúng date range của f_w_dates)."
        )

    df = pd.DataFrame(rows)

    # ── Cumulative returns ────────────────────────────────────────────────────
    cumulative_factors = np.cumprod(1.0 + df["Daily_Return"].values)
    df["Strategy_Cumulative"] = initial_capital * cumulative_factors

    # Buy & Hold: mua tại Close(F_W) của tuần đầu tiên, theo dõi Close(F_{W+1})
    first_close = df["Close"].iloc[0]
    df["BuyHold_Cumulative"] = initial_capital * (
        df["Close_Next"].values / (first_close + eps)
    )

    return df


# =============================================================================
# 3. COMPUTE TRADING METRICS
# =============================================================================

def compute_trading_metrics(
    trade_df: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Tính các trading performance metrics từ kết quả simulation.

    Dùng chung cho cả `simulate_trading()` (daily) và
    `simulate_trading_weekly()` (weekly) — chỉ khác `periods_per_year`
    dùng để annualize Sharpe Ratio.

    Args:
        trade_df: DataFrame trả về từ simulate_trading() hoặc
                  simulate_trading_weekly(). Phải có columns:
                  Daily_Return, Strategy_Cumulative, BuyHold_Cumulative, Prediction.
        periods_per_year: Số kỳ giao dịch / năm dùng để annualize Sharpe.
                  - Daily  (simulate_trading)        : TRADING_DAYS_PER_YEAR  = 252
                  - Weekly (simulate_trading_weekly) : TRADING_WEEKS_PER_YEAR = 52

    Returns:
        dict với 5 keys:
          "Cumulative_Return" (float): Strategy_Cumulative.iloc[-1] - 1.
          "BuyHold_Return"    (float): BuyHold_Cumulative.iloc[-1] - 1.
          "Sharpe_Ratio"      (float): mean(returns)/std(returns, ddof=1) × √periods_per_year.
                              Tính trên TẤT CẢ kỳ (kể cả kỳ không trade → return=0).
                              Nếu std≈0 → Sharpe=0.0.
          "Max_Drawdown"      (float): max drawdown của Strategy_Cumulative, ∈ [0,1].
          "Win_Rate"          (float): tỉ lệ kỳ trade thắng / tổng kỳ trade
                              (Prediction==1 AND Daily_Return>0) / (Prediction==1).
                              Nếu không có kỳ trade nào → 0.0.

    Raises:
        ValueError: Nếu trade_df rỗng hoặc thiếu columns cần thiết.
    """
    required_cols = {"Daily_Return", "Strategy_Cumulative", "BuyHold_Cumulative", "Prediction"}
    missing = required_cols - set(trade_df.columns)
    if missing:
        raise ValueError(f"compute_trading_metrics: thiếu columns {missing}")
    if trade_df.empty:
        raise ValueError("compute_trading_metrics: trade_df rỗng.")

    period_returns = trade_df["Daily_Return"].values.astype(np.float64)
    strategy_cum   = trade_df["Strategy_Cumulative"].values.astype(np.float64)
    buyhold_cum    = trade_df["BuyHold_Cumulative"].values.astype(np.float64)
    predictions    = trade_df["Prediction"].values.astype(np.int32)

    # ── 1. Cumulative Return ──────────────────────────────────────────────────
    cumulative_return = float(strategy_cum[-1] - 1.0)

    # ── 2. Buy & Hold Return ──────────────────────────────────────────────────
    buyhold_return = float(buyhold_cum[-1] - 1.0)

    # ── 3. Sharpe Ratio (annualized) ──────────────────────────────────────────
    mean_ret = np.mean(period_returns)
    std_ret  = np.std(period_returns, ddof=1) if len(period_returns) > 1 else 0.0
    if std_ret < 1e-10:
        sharpe = 0.0
    else:
        sharpe = float((mean_ret / std_ret) * np.sqrt(periods_per_year))

    # ── 4. Max Drawdown ───────────────────────────────────────────────────────
    peak         = np.maximum.accumulate(strategy_cum)
    drawdowns    = 1.0 - strategy_cum / np.where(peak > 0, peak, 1.0)
    max_drawdown = float(np.max(drawdowns))
    max_drawdown = max(0.0, max_drawdown)

    # ── 5. Win Rate ───────────────────────────────────────────────────────────
    trade_mask = predictions == 1
    n_trades   = int(trade_mask.sum())
    if n_trades == 0:
        win_rate = 0.0
    else:
        win_mask = trade_mask & (period_returns > 0)
        win_rate = float(int(win_mask.sum()) / n_trades)

    return {
        "Cumulative_Return" : cumulative_return,
        "BuyHold_Return"    : buyhold_return,
        "Sharpe_Ratio"      : sharpe,
        "Max_Drawdown"      : max_drawdown,
        "Win_Rate"          : win_rate,
    }


# =============================================================================
# 4. RUN SIMULATION FOR ALL MODELS — WEEKLY (Task C chính thức)
# =============================================================================

def run_trading_simulation_all_models(
    ticker  : str,
    currency: str,
    fold_idx: int,
) -> pd.DataFrame:
    """
    Chạy WEEKLY trading simulation (Phương án D) cho tất cả 5 models ×
    2 wavelet conditions, dùng predictions.npz (weekly) từ Task B.

    Nguồn dữ liệu:
      - predictions.npz: experiments/{ticker}_{currency}_{cond}_{model}_classification
        /fold_{fold_idx}/predictions.npz — cần có "y_pred" và "dates" (F_W).
        ("dates" chỉ tồn tại từ sau bản chạy lại Task B weekly — Phase 2.)
      - Close GỐC (unscaled): data/processed/{ticker}_{currency}_{wavelet|nowave}.pkl
        → toàn bộ df_full (KHÔNG filter theo test period) — cần để tìm
        F_{W+1} cho sample cuối cùng (có thể nằm ngoài test_end).

    Args:
        ticker:   "VCB" hoặc "VIC".
        currency: "VND" hoặc "USD".
        fold_idx: 1-based fold index (1, 2, 3).

    Returns:
        pd.DataFrame, mỗi row = 1 model × 1 wavelet condition:
          [ticker, currency, wavelet, model, fold,
           Cumulative_Return, BuyHold_Return, Sharpe_Ratio, Max_Drawdown, Win_Rate,
           n_predictions, n_trade_days, status]

        - "n_predictions": số tuần (sample) thực sự dùng được sau alignment.
        - "n_trade_days" : số TUẦN model predict UP (tên field giữ nguyên để
          tương thích frontend hiện tại — sẽ đổi label hiển thị ở Phase 4).
        - Sharpe_Ratio annualize ×√52 (TRADING_WEEKS_PER_YEAR).
        - Rows với status != 'ok' nghĩa là predictions.npz thiếu/lỗi/chưa
          có "dates" (cần chạy lại Task B weekly).

    Raises:
        ValueError: Nếu fold_idx không hợp lệ.
    """
    fold_map = {f["fold_id"]: f for f in FOLDS}
    if fold_idx not in fold_map:
        raise ValueError(
            f"run_trading_simulation_all_models: fold_idx={fold_idx} không tồn tại. "
            f"Hợp lệ: {list(fold_map.keys())}"
        )

    rows = []

    for use_wavelet in [True, False]:
        cond_str = "wavelet" if use_wavelet else "nowave"

        # ── Load processed pkl để lấy Close GỐC (toàn bộ dataset) ────────────
        pkl_path = PROCESSED_DIR / f"{ticker}_{currency}_{cond_str}.pkl"
        if not pkl_path.exists():
            logger.warning(
                "[run_trading_simulation_all_models] pkl không tồn tại: '%s'. "
                "Bỏ qua tất cả models với cond=%s.",
                pkl_path, cond_str,
            )
            for model_name in MODELS:
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, "pkl_missing"
                ))
            continue

        try:
            df_full = _load_processed_df(pkl_path)
        except Exception as exc:
            logger.warning(
                "[run_trading_simulation_all_models] Lỗi load pkl '%s': %s", pkl_path, exc
            )
            for model_name in MODELS:
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, f"pkl_error:{exc}"
                ))
            continue

        if "Close" not in df_full.columns:
            logger.warning(
                "[run_trading_simulation_all_models] pkl '%s' thiếu cột 'Close'.", pkl_path
            )
            for model_name in MODELS:
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, "close_col_missing"
                ))
            continue

        # Close GỐC (unscaled) — toàn bộ dataset, dùng cho cả F_W và F_{W+1}
        close_series = df_full["Close"].astype(np.float64)

        # ── Loop qua 5 models ─────────────────────────────────────────────────
        for model_name in MODELS:
            exp_id   = f"{ticker}_{currency}_{cond_str}_{model_name}_classification"
            npz_path = EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "predictions.npz"

            if not npz_path.exists():
                logger.debug(
                    "[run_trading_simulation_all_models] predictions.npz không tồn tại: '%s'",
                    npz_path,
                )
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, "npz_missing"
                ))
                continue

            try:
                npz_data = np.load(str(npz_path), allow_pickle=False)

                if "dates" not in npz_data.files:
                    # predictions.npz cũ (daily, trước Phase 2) — chưa có F_W dates
                    rows.append(_make_error_row(
                        ticker, currency, use_wavelet, model_name, fold_idx,
                        "dates_missing_rerun_required",
                    ))
                    continue

                y_pred    = npz_data["y_pred"].astype(np.int32).flatten()       # (N,)
                f_w_dates = pd.DatetimeIndex(npz_data["dates"])                  # (N,) F_W
            except Exception as exc:
                logger.warning(
                    "[run_trading_simulation_all_models] Lỗi load '%s': %s", npz_path, exc
                )
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, f"load_error:{exc}"
                ))
                continue

            # ── Simulate (weekly) & compute metrics ─────────────────────────────
            try:
                trade_df = simulate_trading_weekly(
                    y_pred       = y_pred,
                    f_w_dates    = f_w_dates.values,
                    close_series = close_series,
                    initial_capital = 1.0,
                )
                tm = compute_trading_metrics(
                    trade_df, periods_per_year=TRADING_WEEKS_PER_YEAR
                )
            except Exception as exc:
                logger.warning(
                    "[run_trading_simulation_all_models] Lỗi simulation "
                    "(ticker=%s model=%s fold=%d): %s",
                    ticker, model_name, fold_idx, exc,
                )
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, f"sim_error:{exc}"
                ))
                continue

            n_predictions = len(trade_df)
            n_trade_weeks = int((trade_df["Prediction"] == 1).sum())

            rows.append({
                "ticker"             : ticker,
                "currency"           : currency,
                "wavelet"            : use_wavelet,
                "model"              : model_name,
                "fold"               : fold_idx,
                "Cumulative_Return"  : tm["Cumulative_Return"],
                "BuyHold_Return"     : tm["BuyHold_Return"],
                "Sharpe_Ratio"       : tm["Sharpe_Ratio"],
                "Max_Drawdown"       : tm["Max_Drawdown"],
                "Win_Rate"           : tm["Win_Rate"],
                "n_predictions"      : n_predictions,
                "n_trade_days"       : n_trade_weeks,   # giữ tên field cũ (= số tuần UP)
                "status"             : "ok",
            })

            logger.info(
                "[run_trading_simulation_all_models] %s | %s cond=%s fold=%d (weekly) | "
                "CumRet=%.4f BnH=%.4f Sharpe=%.2f MDD=%.4f WinRate=%.4f | "
                "N=%d trade_weeks=%d",
                model_name, ticker, cond_str, fold_idx,
                tm["Cumulative_Return"], tm["BuyHold_Return"],
                tm["Sharpe_Ratio"], tm["Max_Drawdown"], tm["Win_Rate"],
                n_predictions, n_trade_weeks,
            )

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows).sort_values(
        ["wavelet", "model"]
    ).reset_index(drop=True)

    return df_result


# =============================================================================
# HELPERS
# =============================================================================

def _make_error_row(
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    model_name : str,
    fold_idx   : int,
    status     : str,
) -> dict:
    """Tạo placeholder row với NaN metrics khi simulation bị lỗi/bỏ qua."""
    return {
        "ticker"             : ticker,
        "currency"           : currency,
        "wavelet"            : use_wavelet,
        "model"              : model_name,
        "fold"               : fold_idx,
        "Cumulative_Return"  : np.nan,
        "BuyHold_Return"     : np.nan,
        "Sharpe_Ratio"       : np.nan,
        "Max_Drawdown"       : np.nan,
        "Win_Rate"           : np.nan,
        "n_predictions"      : np.nan,
        "n_trade_days"       : np.nan,
        "status"             : status,
    }