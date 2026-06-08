"""
app/services/trading_service.py
==================================
Task C — Trading Simulation cho VNSP.

Strategy từ bài báo (Li et al., EAAI 2026):
  - Mua cuối ngày t nếu model classify UP (Close(t+1) > Close(t))
  - Không giao dịch nếu classify DOWN
  - Không short selling, không transaction cost (baseline)
  - Source of truth: predictions.npz từ Task B (Classification)

Pipeline:
  predictions.npz (y_pred)  →  simulate_trading()  →  trade_df
  trade_df                   →  compute_trading_metrics()  →  metrics dict
  run_trading_simulation_all_models()  →  comparison DataFrame (5 models × 2 wavelet)

Alignment giữa predictions và prices:
  - Mỗi prediction[i] = direction(Close[i+seq_len] vs Close[i+seq_len-1])
  - Trading: buy tại Close[i+seq_len-1], sell tại Close[i+seq_len]
  - actual_prices phải là shape (N+1,) = Close[seq_len-1 : seq_len-1+N+1] từ test DataFrame
  - seq_len = len(df_test) - N  (tự suy ra từ predictions length)

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 4.3: Trading simulation methodology.
"""

from __future__ import annotations

import json
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

# Số ngày giao dịch trong 1 năm — dùng để annualize Sharpe Ratio
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# 1. SIMULATE TRADING
# =============================================================================

def simulate_trading(
    predictions    : np.ndarray,
    actual_prices  : np.ndarray,
    initial_capital: float = 1.0,
    dates          : Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Mô phỏng chiến lược giao dịch theo bài báo.

    Strategy:
      - predictions[i] == 1 (UP): mua tại actual_prices[i], bán tại actual_prices[i+1]
        → daily_return[i] = (actual_prices[i+1] - actual_prices[i]) / actual_prices[i]
      - predictions[i] == 0 (DOWN): không giao dịch
        → daily_return[i] = 0.0
      - cumulative[i] = initial_capital × ∏(1 + daily_return[0..i])
      - buy_hold[i]   = initial_capital × actual_prices[i+1] / actual_prices[0]

    Args:
        predictions:     Binary labels {0=DOWN, 1=UP}, shape (N,).
                         Là y_pred từ predictions.npz của Task B.
        actual_prices:   Close prices liên tiếp, shape (N+1,).
                         actual_prices[i]   = Close của ngày t   (mua nếu predict UP)
                         actual_prices[i+1] = Close của ngày t+1 (bán)
                         Phải là giá thực (VND/USD), không phải giá đã scale.
        initial_capital: Vốn ban đầu, mặc định 1.0 (normalize). Ví dụ: 1_000_000 VND.
        dates:           Optional array of dates, shape (N,), cho cột Date của output.
                         Nếu None → dùng integer index 0..N-1.

    Returns:
        pd.DataFrame với N rows và columns:
          Date              : date của ngày trading (None → 0..N-1)
          Close             : actual_prices[i] — giá Close "hôm nay" (ngày quyết định)
          Prediction        : 0 (DOWN/no-trade) hoặc 1 (UP/trade)
          Daily_Return      : lợi nhuận ngày i nếu trade, 0.0 nếu không trade
          Strategy_Cumulative: vốn tích lũy của strategy (bắt đầu từ initial_capital)
          BuyHold_Cumulative : vốn tích lũy của Buy & Hold (bắt đầu từ initial_capital)

    Notes:
        - actual_prices phải có length = len(predictions) + 1.
        - Buy & Hold: mua vào đầu test period, nắm giữ đến cuối.
        - Không short selling — DOWN signal = đứng ngoài thị trường (return = 0).
        - Không tính transaction cost (theo bài báo, baseline setup).
        - Nếu actual_prices[i] == 0: daily_return = 0 (safeguard division-by-zero).

    Raises:
        ValueError: Nếu len(actual_prices) != len(predictions) + 1.
        ValueError: Nếu predictions không phải binary {0, 1}.
    """
    predictions   = np.asarray(predictions,   dtype=np.int32  ).flatten()
    actual_prices = np.asarray(actual_prices, dtype=np.float64).flatten()

    n = len(predictions)

    # ── Validation ────────────────────────────────────────────────────────────
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

    # ── Tính daily returns ────────────────────────────────────────────────────
    # price_changes[i] = (price[i+1] - price[i]) / price[i]
    # Dùng eps để tránh division-by-zero (giá cổ phiếu thực tế luôn > 0)
    eps = 1e-8
    price_changes = (actual_prices[1:] - actual_prices[:-1]) / (
        np.abs(actual_prices[:-1]) + eps
    )  # shape (N,)

    # Strategy return: chỉ nhận return khi predict UP (=1)
    daily_returns = np.where(predictions == 1, price_changes, 0.0)

    # ── Cumulative returns ────────────────────────────────────────────────────
    # Strategy: tích lũy từ initial_capital
    cumulative_factors = np.cumprod(1.0 + daily_returns)        # shape (N,)
    strategy_cum = initial_capital * cumulative_factors

    # Buy & Hold: mua ở actual_prices[0], theo dõi đến actual_prices[1..N]
    buyhold_cum = initial_capital * (actual_prices[1:] / (actual_prices[0] + eps))

    # ── Build DataFrame ───────────────────────────────────────────────────────
    if dates is not None:
        date_col = np.asarray(dates).flatten()[:n]
    else:
        date_col = np.arange(n)

    df = pd.DataFrame({
        "Date"                : date_col,
        "Close"               : actual_prices[:n],   # "hôm nay" = ngày quyết định
        "Prediction"          : predictions,
        "Daily_Return"        : daily_returns,
        "Strategy_Cumulative" : strategy_cum,
        "BuyHold_Cumulative"  : buyhold_cum,
    })

    return df


# =============================================================================
# 2. COMPUTE TRADING METRICS
# =============================================================================

def compute_trading_metrics(trade_df: pd.DataFrame) -> dict:
    """
    Tính các trading performance metrics từ kết quả simulation.

    Args:
        trade_df: DataFrame trả về từ simulate_trading().
                  Phải có columns: Daily_Return, Strategy_Cumulative, BuyHold_Cumulative, Prediction.

    Returns:
        dict với 5 keys:
          "Cumulative_Return" (float): Tổng return của strategy tại cuối kỳ.
                              = trade_df['Strategy_Cumulative'].iloc[-1] / initial_capital - 1
                              Ví dụ: 0.15 nghĩa là +15%.
                              Vì initial_capital = 1.0, tính đơn giản = iloc[-1] - 1.

          "BuyHold_Return"    (float): Return của Buy & Hold baseline.
                              = trade_df['BuyHold_Cumulative'].iloc[-1] - 1.

          "Sharpe_Ratio"      (float): Annualized Sharpe Ratio.
                              = mean(daily_returns) / std(daily_returns) × √252
                              Tính trên TẤT CẢ ngày (kể cả ngày không trade → return=0).
                              std dùng ddof=1 (sample std). Nếu std≈0 → Sharpe=0.0.

          "Max_Drawdown"      (float): Max drawdown của strategy, dương, range [0, 1].
                              = max(1 - Strategy_Cumulative / Strategy_Cumulative.cummax())
                              Ví dụ: 0.20 nghĩa là drawdown tối đa 20%.

          "Win_Rate"          (float): Tỉ lệ ngày trade thắng / tổng ngày trade.
                              Trade day: Prediction == 1.
                              Win day: Prediction == 1 AND Daily_Return > 0.
                              Nếu không có ngày trade nào → Win_Rate = 0.0.

    Notes:
        - Sharpe không trừ risk-free rate (theo convention của bài báo — đơn giản hóa).
        - Max Drawdown: dùng Strategy_Cumulative (bao gồm initial_capital).
        - Annualization factor 252 = số ngày trading/năm theo chuẩn thị trường.

    Raises:
        ValueError: Nếu trade_df rỗng hoặc thiếu columns cần thiết.
    """
    required_cols = {"Daily_Return", "Strategy_Cumulative", "BuyHold_Cumulative", "Prediction"}
    missing = required_cols - set(trade_df.columns)
    if missing:
        raise ValueError(f"compute_trading_metrics: thiếu columns {missing}")
    if trade_df.empty:
        raise ValueError("compute_trading_metrics: trade_df rỗng.")

    daily_returns  = trade_df["Daily_Return"].values.astype(np.float64)
    strategy_cum   = trade_df["Strategy_Cumulative"].values.astype(np.float64)
    buyhold_cum    = trade_df["BuyHold_Cumulative"].values.astype(np.float64)
    predictions    = trade_df["Prediction"].values.astype(np.int32)

    # ── 1. Cumulative Return ──────────────────────────────────────────────────
    # Vì initial_capital=1.0 trong simulate_trading mặc định
    cumulative_return = float(strategy_cum[-1] - 1.0)

    # ── 2. Buy & Hold Return ──────────────────────────────────────────────────
    buyhold_return = float(buyhold_cum[-1] - 1.0)

    # ── 3. Sharpe Ratio (annualized) ──────────────────────────────────────────
    # Tính trên tất cả ngày (kể cả ngày không trade với return=0)
    mean_ret = np.mean(daily_returns)
    std_ret  = np.std(daily_returns, ddof=1)  # sample std
    if std_ret < 1e-10:
        # std ≈ 0: hoặc không trade ngày nào, hoặc mọi return đều giống nhau
        sharpe = 0.0
    else:
        sharpe = float((mean_ret / std_ret) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ── 4. Max Drawdown ───────────────────────────────────────────────────────
    # Drawdown tại mỗi điểm = 1 - value / peak_so_far
    peak        = np.maximum.accumulate(strategy_cum)   # running max
    drawdowns   = 1.0 - strategy_cum / np.where(peak > 0, peak, 1.0)
    max_drawdown = float(np.max(drawdowns))
    max_drawdown = max(0.0, max_drawdown)  # đảm bảo không âm

    # ── 5. Win Rate ───────────────────────────────────────────────────────────
    trade_mask = predictions == 1
    n_trades   = int(trade_mask.sum())
    if n_trades == 0:
        win_rate = 0.0
    else:
        win_mask  = trade_mask & (daily_returns > 0)
        n_wins    = int(win_mask.sum())
        win_rate  = float(n_wins / n_trades)

    return {
        "Cumulative_Return" : cumulative_return,
        "BuyHold_Return"    : buyhold_return,
        "Sharpe_Ratio"      : sharpe,
        "Max_Drawdown"      : max_drawdown,
        "Win_Rate"          : win_rate,
    }


# =============================================================================
# 3. RUN SIMULATION FOR ALL MODELS
# =============================================================================

def run_trading_simulation_all_models(
    ticker  : str,
    currency: str,
    fold_idx: int,
) -> pd.DataFrame:
    """
    Chạy trading simulation cho tất cả 5 models × 2 wavelet conditions,
    sử dụng predictions đã lưu từ Task B (classification).

    Lấy predictions từ:
      experiments/{ticker}_{currency}_{cond}_{model}_classification/fold_{fold_idx}/predictions.npz

    Lấy actual close prices từ:
      data/processed/{ticker}_{currency}_{wavelet|nowave}.pkl
      → Close column (unscaled), filter theo test period của fold_idx.

    Args:
        ticker:   "VCB" hoặc "VIC".
        currency: "VND" hoặc "USD".
        fold_idx: 1-based fold index (1, 2, 3).

    Returns:
        pd.DataFrame, mỗi row = 1 model × 1 wavelet condition:
          [ticker, currency, wavelet, model, fold,
           Cumulative_Return, BuyHold_Return, Sharpe_Ratio, Max_Drawdown, Win_Rate,
           n_predictions, n_trade_days, status]

        Rows với status != 'ok' có nghĩa predictions.npz bị thiếu hoặc lỗi.
        BuyHold_Return giống nhau cho tất cả models trong cùng (ticker, currency, wavelet, fold).

    Notes:
        - seq_len tự động suy ra từ len(df_test) - len(predictions) → không cần đọc HPO params.
        - Nếu một model chưa có predictions.npz (experiment chưa chạy), row đó có NaN metrics.
        - Cả regression và classification dùng chung test period, nhưng seq_len có thể khác nhau
          vì HPO chọn seq_len riêng → actual_prices cũng khác nhau. Mỗi model tự align.
    """
    # ── Lookup fold definition ────────────────────────────────────────────────
    fold_map = {f["fold_id"]: f for f in FOLDS}
    if fold_idx not in fold_map:
        raise ValueError(
            f"run_trading_simulation_all_models: fold_idx={fold_idx} không tồn tại. "
            f"Hợp lệ: {list(fold_map.keys())}"
        )
    fold_def   = fold_map[fold_idx]
    test_start = pd.Timestamp(fold_def["test_start"])
    test_end   = pd.Timestamp(fold_def["test_end"])

    rows = []

    for use_wavelet in [True, False]:
        cond_str = "wavelet" if use_wavelet else "nowave"

        # ── Load processed pkl để lấy Close prices ───────────────────────────
        pkl_path = PROCESSED_DIR / f"{ticker}_{currency}_{cond_str}.pkl"
        if not pkl_path.exists():
            logger.warning(
                "[run_trading_simulation_all_models] pkl không tồn tại: '%s'. "
                "Bỏ qua tất cả models với cond=%s.",
                pkl_path, cond_str,
            )
            # Ghi placeholder rows cho các models bị thiếu pkl
            for model_name in MODELS:
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, "pkl_missing"
                ))
            continue

        with open(pkl_path, "rb") as f:
            df_full = pickle.load(f)

        # Filter về test period
        df_test = df_full[
            (df_full.index >= test_start) & (df_full.index <= test_end)
        ].copy()

        if df_test.empty:
            logger.warning(
                "[run_trading_simulation_all_models] df_test rỗng cho "
                "ticker=%s currency=%s cond=%s fold=%d.",
                ticker, currency, cond_str, fold_idx,
            )
            for model_name in MODELS:
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, "empty_test"
                ))
            continue

        close_prices_full = df_test["Close"].values.astype(np.float64)  # (T,)

        # ── Loop qua 5 models ─────────────────────────────────────────────────
        for model_name in MODELS:
            exp_id  = f"{ticker}_{currency}_{cond_str}_{model_name}_classification"
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

            # Load predictions (y_pred = binary labels từ Task B)
            try:
                npz_data    = np.load(str(npz_path), allow_pickle=False)
                y_pred      = npz_data["y_pred"].astype(np.int32).flatten()   # (N,)
            except Exception as exc:
                logger.warning(
                    "[run_trading_simulation_all_models] Lỗi load '%s': %s", npz_path, exc
                )
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, f"load_error:{exc}"
                ))
                continue

            n_pred = len(y_pred)
            T      = len(close_prices_full)

            if n_pred >= T:
                logger.warning(
                    "[run_trading_simulation_all_models] n_pred=%d >= T=%d "
                    "(ticker=%s currency=%s cond=%s model=%s fold=%d). "
                    "Không thể suy ra seq_len hợp lệ.",
                    n_pred, T, ticker, currency, cond_str, model_name, fold_idx,
                )
                rows.append(_make_error_row(
                    ticker, currency, use_wavelet, model_name, fold_idx, "alignment_error"
                ))
                continue

            # seq_len tự suy ra: T - N = số timesteps trong window
            seq_len = T - n_pred   # >= 1 (đã kiểm tra n_pred < T)

            # actual_prices: shape (N+1,)
            # [seq_len-1 : seq_len-1+N+1] = [seq_len-1 : seq_len+N] = [seq_len-1 : T]
            actual_prices = close_prices_full[seq_len - 1:]   # shape (N+1,)

            # Dates: ngày quyết định trade = df_test.index[seq_len-1 : seq_len-1+N]
            dates = df_test.index[seq_len - 1 : seq_len - 1 + n_pred]

            # ── Simulate & compute metrics ─────────────────────────────────────
            try:
                trade_df = simulate_trading(
                    predictions    = y_pred,
                    actual_prices  = actual_prices,
                    initial_capital = 1.0,
                    dates          = dates.values if hasattr(dates, "values") else dates,
                )
                tm = compute_trading_metrics(trade_df)
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

            n_trade_days = int((y_pred == 1).sum())

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
                "n_predictions"      : n_pred,
                "n_trade_days"       : n_trade_days,
                "status"             : "ok",
            })

            logger.info(
                "[run_trading_simulation_all_models] %s | %s cond=%s fold=%d | "
                "CumRet=%.4f BnH=%.4f Sharpe=%.2f MDD=%.4f WinRate=%.4f",
                model_name, ticker, cond_str, fold_idx,
                tm["Cumulative_Return"], tm["BuyHold_Return"],
                tm["Sharpe_Ratio"], tm["Max_Drawdown"], tm["Win_Rate"],
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