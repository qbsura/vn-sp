"""
app/services/data_service.py
=============================
Thu thập và quản lý dữ liệu cổ phiếu VCB & VIC, tỷ giá USD/VND.

Nguyên tắc:
  - Tải 1 lần → lưu CSV → các lần sau chỉ đọc CSV (không gọi API lại)
  - vnstock v4.x API: Quote(symbol, source).history(start, end, interval)
  - TCBS không còn hỗ trợ từ v4 — dùng source='VCI' hoặc 'KBS'
  - Columns đầu ra chuẩn hóa: Date (index), Open, High, Low, Close, Volume
"""

import logging
from pathlib import Path

import pandas as pd

from app.config import (
    DATE_START,
    DATE_END,
    PATHS,
    TICKERS,
    VNSTOCK_INTERVAL,
    VNSTOCK_SOURCE,
)

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(PATHS["raw"])

# =============================================================================
# PHẦN 1 — CỔ PHIẾU (Task 1.1)
# =============================================================================

def download_stock_data(ticker: str) -> pd.DataFrame:
    """
    Tải dữ liệu OHLCV của một mã cổ phiếu từ vnstock v4.x.

    vnstock v4 (28-04-2026): TCBS không còn được hỗ trợ.
    Source hợp lệ: 'VCI' (đầy đủ, khuyến nghị) hoặc 'KBS' (ổn định hơn trên Colab).

    Lưu ý đơn vị giá:
      - VCI trả về đơn vị nghìn VND (35.09 = 35,090 VND) → _normalize_ohlcv ×1000
      - KBS trả về đơn vị VND đầy đủ (34791.0 = 34,791 VND) → giữ nguyên

    Args:
        ticker: Mã cổ phiếu viết hoa, ví dụ "VCB" hoặc "VIC".

    Returns:
        DataFrame với index=Date (DatetimeIndex),
        columns=[Open, High, Low, Close, Volume].
        Giá đơn vị VND đầy đủ (đã chuẩn hoá nếu source=VCI).
        Đồng thời lưu ra data/raw/{ticker}_raw.csv.

    Raises:
        RuntimeError: Nếu API thất bại hoặc trả về dữ liệu rỗng.
    """
    ticker = ticker.upper()
    logger.info(f"[{ticker}] Đang tải từ vnstock v4 (source={VNSTOCK_SOURCE})...")

    try:
        # vnstock v4: Quote(symbol, source).history(start, end, interval)
        # source hợp lệ: vci, kbs, msn, dnse, binance, fmp, fmarket
        # TCBS đã bị loại bỏ từ v4 (28-04-2026)
        from vnstock import Quote  # lazy import — không crash nếu chưa cài
        quote = Quote(symbol=ticker, source=VNSTOCK_SOURCE)
        df_raw = quote.history(
            start=DATE_START,
            end=DATE_END,
            interval=VNSTOCK_INTERVAL,  # "1D"
        )
    except Exception as e:
        raise RuntimeError(
            f"[{ticker}] Lỗi gọi vnstock API: {e}\n"
            f"Source hiện tại: '{VNSTOCK_SOURCE}'. "
            "Source hợp lệ v4: vci, kbs, msn, dnse, binance, fmp, fmarket. "
            "TCBS đã bị loại bỏ từ v4.\n"
            "Kiểm tra kết nối mạng hoặc log của vnstock."
        ) from e

    if df_raw is None or df_raw.empty:
        raise RuntimeError(
            f"[{ticker}] API trả về dữ liệu rỗng. "
            "Kiểm tra ticker, source, hoặc date range."
        )

    df = _normalize_ohlcv(df_raw, ticker)

    # Lưu CSV
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{ticker}_raw.csv"
    df.to_csv(out_path)

    logger.info(
        f"[{ticker}] ✅ Tải thành công | "
        f"Rows: {len(df):,} | "
        f"Range: {df.index[0].date()} → {df.index[-1].date()} | "
        f"Close min/max: {df['Close'].min():,.0f} / {df['Close'].max():,.0f} | "
        f"Lưu: {out_path}"
    )
    return df


def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load dữ liệu cổ phiếu.
    - CSV đã tồn tại → đọc CSV (không gọi API).
    - Chưa có CSV → gọi download_stock_data().
    Chạy validation sau khi load.

    Args:
        ticker: Mã cổ phiếu viết hoa.

    Returns:
        DataFrame đã validate, index=Date, columns=OHLCV.
    """
    ticker = ticker.upper()
    csv_path = RAW_DIR / f"{ticker}_raw.csv"

    if csv_path.exists():
        logger.info(f"[{ticker}] CSV tồn tại — load từ {csv_path} (bỏ qua API).")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df = df.sort_index()
    else:
        logger.info(f"[{ticker}] CSV chưa có — tải từ vnstock API...")
        df = download_stock_data(ticker)

    # Clip về đúng DATE_START — vnstock VCI có thể trả về data sớm hơn yêu cầu
    start_ts = pd.Timestamp(DATE_START)
    end_ts   = pd.Timestamp(DATE_END)
    before_clip = len(df)
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    if len(df) < before_clip:
        logger.info(
            f"[{ticker}] Clip date range: {before_clip:,} → {len(df):,} rows "
            f"({DATE_START} → {DATE_END})."
        )

    _validate_data(df, ticker)
    return df


def download_all() -> dict[str, pd.DataFrame]:
    """
    Tải dữ liệu cho tất cả tickers trong config (VCB, VIC).
    In bảng summary sau khi hoàn tất.

    Returns:
        dict: {"VCB": df_vcb, "VIC": df_vic}
    """
    results: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for ticker in TICKERS:
        try:
            # Dùng load_stock_data để tận dụng CSV cache
            # Nếu CSV đã có → đọc ngay, không gọi API lại
            results[ticker] = load_stock_data(ticker)
        except Exception as e:
            logger.error(f"[{ticker}] ❌ Lỗi: {e}")
            errors[ticker] = str(e)

    # Summary table
    print("\n" + "=" * 67)
    print(f"{'Ticker':<8} {'Rows':>6}  {'Start':<12} {'End':<12}  "
          f"{'Close Min':>10}  {'Close Max':>10}")
    print("-" * 67)
    for ticker, df in results.items():
        print(
            f"{ticker:<8} {len(df):>6,}  "
            f"{str(df.index[0].date()):<12} {str(df.index[-1].date()):<12}  "
            f"{df['Close'].min():>10,.0f}  "
            f"{df['Close'].max():>10,.0f}"
        )
    if errors:
        print("\n❌ Lỗi:")
        for ticker, msg in errors.items():
            print(f"  {ticker}: {msg}")
    print("=" * 67 + "\n")

    return results

# =============================================================================
# PHẦN 3 — HELPERS PRIVATE
# =============================================================================

def _normalize_ohlcv(df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Chuẩn hóa DataFrame thô từ vnstock về format chuẩn:
    index=Date (DatetimeIndex), columns=[Open, High, Low, Close, Volume].

    vnstock v3.x có thể trả về cột tên 'time', 'date', hoặc 'tradingDate'.
    """
    df = df_raw.copy()

    # Tìm cột ngày (thử các tên phổ biến)
    date_col = None
    for candidate in ["time", "date", "tradingDate"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "Date"
        else:
            raise RuntimeError(
                f"[{ticker}] Không tìm thấy cột ngày. "
                f"Columns: {list(df.columns)}"
            )
    else:
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    # Rename OHLCV về dạng chuẩn
    rename_map = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename_map)

    # Giữ đúng 5 cột cần thiết
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{ticker}] Thiếu columns: {missing}. "
            f"Columns hiện tại: {list(df.columns)}"
        )
    df = df[required].sort_index()

    # Đảm bảo kiểu số
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Chuẩn hoá đơn vị giá theo source ────────────────────────────────────
    # VCI trả về giá đơn vị nghìn VND: 35.09 thực ra là 35,090 VND
    # KBS trả về giá VND đầy đủ: 34791.0 = 34,791 VND
    # Cách phân biệt: nếu Close trung vị < 1000 → nhiều khả năng là đơn vị nghìn
    price_cols = ["Open", "High", "Low", "Close"]
    median_close = df["Close"].median()
    if median_close < 1000:
        logger.info(
            f"[{ticker}] Phát hiện giá đơn vị nghìn VND "
            f"(median Close={median_close:.2f}) → nhân ×1000 để ra VND đầy đủ."
        )
        df[price_cols] = df[price_cols] * 1000

    return df


def _validate_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Kiểm tra chất lượng dữ liệu sau khi load.
    Chỉ log warning, không raise — không chặn pipeline.

    Checks:
      1. Duplicate dates
      2. Gap > 7 ngày liên tiếp (holiday Tết có thể đến 6-7 ngày)
      3. NaN values
    """
    # 1. Duplicate index
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        logger.warning(f"[{ticker}] ⚠️ {n_dup} ngày bị trùng (duplicate index).")

    # 2. Gap lớn giữa các ngày giao dịch
    MAX_GAP_DAYS = 7  # Tết VN có thể nghỉ 6-7 ngày liên tiếp
    date_diffs = df.index.to_series().diff().dt.days.dropna()
    large_gaps = date_diffs[date_diffs > MAX_GAP_DAYS]
    if not large_gaps.empty:
        logger.warning(
            f"[{ticker}] ⚠️ {len(large_gaps)} khoảng trống >{MAX_GAP_DAYS} ngày:\n"
            + "\n".join(
                f"  • {str(d.date())}: gap {int(g)} ngày"
                for d, g in large_gaps.items()
            )
        )

    # 3. NaN values
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        logger.warning(
            f"[{ticker}] ⚠️ {total_nan} NaN: "
            + ", ".join(f"{c}={n}" for c, n in nan_counts.items() if n > 0)
        )

    logger.info(
        f"[{ticker}] Validation OK — {len(df):,} rows, {total_nan} NaN."
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    download_all()