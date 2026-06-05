"""
app/services/data_service.py
=============================
Thu thập và quản lý dữ liệu cổ phiếu VCB & VIC, tỷ giá USD/VND.

Nguyên tắc:
  - Tải 1 lần → lưu CSV → các lần sau chỉ đọc CSV (không gọi API lại)
  - vnstock v4.x API: Quote(symbol, source).history(start, end, interval)
  - TCBS không còn hỗ trợ từ v4 — dùng source='VCI' hoặc 'KBS'
  - Columns đầu ra chuẩn hóa: Date (index), Open, High, Low, Close, Volume
  - Tỷ giá USDVND: đọc từ file CSV tải thủ công từ investing.com
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

# Tên file tỷ giá (tải thủ công từ investing.com)
USDVND_CSV = RAW_DIR / "USDVND.csv"


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
# PHẦN 2 — TỶ GIÁ USD/VND (Task 1.2)
# =============================================================================

def load_usdvnd() -> pd.DataFrame:
    """
    Load tỷ giá USD/VND từ file CSV tải thủ công từ investing.com.

    File CSV format (investing.com):
        "Date","Price","Open","High","Low","Change %"
        "Jun 05, 2026","25,480.00","25,460.00","25,500.00","25,440.00","0.08%"

    Chỉ lấy 2 cột: Date và Price (= tỷ giá 1 USD = x VND).

    Returns:
        DataFrame với index=Date (DatetimeIndex), column=[Rate].
        Rate = số VND trên 1 USD (ví dụ: 25480.0).

    Raises:
        FileNotFoundError: Nếu USDVND.csv chưa tồn tại.
        ValueError: Nếu format CSV không đúng kỳ vọng.
    """
    if not USDVND_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file tỷ giá: {USDVND_CSV}\n"
            "Hướng dẫn tải thủ công:\n"
            "  1. Truy cập: https://www.investing.com/currencies/usd-vnd-historical-data\n"
            "  2. Đăng nhập (tài khoản free)\n"
            "  3. Chọn date range: 01/01/2012 → 31/12/2024\n"
            "  4. Nhấn 'Download' (biểu tượng mũi tên xuống)\n"
            f"  5. Đổi tên file thành 'USDVND.csv' → đặt vào {USDVND_CSV}"
        )

    logger.info(f"[USDVND] Load tỷ giá từ {USDVND_CSV}...")

    try:
        df_raw = pd.read_csv(USDVND_CSV)
    except Exception as e:
        raise ValueError(f"[USDVND] Lỗi đọc CSV: {e}") from e

    logger.info(f"[USDVND] Columns trong file: {list(df_raw.columns)}")

    # ── Tìm cột Date ──────────────────────────────────────────────────────────
    # investing.com dùng "Date" (đúng tên)
    if "Date" not in df_raw.columns:
        raise ValueError(
            f"[USDVND] Không tìm thấy cột 'Date'. "
            f"Columns: {list(df_raw.columns)}"
        )

    # ── Tìm cột giá đóng cửa (Rate) ──────────────────────────────────────────
    # investing.com có thể đặt tên "Price" hoặc "Close" hoặc tên khác
    # Ưu tiên: Price > Close > cột số thứ 2 (index 1)
    price_col = None
    for candidate in ["Price", "Close", "close", "price"]:
        if candidate in df_raw.columns:
            price_col = candidate
            break
    if price_col is None:
        # Fallback: lấy cột thứ 2 (sau Date)
        non_date_cols = [c for c in df_raw.columns if c != "Date"]
        if non_date_cols:
            price_col = non_date_cols[0]
            logger.warning(
                f"[USDVND] Không tìm thấy cột 'Price'/'Close' — "
                f"dùng cột '{price_col}' thay thế."
            )
        else:
            raise ValueError(
                f"[USDVND] Không tìm thấy cột giá. Columns: {list(df_raw.columns)}"
            )

    # Chỉ giữ Date và cột giá
    df = df_raw[["Date", price_col]].copy()
    df = df.rename(columns={price_col: "Rate_raw"})

    # ── Parse Date ────────────────────────────────────────────────────────────
    # investing.com format: "Jan 02, 2012"  →  %b %d, %Y
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y", errors="coerce")

    # Fallback cho format khác (không dùng infer_datetime_format — deprecated pandas 2.x)
    nat_mask = df["Date"].isna()
    if nat_mask.any():
        # Thử các format phổ biến khác
        for fmt in ["%b %d,%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]:
            still_nat = df["Date"].isna()
            if not still_nat.any():
                break
            df.loc[still_nat, "Date"] = pd.to_datetime(
                df_raw.loc[still_nat, "Date"], format=fmt, errors="coerce"
            )

    # Còn NaT → drop và cảnh báo
    remaining_nat = df["Date"].isna().sum()
    if remaining_nat > 0:
        logger.warning(
            f"[USDVND] ⚠️ {remaining_nat} dòng không parse được Date → drop."
        )
        df = df.dropna(subset=["Date"])

    # Parse Rate — có thể có dấu phẩy hàng nghìn: "21,031.0"
    df["Rate"] = (
        df["Rate_raw"]
        .astype(str)
        .str.replace(",", "", regex=False)  # bỏ dấu phẩy nghìn
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Drop cột trung gian, set index
    df = df.drop(columns=["Rate_raw"])
    df = df.set_index("Date")
    df = df.sort_index()   # ascending

    # Validate range
    if df["Rate"].isna().any():
        n_nan = df["Rate"].isna().sum()
        logger.warning(f"[USDVND] ⚠️ {n_nan} giá trị Rate bị NaN — sẽ được forward-fill.")
        df["Rate"] = df["Rate"].ffill()

    logger.info(
        f"[USDVND] ✅ Load thành công | "
        f"Rows: {len(df):,} | "
        f"Range: {df.index[0].date()} → {df.index[-1].date()} | "
        f"Rate min/max: {df['Rate'].min():,.0f} / {df['Rate'].max():,.0f} VND/USD"
    )
    return df


def convert_to_usd(
    df_vnd: pd.DataFrame,
    df_fx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuyển đổi DataFrame giá cổ phiếu từ VND sang USD.

    Cơ chế:
      - Join df_vnd với df_fx (tỷ giá) theo Date
      - Forward-fill tỷ giá cho ngày không có (weekend/holiday)
      - Chia Open, High, Low, Close (và Deviation nếu có) cho tỷ giá
      - Volume giữ nguyên (đơn vị cổ phiếu, không đổi)

    Args:
        df_vnd: DataFrame VND, index=Date, cần có [Open, High, Low, Close, Volume].
                Có thể có thêm cột Deviation.
        df_fx:  DataFrame tỷ giá, index=Date, cột [Rate] (VND per 1 USD).

    Returns:
        DataFrame USD, cùng cấu trúc với df_vnd,
        thêm cột 'currency'='USD' để phân biệt.
    """
    df = df_vnd.copy()

    # Reindex tỷ giá theo ngày của df_vnd, forward-fill (cuối tuần/lễ)
    # Dùng reindex + ffill thay vì merge để giữ nguyên index của df_vnd
    rate_series = (
        df_fx["Rate"]
        .reindex(df.index, method="ffill")  # forward-fill ngày thiếu
    )

    # Nếu vẫn còn NaN (ngày trước range của df_fx) → backward-fill
    rate_series = rate_series.bfill()

    if rate_series.isna().any():
        n_missing = rate_series.isna().sum()
        logger.warning(
            f"[convert_to_usd] ⚠️ Vẫn còn {n_missing} ngày không có tỷ giá "
            "sau ffill+bfill. Kiểm tra lại range của USDVND.csv."
        )

    # Các cột cần chia (Price cols) — Volume KHÔNG chia
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Deviation"]
                  if c in df.columns]

    # Chia cho tỷ giá (VND → USD)
    df[price_cols] = df[price_cols].div(rate_series, axis=0)

    # Đánh dấu currency
    df["currency"] = "USD"

    logger.info(
        f"[convert_to_usd] ✅ Đã convert {len(df):,} rows | "
        f"Tỷ giá trung bình: {rate_series.mean():,.0f} VND/USD"
    )
    return df


def prepare_both_currencies(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Load dữ liệu cổ phiếu và trả về cả hai phiên bản VND và USD.

    Args:
        ticker: Mã cổ phiếu viết hoa, ví dụ "VCB".

    Returns:
        dict với 2 keys:
          "VND" → DataFrame giá gốc VND
          "USD" → DataFrame đã convert sang USD
    """
    # Load cổ phiếu (VND)
    df_vnd = load_stock_data(ticker)

    # Load tỷ giá
    df_fx = load_usdvnd()

    # Convert sang USD
    df_usd = convert_to_usd(df_vnd, df_fx)

    logger.info(
        f"[{ticker}] prepare_both_currencies OK | "
        f"VND rows: {len(df_vnd):,} | USD rows: {len(df_usd):,}"
    )

    return {
        "VND": df_vnd,
        "USD": df_usd,
    }


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