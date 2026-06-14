"""
scripts/download_data.py
=========================
Standalone script — tải và kiểm tra toàn bộ raw data files.

Chạy: uv run python scripts/download_data.py

Logic:
  1. Kiểm tra VCB_raw.csv / VIC_raw.csv — nếu thiếu: tải từ vnstock
  2. Nếu đã có tất cả: bỏ qua download, chỉ in summary
  3. In summary table cuối cùng
  4. In hướng dẫn bước tiếp theo

Ghi chú: USDVND.csv không còn cần thiết — chỉ hỗ trợ VND.
"""

import sys
from pathlib import Path

# ── Thêm project root vào sys.path để import được app.*  ──────────────────────
# Script nằm ở scripts/, project root là thư mục cha
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import date

import pandas as pd

from app.config import DATE_START, DATE_END, PATHS, TICKERS

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
# Helpers
# =============================================================================

def _download_ticker(ticker: str) -> bool:
    """
    Tải dữ liệu 1 ticker nếu CSV chưa có.
    Return True nếu thành công (hoặc đã có sẵn).
    """
    from app.services.data_service import download_stock_data

    csv_path = RAW_DIR / f"{ticker}_raw.csv"
    if csv_path.exists():
        logger.info(f"[{ticker}] ✅ CSV đã tồn tại — bỏ qua download.")
        return True

    logger.info(f"[{ticker}] CSV chưa có — bắt đầu tải từ vnstock...")
    try:
        download_stock_data(ticker)
        return True
    except Exception as e:
        logger.error(f"[{ticker}] ❌ Lỗi khi tải: {e}")
        return False


def _count_missing_days(df: pd.DataFrame) -> int:
    """
    Đếm số 'khoảng trống' lớn (> 7 ngày) trong dữ liệu giao dịch.
    Dùng để phát hiện dữ liệu bị thiếu bất thường (không phải weekend/lễ).
    """
    diffs = df.index.to_series().diff().dt.days.dropna()
    # > 7 ngày: weekend (2-3 ngày) + Tết tối đa 6-7 ngày → ngưỡng 7 hợp lý
    return int((diffs > 7).sum())


def _print_summary(results: dict[str, pd.DataFrame]) -> None:
    """
    In bảng tóm tắt tất cả các file dữ liệu đã có.
    Columns: ticker, date_range, rows, missing_gaps.
    """
    print("\n" + "=" * 72)
    print(f"{'Ticker':<8}  {'Start':<12}  {'End':<12}  {'Rows':>6}  {'Gaps>7d':>8}  {'Notes'}")
    print("-" * 72)

    for ticker, df in results.items():
        start = str(df.index[0].date())
        end   = str(df.index[-1].date())
        rows  = len(df)
        gaps  = _count_missing_days(df)

        # Kiểm tra cover đủ range yêu cầu
        expected_start = pd.Timestamp(DATE_START)
        expected_end   = pd.Timestamp(DATE_END)
        notes = []
        if df.index[0] > expected_start:
            notes.append(f"start muộn hơn {DATE_START}")
        if df.index[-1] < expected_end:
            notes.append(f"end sớm hơn {DATE_END}")
        note_str = "; ".join(notes) if notes else "OK"

        print(
            f"{ticker:<8}  {start:<12}  {end:<12}  {rows:>6,}  "
            f"{gaps:>8}  {note_str}"
        )

    print("=" * 72 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Entry point chính của script.
    Thứ tự:
      1. Download cổ phiếu nếu thiếu (VND only — không cần USDVND.csv)
      2. Load lại để in summary
      3. In hướng dẫn bước tiếp theo
    """
    print("\n" + "═" * 65)
    print("  VNSP — Data Download Script (VND only)")
    print(f"  Range: {DATE_START} → {DATE_END}")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print("═" * 65)

    # Đảm bảo thư mục raw tồn tại
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Bước 1: Cổ phiếu ─────────────────────────────────────────────────────
    all_csv_exist = all(
        (RAW_DIR / f"{t}_raw.csv").exists() for t in TICKERS
    )

    if all_csv_exist:
        print("\nAll data files present. Skipping download.\n")
    else:
        print()
        success_flags = [_download_ticker(t) for t in TICKERS]
        if not all(success_flags):
            failed = [t for t, ok in zip(TICKERS, success_flags) if not ok]
            logger.error(f"❌ Tải thất bại cho: {failed}. Kiểm tra kết nối và chạy lại.")
            sys.exit(1)

    # ── Bước 2: Load và in summary ────────────────────────────────────────────
    results: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        csv_path = RAW_DIR / f"{ticker}_raw.csv"
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df = df.sort_index()
            results[ticker] = df
        except Exception as e:
            logger.error(f"[{ticker}] Không đọc được CSV sau khi tải: {e}")

    _print_summary(results)

    # ── Bước 3: Hướng dẫn tiếp theo ──────────────────────────────────────────
    print("✅ Data ready. Run preprocessing next:")
    print("   uv run python scripts/preprocess.py\n")


if __name__ == "__main__":
    main()