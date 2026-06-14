"""
scripts/preprocess.py
======================
Standalone script — tạo processed data cho tất cả 4 combinations.

Chạy: uv run python scripts/preprocess.py

Pipeline cho mỗi combination (ticker × wavelet_condition):
  1. Load raw data từ CSV (không gọi API)
  2. Add Deviation feature (Close - Open)
  3. Nếu use_wavelet=True:
       a. Apply SWT db4 level-1 → 10 wavelet coefficients + Close (11 cols)
       b. Feature selection by correlation (threshold=0.95)
          → fit trên Fold 1 training data (2012–2017) để tránh data leakage
          → áp dụng kết quả lên toàn bộ dataset
     Nếu use_wavelet=False:
       Giữ 6 cột: Open, High, Low, Volume, Deviation (features) + Close (target)
  4. Lưu pkl: {"df": df_processed, "feature_cols": list, "target_col": "Close"}

Output: data/processed/{ticker}_VND_{wavelet|nowave}.pkl

4 combinations tổng cộng:
  VCB_VND_wavelet, VCB_VND_nowave
  VIC_VND_wavelet, VIC_VND_nowave

Ghi chú: Chỉ hỗ trợ VND — USD đã bị loại theo yêu cầu giảng viên.
"""

import logging
import pickle
import sys
import time
from itertools import product
from pathlib import Path

# ── Thêm project root vào sys.path để import app.* ──────────────────────────
# Script nằm ở scripts/, project root là thư mục cha
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.config import (
    CORRELATION_THRESHOLD,
    CURRENCIES,
    FOLDS,
    PATHS,
    TARGET_COL,
    TICKERS,
    WAVELET_CONDITIONS,
)
from app.services.data_service import load_stock_data
from app.services.preprocessing import (
    add_deviation_feature,
    select_features_by_correlation,
)
from app.services.wavelet_service import decompose_all_features

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(PATHS["processed"])
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Hằng số ───────────────────────────────────────────────────────────────────
# Fold 1 train end — dùng để fit correlation selection, tránh data leakage
# (chỉ dùng dữ liệu lịch sử để quyết định features giữ lại)
FOLD1_TRAIN_END: str = FOLDS[0]["train_end"]  # "2017-12-31"

# Input features cho no-wavelet case (Close là target, không tính là feature)
RAW_FEATURE_COLS: list[str] = ["Open", "High", "Low", "Volume", "Deviation"]

# Columns chuẩn OHLCV + Deviation — dùng để lọc sau add_deviation
BASE_COLS: list[str] = ["Open", "High", "Low", "Close", "Volume", "Deviation"]


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def run_preprocessing(ticker: str, currency: str, use_wavelet: bool) -> dict:
    """
    Chạy toàn bộ preprocessing pipeline cho một combination.

    Quy trình:
      1. Load raw CSV → DataFrame (index=Date, cols=OHLCV)
      2. Thêm Deviation = Close - Open
      3a. Wavelet: SWT decompose → feature selection → df_processed
      3b. No-wavelet: select 6 raw cols → df_processed
      4. Trả về dict để lưu pkl

    Args:
        ticker:      Mã cổ phiếu ("VCB" hoặc "VIC").
        currency:    Đơn vị tiền tệ — luôn là "VND".
        use_wavelet: True → pipeline SWT + feature selection.
                     False → giữ 6 raw features.

    Returns:
        dict:
          "df"          : DataFrame đã xử lý, index=DatetimeIndex,
                          columns = feature_cols + [target_col].
          "feature_cols": List tên input features (KHÔNG gồm "Close").
          "target_col"  : "Close" (giá đóng cửa ngày t+1, mục tiêu dự báo).

    Raises:
        FileNotFoundError: Nếu CSV raw chưa tồn tại.
        KeyError:          Nếu columns cần thiết không có.
    """
    label = f"{ticker}_{currency}_{'wavelet' if use_wavelet else 'nowave'}"

    # ── Bước 1: Load raw data từ CSV ─────────────────────────────────────────
    # load_stock_data() tự dùng CSV cache, KHÔNG gọi API lại
    df_raw = load_stock_data(ticker)
    logger.info(f"[{label}] Loaded {len(df_raw):,} rows | "
                f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    df = df_raw.copy()

    # ── Bước 2: Thêm Deviation feature ───────────────────────────────────────
    df = add_deviation_feature(df)

    # Chỉ giữ các cột chuẩn (loại bỏ cột thừa nếu có từ data source)
    df = df[[c for c in BASE_COLS if c in df.columns]]

    missing_base = [c for c in BASE_COLS if c not in df.columns]
    if missing_base:
        raise KeyError(f"[{label}] Thiếu columns sau add_deviation: {missing_base}")

    # ── Bước 3: Wavelet hoặc No-wavelet ──────────────────────────────────────
    if use_wavelet:
        df_processed, feature_cols = _pipeline_wavelet(df, label)
    else:
        df_processed, feature_cols = _pipeline_nowave(df, label)

    # ── Validate output ───────────────────────────────────────────────────────
    assert TARGET_COL in df_processed.columns, (
        f"[{label}] '{TARGET_COL}' không có trong df_processed!"
    )
    assert len(feature_cols) > 0, f"[{label}] feature_cols rỗng!"
    assert not df_processed.isnull().any().any(), (
        f"[{label}] df_processed có NaN — kiểm tra pipeline!"
    )

    logger.info(
        f"[{label}] ✅ Hoàn tất | "
        f"rows={len(df_processed):,} | "
        f"features={len(feature_cols)} {feature_cols} | "
        f"NaN=0"
    )

    return {
        "df":           df_processed,
        "feature_cols": feature_cols,
        "target_col":   TARGET_COL,
    }


def _pipeline_wavelet(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Sub-pipeline cho wavelet case:
      SWT decompose → feature selection → return (df_processed, feature_cols)

    Feature selection được fit trên Fold 1 training data (2012–2017)
    để tránh data leakage, sau đó áp dụng lên toàn bộ dataset.

    Args:
        df:    DataFrame OHLCV + Deviation (chưa decompose).
        label: Nhãn log (ticker_VND_wavelet).

    Returns:
        (df_processed, feature_cols):
          df_processed: DataFrame với wavelet features đã select + Close.
          feature_cols: List tên các input features được giữ.
    """
    # 3a. Apply SWT db4 level-1 cho 5 features (Open, High, Low, Volume, Deviation)
    # → 10 wavelet coefficients (Approx + Detail × 5) + Close (giữ nguyên)
    logger.info(f"[{label}] SWT decompose...")
    df_wav = decompose_all_features(df)
    # df_wav columns (11 cols):
    #   Open_Approx, Open_Detail, High_Approx, High_Detail, Low_Approx, Low_Detail,
    #   Volume_Approx, Volume_Detail, Deviation_Approx, Deviation_Detail, Close

    # 3b. Feature selection by correlation
    #     → fit trên Fold 1 train data (đến 2017-12-31) để tránh leakage
    df_fold1_train = df_wav.loc[:FOLD1_TRAIN_END]

    if len(df_fold1_train) == 0:
        raise ValueError(
            f"[{label}] Fold 1 train data rỗng (đến {FOLD1_TRAIN_END}). "
            "Kiểm tra lại date range của dữ liệu."
        )

    logger.info(
        f"[{label}] Feature selection trên Fold 1 train: "
        f"{len(df_fold1_train):,} rows (đến {FOLD1_TRAIN_END})"
    )

    df_fold1_selected, dropped = select_features_by_correlation(
        df_fold1_train,
        threshold=CORRELATION_THRESHOLD,
        target_col=TARGET_COL,
    )

    # feature_cols = các cột được giữ (không gồm Close)
    feature_cols = [c for c in df_fold1_selected.columns if c != TARGET_COL]

    # Áp dụng selection lên toàn bộ dataset (giữ cùng cột)
    df_processed = df_wav[feature_cols + [TARGET_COL]].copy()

    logger.info(
        f"[{label}] Feature selection: "
        f"{len(feature_cols)} kept, "
        f"{len(dropped)} dropped: {dropped}"
    )

    return df_processed, feature_cols


def _pipeline_nowave(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Sub-pipeline cho no-wavelet case:
      Giữ 6 raw columns (Open, High, Low, Volume, Deviation, Close).

    Không cần feature selection vì raw features ít (5 inputs) và
    không có redundancy Approx/Detail như wavelet case.

    Args:
        df:    DataFrame OHLCV + Deviation.
        label: Nhãn log.

    Returns:
        (df_processed, feature_cols):
          df_processed: DataFrame với 5 input features + Close.
          feature_cols: ["Open", "High", "Low", "Volume", "Deviation"].
    """
    feature_cols = RAW_FEATURE_COLS.copy()  # ["Open", "High", "Low", "Volume", "Deviation"]
    df_processed = df[feature_cols + [TARGET_COL]].copy()

    logger.info(
        f"[{label}] No-wavelet: giữ {len(feature_cols)} raw features: {feature_cols}"
    )

    return df_processed, feature_cols


# =============================================================================
# SAVE HELPER
# =============================================================================

def save_pickle(
    data: dict,
    ticker: str,
    currency: str,
    use_wavelet: bool,
) -> Path:
    """
    Lưu processed data dict ra file pkl.

    Naming convention: {ticker}_{currency}_{wavelet|nowave}.pkl
    Ví dụ: VCB_VND_wavelet.pkl, VIC_VND_nowave.pkl

    Args:
        data:        Dict {"df": ..., "feature_cols": ..., "target_col": ...}.
        ticker:      Mã cổ phiếu.
        currency:    Tiền tệ (luôn là "VND").
        use_wavelet: True → "wavelet", False → "nowave".

    Returns:
        Path object của file đã lưu.
    """
    suffix = "wavelet" if use_wavelet else "nowave"
    filename = f"{ticker}_{currency}_{suffix}.pkl"
    output_path = PROCESSED_DIR / filename

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"[save] → {output_path} ({size_kb:.1f} KB)")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Chạy toàn bộ 4 combinations (2T × 1C × 2W), in tiến độ và bảng tóm tắt.
    """
    combos = list(product(TICKERS, CURRENCIES, WAVELET_CONDITIONS))
    n_combos = len(combos)

    print("\n" + "=" * 72)
    print("VNSP — Phase 2: Preprocessing Pipeline (VND only)")
    print("=" * 72)
    print(f"Output dir    : {PROCESSED_DIR.resolve()}")
    print(f"Combinations  : {len(TICKERS)} tickers × {len(CURRENCIES)} currency "
          f"× {len(WAVELET_CONDITIONS)} wavelet conditions = {n_combos}")
    print(f"Fold 1 cutoff : {FOLD1_TRAIN_END} (dùng cho feature selection)")
    print(f"Corr threshold: {CORRELATION_THRESHOLD}")
    print()

    summary_rows: list[dict] = []
    total_start = time.time()

    for idx, (ticker, currency, use_wavelet) in enumerate(combos, start=1):
        label = f"{ticker}_{currency}_{'wavelet' if use_wavelet else 'nowave'}"
        print(f"  [{idx:>2}/{n_combos}] {label:<30}", end=" ", flush=True)
        t_start = time.time()

        try:
            # Chạy pipeline
            result = run_preprocessing(ticker, currency, use_wavelet)

            # Lưu pkl
            out_path = save_pickle(result, ticker, currency, use_wavelet)

            elapsed = time.time() - t_start
            df = result["df"]
            n_feat = len(result["feature_cols"])

            # Lưu vào summary
            summary_rows.append({
                "ticker":     ticker,
                "currency":   currency,
                "wavelet":    "Yes" if use_wavelet else "No",
                "n_features": n_feat,
                "n_rows":     len(df),
                "date_start": str(df.index[0].date()),
                "date_end":   str(df.index[-1].date()),
                "file_kb":    f"{out_path.stat().st_size / 1024:.1f}",
                "elapsed_s":  f"{elapsed:.1f}s",
                "status":     "OK",
            })
            print(f"✅  {n_feat} features | {len(df):,} rows | {elapsed:.1f}s")

        except Exception as exc:
            elapsed = time.time() - t_start
            logger.error(f"[{label}] ❌ {exc}", exc_info=True)
            summary_rows.append({
                "ticker":     ticker,
                "currency":   currency,
                "wavelet":    "Yes" if use_wavelet else "No",
                "n_features": "-",
                "n_rows":     "-",
                "date_start": "-",
                "date_end":   "-",
                "file_kb":    "-",
                "elapsed_s":  f"{elapsed:.1f}s",
                "status":     f"ERROR: {exc}",
            })
            print(f"❌  {exc}")

    total_elapsed = time.time() - total_start

    # ── Bảng tóm tắt ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BẢNG TÓM TẮT")
    print("=" * 72)
    hdr = (
        f"{'Ticker':<6}  {'Currency':<8}  {'Wavelet':<7}  "
        f"{'Feats':>5}  {'Rows':>6}  "
        f"{'Start':<12}  {'End':<12}  "
        f"{'KB':>7}  {'Time':>6}  Status"
    )
    print(hdr)
    print("-" * 72)

    for row in summary_rows:
        print(
            f"{row['ticker']:<6}  {row['currency']:<8}  {row['wavelet']:<7}  "
            f"{str(row['n_features']):>5}  {str(row['n_rows']):>6}  "
            f"{row['date_start']:<12}  {row['date_end']:<12}  "
            f"{str(row['file_kb']):>7}  {row['elapsed_s']:>6}  {row['status']}"
        )

    n_ok  = sum(1 for r in summary_rows if r["status"] == "OK")
    n_err = len(summary_rows) - n_ok
    print("=" * 72)
    print(f"Kết quả: {n_ok}/{n_combos} OK  |  {n_err} lỗi  |  "
          f"Tổng thời gian: {total_elapsed:.1f}s")

    if n_ok == n_combos:
        print()
        print("✅ Tất cả processed files đã sẵn sàng!")
        print(f"   Kiểm tra: {PROCESSED_DIR.resolve()}")
        print()
        print("Bước tiếp theo:")
        print("  uv run python scripts/run_experiments.py  (Phase 4 — Experiments)")
    else:
        print()
        print(f"⚠️  {n_err} combination(s) bị lỗi. Xem log chi tiết ở trên.")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()