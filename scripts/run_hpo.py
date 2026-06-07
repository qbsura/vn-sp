"""
scripts/run_hpo.py
==================
Standalone script chạy Hyperparameter Optimization (HPO) cho BiLSTM.

PHẢI CHẠY TRƯỚC scripts/run_experiments.py.

Workflow đúng:
  1. scripts/preprocess.py     → data/processed/*.pkl
  2. scripts/run_hpo.py        → experiments/{ticker}_{currency}_{cond}/fold_{i}/best_params.json
  3. scripts/run_experiments.py → experiments/{exp_id}/fold_{i}/metrics.json

HPO chỉ chạy cho BiLSTM (30 trials × 3 folds = 90 Optuna trials / combination).
DNN, RNN, GRU, LSTM dùng chung best_params của BiLSTM (cùng fold + condition).

Ước tính thời gian (CPU):
  ~30–90 phút / combination (tuỳ data size và hardware)
  8 combinations tổng (2 tickers × 2 currencies × 2 wavelet) → ~4–12 giờ full run

Chạy:
  uv run python scripts/run_hpo.py                              # tất cả 8 combinations
  uv run python scripts/run_hpo.py --ticker VCB                 # chỉ VCB (4 combos)
  uv run python scripts/run_hpo.py --ticker VCB --currency VND --wavelet true
  uv run python scripts/run_hpo.py --trials 5                   # 5 trials (debug nhanh)

Resume: Skip combination nếu cả 3 fold best_params.json đã tồn tại.
"""

import argparse
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

# ── Add project root to sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CURRENCIES,
    FOLDS,
    OPTUNA_TRIALS,
    PATHS,
    TICKERS,
    WAVELET_CONDITIONS,
)
from app.services.hpo_service import load_best_params, run_full_hpo

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logging.getLogger("optuna").setLevel(logging.WARNING)

EXPERIMENTS_DIR = Path(PATHS["experiments"])


def _all_folds_done(ticker: str, currency: str, use_wavelet: bool) -> bool:
    """
    Kiểm tra tất cả 3 fold best_params.json đã tồn tại cho combination này.

    Resume logic: nếu cả 3 folds đã có → skip toàn bộ combination.
    """
    return all(
        (EXPERIMENTS_DIR
         / f"{ticker}_{currency}_{'wavelet' if use_wavelet else 'nowave'}"
         / f"fold_{fold['fold_id']}"
         / "best_params.json").exists()
        for fold in FOLDS
    )


def _parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{s}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VNSP HPO Runner — chạy Optuna HPO cho BiLSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy tất cả 8 combinations (full run, ~4-12 giờ CPU)
  uv run python scripts/run_hpo.py

  # Chỉ VCB + VND + wavelet (1 combination, ~30-90 phút)
  uv run python scripts/run_hpo.py --ticker VCB --currency VND --wavelet true

  # Debug nhanh: 5 trials/fold
  uv run python scripts/run_hpo.py --ticker VCB --currency VND --wavelet true --trials 5

  # Sau khi HPO xong, chạy experiments:
  uv run python scripts/run_experiments.py
        """,
    )
    parser.add_argument(
        "--ticker",
        choices=["VCB", "VIC"],
        default=None,
        help="Lọc theo ticker. None = cả VCB và VIC.",
    )
    parser.add_argument(
        "--currency",
        choices=["VND", "USD"],
        default=None,
        help="Lọc theo currency. None = cả VND và USD.",
    )
    parser.add_argument(
        "--wavelet",
        type=_parse_bool,
        default=None,
        metavar="true|false",
        help="Lọc wavelet condition. None = cả hai.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=OPTUNA_TRIALS,
        help=f"Số Optuna trials mỗi fold. Default: {OPTUNA_TRIALS}.",
    )

    args = parser.parse_args()

    # ── Build combination list ─────────────────────────────────────────────────
    tickers    = [args.ticker]   if args.ticker   else TICKERS
    currencies = [args.currency] if args.currency else CURRENCIES
    wavelets   = [args.wavelet]  if args.wavelet is not None else WAVELET_CONDITIONS
    combos     = list(product(tickers, currencies, wavelets))
    total      = len(combos)

    print(f"\n{'='*72}")
    print(f"VNSP — HPO Runner (BiLSTM)")
    print(f"{'='*72}")
    print(f"Combinations  : {total}  ({len(tickers)}T × {len(currencies)}C × {len(wavelets)}W)")
    print(f"Trials / fold : {args.trials}")
    print(f"Folds         : {len(FOLDS)}")
    print(f"Total trials  : {total * len(FOLDS) * args.trials:,}")
    print(f"Output dir    : {EXPERIMENTS_DIR.resolve()}")
    print(f"{'='*72}\n")

    done    = 0
    skipped = 0
    errors  = 0
    global_start = time.time()

    for idx, (ticker, currency, use_wavelet) in enumerate(combos, 1):
        cond_str = "wavelet" if use_wavelet else "nowave"
        label    = f"[{idx}/{total}] {ticker}_{currency}_{cond_str}"

        # ── Resume check ──────────────────────────────────────────────────────
        if _all_folds_done(ticker, currency, use_wavelet):
            skipped += 1
            print(f"  {label} — ⏭  SKIP (all 3 folds done)")
            continue

        print(f"  {label} | trials={args.trials} × {len(FOLDS)} folds ...", flush=True)
        t_start = time.time()

        try:
            all_params = run_full_hpo(
                ticker      = ticker,
                currency    = currency,
                use_wavelet = use_wavelet,
                n_trials    = args.trials,
            )
            elapsed = time.time() - t_start
            done += 1

            # Print best_val_loss per fold
            fold_summary = " | ".join(
                f"F{i}: {p['_meta']['best_val_loss']:.5f}"
                for i, p in enumerate(all_params, 1)
            )
            print(f"  {label} — ✅  {fold_summary} | {elapsed:.0f}s")

        except FileNotFoundError as exc:
            errors += 1
            print(f"  {label} — ❌  {exc}")
        except Exception as exc:
            errors += 1
            logging.getLogger(__name__).error(
                "%s — ERROR: %s", label, exc, exc_info=True
            )
            print(f"  {label} — ❌  ERROR: {exc}")

    total_elapsed = time.time() - global_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s   = divmod(rem, 60)
    time_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

    print(f"\n{'='*72}")
    print(f"KẾT QUẢ HPO:")
    print(f"  ✅  {done}/{total} combinations hoàn thành")
    print(f"  ⏭  {skipped} skipped (đã có kết quả)")
    print(f"  ❌  {errors} errors")
    print(f"  ⏱  Tổng thời gian: {time_str}")
    print(f"{'='*72}")

    if done > 0 or skipped == total:
        print(f"\nBước tiếp theo:")
        print(f"  uv run python scripts/run_experiments.py")
    print()


if __name__ == "__main__":
    main()