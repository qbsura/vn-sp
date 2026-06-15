"""
scripts/run_hpo.py
==================
Standalone script chay Hyperparameter Optimization (HPO) cho BiLSTM.

PHAI CHAY TRUOC scripts/run_experiments.py.

Workflow dung:
  1. scripts/preprocess.py      -> data/processed/*.pkl
  2. scripts/run_hpo.py         -> experiments/.../best_params.json
                                -> experiments/.../best_params_classification.json
  3. scripts/run_experiments.py -> experiments/{exp_id}/fold_{i}/metrics.json

HPO chi chay cho BiLSTM (30 trials x 3 folds = 90 Optuna trials / combination).
DNN, RNN, GRU, LSTM dung chung best_params cua BiLSTM (cung fold + condition).

Task:
  --task regression      (default) Optimize MSE tren daily sequences
  --task classification  Optimize BCE loss tren weekly sequences (T2-T6 direction)
                         -> luu vao best_params_classification.json (rieng biet)

Chay (classification - 4 lenh can thiet):
  uv run python scripts/run_hpo.py --ticker VCB --wavelet true  --task classification
  uv run python scripts/run_hpo.py --ticker VCB --wavelet false --task classification
  uv run python scripts/run_hpo.py --ticker VIC --wavelet true  --task classification
  uv run python scripts/run_hpo.py --ticker VIC --wavelet false --task classification

Resume: Skip combination neu ca 3 fold best_params*.json da ton tai (task-specific).
"""

import argparse
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

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

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logging.getLogger("optuna").setLevel(logging.WARNING)

EXPERIMENTS_DIR = Path(PATHS["experiments"])


def _params_filename(task: str) -> str:
    """Ten file best_params dua tren task."""
    return (
        "best_params_classification.json"
        if task == "classification"
        else "best_params.json"
    )


def _all_folds_done(ticker: str, currency: str, use_wavelet: bool, task: str) -> bool:
    """
    Kiem tra tat ca 3 fold da co ket qua HPO HOP LE cho combination + task nay.

    Resume logic: chi SKIP khi ca 3 folds:
      1. File best_params*.json ton tai
      2. best_val_loss doc duoc tu file < 1e10 (khong phai inf / ket qua buggy)

    Neu bat ky fold nao thieu file HOAC co best_val_loss >= 1e10 (buggy):
      → tra ve False → HPO se chay lai cho combination do.
    """
    fname = _params_filename(task)
    cond_str = "wavelet" if use_wavelet else "nowave"

    for fold in FOLDS:
        path = (
            EXPERIMENTS_DIR
            / f"{ticker}_{currency}_{cond_str}"
            / f"fold_{fold['fold_id']}"
            / fname
        )

        # Kiem tra file ton tai
        if not path.exists():
            return False

        # Doc file va kiem tra best_val_loss hop le
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            val_loss = d.get("_meta", {}).get("best_val_loss", float("inf"))
            if val_loss >= 1e10:  # inf hoac qua lon = ket qua buggy (run loi truoc)
                return False
        except Exception:
            # JSON corrupt hoac khong doc duoc → chay lai
            return False

    return True


def _parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{s}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VNSP HPO Runner -- chay Optuna HPO cho BiLSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker",   choices=["VCB", "VIC"], default=None,
                        help="Loc theo ticker. None = ca VCB va VIC.")
    parser.add_argument("--currency", choices=["VND"], default=None,
                        help="Loc theo currency. Chi ho tro VND.")
    parser.add_argument("--wavelet",  type=_parse_bool, default=None, metavar="true|false",
                        help="Loc wavelet condition. None = ca hai.")
    parser.add_argument("--trials",   type=int, default=OPTUNA_TRIALS,
                        help=f"So Optuna trials moi fold. Default: {OPTUNA_TRIALS}.")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help=(
            "Task HPO. 'regression' (default): optimize MSE tren daily data. "
            "'classification': optimize BCE loss tren weekly data. "
            "Ket qua luu vao best_params.json hoac best_params_classification.json."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help=(
            "Xoa file best_params*.json cu (ke ca ket qua buggy co best_val_loss=inf) "
            "va chay lai HPO tu dau cho cac combination duoc chi dinh. "
            "Dung khi can override ket qua tu lan chay truoc bi loi."
        ),
    )

    args = parser.parse_args()

    tickers    = [args.ticker]   if args.ticker   else TICKERS
    currencies = [args.currency] if args.currency else CURRENCIES
    wavelets   = [args.wavelet]  if args.wavelet is not None else WAVELET_CONDITIONS
    combos     = list(product(tickers, currencies, wavelets))
    total      = len(combos)

    fname_out = _params_filename(args.task)

    print(f"\n{'='*72}")
    print(f"VNSP -- HPO Runner (BiLSTM | task={args.task})")
    print(f"{'='*72}")
    print(f"Combinations  : {total}  ({len(tickers)}T x {len(currencies)}C x {len(wavelets)}W)")
    print(f"Trials / fold : {args.trials}")
    print(f"Folds         : {len(FOLDS)}")
    print(f"Total trials  : {total * len(FOLDS) * args.trials:,}")
    print(f"Output file   : {fname_out}")
    print(f"Output dir    : {EXPERIMENTS_DIR.resolve()}")
    print(f"{'='*72}\n")

    done    = 0
    skipped = 0
    errors  = 0
    global_start = time.time()

    for idx, (ticker, currency, use_wavelet) in enumerate(combos, 1):
        cond_str = "wavelet" if use_wavelet else "nowave"
        label    = f"[{idx}/{total}] {ticker}_{currency}_{cond_str} (task={args.task})"

        # --force-rerun: xoa file best_params*.json cu cho combination nay
        # (ke ca ket qua buggy co best_val_loss=inf)
        if args.force_rerun:
            fname = _params_filename(args.task)
            deleted_any = False
            for fold in FOLDS:
                stale_path = (
                    EXPERIMENTS_DIR
                    / f"{ticker}_{currency}_{cond_str}"
                    / f"fold_{fold['fold_id']}"
                    / fname
                )
                if stale_path.exists():
                    stale_path.unlink()
                    deleted_any = True
                    logging.getLogger(__name__).info(
                        "[force-rerun] Deleted stale: %s", stale_path
                    )
            if deleted_any:
                print(f"  {label} -- force-rerun: da xoa file {fname} cu")

        # Resume check (task-aware, kiem tra best_val_loss hop le)
        if _all_folds_done(ticker, currency, use_wavelet, args.task):
            skipped += 1
            print(f"  {label} -- SKIP (all 3 folds done for task={args.task})")
            continue

        print(f"  {label} | trials={args.trials} x {len(FOLDS)} folds ...", flush=True)
        t_start = time.time()

        try:
            # run_full_hpo accepts task= param to switch regression/classification HPO
            all_params = run_full_hpo(
                ticker      = ticker,
                currency    = currency,
                use_wavelet = use_wavelet,
                n_trials    = args.trials,
                task        = args.task,       # NEW: pass task to HPO service
            )
            elapsed = time.time() - t_start
            done += 1

            fold_summary = " | ".join(
                f"F{i}: {p['_meta']['best_val_loss']:.5f}"
                for i, p in enumerate(all_params, 1)
            )
            print(f"  {label} -- OK  {fold_summary} | {elapsed:.0f}s")

        except FileNotFoundError as exc:
            errors += 1
            print(f"  {label} -- FAIL (FileNotFoundError): {exc}")
        except Exception as exc:
            errors += 1
            logging.getLogger(__name__).error("%s -- ERROR: %s", label, exc, exc_info=True)
            print(f"  {label} -- FAIL: {exc}")

    total_elapsed = time.time() - global_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s   = divmod(rem, 60)
    time_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

    print(f"\n{'='*72}")
    print(f"KET QUA HPO ({args.task.upper()}):")
    print(f"  OK    : {done}/{total} combinations hoan thanh")
    print(f"  SKIP  : {skipped} skipped (da co ket qua)")
    print(f"  FAIL  : {errors} errors")
    print(f"  TIME  : {time_str}")
    print(f"{'='*72}")

    if done > 0 or skipped == total:
        print(f"\nBuoc tiep theo (re-run classification experiments):")
        print(f"  uv run python scripts/run_experiments.py --task classification --force-rerun")
    print()


if __name__ == "__main__":
    main()