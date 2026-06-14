"""
scripts/run_experiments.py
============================
Standalone script chạy toàn bộ (hoặc một phần) experiment matrix VNSP.

Chạy:
  uv run python scripts/run_experiments.py                        # tất cả 240 runs
  uv run python scripts/run_experiments.py --ticker VCB           # chỉ VCB
  uv run python scripts/run_experiments.py --ticker VCB --currency VND --model BiLSTM
  uv run python scripts/run_experiments.py --task regression --wavelet true
  uv run python scripts/run_experiments.py --fold 1               # chỉ fold 1

  # Re-run Task B (classification) sau khi đổi target sang WEEKLY (T2-T6):
  # --force-rerun để overwrite 120 kết quả classification daily cũ.
  # Task A (regression) KHÔNG cần re-run.
  uv run python scripts/run_experiments.py --task classification --force-rerun

Lưu ý:
  - Chạy HPO trước (hpo_service.run_full_hpo) để có best_params.json.
  - Script này sẽ skip experiments đã có metrics.json (resume safe).
  - Mỗi experiment ≈ 1-5 phút tùy model/data size. Tổng ≈ 4-20 giờ.

Prerequisites:
  data/processed/*.pkl  — chạy scripts/preprocess.py trước
  experiments/*/fold_*/best_params.json — chạy HPO trước
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.experiment_runner import run_all_experiments

# ── Logging setup ─────────────────────────────────────────────────────────────
# Ghi log ra CẢ console VÀ file (logs/run_experiments_<timestamp>.log) — đảm bảo
# traceback đầy đủ của các experiment lỗi (logger.error(..., exc_info=True))
# không bị mất khi terminal scroll qua / không capture output.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"run_experiments_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                       # console (như cũ)
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # file — lưu traceback
    ],
)
print(f"[run_experiments] Log file: {LOG_FILE}")
# Suppress noisy loggers
logging.getLogger("optuna").setLevel(logging.WARNING)


def _parse_bool(s: str) -> bool:
    """Parse 'true'/'false'/'1'/'0' thành bool."""
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{s}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VNSP Experiment Runner — chạy training và evaluation matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy tất cả (cần HPO đã xong trước)
  uv run python scripts/run_experiments.py

  # Chỉ BiLSTM regression, tất cả tickers/currencies/wavelet/folds
  uv run python scripts/run_experiments.py --model BiLSTM --task regression

  # VCB, VND, wavelet, tất cả models, regression
  uv run python scripts/run_experiments.py --ticker VCB --currency VND \\
      --wavelet true --task regression

  # Test nhanh: 1 ticker, 1 currency, 1 wavelet, 1 model, 1 task, 1 fold
  uv run python scripts/run_experiments.py \\
      --ticker VCB --currency VND --wavelet true \\
      --model BiLSTM --task regression --fold 1

  # Re-run Task B (weekly classification, T2-T6) — overwrite kết quả cũ
  uv run python scripts/run_experiments.py --task classification --force-rerun
        """,
    )

    parser.add_argument(
        "--ticker",
        choices=["VCB", "VIC"],
        default=None,
        help="Lọc theo mã cổ phiếu. None = cả VCB và VIC.",
    )
    parser.add_argument(
        "--currency",
        choices=["VND"],     
        default=None,
        help="Lọc theo tiền tệ. Chỉ hỗ trợ VND.",
    )
    parser.add_argument(
        "--wavelet",
        type=_parse_bool,
        default=None,
        metavar="true|false",
        help="Lọc theo wavelet condition. None = cả hai.",
    )
    parser.add_argument(
        "--model",
        choices=["DNN", "RNN", "GRU", "LSTM", "BiLSTM"],
        default=None,
        help="Lọc theo model. None = tất cả 5 models.",
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default=None,
        help="Lọc theo task. None = cả regression và classification.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Lọc theo fold index (1, 2, hoặc 3). None = cả 3 folds.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Bỏ qua resume-skip, LUÔN chạy lại và overwrite metrics.json/"
            "predictions.npz/best_model.pt cũ. Dùng khi đổi target/logic "
            "(ví dụ Task B chuyển sang weekly T2-T6)."
        ),
    )

    args = parser.parse_args()

    results = run_all_experiments(
        ticker_filter   = args.ticker,
        currency_filter = args.currency,
        wavelet_filter  = args.wavelet,
        model_filter    = args.model,
        task_filter     = args.task,
        fold_filter     = args.fold,
        force_rerun     = args.force_rerun,
    )

    print(f"\nDone. {len(results)} experiments completed.")


if __name__ == "__main__":
    main()