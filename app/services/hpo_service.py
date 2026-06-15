"""
app/services/hpo_service.py
============================
Hyperparameter Optimization (HPO) cho BiLSTM dùng Optuna TPE sampler.

Task 4.2 trong Phase 4 — Huấn luyện & Hyperparameter Optimization.

THIẾT KẾ HPO:
  - Chạy HPO riêng cho BiLSTM trên regression task (Task A).
  - DNN, RNN, GRU, LSTM dùng chung best_params của BiLSTM (cùng fold).
  - 3 fold × 1 set best_params = 3 JSON files per (ticker, currency, wavelet).

Search space (theo project spec / config.py HPO_SEARCH_SPACE):
  num_layers      : suggest_int(1, 3)
  hidden_units    : suggest_categorical([32, 64, 128, 256])
  dropout_rate    : suggest_categorical([0.1, 0.2, 0.3, 0.4, 0.5])
  learning_rate   : suggest_categorical([1e-4, 5e-4, 1e-3])
  batch_size      : suggest_categorical([32, 64, 128])
  sequence_length : suggest_categorical([15, 20, 30, 50])

HPO Validation split (cả regression lẫn classification):
  - Lấy fold's training period (ví dụ Fold 1: 2012→2017)
  - Chia 90% train / 10% val theo thứ tự thời gian (không shuffle)
  - Fit scaler chỉ trên 90% để tránh leakage trong HPO

Optimization targets:
  - task='regression':     Minimize MSE val_loss (scaled Close, daily t+1)
  - task='classification': Minimize BCE val_loss (binary direction, daily t+1)
    THIẾT KẾ (2026-06): Classification HPO dùng daily T+1 binary sequences
    thay vì weekly sequences để có đủ training samples (~1500/fold thay vì
    ~280/fold weekly). Kết quả lưu vào best_params_classification.json.
    Experiments thực tế VẪN dùng weekly sequences từ dataset_builder.

Public API:
  build_feature_info()  — tạo feature_info dict từ feature_cols list
  create_objective()    — Optuna objective cho regression (MSE)
  run_hpo_for_fold()    — chạy HPO, lưu best_params*.json
  run_full_hpo()        — chạy HPO cho cả 3 folds, load pkl tự động
  load_best_params()    — load best_params từ JSON (với fallback logic)

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 4.2: Hyperparameter Optimization — search space.
  Akiba et al. (2019): Optuna: A Next-generation HPO Framework.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable

import optuna
import pandas as pd

from app.config import (
    FOLDS,
    HPO_SEARCH_SPACE,
    OPTUNA_HPO_EPOCHS,
    OPTUNA_TRIALS,
    PATHS,
    SEED,
    TARGET_COL,
)
from app.models.bilstm import BiLSTM
from app.services.dataset_builder import StockDataset, build_sequences
from app.services.preprocessing import FeatureScaler
from app.services.training_service import train_model

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# Suppress Optuna info logs (chỉ hiện WARNING+)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = Path(PATHS["processed"])
EXPERIMENTS_DIR = Path(PATHS["experiments"])


# =============================================================================
# HELPERS
# =============================================================================

def _get_params_filename(task: str) -> str:
    """
    Tên file best_params dựa trên task.
    Classification HPO lưu riêng để không ghi đè regression params.
    """
    return (
        "best_params_classification.json"
        if task == "classification"
        else "best_params.json"
    )


def build_feature_info(feature_cols: list[str]) -> dict:
    """
    Tạo feature_info dict từ danh sách tên feature columns.

    Tự động tìm approx_indices và detail_indices bằng cách kiểm tra
    suffix "_Approx" / "_Detail" của tên cột.

    Args:
        feature_cols: List tên input features (KHÔNG gồm "Close").
                      Wavelet case: [..., "High_Approx", "High_Detail", ...]
                      No-wavelet:   ["Open", "High", "Low", "Volume", "Deviation"]

    Returns:
        dict với keys:
          "feature_cols"    (list[str])
          "n_features"      (int)
          "approx_indices"  (list[int])  : 0-based indices của _Approx cols
          "detail_indices"  (list[int])  : 0-based indices của _Detail cols
          "use_wavelet"     (bool)
    """
    approx_indices: list[int] = [
        i for i, col in enumerate(feature_cols) if col.endswith("_Approx")
    ]
    detail_indices: list[int] = [
        i for i, col in enumerate(feature_cols) if col.endswith("_Detail")
    ]
    use_wavelet: bool = len(approx_indices) > 0 or len(detail_indices) > 0

    return {
        "feature_cols":   feature_cols,
        "n_features":     len(feature_cols),
        "approx_indices": approx_indices,
        "detail_indices": detail_indices,
        "use_wavelet":    use_wavelet,
    }


def _save_best_params(
    best_params: dict,
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    fold_idx   : int,
    task       : str = "regression",
) -> Path:
    """
    Lưu best_params dict ra file JSON.

    Path:
      regression    : experiments/{cond}/fold_{i}/best_params.json
      classification: experiments/{cond}/fold_{i}/best_params_classification.json
    """
    cond_str = "wavelet" if use_wavelet else "nowave"
    fname    = _get_params_filename(task)
    out_dir  = EXPERIMENTS_DIR / f"{ticker}_{currency}_{cond_str}" / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    logger.info("[_save_best_params] Saved %s → %s", fname, out_path)
    return out_path


def load_best_params(
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    fold_idx   : int,
    task       : str = "regression",
) -> dict:
    """
    Load best_params từ file JSON.

    Logic:
      - task='regression'     → load best_params.json
      - task='classification' → load best_params_classification.json nếu tồn tại,
                                fallback sang best_params.json với cảnh báo.

    Raises:
        FileNotFoundError: Nếu cả 2 files đều không tồn tại.
    """
    cond_str = "wavelet" if use_wavelet else "nowave"
    base_dir = EXPERIMENTS_DIR / f"{ticker}_{currency}_{cond_str}" / f"fold_{fold_idx}"

    # ── Classification: try specific file first, fallback to regression ────────
    if task == "classification":
        cls_path = base_dir / "best_params_classification.json"
        if cls_path.exists():
            logger.debug("[load_best_params] Loading classification params: %s", cls_path)
            with open(cls_path, encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(
                "[load_best_params] best_params_classification.json không tồn tại "
                "(%s fold %d). Fallback sang regression params. "
                "Chạy: uv run python scripts/run_hpo.py --task classification "
                "--ticker %s --wavelet %s",
                f"{ticker}_{currency}_{cond_str}", fold_idx, ticker,
                "true" if use_wavelet else "false",
            )

    # ── Regression (default) hoặc classification fallback ─────────────────────
    reg_path = base_dir / "best_params.json"
    if not reg_path.exists():
        raise FileNotFoundError(
            f"best_params.json không tồn tại: {reg_path}. "
            f"Chạy HPO trước: uv run python scripts/run_hpo.py "
            f"--ticker {ticker} --wavelet {'true' if use_wavelet else 'false'}"
        )
    with open(reg_path, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 1. REGRESSION OBJECTIVE (daily t+1, MSE loss)
# =============================================================================

def create_objective(
    df_processed  : pd.DataFrame,
    fold          : dict,
    feature_cols  : list[str],
    n_features    : int,
    use_wavelet   : bool,
    approx_indices: list[int],
    detail_indices: list[int],
) -> Callable:
    """
    Tạo Optuna objective function để tối ưu BiLSTM hyperparameters (regression).

    Validation split:
      - Lấy fold's training portion (index <= fold["train_end"])
      - Chronological 90% / 10% split (không shuffle)
      - Fit FeatureScaler CHỈ trên 90% train (anti-leakage)
      - Max OPTUNA_HPO_EPOCHS epochs per trial

    Returns float("inf") nếu trial fail → Optuna tiếp tục với trial khác.

    Returns:
        objective(trial) -> float (MSE val_loss, minimize).
    """
    train_end = pd.Timestamp(fold["train_end"])
    fold_id   = fold.get("fold_id", "?")

    def objective(trial: optuna.Trial) -> float:
        # ── 1. Suggest hyperparameters ────────────────────────────────────────
        params: dict = {
            "num_layers":      trial.suggest_int(
                "num_layers",
                min(HPO_SEARCH_SPACE["num_layers"]),
                max(HPO_SEARCH_SPACE["num_layers"]),
            ),
            "hidden_units":    trial.suggest_categorical(
                "hidden_units", HPO_SEARCH_SPACE["hidden_units"]
            ),
            "dropout_rate":    trial.suggest_categorical(
                "dropout_rate", HPO_SEARCH_SPACE["dropout_rate"]
            ),
            "learning_rate":   trial.suggest_categorical(
                "learning_rate", HPO_SEARCH_SPACE["learning_rate"]
            ),
            "batch_size":      trial.suggest_categorical(
                "batch_size", HPO_SEARCH_SPACE["batch_size"]
            ),
            "sequence_length": trial.suggest_categorical(
                "sequence_length", HPO_SEARCH_SPACE["sequence_length"]
            ),
        }
        params["use_wavelet"] = use_wavelet
        if use_wavelet:
            params["approx_indices"] = approx_indices
            params["detail_indices"] = detail_indices

        seq_len: int = params["sequence_length"]

        try:
            # ── 2. Lấy training portion của fold ─────────────────────────────
            df_train_full = df_processed[df_processed.index <= train_end].copy()
            if len(df_train_full) == 0:
                return float("inf")

            # ── 3. Chronological 90%/10% split ───────────────────────────────
            n_total   = len(df_train_full)
            split_idx = int(n_total * 0.9)
            df_hpo_train = df_train_full.iloc[:split_idx]
            df_hpo_val   = df_train_full.iloc[split_idx:]

            min_rows = seq_len + 1
            if len(df_hpo_train) <= min_rows or len(df_hpo_val) <= min_rows:
                return float("inf")

            # ── 4. Scale: fit CHỈ trên HPO train ─────────────────────────────
            scaler = FeatureScaler()
            df_hpo_train_scaled = scaler.fit_transform(df_hpo_train)
            df_hpo_val_scaled   = scaler.transform(df_hpo_val)

            # ── 5. Build daily regression sequences ───────────────────────────
            X_train, y_train = build_sequences(
                df_hpo_train_scaled, seq_len,
                target_col = TARGET_COL,
                task       = "regression",
            )
            X_val, y_val = build_sequences(
                df_hpo_val_scaled, seq_len,
                target_col = TARGET_COL,
                task       = "regression",
            )

            train_ds = StockDataset(X_train, y_train, task="regression")
            val_ds   = StockDataset(X_val,   y_val,   task="regression")

            # ── 6. Build BiLSTM và train ──────────────────────────────────────
            model = BiLSTM(
                task       = "regression",
                n_features = n_features,
                params     = params,
            )
            result = train_model(
                model         = model,
                train_dataset = train_ds,
                val_dataset   = val_ds,
                params        = params,
                task          = "regression",
                max_epochs    = OPTUNA_HPO_EPOCHS,
                save_path     = None,
            )

            val_loss: float = result["best_val_loss"]
            logger.debug(
                "[HPO reg Fold %s | Trial %d] MSE=%.6f | params=%s",
                fold_id, trial.number, val_loss,
                {k: v for k, v in params.items()
                 if k not in ("approx_indices", "detail_indices", "use_wavelet")},
            )
            return val_loss

        except Exception as exc:
            logger.warning("[HPO reg Fold %s | Trial %d] Failed: %s",
                           fold_id, trial.number, exc)
            return float("inf")

    return objective


# =============================================================================
# 2. CLASSIFICATION OBJECTIVE (daily T+1, BCE loss)
# =============================================================================

def _create_classification_objective(
    df_processed : pd.DataFrame,
    fold         : dict,
    feature_info : dict,
    use_wavelet  : bool,
) -> Callable:
    """
    Tạo Optuna objective function cho classification HPO.

    THIẾT KẾ (2026-06 — Daily T+1 thay vì Weekly):
      Vấn đề cũ: weekly HPO chỉ có ~280 train samples/fold → Optuna không hội tụ
      → best_val_loss=inf → toàn bộ kết quả classification vô nghĩa.

      Giải pháp: dùng daily T+1 binary sequences (cùng cấu trúc với regression):
        - ~1500 daily train samples/fold (Fold 1: 2012→2017) → HPO hội tụ tốt
        - Target: y[i] = 1 nếu Close(t+1) > Close(t), else 0
          Dùng giá GỐC (unscaled) để tính direction (tránh distortion từ scaling)
        - Loss: Binary Cross-Entropy (BCE) — phù hợp binary classification
        - Lưu: best_params_classification.json (không ghi đè regression params)

      Lý luận về proxy: hyperparams tốt cho daily binary direction prediction
      là proxy tốt cho weekly direction prediction — cùng model architecture,
      cùng feature space, chỉ khác temporal granularity của target.
      Experiments thực tế (run_experiments.py) vẫn dùng weekly sequences.

    Args:
        df_processed:  Full processed DataFrame (chưa scale, DatetimeIndex).
        fold:          Fold definition dict (fold_id, train_end, test_start, test_end).
        feature_info:  Dict từ build_feature_info() — n_features, approx/detail indices.
        use_wavelet:   True = dual-branch BiLSTM.

    Returns:
        objective(trial: optuna.Trial) -> float (BCE val_loss, minimize)
    """
    train_end = pd.Timestamp(fold["train_end"])
    fold_id   = fold.get("fold_id", "?")
    n_feat    = feature_info["n_features"]

    def objective(trial: optuna.Trial) -> float:
        # ── 1. Sample hyperparameters ─────────────────────────────────────────
        params: dict = {
            "num_layers":      trial.suggest_int(
                "num_layers",
                min(HPO_SEARCH_SPACE["num_layers"]),
                max(HPO_SEARCH_SPACE["num_layers"]),
            ),
            "hidden_units":    trial.suggest_categorical(
                "hidden_units", HPO_SEARCH_SPACE["hidden_units"]
            ),
            "dropout_rate":    trial.suggest_categorical(
                "dropout_rate", HPO_SEARCH_SPACE["dropout_rate"]
            ),
            "learning_rate":   trial.suggest_categorical(
                "learning_rate", HPO_SEARCH_SPACE["learning_rate"]
            ),
            "batch_size":      trial.suggest_categorical(
                "batch_size", HPO_SEARCH_SPACE["batch_size"]
            ),
            "sequence_length": trial.suggest_categorical(
                "sequence_length", HPO_SEARCH_SPACE["sequence_length"]
            ),
        }
        params["use_wavelet"] = use_wavelet
        if use_wavelet:
            params["approx_indices"] = feature_info["approx_indices"]
            params["detail_indices"] = feature_info["detail_indices"]

        seq_len: int = params["sequence_length"]

        try:
            # ── 2. Lấy training portion của fold ─────────────────────────────
            df_train_full = df_processed[df_processed.index <= train_end].copy()
            if len(df_train_full) == 0:
                logger.warning(
                    "[HPO cls Fold %s | Trial %d] Training data rỗng (train_end=%s)",
                    fold_id, trial.number, train_end,
                )
                return float("inf")

            # ── 3. Chronological 90%/10% split ───────────────────────────────
            # KHÔNG shuffle — time series order phải được bảo toàn
            n_total   = len(df_train_full)
            split_idx = int(n_total * 0.9)
            df_hpo_train = df_train_full.iloc[:split_idx]
            df_hpo_val   = df_train_full.iloc[split_idx:]

            min_rows = seq_len + 1
            if len(df_hpo_train) <= min_rows:
                logger.debug(
                    "[HPO cls Fold %s | Trial %d] HPO train quá nhỏ: %d <= %d",
                    fold_id, trial.number, len(df_hpo_train), min_rows,
                )
                return float("inf")
            if len(df_hpo_val) <= min_rows:
                logger.debug(
                    "[HPO cls Fold %s | Trial %d] HPO val quá nhỏ: %d <= %d",
                    fold_id, trial.number, len(df_hpo_val), min_rows,
                )
                return float("inf")

            # ── 4. Lưu Close GỐC TRƯỚC khi scale ────────────────────────────
            # Target: y[i] = (Close[t+1] > Close[t]) ? 1 : 0
            # Dùng giá THỰC (unscaled) — scaling có thể flip dấu chênh lệch nhỏ
            orig_close_train = df_hpo_train[TARGET_COL].values.copy()
            orig_close_val   = df_hpo_val[TARGET_COL].values.copy()

            # ── 5. Scale: fit CHỈ trên 90% HPO train (anti-leakage) ──────────
            scaler = FeatureScaler()
            df_hpo_train_scaled = scaler.fit_transform(df_hpo_train)
            df_hpo_val_scaled   = scaler.transform(df_hpo_val)

            # ── 6. Build daily T+1 binary classification sequences ────────────
            # build_sequences(task='classification') dùng original_close để tính
            # y[i] = float(close[t+1] > close[t]) — binary direction label
            X_train, y_train = build_sequences(
                df_hpo_train_scaled, seq_len,
                target_col     = TARGET_COL,
                task           = "classification",
                original_close = orig_close_train,   # unscaled Close cho direction
            )
            X_val, y_val = build_sequences(
                df_hpo_val_scaled, seq_len,
                target_col     = TARGET_COL,
                task           = "classification",
                original_close = orig_close_val,     # unscaled Close cho direction
            )

            train_ds = StockDataset(X_train, y_train, task="classification")
            val_ds   = StockDataset(X_val,   y_val,   task="classification")

            if len(train_ds) < 2:
                return float("inf")

            # ── 7. Build BiLSTM (classification head) ────────────────────────
            model = BiLSTM(
                task       = "classification",
                n_features = n_feat,
                params     = params,
            )

            # ── 8. Train với BCE loss ─────────────────────────────────────────
            # train_model với task='classification' dùng BCELoss + Sigmoid output
            result = train_model(
                model         = model,
                train_dataset = train_ds,
                val_dataset   = val_ds,
                params        = params,
                task          = "classification",
                max_epochs    = OPTUNA_HPO_EPOCHS,  # 50 epochs — nhanh cho HPO
                save_path     = None,
            )

            val_loss: float = result["best_val_loss"]  # BCE loss
            logger.debug(
                "[HPO cls Fold %s | Trial %d] BCE=%.6f | params=%s",
                fold_id, trial.number, val_loss,
                {k: v for k, v in params.items()
                 if k not in ("approx_indices", "detail_indices", "use_wavelet")},
            )
            return val_loss

        except Exception as exc:
            logger.warning("[HPO cls Fold %s | Trial %d] Failed: %s",
                           fold_id, trial.number, exc)
            return float("inf")

    return objective


# =============================================================================
# 3. RUN HPO FOR FOLD (dispatch regression / classification)
# =============================================================================

def run_hpo_for_fold(
    df_processed : pd.DataFrame,
    fold         : dict,
    feature_info : dict,
    n_trials     : int,
    fold_idx     : int,
    ticker       : str,
    currency     : str,
    use_wavelet  : bool,
    task         : str = "regression",
) -> dict:
    """
    Chạy Optuna HPO cho BiLSTM trên một fold, dispatch theo task.

    task='regression':
      - create_objective() — daily sequences, minimize MSE → best_params.json

    task='classification':
      - _create_classification_objective() — daily T+1 binary, minimize BCE
      - Output: best_params_classification.json
      - Experiments thực tế vẫn dùng weekly sequences (dataset_builder không đổi)

    Returns:
        best_params dict (đã được lưu vào JSON).
    """
    # Dispatch: chọn đúng objective function theo task
    if task == "regression":
        objective = create_objective(
            df_processed   = df_processed,
            fold           = fold,
            feature_cols   = feature_info["feature_cols"],
            n_features     = feature_info["n_features"],
            use_wavelet    = use_wavelet,
            approx_indices = feature_info["approx_indices"],
            detail_indices = feature_info["detail_indices"],
        )
    elif task == "classification":
        # Daily T+1 binary sequences, BCE loss
        objective = _create_classification_objective(
            df_processed = df_processed,
            fold         = fold,
            feature_info = feature_info,
            use_wavelet  = use_wavelet,
        )
    else:
        raise ValueError(
            f"run_hpo_for_fold: task='{task}' không hợp lệ. "
            f"Chọn: 'regression' hoặc 'classification'."
        )

    # ── Run Optuna study ──────────────────────────────────────────────────────
    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # ── Extract best params ───────────────────────────────────────────────────
    best = study.best_trial
    best_params = {
        "num_layers"     : best.params["num_layers"],
        "hidden_units"   : best.params["hidden_units"],
        "dropout_rate"   : best.params["dropout_rate"],
        "learning_rate"  : best.params["learning_rate"],
        "batch_size"     : best.params["batch_size"],
        "sequence_length": best.params["sequence_length"],
        "use_wavelet"    : use_wavelet,
        # feature_info keys (approx/detail indices) — cần cho BiLSTM wavelet
        **{k: v for k, v in feature_info.items() if k != "n_features"},
        "_meta": {
            "ticker"        : ticker,
            "currency"      : currency,
            "use_wavelet"   : use_wavelet,
            "fold_idx"      : fold_idx,
            "task"          : task,
            "n_trials"      : n_trials,
            "best_val_loss" : best.value,
            "best_trial_num": best.number,
        },
    }

    # Lưu vào đúng file theo task
    _save_best_params(
        best_params = best_params,
        ticker      = ticker,
        currency    = currency,
        use_wavelet = use_wavelet,
        fold_idx    = fold_idx,
        task        = task,
    )
    return best_params


# =============================================================================
# 4. RUN FULL HPO (all 3 folds)
# =============================================================================

def run_full_hpo(
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    n_trials   : int = OPTUNA_TRIALS,
    task       : str = "regression",
) -> list[dict]:
    """
    Chạy HPO cho tất cả 3 folds của một (ticker, currency, wavelet) combination.

    Args:
        ticker, currency, use_wavelet: Xác định dataset pkl.
        n_trials: Số Optuna trials mỗi fold. Default: OPTUNA_TRIALS (30).
        task: 'regression' → best_params.json
              'classification' → best_params_classification.json (daily T+1 BCE HPO)

    Returns:
        List 3 best_params dicts (một mỗi fold).

    Raises:
        FileNotFoundError: Nếu pkl chưa được tạo (chạy preprocess.py trước).
    """
    cond_str = "wavelet" if use_wavelet else "nowave"
    label    = f"{ticker}_{currency}_{cond_str}"
    fname    = _get_params_filename(task)

    # Load processed data từ pkl — cấu trúc: {"df": DataFrame, "feature_cols": list, ...}
    pkl_path = PROCESSED_DIR / f"{label}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"run_full_hpo: Không tìm thấy {pkl_path}. "
            f"Chạy 'uv run python scripts/preprocess.py' trước."
        )

    logger.info("[run_full_hpo] task=%s | Load: %s", task, pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df_processed : pd.DataFrame = data["df"]           # unpack dict
    feature_cols : list[str]    = data["feature_cols"]
    feature_info : dict         = build_feature_info(feature_cols)

    logger.info(
        "[run_full_hpo] %s | task=%s | rows=%d | n_features=%d | output=%s",
        label, task, len(df_processed), feature_info["n_features"], fname,
    )

    all_best_params: list[dict] = []
    total_start = time.time()

    for fold_idx, fold in enumerate(FOLDS, start=1):
        logger.info(
            "[run_full_hpo] === Fold %d/%d | task=%s | Train→%s | Test: %s→%s ===",
            fold_idx, len(FOLDS), task,
            fold["train_end"], fold["test_start"], fold["test_end"],
        )

        fold_start  = time.time()
        best_params = run_hpo_for_fold(
            df_processed = df_processed,
            fold         = fold,
            feature_info = feature_info,
            n_trials     = n_trials,
            fold_idx     = fold_idx,
            ticker       = ticker,
            currency     = currency,
            use_wavelet  = use_wavelet,
            task         = task,
        )
        fold_elapsed = time.time() - fold_start
        all_best_params.append(best_params)

        logger.info(
            "[run_full_hpo] Fold %d done | task=%s | time=%.1fs | best_val_loss=%.6f",
            fold_idx, task, fold_elapsed, best_params["_meta"]["best_val_loss"],
        )

    logger.info(
        "[run_full_hpo] All 3 folds done | %s | task=%s | total_time=%.1fs",
        label, task, time.time() - total_start,
    )
    return all_best_params