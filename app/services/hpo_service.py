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

HPO Validation split:
  - Lấy fold's training period (ví dụ Fold 1: 2012→2017)
  - Chia 90% train / 10% val theo thứ tự thời gian (không shuffle)
  - Fit scaler chỉ trên 90% để tránh leakage trong HPO

Optimization target:
  - Minimize MSE val_loss (regression task)
  - Reason: MSE là metric chính trong bài báo (Table 1), và regression loss
    phản ánh trực tiếp model capacity; params tối ưu cho regression
    cũng tốt cho classification task.

Public API:
  build_feature_info()  — tạo feature_info dict từ feature_cols list
  create_objective()    — tạo Optuna objective function cho một fold
  run_hpo_for_fold()    — chạy HPO và lưu best_params.json
  run_full_hpo()        — chạy HPO cho cả 3 folds, load pkl tự động

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 4.2: Hyperparameter Optimization — search space.
  Akiba et al. (2019): Optuna: A Next-generation HPO Framework.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

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

# Suppress Optuna's info logs (chỉ hiện WARNING trở lên) để giảm noise
# Caller có thể re-enable bằng optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = Path(PATHS["processed"])
EXPERIMENTS_DIR = Path(PATHS["experiments"])


# =============================================================================
# HELPER — Build feature_info dict
# =============================================================================

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
          "feature_cols"    (list[str])  : input feature names
          "n_features"      (int)        : len(feature_cols)
          "approx_indices"  (list[int])  : 0-based indices của _Approx cols
          "detail_indices"  (list[int])  : 0-based indices của _Detail cols
          "use_wavelet"     (bool)       : True nếu có _Approx/_Detail cols

    Example:
        >>> cols = ["High_Approx", "High_Detail", "Volume_Approx", "Volume_Detail",
        ...          "Deviation_Approx", "Deviation_Detail", "Open_Detail", "Volume_Approx2"]
        >>> info = build_feature_info(cols)
        >>> info["approx_indices"]
        [0, 2, 4]
        >>> info["detail_indices"]
        [1, 3, 5, 6]
    """
    approx_indices: list[int] = [
        i for i, col in enumerate(feature_cols) if col.endswith("_Approx")
    ]
    detail_indices: list[int] = [
        i for i, col in enumerate(feature_cols) if col.endswith("_Detail")
    ]
    # use_wavelet: True nếu có ít nhất 1 wavelet feature
    use_wavelet: bool = len(approx_indices) > 0 or len(detail_indices) > 0

    return {
        "feature_cols":   feature_cols,
        "n_features":     len(feature_cols),
        "approx_indices": approx_indices,
        "detail_indices": detail_indices,
        "use_wavelet":    use_wavelet,
    }


# =============================================================================
# 1. CREATE OBJECTIVE
# =============================================================================

def create_objective(
    df_processed: pd.DataFrame,
    fold: dict,
    feature_cols: list[str],
    n_features: int,
    use_wavelet: bool,
    approx_indices: list[int],
    detail_indices: list[int],
) -> Callable:
    """
    Tạo Optuna objective function để tối ưu BiLSTM hyperparameters cho 1 fold.

    Objective được tạo dưới dạng closure để capture dữ liệu fold và
    feature metadata, cho phép Optuna gọi nhiều lần với params khác nhau.

    Validation split bên trong HPO:
      - Lấy fold's training portion (df_processed.index <= fold["train_end"])
      - Chronological 90% / 10% split (không random shuffle — time series!)
      - Fit FeatureScaler CHỈ trên 90% train (anti-leakage)
      - Max OPTUNA_HPO_EPOCHS epochs per trial (nhanh hơn full training)

    Exception handling:
      - Nếu trial fail (ví dụ: val quá nhỏ, NaN loss), return float("inf")
      - Optuna sẽ đánh giá trial này tệ nhất và tiếp tục với params khác

    Args:
        df_processed:   DataFrame đầy đủ (feature_cols + "Close", chưa scale).
                        Index là DatetimeIndex chronological.
        fold:           Dict fold definition với keys: fold_id, train_end,
                        test_start, test_end.
        feature_cols:   List tên input features (không gồm "Close").
        n_features:     len(feature_cols) — truyền rõ ràng để tránh recompute.
        use_wavelet:    True = dual-branch BiLSTM.
        approx_indices: 0-based indices của _Approx features trong feature_cols.
        detail_indices: 0-based indices của _Detail features trong feature_cols.

    Returns:
        objective(trial: optuna.Trial) -> float
        → val_loss (MSE), minimize.
    """
    # Pre-compute fold boundary (capture trong closure)
    train_end = pd.Timestamp(fold["train_end"])
    fold_id   = fold.get("fold_id", "?")

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective: build BiLSTM với trial params, train, return val_loss.

        Returns float("inf") nếu trial không hợp lệ (thay vì raise exception)
        để Optuna có thể tiếp tục với trials khác.
        """
        # ── 1. Suggest hyperparameters từ search space ────────────────────────
        params: dict = {
            "num_layers":     trial.suggest_int(
                "num_layers",
                min(HPO_SEARCH_SPACE["num_layers"]),
                max(HPO_SEARCH_SPACE["num_layers"]),
            ),
            "hidden_units":   trial.suggest_categorical(
                "hidden_units", HPO_SEARCH_SPACE["hidden_units"]
            ),
            "dropout_rate":   trial.suggest_categorical(
                "dropout_rate", HPO_SEARCH_SPACE["dropout_rate"]
            ),
            "learning_rate":  trial.suggest_categorical(
                "learning_rate", HPO_SEARCH_SPACE["learning_rate"]
            ),
            "batch_size":     trial.suggest_categorical(
                "batch_size", HPO_SEARCH_SPACE["batch_size"]
            ),
            "sequence_length": trial.suggest_categorical(
                "sequence_length", HPO_SEARCH_SPACE["sequence_length"]
            ),
        }

        # BiLSTM-specific params (không phải hyperparameters, truyền qua params dict)
        params["use_wavelet"] = use_wavelet
        if use_wavelet:
            params["approx_indices"] = approx_indices
            params["detail_indices"] = detail_indices

        seq_len: int = params["sequence_length"]

        try:
            # ── 2. Lấy training portion của fold ─────────────────────────────
            df_train_full = df_processed[df_processed.index <= train_end].copy()

            if len(df_train_full) == 0:
                logger.warning(
                    "[HPO Fold %s | Trial %d] Training data rỗng (train_end=%s)",
                    fold_id, trial.number, train_end,
                )
                return float("inf")

            # ── 3. Chronological 90%/10% split ───────────────────────────────
            # KHÔNG shuffle: time series — thứ tự quan trọng
            n_total  : int = len(df_train_full)
            split_idx: int = int(n_total * 0.9)

            df_hpo_train = df_train_full.iloc[:split_idx]
            df_hpo_val   = df_train_full.iloc[split_idx:]

            # Validate đủ rows để tạo sequences (cần seq_len + 1 rows tối thiểu)
            min_rows: int = seq_len + 1
            if len(df_hpo_train) <= min_rows:
                logger.debug(
                    "[HPO Fold %s | Trial %d] HPO train quá nhỏ: %d <= %d",
                    fold_id, trial.number, len(df_hpo_train), min_rows,
                )
                return float("inf")
            if len(df_hpo_val) <= min_rows:
                logger.debug(
                    "[HPO Fold %s | Trial %d] HPO val quá nhỏ: %d <= %d",
                    fold_id, trial.number, len(df_hpo_val), min_rows,
                )
                return float("inf")

            # ── 4. Scaling: fit CHỈ trên HPO train ───────────────────────────
            # Anti-leakage: val statistics không được leak vào scaler
            scaler = FeatureScaler()
            df_hpo_train_scaled = scaler.fit_transform(df_hpo_train)
            df_hpo_val_scaled   = scaler.transform(df_hpo_val)

            # ── 5. Build sequences (regression task cho HPO) ──────────────────
            # Regression: y = scaled Close(t+1), minimize MSE
            X_train, y_train = build_sequences(
                df_hpo_train_scaled, seq_len, target_col=TARGET_COL,
                task="regression",
            )
            X_val, y_val = build_sequences(
                df_hpo_val_scaled, seq_len, target_col=TARGET_COL,
                task="regression",
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
                model          = model,
                train_dataset  = train_ds,
                val_dataset    = val_ds,
                params         = params,
                task           = "regression",
                max_epochs     = OPTUNA_HPO_EPOCHS,  # 50 epochs (quick HPO)
                save_path      = None,               # không lưu checkpoint per trial
            )

            val_loss: float = result["best_val_loss"]

            logger.debug(
                "[HPO Fold %s | Trial %d] val_loss=%.6f | params=%s",
                fold_id, trial.number, val_loss,
                {k: v for k, v in params.items()
                 if k not in ("approx_indices", "detail_indices", "use_wavelet")},
            )

            return val_loss

        except Exception as exc:
            logger.warning(
                "[HPO Fold %s | Trial %d] Failed: %s",
                fold_id, trial.number, exc,
            )
            return float("inf")

    return objective


# =============================================================================
# 2. RUN HPO FOR FOLD
# =============================================================================

def run_hpo_for_fold(
    df_processed: pd.DataFrame,
    fold: dict,
    feature_info: dict,
    n_trials: int = OPTUNA_TRIALS,
    fold_idx: int = 1,
    ticker: str = "",
    currency: str = "",
    use_wavelet: bool = True,
) -> dict:
    """
    Chạy Optuna HPO cho BiLSTM trên một Walk-Forward fold.

    Pipeline:
      1. Tạo objective function từ fold data + feature info
      2. Tạo Optuna study (minimize, TPESampler seed=42)
      3. Chạy n_trials trials
      4. Lưu best_params → experiments/{ticker}_{currency}_{cond}/fold_{i}/best_params.json
      5. Return best_params dict

    Best_params saved format (JSON):
      {
        "num_layers": 2,
        "hidden_units": 128,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "sequence_length": 20,
        "use_wavelet": true,
        "approx_indices": [...],
        "detail_indices": [...],
        "_meta": {
          "ticker": "VCB", "currency": "VND", "fold_idx": 1,
          "n_trials": 30, "best_val_loss": 0.012345,
          "n_trials_completed": 30
        }
      }

    Args:
        df_processed: Full preprocessed DataFrame (feature_cols + "Close").
        fold:         Fold definition dict (fold_id, train_end, test_start, test_end).
        feature_info: Dict từ build_feature_info():
                        feature_cols, n_features, approx_indices, detail_indices,
                        use_wavelet.
        n_trials:     Số Optuna trials. Default: OPTUNA_TRIALS=30 (config).
        fold_idx:     1-based fold index — dùng để đặt tên thư mục và log.
        ticker:       Mã cổ phiếu — dùng để đặt tên thư mục.
        currency:     Tiền tệ — dùng để đặt tên thư mục.
        use_wavelet:  True = wavelet condition — dùng để đặt tên thư mục.

    Returns:
        best_params dict (JSON-serializable) với tất cả hyperparameters
        + BiLSTM-specific params (use_wavelet, approx_indices, detail_indices).
        Cùng format với params dict được truyền vào train_model().

    Raises:
        ValueError: Nếu feature_info thiếu required keys.
        RuntimeError: Nếu tất cả trials fail (best_val_loss = inf).
    """
    # ── Validate feature_info ─────────────────────────────────────────────────
    required_info_keys = {"feature_cols", "n_features", "approx_indices",
                          "detail_indices", "use_wavelet"}
    missing = required_info_keys - set(feature_info.keys())
    if missing:
        raise ValueError(
            f"run_hpo_for_fold: feature_info thiếu keys {sorted(missing)}."
        )

    feature_cols   : list[str] = feature_info["feature_cols"]
    n_features     : int       = feature_info["n_features"]
    approx_indices : list[int] = feature_info["approx_indices"]
    detail_indices : list[int] = feature_info["detail_indices"]
    # use_wavelet từ tham số trực tiếp (ưu tiên hơn feature_info["use_wavelet"])

    fold_id  = fold.get("fold_id", fold_idx)
    cond_str = "wavelet" if use_wavelet else "nowave"

    logger.info(
        "[HPO] Start | %s_%s_%s | Fold %d | trials=%d | "
        "n_features=%d | use_wavelet=%s",
        ticker, currency, cond_str, fold_idx, n_trials,
        n_features, use_wavelet,
    )

    t_start = time.time()

    # ── Tạo objective function ────────────────────────────────────────────────
    objective = create_objective(
        df_processed   = df_processed,
        fold           = fold,
        feature_cols   = feature_cols,
        n_features     = n_features,
        use_wavelet    = use_wavelet,
        approx_indices = approx_indices,
        detail_indices = detail_indices,
    )

    # ── Tạo Optuna study ──────────────────────────────────────────────────────
    # TPESampler(seed=42): Tree-structured Parzen Estimator, reproducible
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction = "minimize",   # minimize val_loss (MSE)
        sampler   = sampler,
        study_name = f"{ticker}_{currency}_{cond_str}_fold{fold_idx}",
    )

    # ── Chạy optimization ─────────────────────────────────────────────────────
    study.optimize(
        objective,
        n_trials    = n_trials,
        show_progress_bar = False,   # tắt tqdm để log sạch hơn
    )

    elapsed = time.time() - t_start

    # ── Lấy best trial ────────────────────────────────────────────────────────
    best_trial = study.best_trial

    if best_trial.value == float("inf"):
        raise RuntimeError(
            f"run_hpo_for_fold: tất cả {n_trials} trials đều fail "
            f"(best_val_loss = inf). Kiểm tra data và model setup."
        )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])

    # ── Xây dựng best_params dict ─────────────────────────────────────────────
    best_params: dict = dict(best_trial.params)

    # Thêm BiLSTM-specific params (không trong Optuna search space)
    best_params["use_wavelet"] = use_wavelet
    if use_wavelet:
        best_params["approx_indices"] = approx_indices
        best_params["detail_indices"] = detail_indices

    # Metadata cho traceability
    best_params["_meta"] = {
        "ticker":            ticker,
        "currency":          currency,
        "use_wavelet":       use_wavelet,
        "fold_idx":          fold_idx,
        "n_trials":          n_trials,
        "n_trials_completed": n_completed,
        "best_val_loss":     best_trial.value,
        "elapsed_seconds":   round(elapsed, 2),
    }

    logger.info(
        "[HPO] Done  | %s_%s_%s | Fold %d | "
        "best_val_loss=%.6f | best_epoch_params=%s | "
        "time=%.1fs | %d/%d trials completed",
        ticker, currency, cond_str, fold_idx,
        best_trial.value,
        {k: v for k, v in best_params.items()
         if k not in ("approx_indices", "detail_indices", "use_wavelet", "_meta")},
        elapsed, n_completed, n_trials,
    )

    # ── Lưu best_params.json ─────────────────────────────────────────────────
    _save_best_params(
        best_params = best_params,
        ticker      = ticker,
        currency    = currency,
        use_wavelet = use_wavelet,
        fold_idx    = fold_idx,
    )

    return best_params


# =============================================================================
# 3. RUN FULL HPO (all 3 folds)
# =============================================================================

def run_full_hpo(
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    n_trials   : int = OPTUNA_TRIALS,
) -> list[dict]:
    """
    Chạy HPO cho BiLSTM trên cả 3 Walk-Forward folds.

    Load processed data từ disk, build feature_info, rồi gọi
    run_hpo_for_fold() cho mỗi fold.

    Best_params được tự động lưu vào:
      experiments/{ticker}_{currency}_{cond}/fold_{i}/best_params.json

    Sau khi chạy xong, DNN/RNN/GRU/LSTM sẽ load các file này để
    dùng chung hyperparameters (cùng fold, cùng condition).

    Args:
        ticker:      "VCB" hoặc "VIC".
        currency:    "VND" hoặc "USD".
        use_wavelet: True = wavelet pipeline.
        n_trials:    Số Optuna trials per fold. Default: 30 (config).

    Returns:
        list[dict] gồm 3 phần tử:
          [best_params_fold1, best_params_fold2, best_params_fold3]
        Mỗi dict là best_params từ run_hpo_for_fold() (JSON-serializable).

    Raises:
        FileNotFoundError: Nếu processed pkl file chưa tồn tại.
                           Chạy scripts/preprocess.py trước.
        RuntimeError:      Nếu HPO fail cho bất kỳ fold nào.
    """
    cond_str = "wavelet" if use_wavelet else "nowave"
    label    = f"{ticker}_{currency}_{cond_str}"

    # ── Load processed data từ pkl ─────────────────────────────────────────────
    pkl_path = PROCESSED_DIR / f"{label}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"run_full_hpo: Không tìm thấy {pkl_path}. "
            f"Chạy 'uv run python scripts/preprocess.py' trước."
        )

    logger.info("[run_full_hpo] Load: %s", pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df_processed : pd.DataFrame = data["df"]
    feature_cols : list[str]    = data["feature_cols"]
    feature_info : dict         = build_feature_info(feature_cols)

    logger.info(
        "[run_full_hpo] %s | rows=%d | n_features=%d | "
        "approx=%d | detail=%d",
        label, len(df_processed), feature_info["n_features"],
        len(feature_info["approx_indices"]),
        len(feature_info["detail_indices"]),
    )

    # ── Chạy HPO cho từng fold ─────────────────────────────────────────────────
    all_best_params: list[dict] = []
    total_start = time.time()

    for fold_idx, fold in enumerate(FOLDS, start=1):
        logger.info(
            "[run_full_hpo] === Fold %d/%d | Train→%s | Test: %s→%s ===",
            fold_idx, len(FOLDS),
            fold["train_end"], fold["test_start"], fold["test_end"],
        )

        fold_start = time.time()

        best_params = run_hpo_for_fold(
            df_processed = df_processed,
            fold         = fold,
            feature_info = feature_info,
            n_trials     = n_trials,
            fold_idx     = fold_idx,
            ticker       = ticker,
            currency     = currency,
            use_wavelet  = use_wavelet,
        )

        fold_elapsed = time.time() - fold_start
        all_best_params.append(best_params)

        logger.info(
            "[run_full_hpo] Fold %d done | time=%.1fs | "
            "best_val_loss=%.6f",
            fold_idx, fold_elapsed,
            best_params["_meta"]["best_val_loss"],
        )

    total_elapsed = time.time() - total_start
    logger.info(
        "[run_full_hpo] All 3 folds done | %s | total_time=%.1fs",
        label, total_elapsed,
    )

    return all_best_params


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _save_best_params(
    best_params: dict,
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    fold_idx   : int,
) -> Path:
    """
    Lưu best_params dict ra file JSON.

    Path: experiments/{ticker}_{currency}_{cond}/fold_{fold_idx}/best_params.json

    Args:
        best_params: Dict hyperparameters + metadata + BiLSTM-specific params.
        ticker:      Mã cổ phiếu.
        currency:    Tiền tệ.
        use_wavelet: True → cond = "wavelet", False → cond = "nowave".
        fold_idx:    1-based fold index.

    Returns:
        Path object của file đã lưu.
    """
    cond_str = "wavelet" if use_wavelet else "nowave"
    out_dir  = EXPERIMENTS_DIR / f"{ticker}_{currency}_{cond_str}" / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_params.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    logger.info("[HPO] best_params.json saved → %s", out_path)
    return out_path


def load_best_params(
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    fold_idx   : int,
) -> dict:
    """
    Load best_params.json đã lưu từ run_hpo_for_fold().

    Dùng bởi DNN/RNN/GRU/LSTM để lấy shared hyperparameters của BiLSTM.

    Args:
        ticker:      Mã cổ phiếu.
        currency:    Tiền tệ.
        use_wavelet: Wavelet condition.
        fold_idx:    1-based fold index.

    Returns:
        best_params dict (JSON đã load).

    Raises:
        FileNotFoundError: Nếu JSON file chưa tồn tại.
                           Chạy run_hpo_for_fold() trước.
    """
    cond_str = "wavelet" if use_wavelet else "nowave"
    json_path = (
        EXPERIMENTS_DIR
        / f"{ticker}_{currency}_{cond_str}"
        / f"fold_{fold_idx}"
        / "best_params.json"
    )

    if not json_path.exists():
        raise FileNotFoundError(
            f"load_best_params: Không tìm thấy {json_path}. "
            f"Chạy run_hpo_for_fold() cho Fold {fold_idx} trước."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    logger.info("[HPO] best_params loaded ← %s", json_path)
    return params