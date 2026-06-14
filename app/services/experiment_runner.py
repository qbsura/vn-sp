"""
app/services/experiment_runner.py
===================================
Experiment runner cho toàn bộ ma trận thực nghiệm VNSP.

Task 4.3 trong Phase 4 — Huấn luyện & Hyperparameter Optimization.

Ma trận đầy đủ:
  2 tickers × 2 currencies × 2 wavelet × 5 models × 2 tasks × 3 folds = 240 runs

Experiment ID convention:
  {ticker}_{currency}_{wavelet|nowave}_{model_name}_{task}
  Ví dụ: "VCB_VND_wavelet_BiLSTM_regression"

Directory structure:
  experiments/
    {ticker}_{currency}_{cond}/
      fold_{i}/
        best_params.json          ← từ hpo_service.py
    {exp_id}/
      fold_{i}/
        best_model.pt             ← best checkpoint (từ train_model)
        metrics.json              ← metrics + metadata
        predictions.npz           ← y_pred, y_true, y_prob (optional)

Resume capability:
  run_all_experiments() skip nếu metrics.json đã tồn tại.
  → force_rerun=True để bỏ qua skip-check (dùng khi re-run Task B sau khi
    đổi target sang weekly — xem ghi chú "Task B WEEKLY" dưới đây).

Task B — WEEKLY CLASSIFICATION (Phương án D, 2026-06):
  Theo feedback giảng viên, Task B chuyển từ "hướng đi ngày kế tiếp (t+1)"
  sang "hướng đi TUẦN kế tiếp" (T2→T6 theo lịch, 1 sample/tuần). Thay đổi
  này nằm trong dataset_builder.build_weekly_sequences() / prepare_fold_data()
  — run_single_experiment() KHÔNG cần sửa logic train/eval, chỉ:
    (a) predictions.npz lưu thêm "dates" (F_W) — dùng cho Task C (trading).
    (b) run_all_experiments(task_filter="classification", force_rerun=True)
        để overwrite 120 kết quả classification cũ (đã tính theo t+1 daily).
  Task A (regression) HOÀN TOÀN KHÔNG ĐỔI — không cần re-run.

HPO policy:
  BiLSTM chạy HPO riêng → best_params.json
  DNN/RNN/GRU/LSTM load chung best_params của BiLSTM (cùng fold, cùng condition)
  Task B weekly TÁI SỬ DỤNG best_params này (không HPO riêng).

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Table 1: Metrics — MSE, MAE, MAPE (regression).
  Section 4.2: Classification metrics.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from app.config import (
    CURRENCIES,
    FOLDS,
    MAX_TRAIN_EPOCHS,
    MODELS,
    PATHS,
    SEED,
    TASKS,
    TICKERS,
    WAVELET_CONDITIONS,
)
from app.models.base import build_model
from app.services.dataset_builder import prepare_fold_data
from app.services.hpo_service import build_feature_info, load_best_params
from app.services.training_service import get_predictions, train_model
from app.utils.seeds import set_all_seeds

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = Path(PATHS["processed"])
EXPERIMENTS_DIR = Path(PATHS["experiments"])


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    y_pred : np.ndarray,
    y_true : np.ndarray,
    task   : str,
    y_prob : Optional[np.ndarray] = None,
) -> dict:
    """
    Tính evaluation metrics cho Task A (regression) hoặc Task B (classification).

    Regression metrics (Task A):
      MSE, MAE, RMSE, MAPE (%), R²
      Tất cả tính trên giá thực (đã inverse transform).

    Classification metrics (Task B — WEEKLY, T2→T6):
      Accuracy, Precision, Recall, F1 (binary), AUC-ROC.
      N (số sample) nay là số TUẦN trong test set (~48-156/fold), không
      phải số ngày. Công thức metric không đổi (frequency-agnostic).

    Args:
        y_pred: Predicted values, shape (N,).
                Regression: giá Close thực (VND/USD).
                Classification: class labels {0=DOWN, 1=UP}.
        y_true: Actual values, shape (N,).
        task:   "regression" hoặc "classification".
        y_prob: Probability P(UP), shape (N,). Chỉ dùng cho classification/AUC.

    Returns:
        dict với metric names → float values (Python float, JSON-serializable).

    Notes:
        MAPE: Nếu y_true=0 tại một vài điểm, dùng epsilon=1e-8 để tránh division by zero.
        Precision/Recall/F1: zero_division=0 để tránh warning khi một class không xuất hiện.
    """
    # Đảm bảo numpy arrays 1-D float64
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    y_true = np.asarray(y_true, dtype=np.float64).flatten()

    if task == "regression":
        mse  = float(mean_squared_error(y_true, y_pred))
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        # MAPE: epsilon tránh division by zero khi giá = 0
        eps  = 1e-8
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
        r2   = float(r2_score(y_true, y_pred))
        return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    elif task == "classification":
        # y_pred phải là integer labels cho sklearn
        y_pred_int = y_pred.astype(int)
        y_true_int = y_true.astype(int)

        acc  = float(accuracy_score(y_true_int, y_pred_int))
        prec = float(precision_score(
            y_true_int, y_pred_int, zero_division=0, average="binary"
        ))
        rec  = float(recall_score(
            y_true_int, y_pred_int, zero_division=0, average="binary"
        ))
        f1   = float(f1_score(
            y_true_int, y_pred_int, zero_division=0, average="binary"
        ))
        # AUC-ROC: cần y_prob, fallback 0.5 nếu không có
        if y_prob is not None:
            try:
                auc = float(roc_auc_score(y_true_int, np.asarray(y_prob).flatten()))
            except ValueError:
                # Xảy ra khi chỉ có 1 class trong y_true
                auc = 0.5
        else:
            auc = 0.5
            logger.warning("compute_metrics: y_prob=None → AUC-ROC = 0.5 (default)")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}

    else:
        raise ValueError(f"compute_metrics: task='{task}' không hợp lệ.")


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_single_experiment(
    ticker     : str,
    currency   : str,
    use_wavelet: bool,
    model_name : str,
    task       : str,
    fold_idx   : int,
    best_params: dict,
) -> dict:
    """
    Chạy một experiment đầy đủ: load data → train → evaluate → save metrics.

    Pipeline:
      1. Load processed data từ data/processed/{ticker}_{currency}_{cond}.pkl
      2. Build feature_info (approx/detail indices cho BiLSTM)
      3. Prepare fold data: split → scale → build sequences (via prepare_fold_data)
      4. Build model (build_model factory)
      5. Train model (train_model với early stopping, best checkpoint)
      6. Get predictions (get_predictions với inverse transform cho regression)
      7. Compute metrics (compute_metrics)
      8. Save metrics.json + predictions.npz → experiments/{exp_id}/fold_{i}/
      9. Return metrics dict

    Checkpoint path: experiments/{exp_id}/fold_{fold_idx}/best_model.pt
    Metrics path:    experiments/{exp_id}/fold_{fold_idx}/metrics.json

    Note về validation:
      Dùng test_dataset làm validation set cho early stopping.
      Walk-forward: test set nằm SAU train set về mặt thời gian → không có data leakage.

    Args:
        ticker:       "VCB" hoặc "VIC".
        currency:     "VND" hoặc "USD".
        use_wavelet:  True = wavelet pipeline.
        model_name:   "DNN", "RNN", "GRU", "LSTM", "BiLSTM".
        task:         "regression" hoặc "classification".
        fold_idx:     1-based fold index (1, 2, 3).
        best_params:  Hyperparameter dict từ load_best_params(). Phải có:
                        num_layers, hidden_units, dropout_rate, batch_size,
                        learning_rate, sequence_length.
                      BiLSTM wavelet: thêm use_wavelet, approx_indices, detail_indices.

    Returns:
        dict với keys:
          "metrics"   (dict)  : evaluation metrics (task-specific)
          "exp_id"    (str)   : experiment identifier
          "fold_idx"  (int)   : fold index
          "model_name"(str)   : model name
          "task"      (str)   : task name
          "train_history" (dict): {"train_losses", "val_losses", "best_epoch",
                                    "best_val_loss", "stopped_early"}
          "n_test_samples" (int): số test samples

    Raises:
        FileNotFoundError: Nếu processed pkl không tồn tại.
        ValueError:       Nếu fold_idx không hợp lệ.
    """
    set_all_seeds(SEED)   # reproducibility mỗi experiment

    cond_str = "wavelet" if use_wavelet else "nowave"
    exp_id   = f"{ticker}_{currency}_{cond_str}_{model_name}_{task}"
    label    = f"{exp_id} | fold {fold_idx}"

    # ── 1. Load processed data ────────────────────────────────────────────────
    pkl_path = PROCESSED_DIR / f"{ticker}_{currency}_{cond_str}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"run_single_experiment: Không tìm thấy {pkl_path}. "
            f"Chạy scripts/preprocess.py trước."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df_processed : object  = data["df"]
    feature_cols : list    = data["feature_cols"]

    # ── 2. Build feature_info ──────────────────────────────────────────────────
    feature_info = build_feature_info(feature_cols)
    n_features   = feature_info["n_features"]

    # ── 3. Prepare fold data ──────────────────────────────────────────────────
    if fold_idx < 1 or fold_idx > len(FOLDS):
        raise ValueError(
            f"fold_idx={fold_idx} không hợp lệ. Phải trong [1, {len(FOLDS)}]."
        )
    fold = FOLDS[fold_idx - 1]   # FOLDS là 0-based list

    seq_len = int(best_params.get("sequence_length", 20))

    fold_data = prepare_fold_data(
        df              = df_processed,
        fold            = fold,
        sequence_length = seq_len,
        task            = task,
    )
    train_dataset = fold_data["train_dataset"]
    test_dataset  = fold_data["test_dataset"]
    scaler        = fold_data["scaler"]

    logger.info(
        "[%s] Fold data ready | train=%d | test=%d | n_features=%d",
        label, len(train_dataset), len(test_dataset), n_features,
    )

    # ── 4. Build model ─────────────────────────────────────────────────────────
    # Chuẩn bị params đầy đủ cho BiLSTM (thêm indices nếu wavelet)
    model_params = dict(best_params)

    if model_name == "BiLSTM":
        # Đảm bảo BiLSTM có đầy đủ params theo experiment condition
        model_params["use_wavelet"] = use_wavelet
        if use_wavelet:
            model_params["approx_indices"] = feature_info["approx_indices"]
            model_params["detail_indices"] = feature_info["detail_indices"]
        # else: no_wavelet BiLSTM không cần approx/detail
    # Với DNN/RNN/GRU/LSTM: extra keys (use_wavelet, approx_indices...) bị ignore

    model = build_model(model_name, task, n_features, model_params)

    logger.info(
        "[%s] Model built | %s",
        label, model.summary(),
    )

    # ── 5. Train model ─────────────────────────────────────────────────────────
    # Tạo output dir sớm để save checkpoint
    out_dir = EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(out_dir / "best_model.pt")

    t_train_start = time.time()
    train_history = train_model(
        model         = model,
        train_dataset = train_dataset,
        val_dataset   = test_dataset,   # test làm val: walk-forward, không leak
        params        = model_params,
        task          = task,
        max_epochs    = MAX_TRAIN_EPOCHS,
        save_path     = checkpoint_path,
    )
    train_elapsed = time.time() - t_train_start

    logger.info(
        "[%s] Training done | epochs=%d | best_epoch=%d | "
        "best_val_loss=%.6f | stopped_early=%s | time=%.1fs",
        label,
        len(train_history["train_losses"]),
        train_history["best_epoch"],
        train_history["best_val_loss"],
        train_history["stopped_early"],
        train_elapsed,
    )

    # ── 6. Get predictions ────────────────────────────────────────────────────
    preds = get_predictions(
        model        = model,
        test_dataset = test_dataset,
        task         = task,
        scaler       = scaler,   # dùng inverse transform cho regression
    )

    # ── 7. Compute metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(
        y_pred = preds["y_pred"],
        y_true = preds["y_true"],
        task   = task,
        y_prob = preds.get("y_prob"),   # None cho regression
    )

    logger.info(
        "[%s] Metrics: %s",
        label,
        {k: f"{v:.4f}" for k, v in metrics.items()},
    )

    # ── 8. Save metrics.json + predictions.npz ────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    result_dict  = {
        "metrics"         : metrics,
        "exp_id"          : exp_id,
        "fold_idx"        : fold_idx,
        "model_name"      : model_name,
        "task"            : task,
        "ticker"          : ticker,
        "currency"        : currency,
        "use_wavelet"     : use_wavelet,
        "n_test_samples"  : int(len(test_dataset)),
        "train_elapsed_s" : round(train_elapsed, 2),
        "train_history"   : {
            "best_epoch"      : train_history["best_epoch"],
            "best_val_loss"   : float(train_history["best_val_loss"]),
            "stopped_early"   : train_history["stopped_early"],
            "n_epochs_run"    : len(train_history["train_losses"]),
            "train_losses"    : [float(x) for x in train_history["train_losses"]],
            "val_losses"      : [float(x) for x in train_history["val_losses"]],
        },
        "_params_used"    : {
            k: v for k, v in model_params.items()
            if k not in ("approx_indices", "detail_indices", "_meta")
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    # Lưu predictions cho visualization (predictions.npz)
    pred_save = {"y_pred": preds["y_pred"], "y_true": preds["y_true"]}
    if "y_prob" in preds:
        pred_save["y_prob"] = preds["y_prob"]

    if task == "classification":
        # Task B (weekly): mỗi prediction ứng với F_W (phiên cuối tuần W,
        # thời điểm dự đoán cho tuần W+1). Lưu lại để Task C (trading
        # simulation weekly, Phase 3) tính return theo đúng mốc thời gian
        # mà KHÔNG cần train lại.
        pred_save["dates"] = fold_data["test_dates"].values.astype("datetime64[ns]")

    np.savez_compressed(str(out_dir / "predictions.npz"), **pred_save)

    logger.info("[%s] Saved → %s", label, out_dir)

    return result_dict


# =============================================================================
# FULL EXPERIMENT MATRIX
# =============================================================================

def run_all_experiments(
    ticker_filter    : Optional[str]  = None,
    currency_filter  : Optional[str]  = None,
    wavelet_filter   : Optional[bool] = None,
    model_filter     : Optional[str]  = None,
    task_filter      : Optional[str]  = None,
    fold_filter      : Optional[int]  = None,
    force_rerun      : bool           = False,
) -> list[dict]:
    """
    Chạy toàn bộ experiment matrix với resume capability.

    Ma trận đầy đủ (không filter):
      2 tickers × 2 currencies × 2 wavelet × 5 models × 2 tasks × 3 folds = 240 runs

    Resume: Skip experiment nếu experiments/{exp_id}/fold_{i}/metrics.json đã tồn tại.
      force_rerun=True → bỏ qua skip-check, LUÔN chạy lại và overwrite
      metrics.json/predictions.npz/best_model.pt cũ.

    Filter arguments (None = không lọc):
      ticker_filter:   "VCB" hoặc "VIC"
      currency_filter: "VND" hoặc "USD"
      wavelet_filter:  True/False
      model_filter:    "DNN", "RNN", "GRU", "LSTM", "BiLSTM"
      task_filter:     "regression" hoặc "classification"
      fold_filter:     1, 2, hoặc 3
      force_rerun:     True → re-run dù đã có metrics.json (dùng khi đổi
                       target/logic, ví dụ Task B chuyển sang weekly — xem
                       module docstring "Task B — WEEKLY CLASSIFICATION").

    Time estimation:
      Sau mỗi experiment hoàn thành, tính rolling average time và ước tính
      thời gian còn lại.

    Returns:
        list[dict] — kết quả của tất cả experiments đã chạy thành công.
        Experiments bị skip (đã tồn tại, force_rerun=False) không được include.
    """
    # ── Build filtered experiment list ────────────────────────────────────────
    tickers    = [ticker_filter]    if ticker_filter    else TICKERS
    currencies = [currency_filter]  if currency_filter  else CURRENCIES
    wavelets   = [wavelet_filter]   if wavelet_filter is not None else WAVELET_CONDITIONS
    models     = [model_filter]     if model_filter     else MODELS
    tasks_list = [task_filter]      if task_filter      else TASKS
    folds_list = [fold_filter]      if fold_filter      else [f["fold_id"] for f in FOLDS]

    # Cartesian product: tất cả combinations cần chạy
    all_combos = list(product(tickers, currencies, wavelets, models, tasks_list, folds_list))
    total      = len(all_combos)

    logger.info(
        "[run_all_experiments] Total: %d experiments | "
        "tickers=%s | currencies=%s | wavelets=%s | models=%s | tasks=%s | folds=%s",
        total, tickers, currencies, wavelets, models, tasks_list, folds_list,
    )
    print(f"\n{'='*72}")
    print(f"VNSP — Experiment Matrix Runner")
    print(f"{'='*72}")
    print(f"Total experiments : {total}")
    print(f"Matrix            : {len(tickers)}T × {len(currencies)}C × "
          f"{len(wavelets)}W × {len(models)}M × {len(tasks_list)}Tk × {len(folds_list)}F")
    print(f"Output dir        : {EXPERIMENTS_DIR.resolve()}")
    print(f"{'='*72}\n")

    results    : list[dict] = []
    failed     : list[dict] = []   # chi tiết các experiment lỗi — để debug sau
    skipped    : int        = 0
    errors     : int        = 0
    elapsed_log: list[float]= []   # thời gian mỗi run để estimate remaining
    global_start = time.time()

    for run_idx, (ticker, currency, use_wavelet, model_name, task, fold_idx) in enumerate(
        all_combos, start=1
    ):
        cond_str = "wavelet" if use_wavelet else "nowave"
        exp_id   = f"{ticker}_{currency}_{cond_str}_{model_name}_{task}"
        label    = f"[{run_idx:>3}/{total}] {exp_id} | fold {fold_idx}"

        # ── Check resume ──────────────────────────────────────────────────────
        metrics_path = EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "metrics.json"
        if metrics_path.exists() and not force_rerun:
            skipped += 1
            logger.debug("%s — SKIP (metrics.json exists)", label)
            print(f"  {label} — ⏭  SKIP")
            continue

        # ── Force-rerun: dọn sạch kết quả CŨ trước khi chạy lại ─────────────────
        # Nếu không xoá: trường hợp experiment lỗi TRƯỚC bước ghi metrics.json
        # (vd. lỗi ở load data/build model/training) sẽ để lại metrics.json
        # CŨ (ví dụ từ Task B daily trước khi đổi sang weekly) — lần resume
        # sau (không force_rerun) sẽ hiểu nhầm "đã có kết quả" và SKIP, khiến
        # experiment lỗi này không bao giờ được retry và không hiện lỗi nữa.
        if force_rerun:
            exp_dir = EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}"
            for fname in ("metrics.json", "predictions.npz", "best_model.pt"):
                fpath = exp_dir / fname
                if fpath.exists():
                    fpath.unlink()

        # ── Time estimate ──────────────────────────────────────────────────────
        if elapsed_log:
            avg_time   = np.mean(elapsed_log[-10:])   # rolling window 10
            remaining  = avg_time * (total - run_idx + 1 - skipped)
            eta_str    = _format_seconds(remaining)
        else:
            eta_str = "..."

        print(f"  {label} | ETA: {eta_str}", flush=True)
        logger.info("[run_all_experiments] Running %s", label)

        t_run_start = time.time()

        try:
            # ── Load best_params ──────────────────────────────────────────────
            # BiLSTM: best_params từ HPO của chính nó
            # DNN/RNN/GRU/LSTM: best_params từ BiLSTM HPO (cùng fold/cond)
            # Classification: ưu tiên best_params_classification.json nếu tồn tại,
            #                 fallback sang best_params.json (regression params)
            best_params = load_best_params(ticker, currency, use_wavelet, fold_idx, task=task)

            # ── Run experiment ────────────────────────────────────────────────
            result = run_single_experiment(
                ticker      = ticker,
                currency    = currency,
                use_wavelet = use_wavelet,
                model_name  = model_name,
                task        = task,
                fold_idx    = fold_idx,
                best_params = best_params,
            )

            run_elapsed = time.time() - t_run_start
            elapsed_log.append(run_elapsed)
            results.append(result)

            # Log metrics
            metrics_str = " | ".join(
                f"{k}={v:.4f}" for k, v in result["metrics"].items()
            )
            print(f"  {label} — ✅  {metrics_str} | {run_elapsed:.1f}s")

        except FileNotFoundError as exc:
            run_elapsed = time.time() - t_run_start
            errors += 1
            logger.error("[run_all_experiments] SKIP (not found): %s | %s", label, exc)
            print(f"  {label} — ⚠️  SKIP (best_params not found — run HPO first)")
            failed.append({
                "exp_id"    : exp_id,
                "fold_idx"  : fold_idx,
                "error_type": "FileNotFoundError",
                "error_msg" : str(exc),
                "traceback" : traceback.format_exc(),
            })

        except Exception as exc:
            run_elapsed = time.time() - t_run_start
            elapsed_log.append(run_elapsed)
            errors += 1
            logger.error("[run_all_experiments] ERROR: %s | %s", label, exc, exc_info=True)
            print(f"  {label} — ❌  ERROR: {type(exc).__name__}: {exc}")
            failed.append({
                "exp_id"    : exp_id,
                "fold_idx"  : fold_idx,
                "error_type": type(exc).__name__,
                "error_msg" : str(exc),
                "traceback" : traceback.format_exc(),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - global_start
    n_done        = len(results)

    print(f"\n{'='*72}")
    print(f"KẾT QUẢ:")
    print(f"  ✅  {n_done} experiments hoàn thành")
    print(f"  ⏭  {skipped} skipped (đã có results)")
    print(f"  ❌  {errors} errors")
    print(f"  ⏱  Tổng thời gian: {_format_seconds(total_elapsed)}")
    print(f"{'='*72}\n")

    # ── Chi tiết lỗi (nếu có) ───────────────────────────────────────────────────
    # In tóm tắt từng lỗi ra console + lưu full traceback ra file JSON để debug
    # (resume: experiment lỗi KHÔNG có metrics.json → lần chạy sau sẽ tự retry).
    if failed:
        print(f"{'='*72}")
        print(f"CHI TIẾT {len(failed)} LỖI:")
        print(f"{'='*72}")
        for f_info in failed:
            print(
                f"  • {f_info['exp_id']} | fold {f_info['fold_idx']} | "
                f"{f_info['error_type']}: {f_info['error_msg']}"
            )

        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        err_log_path = EXPERIMENTS_DIR / "_last_errors.json"
        with open(err_log_path, "w", encoding="utf-8") as ef:
            json.dump(failed, ef, indent=2, ensure_ascii=False)
        print(f"\n  → Traceback đầy đủ đã lưu tại: {err_log_path}")
        print(f"{'='*72}\n")

    return results


# =============================================================================
# HELPERS
# =============================================================================

def _format_seconds(seconds: float) -> str:
    """Format số giây thành chuỗi dễ đọc (Xh Ym Zs)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def load_experiment_results(
    exp_id  : str,
    fold_idx: int,
) -> Optional[dict]:
    """
    Load metrics.json của một experiment đã chạy.

    Args:
        exp_id:   Experiment ID, ví dụ "VCB_VND_wavelet_BiLSTM_regression".
        fold_idx: 1-based fold index.

    Returns:
        Parsed dict từ metrics.json, hoặc None nếu file không tồn tại.
    """
    json_path = EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "metrics.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)