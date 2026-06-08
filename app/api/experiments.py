"""
app/api/experiments.py
=======================
API routes cho Experiment Management — /api/experiments/*

Endpoints:
  GET  /api/experiments/matrix              → trạng thái toàn bộ experiment matrix
  GET  /api/experiments/{exp_id}            → info một experiment
  GET  /api/experiments/{exp_id}/params     → HPO best_params
  GET  /api/experiments/{exp_id}/loss-curves → train/val loss per fold
  POST /api/experiments/run                 → trigger experiment async  ← Task 7.3
  GET  /api/experiments/{job_id}/status     → trạng thái job            ← Task 7.3
  POST /api/experiments/hpo                 → trigger Optuna HPO        ← Task 7.3

Job tracking:
  In-memory dict _JOBS: {job_id: {"status", "type", "progress", "error", ...}}
  Job ID: 8-char UUID prefix — đủ unique cho số lượng jobs thực tế.
  Trạng thái: "running" → "done" | "error"

Tham chiếu:
  Experiment ID format: {ticker}_{currency}_{cond}_{model}_{task}
  Ví dụ: "VCB_VND_wavelet_BiLSTM_regression"
"""

from __future__ import annotations

import json
import logging
import uuid
from itertools import product
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from app.config import CURRENCIES, FOLDS, MODELS, OPTUNA_TRIALS, PATHS, TASKS, TICKERS, WAVELET_CONDITIONS

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Paths ──────────────────────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(PATHS["experiments"])


# =============================================================================
# IN-MEMORY JOB STORE  (Task 7.3)
# =============================================================================

# Dict đơn giản — đủ cho single-process FastAPI server.
# Key: job_id (str), Value: dict chứa status + metadata.
_JOBS: dict[str, dict] = {}


# =============================================================================
# REQUEST MODELS  (Task 7.3)
# =============================================================================

class RunExperimentRequest(BaseModel):
    """Body cho POST /api/experiments/run."""
    ticker      : str
    currency    : str
    use_wavelet : bool
    model       : str
    task        : str
    fold        : Optional[int] = None   # None = chạy tất cả 3 folds


class HPORequest(BaseModel):
    """Body cho POST /api/experiments/hpo."""
    ticker      : str
    currency    : str
    use_wavelet : bool
    n_trials    : int = OPTUNA_TRIALS    # default từ config (30)


# =============================================================================
# BACKGROUND TASK FUNCTIONS  (Task 7.3)
# =============================================================================

def _run_experiment_background(job_id: str, body: RunExperimentRequest) -> None:
    """
    Background task: chạy run_single_experiment cho từng fold.

    Load best_params từ HPO results (phải chạy run_hpo trước).
    Cập nhật _JOBS[job_id] theo tiến độ.
    """
    from app.services.experiment_runner import run_single_experiment
    from app.services.hpo_service import load_best_params

    cond_str = "wavelet" if body.use_wavelet else "nowave"
    exp_id   = f"{body.ticker}_{body.currency}_{cond_str}_{body.model}_{body.task}"

    # Xác định danh sách folds cần chạy
    folds_to_run = (
        [body.fold] if body.fold is not None
        else [f["fold_id"] for f in FOLDS]
    )
    total = len(folds_to_run)

    try:
        for i, fold_idx in enumerate(folds_to_run):
            _JOBS[job_id]["progress"] = f"{i}/{total} folds done (fold {fold_idx} running)"

            # Load best_params từ HPO results
            best_params = load_best_params(
                ticker      = body.ticker,
                currency    = body.currency,
                use_wavelet = body.use_wavelet,
                fold_idx    = fold_idx,
            )

            run_single_experiment(
                ticker      = body.ticker,
                currency    = body.currency,
                use_wavelet = body.use_wavelet,
                model_name  = body.model,
                task        = body.task,
                fold_idx    = fold_idx,
                best_params = best_params,
            )

        # Done
        _JOBS[job_id]["status"]   = "done"
        _JOBS[job_id]["progress"] = f"{total}/{total} folds done"
        _JOBS[job_id]["exp_id"]   = exp_id
        logger.info("[job %s] Experiment done: %s", job_id, exp_id)

    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = str(exc)
        logger.exception("[job %s] Experiment failed: %s — %s", job_id, exp_id, exc)


def _run_hpo_background(job_id: str, body: HPORequest) -> None:
    """
    Background task: chạy run_full_hpo (3 folds) cho BiLSTM.

    Kết quả lưu vào experiments/{ticker}_{currency}_{cond}/fold_{i}/best_params.json.
    """
    from app.services.hpo_service import run_full_hpo

    cond_str = "wavelet" if body.use_wavelet else "nowave"
    label    = f"{body.ticker}_{body.currency}_{cond_str}"

    try:
        _JOBS[job_id]["progress"] = "HPO đang chạy (3 folds × n_trials)..."

        results = run_full_hpo(
            ticker      = body.ticker,
            currency    = body.currency,
            use_wavelet = body.use_wavelet,
            n_trials    = body.n_trials,
        )

        _JOBS[job_id]["status"]   = "done"
        _JOBS[job_id]["progress"] = f"3/3 folds done"
        _JOBS[job_id]["result"]   = {
            "label"  : label,
            "n_folds": len(results),
        }
        logger.info("[job %s] HPO done: %s (%d folds)", job_id, label, len(results))

    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = str(exc)
        logger.exception("[job %s] HPO failed: %s — %s", job_id, label, exc)


# =============================================================================
# EXPERIMENT MATRIX STATUS
# =============================================================================

@router.get("/matrix", summary="Trạng thái toàn bộ experiment matrix")
def get_experiment_matrix(
    ticker  : Optional[str]  = Query(None, description="Filter theo ticker"),
    currency: Optional[str]  = Query(None, description="Filter theo currency"),
    task    : Optional[str]  = Query(None, description="Filter theo task"),
    wavelet : Optional[bool] = Query(None, description="Filter theo wavelet condition"),
) -> dict:
    """
    Trả về trạng thái của toàn bộ 240 experiments (2×2×2×5×2×3 folds).

    Status của mỗi experiment được xác định dựa trên sự tồn tại của metrics.json:
      "done"    — tất cả 3 folds có metrics.json
      "partial" — 1 hoặc 2 folds có metrics.json
      "pending" — không có fold nào có metrics.json

    Returns:
        {"total": int, "done": int, "partial": int, "pending": int,
         "experiments": [{"exp_id": str, "ticker": str, "currency": str,
                          "use_wavelet": bool, "model": str, "task": str,
                          "status": str, "folds_done": int, "folds_total": int}]}
    """
    experiments = []
    n_done = n_partial = n_pending = 0

    cond_list = [True, False] if wavelet is None else [wavelet]

    for t, c, w, m, task_val in product(
        TICKERS    if ticker   is None else [ticker],
        CURRENCIES if currency is None else [currency],
        cond_list,
        MODELS,
        TASKS if task is None else [task],
    ):
        cond_str = "wavelet" if w else "nowave"
        exp_id   = f"{t}_{c}_{cond_str}_{m}_{task_val}"
        exp_dir  = _EXPERIMENTS_DIR / exp_id

        folds_done = sum(
            1 for fold in FOLDS
            if (exp_dir / f"fold_{fold['fold_id']}" / "metrics.json").exists()
        )
        folds_total = len(FOLDS)

        if folds_done == folds_total:
            status = "done"
            n_done += 1
        elif folds_done > 0:
            status = "partial"
            n_partial += 1
        else:
            status = "pending"
            n_pending += 1

        experiments.append({
            "exp_id"      : exp_id,
            "ticker"      : t,
            "currency"    : c,
            "use_wavelet" : w,
            "model"       : m,
            "task"        : task_val,
            "status"      : status,
            "folds_done"  : folds_done,
            "folds_total" : folds_total,
        })

    total = len(experiments)
    return {
        "total"      : total,
        "done"       : n_done,
        "partial"    : n_partial,
        "pending"    : n_pending,
        "experiments": experiments,
    }


# =============================================================================
# RUN EXPERIMENT (async)  — Task 7.3
# =============================================================================

@router.post("/run", summary="Chạy experiment (background task)")
def run_experiment(
    body             : RunExperimentRequest,
    background_tasks : BackgroundTasks,
) -> dict:
    """
    Trigger một experiment trong background.

    Yêu cầu: HPO đã chạy trước (best_params.json phải tồn tại).
    Dùng GET /api/experiments/{job_id}/status để theo dõi tiến độ.

    Args (request body):
        ticker:      "VCB" hoặc "VIC".
        currency:    "VND" hoặc "USD".
        use_wavelet: True / False.
        model:       "DNN", "RNN", "GRU", "LSTM", "BiLSTM".
        task:        "regression" hoặc "classification".
        fold:        1, 2, hoặc 3. None = chạy tất cả 3 folds.

    Returns:
        {"job_id": str, "status": "started", "exp_id": str}
    """
    # Validate inputs
    if body.ticker not in TICKERS:
        raise HTTPException(400, f"ticker='{body.ticker}' không hợp lệ. Chọn: {TICKERS}")
    if body.currency not in CURRENCIES:
        raise HTTPException(400, f"currency='{body.currency}' không hợp lệ. Chọn: {CURRENCIES}")
    if body.model not in MODELS:
        raise HTTPException(400, f"model='{body.model}' không hợp lệ. Chọn: {MODELS}")
    if body.task not in TASKS:
        raise HTTPException(400, f"task='{body.task}' không hợp lệ. Chọn: {TASKS}")
    if body.fold is not None and body.fold not in [f["fold_id"] for f in FOLDS]:
        raise HTTPException(400, f"fold={body.fold} không hợp lệ. Chọn: 1, 2, 3 hoặc None.")

    # Tạo job ID ngắn (8 chars UUID prefix — đủ unique)
    job_id  = uuid.uuid4().hex[:8]
    cond_str = "wavelet" if body.use_wavelet else "nowave"
    exp_id  = f"{body.ticker}_{body.currency}_{cond_str}_{body.model}_{body.task}"

    # Khởi tạo job record
    _JOBS[job_id] = {
        "status"  : "running",
        "type"    : "experiment",
        "exp_id"  : exp_id,
        "progress": "0/3 folds done",
        "error"   : None,
    }

    background_tasks.add_task(_run_experiment_background, job_id, body)
    logger.info("[POST /run] Queued job %s for %s", job_id, exp_id)

    return {
        "job_id" : job_id,
        "status" : "started",
        "exp_id" : exp_id,
    }


# =============================================================================
# JOB STATUS  — Task 7.3
# =============================================================================

@router.get("/{job_id}/status", summary="Trạng thái và tiến độ của một job")
def get_job_status(job_id: str) -> dict:
    """
    Kiểm tra trạng thái của một background job (experiment hoặc HPO).

    Args:
        job_id: Job ID trả về từ POST /run hoặc POST /hpo.

    Returns:
        {"job_id": str, "status": "running|done|error",
         "type": str, "progress": str, "error": str | None,
         "exp_id": str | None}

    Raises:
        404: Nếu job_id không tồn tại.
    """
    if job_id not in _JOBS:
        raise HTTPException(
            status_code = 404,
            detail      = f"job_id='{job_id}' không tồn tại. "
                          "Có thể server đã restart và mất trạng thái in-memory.",
        )

    job = _JOBS[job_id]
    return {
        "job_id"  : job_id,
        "status"  : job.get("status",   "unknown"),
        "type"    : job.get("type",     "unknown"),
        "progress": job.get("progress", ""),
        "error"   : job.get("error",    None),
        "exp_id"  : job.get("exp_id",   None),
        "result"  : job.get("result",   None),
    }


# =============================================================================
# HPO TRIGGER (async)  — Task 7.3
# =============================================================================

@router.post("/hpo", summary="Trigger Optuna HPO cho BiLSTM (background)")
def trigger_hpo(
    body             : HPORequest,
    background_tasks : BackgroundTasks,
) -> dict:
    """
    Trigger HPO cho BiLSTM trong background.

    Chạy Optuna TPE với n_trials trials cho cả 3 folds.
    Kết quả lưu tại: experiments/{ticker}_{currency}_{cond}/fold_{i}/best_params.json

    Yêu cầu: PKL preprocessed data phải tồn tại trước.

    Args (request body):
        ticker:      "VCB" hoặc "VIC".
        currency:    "VND" hoặc "USD".
        use_wavelet: True / False.
        n_trials:    Số Optuna trials per fold. Default: 30.

    Returns:
        {"job_id": str, "status": "started", "label": str}
    """
    if body.ticker not in TICKERS:
        raise HTTPException(400, f"ticker='{body.ticker}' không hợp lệ. Chọn: {TICKERS}")
    if body.currency not in CURRENCIES:
        raise HTTPException(400, f"currency='{body.currency}' không hợp lệ. Chọn: {CURRENCIES}")
    if body.n_trials < 1 or body.n_trials > 200:
        raise HTTPException(400, "n_trials phải trong khoảng [1, 200].")

    # Kiểm tra PKL tồn tại trước khi queue job
    cond_str = "wavelet" if body.use_wavelet else "nowave"
    pkl_path = Path(PATHS["processed"]) / f"{body.ticker}_{body.currency}_{cond_str}.pkl"
    if not pkl_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"{pkl_path.name} không tồn tại. Chạy POST /api/data/preprocess trước.",
        )

    job_id = uuid.uuid4().hex[:8]
    label  = f"{body.ticker}_{body.currency}_{cond_str}"

    _JOBS[job_id] = {
        "status"  : "running",
        "type"    : "hpo",
        "exp_id"  : None,
        "progress": "HPO đang khởi động...",
        "error"   : None,
        "result"  : None,
    }

    background_tasks.add_task(_run_hpo_background, job_id, body)
    logger.info("[POST /hpo] Queued HPO job %s for %s (%d trials)", job_id, label, body.n_trials)

    return {
        "job_id" : job_id,
        "status" : "started",
        "label"  : label,
        "n_trials": body.n_trials,
    }


# =============================================================================
# SINGLE EXPERIMENT INFO
# =============================================================================

@router.get("/{exp_id}", summary="Thông tin một experiment")
def get_experiment_info(exp_id: str) -> dict:
    """
    Trả về metadata và metrics (tất cả folds) của một experiment.

    Path param:
        exp_id: Experiment ID, ví dụ "VCB_VND_wavelet_BiLSTM_regression".

    Returns:
        {"exp_id": str, "folds": [{"fold_idx": int, "metrics": dict, ...}]}

    Raises:
        404: Nếu không có fold nào của experiment này có metrics.json.
    """
    exp_dir = _EXPERIMENTS_DIR / exp_id
    if not exp_dir.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"Experiment '{exp_id}' chưa chạy (thư mục không tồn tại).",
        )

    folds_data = []
    for fold in FOLDS:
        fold_idx  = fold["fold_id"]
        json_path = exp_dir / f"fold_{fold_idx}" / "metrics.json"
        if not json_path.exists():
            folds_data.append({"fold_idx": fold_idx, "status": "pending"})
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        folds_data.append({
            "fold_idx"       : fold_idx,
            "status"         : "done",
            "metrics"        : data.get("metrics",      {}),
            "best_epoch"     : data.get("train_history", {}).get("best_epoch"),
            "best_val_loss"  : data.get("train_history", {}).get("best_val_loss"),
            "stopped_early"  : data.get("train_history", {}).get("stopped_early"),
            "n_epochs_run"   : data.get("train_history", {}).get("n_epochs_run"),
            "n_test_samples" : data.get("n_test_samples"),
            "train_elapsed_s": data.get("train_elapsed_s"),
        })

    if all(f.get("status") == "pending" for f in folds_data):
        raise HTTPException(
            status_code = 404,
            detail      = f"Experiment '{exp_id}' chưa có fold nào hoàn thành.",
        )

    return {"exp_id": exp_id, "folds": folds_data}


# =============================================================================
# HPO BEST PARAMS
# =============================================================================

@router.get("/{exp_id}/params", summary="HPO best params của experiment")
def get_best_params(
    exp_id  : str,
    fold_idx: int = Query(1, ge=1, le=3, description="Fold index 1–3"),
) -> dict:
    """
    Trả về best hyperparameters từ Optuna HPO cho một fold.

    Note: BiLSTM có params riêng; DNN/RNN/GRU/LSTM dùng chung params của BiLSTM
    (cùng ticker/currency/wavelet condition).

    Returns:
        {"exp_id": str, "fold_idx": int, "params": dict}
    """
    # Lấy condition prefix từ exp_id (ticker_currency_cond)
    parts = exp_id.split("_")
    if len(parts) < 5:
        raise HTTPException(
            status_code = 400,
            detail      = f"exp_id='{exp_id}' không hợp lệ. "
                          "Format: ticker_currency_cond_model_task",
        )
    ticker, currency, cond = parts[0], parts[1], parts[2]
    hpo_dir   = _EXPERIMENTS_DIR / f"{ticker}_{currency}_{cond}" / f"fold_{fold_idx}"
    json_path = hpo_dir / "best_params.json"

    if not json_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"best_params.json không tồn tại tại '{json_path}'. "
                          "Chạy POST /api/experiments/hpo trước.",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return {"exp_id": exp_id, "fold_idx": fold_idx, "params": params}


# =============================================================================
# LOSS CURVES
# =============================================================================

@router.get("/{exp_id}/loss-curves", summary="Train/val loss curves data")
def get_loss_curves(
    exp_id  : str,
    fold_idx: int = Query(1, ge=1, le=3),
) -> dict:
    """
    Trả về train/val loss per epoch cho một fold.
    Dùng để render loss curve chart ở frontend.

    Returns:
        {"exp_id": str, "fold_idx": int,
         "train_losses": list[float], "val_losses": list[float],
         "best_epoch": int, "stopped_early": bool, "n_epochs_run": int}
    """
    json_path = _EXPERIMENTS_DIR / exp_id / f"fold_{fold_idx}" / "metrics.json"
    if not json_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"metrics.json chưa tồn tại cho {exp_id}/fold_{fold_idx}.",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("train_history", {})
    return {
        "exp_id"        : exp_id,
        "fold_idx"      : fold_idx,
        "train_losses"  : history.get("train_losses",  []),
        "val_losses"    : history.get("val_losses",    []),
        "best_epoch"    : history.get("best_epoch",    None),
        "stopped_early" : history.get("stopped_early", False),
        "n_epochs_run"  : history.get("n_epochs_run",  0),
    }