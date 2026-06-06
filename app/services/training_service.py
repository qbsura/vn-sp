"""
app/services/training_service.py
==================================
Training loop, early stopping, và prediction utilities cho VNSP.

Task 4.1 trong Phase 4 — Huấn luyện & Hyperparameter Optimization.

Cung cấp 5 components chính:
  1. train_one_epoch()  — train loop 1 epoch, return average loss
  2. evaluate()         — eval mode inference, return average val loss
  3. EarlyStopping      — monitor val loss, trigger stop khi không cải thiện
  4. train_model()      — full training pipeline (optimizer, scheduler, checkpoint)
  5. get_predictions()  — inference + inverse transform + thresholding

Design notes:
  - Device: CPU ONLY — TUYỆT ĐỐI không dùng "mps" (bug BiLSTM) hay "cuda"
  - Loss function:
      Task A (Regression)     : MSELoss — model output raw scaled Close(t+1)
      Task B (Classification) : BCELoss — model output probability ∈ [0,1] via Sigmoid
  - Gradient clipping: clip_grad_norm_(max_norm=1.0) theo bài báo section 4.5
  - LR Scheduler: ReduceLROnPlateau(factor=0.1, patience=10) — theo bài báo
  - Early stopping: patience=20 epochs
  - Best checkpoint: lưu và reload state_dict để model có best weights sau train

Output shape convention (khớp với tất cả model classes):
  Regression     : model → (batch, 1)  | y: (batch, 1)
  Classification : model → (batch, 1)  | y: (batch,) → cần squeeze output khi compute loss

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 4.1.4 : Adam optimizer, Dropout, BatchNorm, ReLU.
  Section 4.5   : gradient clipping, LR scheduling.
"""

from __future__ import annotations

import copy
import io
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.config import (
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_TRAIN_EPOCHS,
)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# CPU only — KHÔNG dùng "mps" (bug BiLSTM bidirectional) hay "cuda"
_DEVICE_OBJ = torch.device(DEVICE)


# =============================================================================
# 1. TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    task: str,
) -> float:
    """
    Chạy một epoch training và return average loss.

    Pipeline mỗi batch:
      zero_grad → forward → compute loss → backward → clip_grad → step

    Gradient clipping (clip_grad_norm_ max_norm=1.0) áp dụng sau backward()
    và trước optimizer.step() để ổn định training (bài báo section 4.5).

    Args:
        model:      nn.Module ở train mode — hàm này tự gọi model.train().
        dataloader: DataLoader, mỗi batch trả về (X_batch, y_batch).
        optimizer:  Adam optimizer (đã được tạo trước khi gọi).
        loss_fn:    MSELoss (regression) hoặc BCELoss (classification).
        task:       "regression" hoặc "classification" — ảnh hưởng shape logic.

    Returns:
        Average loss trên toàn epoch (float).
        Tính bằng tổng loss × batch_size / tổng số samples.

    Notes:
        Classification: model output (batch, 1) cần squeeze(-1) → (batch,)
        trước khi tính BCELoss với y shape (batch,).
    """
    model.train()
    total_loss   : float = 0.0
    total_samples: int   = 0

    for X_batch, y_batch in dataloader:
        # Move to device (luôn CPU ở đây)
        X_batch = X_batch.to(_DEVICE_OBJ)
        y_batch = y_batch.to(_DEVICE_OBJ)

        optimizer.zero_grad()

        # Forward pass
        output = model(X_batch)   # (batch, 1)

        # Shape alignment:
        #   Regression     : output=(batch,1), y=(batch,1) → no change needed
        #   Classification : output=(batch,1), y=(batch,) → squeeze output
        if task == "classification":
            output = output.squeeze(-1)   # (batch, 1) → (batch,)

        loss = loss_fn(output, y_batch)

        # Backward + gradient clip + optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        # Accumulate weighted loss (tính weighted average, không simple average)
        batch_size: int = X_batch.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0


# =============================================================================
# 2. EVALUATE
# =============================================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    task: str,
) -> float:
    """
    Chạy evaluation trên một DataLoader và return average loss.

    Dùng torch.no_grad() để tiết kiệm memory và tăng tốc inference.
    Gọi model.eval() để tắt Dropout và BatchNorm dùng running stats.

    Args:
        model:      nn.Module — hàm tự gọi model.eval() và khôi phục mode sau.
        dataloader: DataLoader (thường là val hoặc test set).
        loss_fn:    MSELoss hoặc BCELoss.
        task:       "regression" hoặc "classification".

    Returns:
        Average loss trên toàn dataloader (float).
    """
    model.eval()
    total_loss   : float = 0.0
    total_samples: int   = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(_DEVICE_OBJ)
            y_batch = y_batch.to(_DEVICE_OBJ)

            output = model(X_batch)   # (batch, 1)

            if task == "classification":
                output = output.squeeze(-1)   # (batch,)

            loss = loss_fn(output, y_batch)

            batch_size     = X_batch.size(0)
            total_loss    += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0


# =============================================================================
# 3. EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping theo val_loss.

    Dừng training khi val_loss không cải thiện hơn min_delta trong `patience` epochs.
    Tránh overfitting và tiết kiệm compute.

    Attributes:
        patience  (int)  : số epochs chờ không cải thiện trước khi dừng.
        min_delta (float): ngưỡng cải thiện tối thiểu (tránh dừng vì noise nhỏ).
        counter   (int)  : số epochs liên tiếp không cải thiện (hiện tại).
        best_loss (float): val_loss tốt nhất đã thấy.

    Usage:
        early_stop = EarlyStopping(patience=20)
        for epoch in range(max_epochs):
            val_loss = evaluate(...)
            if early_stop(val_loss):
                break
        early_stop.reset()  # trước lần train mới
    """

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 1e-6) -> None:
        """
        Args:
            patience:  Số epochs không cải thiện trước khi dừng. Default: 20 (config).
            min_delta: Ngưỡng cải thiện tối thiểu. Default: 1e-6.
        """
        if patience < 1:
            raise ValueError(f"EarlyStopping: patience={patience} phải >= 1.")
        if min_delta < 0:
            raise ValueError(f"EarlyStopping: min_delta={min_delta} phải >= 0.")

        self.patience  : int   = patience
        self.min_delta : float = min_delta
        self.counter   : int   = 0
        self.best_loss : float = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """
        Cập nhật state và quyết định có dừng không.

        Args:
            val_loss: Validation loss của epoch hiện tại.

        Returns:
            True  → nên dừng training (không cải thiện đủ lâu).
            False → tiếp tục training.
        """
        # Cải thiện: val_loss giảm ít nhất min_delta so với best
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False   # tiếp tục training

        # Không cải thiện: tăng counter
        self.counter += 1
        if self.counter >= self.patience:
            logger.debug(
                "[EarlyStopping] Dừng sau %d epochs không cải thiện. "
                "Best val_loss=%.6f",
                self.patience, self.best_loss,
            )
            return True   # dừng

        return False

    def reset(self) -> None:
        """
        Reset state về ban đầu.
        Gọi trước mỗi lần train mới (ví dụ: mỗi fold mới).
        """
        self.counter   = 0
        self.best_loss = float("inf")


# =============================================================================
# 4. TRAIN MODEL (full pipeline)
# =============================================================================

def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    params: dict,
    task: str,
    max_epochs: int = MAX_TRAIN_EPOCHS,
    save_path: Optional[str] = None,
) -> dict:
    """
    Full training pipeline với optimizer, scheduler, early stopping, checkpoint.

    Pipeline mỗi epoch:
      train_one_epoch() → evaluate() → scheduler.step(val_loss)
      → EarlyStopping check → save checkpoint nếu val_loss cải thiện

    Sau khi train xong, model được RESTORE về best weights (từ checkpoint).
    Đảm bảo model.predict() sau train_model() dùng best epoch, không phải last.

    Args:
        model:         nn.Module (đã được build_model() tạo, đang ở CPU).
        train_dataset: StockDataset cho training.
        val_dataset:   StockDataset cho validation (thường là test fold).
        params:        Hyperparameter dict từ Optuna. Keys cần có:
                         batch_size    (int)   : batch size cho DataLoader
                         learning_rate (float) : Adam initial learning rate
        task:          "regression" hoặc "classification".
        max_epochs:    Số epoch tối đa. Default: MAX_TRAIN_EPOCHS=200 (config).
        save_path:     Đường dẫn file .pt để lưu best model. None = không lưu ra disk.
                       Model vẫn được restore về best weights dù save_path=None.

    Returns:
        dict với keys:
          "train_losses"   (list[float]) : average train loss mỗi epoch
          "val_losses"     (list[float]) : average val loss mỗi epoch
          "best_val_loss"  (float)       : val loss tốt nhất đã thấy
          "best_epoch"     (int)         : epoch index (0-based) có best val loss
          "stopped_early"  (bool)        : True nếu early stopping kích hoạt

    Raises:
        ValueError: Nếu params thiếu required keys.
        RuntimeError: Nếu dataset rỗng hoặc không tương thích với model.

    Notes:
        - model.train() / model.eval() được quản lý trong train_one_epoch()/evaluate()
        - BCELoss: dùng cho classification (model đã có Sigmoid bên trong)
        - MSELoss: dùng cho regression (raw scaled output)
        - Gradient clip: max_norm=GRADIENT_CLIP_NORM=1.0 (từ config)
        - LR Scheduler: ReduceLROnPlateau, patience=LR_SCHEDULER_PATIENCE=10
    """
    # ── Validate params ───────────────────────────────────────────────────────
    required = {"batch_size", "learning_rate"}
    missing  = required - set(params.keys())
    if missing:
        raise ValueError(
            f"train_model: params thiếu keys {sorted(missing)}. "
            f"Required: {sorted(required)}"
        )

    batch_size    : int   = int(params["batch_size"])
    learning_rate : float = float(params["learning_rate"])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # Train: shuffle=True để tránh overfitting thứ tự
    # Val:   shuffle=False để đánh giá nhất quán
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False,   # giữ samples cuối (có thể batch nhỏ hơn)
        num_workers = 0,       # CPU-only, không cần multiprocessing
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    # Regression     : MSELoss — output = scaled Close(t+1), y = scaled Close(t+1)
    # Classification : BCELoss — output ∈ [0,1] (Sigmoid inside model), y ∈ {0,1}
    #   Tại sao BCELoss, không BCEWithLogitsLoss:
    #   Tất cả model classes (DNN/RNN/GRU/LSTM/BiLSTM) đã apply Sigmoid bên trong
    #   → output đã là probability. BCEWithLogitsLoss sẽ double-apply Sigmoid.
    if task == "regression":
        loss_fn: nn.Module = nn.MSELoss()
    elif task == "classification":
        loss_fn = nn.BCELoss()
    else:
        raise ValueError(f"train_model: task='{task}' không hợp lệ.")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Adam: adaptive learning rate, hiệu quả cho BiLSTM complex models
    # (bài báo section 4.1.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    # ReduceLROnPlateau: giảm LR ×factor khi val_loss không giảm trong patience epochs
    # Bài báo: reduce by 10× (factor=0.1) nếu val loss không cải thiện 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = LR_SCHEDULER_FACTOR,    # 0.1 từ config
        patience  = LR_SCHEDULER_PATIENCE,  # 10 từ config
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # ── State tracking ────────────────────────────────────────────────────────
    train_losses : list[float] = []
    val_losses   : list[float] = []
    best_val_loss: float       = float("inf")
    best_epoch   : int         = 0
    stopped_early: bool        = False

    # Lưu best weights vào buffer in-memory (hoặc disk nếu save_path có)
    # Dùng BytesIO để không phụ thuộc vào save_path=None
    best_weights_buffer: io.BytesIO = io.BytesIO()

    logger.info(
        "[train_model] Start | task=%s | epochs=%d | batch=%d | lr=%.4e",
        task, max_epochs, batch_size, learning_rate,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        # Train 1 epoch
        train_loss: float = train_one_epoch(
            model, train_loader, optimizer, loss_fn, task
        )

        # Evaluate trên validation set
        val_loss: float = evaluate(model, val_loader, loss_fn, task)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # LR scheduling: step với val_loss (bài báo section 4.5)
        scheduler.step(val_loss)

        # Checkpoint: lưu khi val_loss cải thiện
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch

            # Lưu best weights vào in-memory buffer (luôn)
            best_weights_buffer = io.BytesIO()
            torch.save(model.state_dict(), best_weights_buffer)

            # Lưu ra disk nếu save_path được chỉ định
            if save_path is not None:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
                torch.save(model.state_dict(), save_path)
                logger.debug(
                    "[train_model] Epoch %d | val_loss=%.6f (best) → saved to %s",
                    epoch, val_loss, save_path,
                )
            else:
                logger.debug(
                    "[train_model] Epoch %d | train=%.6f | val=%.6f (best)",
                    epoch, train_loss, val_loss,
                )
        else:
            logger.debug(
                "[train_model] Epoch %d | train=%.6f | val=%.6f",
                epoch, train_loss, val_loss,
            )

        # Early stopping check
        if early_stopping(val_loss):
            stopped_early = True
            logger.info(
                "[train_model] Early stopping tại epoch %d | best_epoch=%d | "
                "best_val_loss=%.6f",
                epoch, best_epoch, best_val_loss,
            )
            break

    # ── Restore best weights ──────────────────────────────────────────────────
    # Sau training, load lại best weights để model sẵn sàng cho inference
    # (tránh dùng "last epoch" weights thay vì "best epoch" weights)
    if best_weights_buffer.tell() > 0 or best_weights_buffer.getvalue():
        best_weights_buffer.seek(0)
        model.load_state_dict(
            torch.load(best_weights_buffer, map_location=_DEVICE_OBJ, weights_only=True)
        )
        logger.info(
            "[train_model] Restored best weights | epoch=%d | val_loss=%.6f",
            best_epoch, best_val_loss,
        )

    logger.info(
        "[train_model] Done | epochs_run=%d | best_epoch=%d | "
        "best_val_loss=%.6f | stopped_early=%s",
        len(train_losses), best_epoch, best_val_loss, stopped_early,
    )

    return {
        "train_losses"  : train_losses,
        "val_losses"    : val_losses,
        "best_val_loss" : best_val_loss,
        "best_epoch"    : best_epoch,
        "stopped_early" : stopped_early,
    }


# =============================================================================
# 5. GET PREDICTIONS
# =============================================================================

def get_predictions(
    model: nn.Module,
    test_dataset: Dataset,
    task: str,
    scaler=None,
    batch_size: int = 256,
) -> dict:
    """
    Chạy inference trên test_dataset và trả về predictions + ground truth.

    Task A (Regression):
      - y_pred: inverse transform → giá Close thực (VND hoặc USD)
      - y_true: inverse transform → giá Close thực
      - Không có y_prob (không liên quan)

    Task B (Classification):
      - y_prob: model output probability ∈ [0, 1] (P(UP))
      - y_pred: threshold 0.5 → class label {0=DOWN, 1=UP}
      - y_true: true labels {0, 1}

    Args:
        model:        nn.Module đã train xong (best weights).
        test_dataset: StockDataset — test fold.
        task:         "regression" hoặc "classification".
        scaler:       FeatureScaler đã fit — cần cho regression inverse transform.
                      Nếu None khi regression: y_pred/y_true là scaled values.
        batch_size:   Batch size khi chạy inference. Default: 256.

    Returns:
        dict:
          Regression:
            "y_pred" (np.ndarray, shape N): predicted Close prices (real-scale)
            "y_true" (np.ndarray, shape N): actual  Close prices (real-scale)
          Classification:
            "y_pred" (np.ndarray, shape N): predicted class {0, 1}
            "y_true" (np.ndarray, shape N): actual   class {0, 1}
            "y_prob" (np.ndarray, shape N): predicted probability P(UP) ∈ [0,1]

    Notes:
        - model.eval() + torch.no_grad() được dùng — safe cho BatchNorm/Dropout.
        - y_true cho regression được inverse transform từ y values trong dataset
          (những giá trị này là scaled Close, cần invert để compare với prediction).
    """
    # ── DataLoader cho inference ──────────────────────────────────────────────
    # Shuffle=False — giữ thứ tự chronological cho time-series evaluation
    loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
    )

    model.eval()
    all_preds : list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(_DEVICE_OBJ)

            output = model(X_batch)   # (batch, 1)

            # Detach, move to CPU, convert to numpy
            preds  = output.cpu().numpy()    # (batch, 1)
            labels = y_batch.numpy()         # (batch, 1) regression | (batch,) classif

            all_preds.append(preds)
            all_labels.append(labels)

    # ── Concatenate tất cả batches ────────────────────────────────────────────
    preds_arr  = np.concatenate(all_preds,  axis=0)   # (N, 1) hoặc (N,)
    labels_arr = np.concatenate(all_labels, axis=0)

    if task == "regression":
        # ── Regression: flatten và inverse transform ──────────────────────────
        preds_flat  = preds_arr.flatten()    # (N,)
        labels_flat = labels_arr.flatten()   # (N,)

        if scaler is not None:
            # Inverse transform scaled predictions → giá thực (VND/USD)
            y_pred = scaler.inverse_transform_target(preds_flat)
            y_true = scaler.inverse_transform_target(labels_flat)
        else:
            # Không có scaler → return scaled values (warning)
            logger.warning(
                "[get_predictions] scaler=None: y_pred/y_true là SCALED values, "
                "không phải giá thực. Truyền scaler để inverse transform."
            )
            y_pred = preds_flat
            y_true = labels_flat

        return {
            "y_pred": y_pred.astype(np.float64),   # (N,) giá Close thực
            "y_true": y_true.astype(np.float64),   # (N,) giá Close thực
        }

    elif task == "classification":
        # ── Classification: threshold + extract probability ───────────────────
        # preds_arr: (N, 1) probability P(UP)
        # labels_arr: (N,) true class {0, 1}

        y_prob  = preds_arr.flatten().astype(np.float64)     # (N,) P(UP)
        y_pred  = (y_prob >= 0.5).astype(np.int32)           # (N,) {0, 1}
        y_true  = labels_arr.flatten().astype(np.int32)      # (N,) {0, 1}

        return {
            "y_pred": y_pred,   # (N,) predicted class
            "y_true": y_true,   # (N,) actual   class
            "y_prob": y_prob,   # (N,) P(UP) ∈ [0,1]
        }

    else:
        raise ValueError(f"get_predictions: task='{task}' không hợp lệ.")