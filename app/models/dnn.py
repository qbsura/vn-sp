"""
app/models/dnn.py
==================
Dense Neural Network (DNN) — baseline model không có temporal modeling.

Task 3.2 trong Phase 3 — Model Architecture.

DNN flatten toàn bộ input sequence trước khi qua Dense layers,
không có temporal ordering. Dùng làm baseline để so sánh với
RNN/GRU/LSTM/BiLSTM và đo "giá trị thêm vào" của temporal modeling.

Architecture:
  (batch, seq_len, n_features)
    └─ Flatten → (batch, seq_len × n_features)
    └─ Block 1 : Linear → BatchNorm1d → ReLU → Dropout
    └─ Block 2 : Linear → BatchNorm1d → ReLU → Dropout  [nếu num_layers >= 2]
    └─ Block 3 : Linear → BatchNorm1d → ReLU → Dropout  [nếu num_layers == 3]
    └─ Output  : Linear(last_hidden, 1)
                 + Sigmoid nếu task="classification"

Neurons giảm dần (factor 0.5, floor = min(hidden_units, 64)):
  hidden_units=256, num_layers=3 → layers=[256, 128, 64]
  hidden_units=128, num_layers=3 → layers=[128,  64, 64]
  hidden_units=64,  num_layers=3 → layers=[64,   64, 64]
  hidden_units=32,  num_layers=2 → layers=[32,   32]      (32 < 64 → floor=32)

Activation: ReLU (bài báo Li et al. section 3.4 recommend ReLU, không tanh).

Output convention:
  Regression     : raw scalar — không activation — dùng với MSELoss
  Classification : probability ∈ [0,1] — sau Sigmoid — dùng với BCELoss
                   (BCELoss, không phải BCEWithLogitsLoss)

Class name: DNN (không phải DNNModel) để khớp với factory map trong base.py:
  _MODEL_MODULE_MAP["DNN"] = ("app.models.dnn", "DNN")

Constructor interface (match build_model factory):
  DNN(task, n_features, params)
  params keys cần có: num_layers, hidden_units, dropout_rate, sequence_length

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.4: Baseline models — Dense Neural Network.
"""

import logging

import torch
import torch.nn as nn

from app.config import DEFAULT_SEQUENCE_LENGTH, DEVICE
from app.models.base import BaseModel

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# CPU only — KHÔNG dùng "mps" (bug BiLSTM) hay "cuda" (không có GPU)
_DEVICE_OBJ = torch.device(DEVICE)

# Kích thước tối thiểu của hidden layer khi shrinking
_MIN_HIDDEN: int = 64


# =============================================================================
# DNN MODEL
# =============================================================================

class DNN(BaseModel):
    """
    Dense Neural Network — baseline prediction model.

    Không có temporal modeling: sequence được flatten hoàn toàn trước khi
    qua các Dense blocks. Dùng để so sánh với recurrent models.

    Attributes:
        seq_len (int)     : window size — cần biết lúc build để size Linear đầu.
        n_features (int)  : số input features.
        net (nn.Sequential): toàn bộ network gồm Dense blocks + output layer.
    """

    def __init__(self, task: str, n_features: int, params: dict) -> None:
        """
        Khởi tạo DNN.

        Args:
            task:       "regression" hoặc "classification".
            n_features: Số input features (wavelet case: 8, no-wavelet: 5).
            params:     Hyperparameter dict từ Optuna. Keys cần có:
                          num_layers      (int)   : số Dense blocks [1, 2, 3]
                          hidden_units    (int)   : neurons block đầu [32,64,128,256]
                          dropout_rate    (float) : dropout probability [0.1–0.5]
                          sequence_length (int)   : window size — dùng tính input size
                                                    nếu thiếu, dùng DEFAULT_SEQUENCE_LENGTH

        Raises:
            ValueError: Nếu params thiếu required keys hoặc giá trị không hợp lệ.
        """
        super().__init__("DNN", task)

        # ── Extract và validate hyperparameters ───────────────────────────────
        _check_required_params(params, {"num_layers", "hidden_units", "dropout_rate"})

        num_layers   : int   = int(params["num_layers"])
        hidden_units : int   = int(params["hidden_units"])
        dropout_rate : float = float(params["dropout_rate"])

        # seq_len cần biết lúc build để tính input_size của Linear đầu tiên
        self.seq_len   : int = int(params.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
        self.n_features: int = n_features

        if num_layers < 1:
            raise ValueError(f"DNN: num_layers={num_layers} phải >= 1.")
        if hidden_units < 1:
            raise ValueError(f"DNN: hidden_units={hidden_units} phải >= 1.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"DNN: dropout_rate={dropout_rate} phải trong [0, 1).")

        # ── Tính kích thước các hidden layers (giảm dần) ─────────────────────
        layer_sizes: list[int] = _compute_layer_sizes(hidden_units, num_layers)

        # ── Xây dựng network ─────────────────────────────────────────────────
        # Input size sau khi flatten: seq_len × n_features
        input_size: int = self.seq_len * n_features
        blocks: list[nn.Module] = []

        in_size = input_size
        for hidden_size in layer_sizes:
            # Dense block: Linear → BatchNorm1d → ReLU → Dropout
            blocks.append(nn.Linear(in_size, hidden_size))
            blocks.append(nn.BatchNorm1d(hidden_size))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(p=dropout_rate))
            in_size = hidden_size

        # ── Output layer ──────────────────────────────────────────────────────
        # Regression     : raw scalar, dùng với MSELoss
        # Classification : probability qua Sigmoid, dùng với BCELoss
        blocks.append(nn.Linear(in_size, 1))
        if task == "classification":
            blocks.append(nn.Sigmoid())

        self.net = nn.Sequential(*blocks)

        logger.debug(
            f"[DNN] task={task} | n_features={n_features} | "
            f"seq_len={self.seq_len} | "
            f"input_size={input_size} | "
            f"layer_sizes={layer_sizes} | "
            f"dropout={dropout_rate:.2f} | "
            f"params={self.get_param_count():,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor shape (batch_size, seq_len, n_features).
               seq_len và n_features phải khớp với lúc khởi tạo.

        Returns:
            Tensor shape (batch_size, 1):
              Regression     : raw scaled Close(t+1)
              Classification : probability ∈ [0, 1] (sau Sigmoid)

        Raises:
            RuntimeError: Nếu x không phải 3-D tensor.
        """
        # Validate 3-D input
        if x.ndim != 3:
            raise RuntimeError(
                f"DNN.forward: expected 3-D input (batch, seq_len, n_features), "
                f"got {x.ndim}-D tensor shape={tuple(x.shape)}."
            )

        # Flatten: (batch, seq_len, n_features) → (batch, seq_len × n_features)
        # x.size(0) = batch_size; -1 cho PyTorch tự tính seq_len × n_features
        x = x.view(x.size(0), -1)

        # Forward qua Sequential: Dense blocks → output → [Sigmoid]
        return self.net(x)


# =============================================================================
# MODULE-LEVEL HELPERS (private)
# =============================================================================

def _compute_layer_sizes(hidden_units: int, num_layers: int) -> list[int]:
    """
    Tính danh sách hidden sizes cho các Dense blocks.

    Rule: giảm factor 0.5 sau mỗi layer, floor = min(hidden_units, _MIN_HIDDEN).
    Nếu hidden_units < _MIN_HIDDEN (ví dụ 32 < 64), floor là hidden_units chính nó
    → tránh tăng kích thước thay vì giảm.

    Args:
        hidden_units: Số neurons block đầu tiên.
        num_layers:   Số Dense blocks.

    Returns:
        List hidden sizes, len == num_layers.

    Examples:
        _compute_layer_sizes(256, 3)  →  [256, 128,  64]
        _compute_layer_sizes(128, 3)  →  [128,  64,  64]
        _compute_layer_sizes(64,  3)  →  [ 64,  64,  64]
        _compute_layer_sizes(32,  2)  →  [ 32,  32]      (floor=32, bellow 64)
        _compute_layer_sizes(128, 1)  →  [128]
    """
    # floor = min để tránh layer nhỏ bị tăng lên khi hidden_units < _MIN_HIDDEN
    floor: int = min(hidden_units, _MIN_HIDDEN)
    sizes: list[int] = []
    h: int = hidden_units

    for _ in range(num_layers):
        sizes.append(h)
        # Giảm 50%, không xuống dưới floor
        h = max(h // 2, floor)

    return sizes


def _check_required_params(params: dict, required: set[str]) -> None:
    """
    Kiểm tra params dict có đủ required keys không.

    Args:
        params:   Hyperparameter dict từ Optuna.
        required: Set tên keys cần thiết.

    Raises:
        ValueError: Nếu có key bị thiếu.
    """
    missing = required - set(params.keys())
    if missing:
        raise ValueError(
            f"params dict thiếu keys: {sorted(missing)}. "
            f"Required: {sorted(required)}"
        )