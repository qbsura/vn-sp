"""
app/models/rnn.py
==================
Vanilla Recurrent Neural Network (RNN) — baseline model.

Task 3.3 trong Phase 3 — Model Architecture.

RNN xử lý input theo thứ tự thời gian, mỗi bước cập nhật hidden state.
Dùng làm baseline để so sánh với GRU/LSTM (có gating) và BiLSTM (bidirectional).

Architecture:
  (batch, seq_len, n_features)
    └─ nn.RNN(n_features → hidden_units, nonlinearity='relu')
    └─ Lấy timestep cuối: out[:, -1, :] → (batch, hidden_units)
    └─ Linear(hidden_units, 64) → ReLU → Dropout
    └─ Linear(64, 1)
    └─ Sigmoid nếu task="classification"

QUAN TRỌNG — Activation trong RNN:
  nonlinearity='relu' thay vì mặc định 'tanh'.
  Bài báo Li et al. section 3.4 ghi nhận: tanh gây degraded performance
  so với relu trên time-series dữ liệu giá cổ phiếu.

Output convention:
  Regression     : raw scalar — không activation — dùng với MSELoss
  Classification : probability ∈ [0,1] — sau Sigmoid — dùng với BCELoss

Class name: RNN (không phải RNNModel) để khớp factory map trong base.py:
  _MODEL_MODULE_MAP["RNN"] = ("app.models.rnn", "RNN")

Constructor interface (match build_model factory):
  RNN(task, n_features, params)
  params keys cần có: num_layers, hidden_units, dropout_rate

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.4: Baseline models — Vanilla RNN with relu activation.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import DEVICE
from app.models.base import BaseModel

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# CPU only — KHÔNG dùng "mps" (bug BiLSTM) hay "cuda" (không có GPU)
_DEVICE_OBJ = torch.device(DEVICE)

# Kích thước fixed của hidden layer giữa (theo bài báo)
_FC_HIDDEN: int = 64


# =============================================================================
# RNN MODEL
# =============================================================================

class RNN(BaseModel):
    """
    Vanilla Recurrent Neural Network — baseline temporal model.

    Xử lý sequence theo thứ tự thời gian với recurrent hidden state.
    Activation trong recurrent cell là ReLU (không phải tanh mặc định).

    Architecture:
      nn.RNN → last timestep → fc1 → ReLU → Dropout → fc2 → [Sigmoid]

    Attributes:
        rnn     (nn.RNN)     : recurrent layer, nonlinearity='relu'
        fc1     (nn.Linear)  : hidden_units → 64
        dropout (nn.Dropout) : dropout sau fc1
        fc2     (nn.Linear)  : 64 → 1
        sigmoid (nn.Sigmoid) : chỉ tồn tại nếu task="classification"
        n_features (int)     : số input features
        hidden_units (int)   : RNN hidden size
    """

    def __init__(self, task: str, n_features: int, params: dict) -> None:
        """
        Khởi tạo RNN.

        Args:
            task:       "regression" hoặc "classification".
            n_features: Số input features (wavelet: 8, no-wavelet: 5).
            params:     Hyperparameter dict từ Optuna. Keys cần có:
                          num_layers   (int)   : số RNN layers [1, 2, 3]
                          hidden_units (int)   : hidden state size [32, 64, 128, 256]
                          dropout_rate (float) : dropout probability [0.1–0.5]

        Raises:
            ValueError: Nếu params thiếu required keys hoặc giá trị không hợp lệ.

        Notes:
            dropout trong nn.RNN chỉ áp dụng khi num_layers > 1.
            Với num_layers=1, nn.RNN bỏ qua dropout (PyTorch convention).
        """
        super().__init__("RNN", task)

        # ── Extract và validate hyperparameters ───────────────────────────────
        _check_required_params(params, {"num_layers", "hidden_units", "dropout_rate"})

        num_layers  : int   = int(params["num_layers"])
        hidden_units: int   = int(params["hidden_units"])
        dropout_rate: float = float(params["dropout_rate"])

        if num_layers < 1:
            raise ValueError(f"RNN: num_layers={num_layers} phải >= 1.")
        if hidden_units < 1:
            raise ValueError(f"RNN: hidden_units={hidden_units} phải >= 1.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"RNN: dropout_rate={dropout_rate} phải trong [0, 1).")

        # Lưu để dùng trong forward và logging
        self.n_features : int = n_features
        self.hidden_units: int = hidden_units

        # ── Recurrent layer ───────────────────────────────────────────────────
        # QUAN TRỌNG: nonlinearity='relu' — bài báo ghi nhận tanh làm degraded
        # performance trên time-series giá cổ phiếu (section 3.4).
        # dropout của nn.RNN chỉ áp dụng giữa các layers khi num_layers > 1.
        self.rnn = nn.RNN(
            input_size   = n_features,
            hidden_size  = hidden_units,
            num_layers   = num_layers,
            batch_first  = True,           # input/output shape: (batch, seq, feature)
            dropout      = dropout_rate if num_layers > 1 else 0.0,
            nonlinearity = "relu",         # KHÔNG dùng "tanh"
        )

        # ── 2 Dense layers sau RNN: hidden_units → 64 → 1 ────────────────────
        self.fc1     = nn.Linear(hidden_units, _FC_HIDDEN)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2     = nn.Linear(_FC_HIDDEN, 1)

        # Sigmoid chỉ tồn tại cho classification
        # Tạo sẵn như module để được đưa vào parameters() nếu cần
        if task == "classification":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None  # type: ignore[assignment]

        logger.debug(
            f"[RNN] task={task} | n_features={n_features} | "
            f"hidden={hidden_units} | layers={num_layers} | "
            f"dropout={dropout_rate:.2f} | "
            f"fc: {hidden_units}→{_FC_HIDDEN}→1 | "
            f"params={self.get_param_count():,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor shape (batch_size, seq_len, n_features).

        Returns:
            Tensor shape (batch_size, 1):
              Regression     : raw scaled Close(t+1)
              Classification : probability ∈ [0, 1] (sau Sigmoid)

        Raises:
            RuntimeError: Nếu x không phải 3-D tensor.
        """
        # ── Validate shape ────────────────────────────────────────────────────
        if x.ndim != 3:
            raise RuntimeError(
                f"RNN.forward: expected 3-D input (batch, seq_len, n_features), "
                f"got {x.ndim}-D tensor shape={tuple(x.shape)}."
            )

        # ── RNN pass ──────────────────────────────────────────────────────────
        # out: (batch, seq_len, hidden_units) — output tại mỗi timestep
        # _:   (num_layers, batch, hidden_units) — hidden state cuối cùng (không dùng)
        out, _ = self.rnn(x)

        # Lấy output tại timestep cuối cùng — tổng hợp thông tin toàn sequence
        # Shape: (batch, seq_len, hidden_units) → (batch, hidden_units)
        out = out[:, -1, :]

        # ── Dense layers: hidden_units → 64 → 1 ──────────────────────────────
        out = F.relu(self.fc1(out))   # Linear + ReLU
        out = self.dropout(out)        # Dropout regularization
        out = self.fc2(out)            # Output: (batch, 1)

        # ── Output activation ────────────────────────────────────────────────
        # Regression: trả về raw (không sigmoid/tanh)
        # Classification: sigmoid → probability ∈ [0, 1]
        if self.sigmoid is not None:
            out = self.sigmoid(out)

        return out


# =============================================================================
# MODULE-LEVEL HELPERS (private)
# =============================================================================

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