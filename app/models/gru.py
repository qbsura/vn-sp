"""
app/models/gru.py
==================
Gated Recurrent Unit (GRU) — baseline model.

Task 3.4 trong Phase 3 — Model Architecture.

GRU là biến thể của LSTM với ít tham số hơn, dùng 2 gates (reset + update)
thay vì 3 (input + forget + output). Thường train nhanh hơn LSTM và cho
kết quả tương đương trên nhiều time-series tasks.

Architecture:
  (batch, seq_len, n_features)
    └─ nn.GRU(n_features → hidden_units, num_layers, batch_first)
    └─ Lấy timestep cuối: out[:, -1, :] → (batch, hidden_units)
    └─ BatchNorm1d(hidden_units)
    └─ Linear(hidden_units, 64) → Dropout → ReLU
    └─ Linear(64, 1)
    └─ Sigmoid nếu task="classification"

Thứ tự Dense head (theo bài báo spec):
  BN → fc1 → Dropout → ReLU → fc2 → [Sigmoid]
  (BatchNorm trước fc1, Dropout trước ReLU — khác với RNN baseline)

Output convention:
  Regression     : raw scalar — không activation — dùng với MSELoss
  Classification : probability ∈ [0,1] — sau Sigmoid — dùng với BCELoss

Training note: dùng Adam optimizer với ReduceLROnPlateau scheduler
  (được set up ở training_service.py, không phải ở đây).

Class name: GRU (không phải GRUModel) để khớp factory map trong base.py:
  _MODEL_MODULE_MAP["GRU"] = ("app.models.gru", "GRU")

Constructor interface (match build_model factory):
  GRU(task, n_features, params)
  params keys cần có: num_layers, hidden_units, dropout_rate

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.4: Baseline models — GRU.
  Cho, Bahdanau et al. (2014): Learning Phrase Representations using RNN.
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

# Kích thước fixed của hidden layer giữa (theo bài báo spec)
_FC_HIDDEN: int = 64


# =============================================================================
# GRU MODEL
# =============================================================================

class GRU(BaseModel):
    """
    Gated Recurrent Unit — baseline temporal model với gating mechanism.

    GRU dùng reset gate và update gate để kiểm soát luồng thông tin,
    giúp capture long-term dependencies hiệu quả hơn vanilla RNN.

    Architecture:
      nn.GRU → last timestep → BN → fc1 → Dropout → ReLU → fc2 → [Sigmoid]

    Attributes:
        gru     (nn.GRU)        : recurrent GRU layer
        bn      (nn.BatchNorm1d): normalize GRU output trước Dense head
        fc1     (nn.Linear)     : hidden_units → 64
        dropout (nn.Dropout)    : dropout trước ReLU (sau fc1)
        fc2     (nn.Linear)     : 64 → 1
        sigmoid (nn.Sigmoid)    : chỉ tồn tại nếu task="classification"
        n_features (int)        : số input features
        hidden_units (int)      : GRU hidden size
    """

    def __init__(self, task: str, n_features: int, params: dict) -> None:
        """
        Khởi tạo GRU.

        Args:
            task:       "regression" hoặc "classification".
            n_features: Số input features (wavelet: 8, no-wavelet: 5).
            params:     Hyperparameter dict từ Optuna. Keys cần có:
                          num_layers   (int)   : số GRU layers [1, 2, 3]
                          hidden_units (int)   : hidden state size [32, 64, 128, 256]
                          dropout_rate (float) : dropout probability [0.1–0.5]

        Raises:
            ValueError: Nếu params thiếu required keys hoặc giá trị không hợp lệ.

        Notes:
            dropout trong nn.GRU chỉ áp dụng giữa các layers khi num_layers > 1.
            BatchNorm1d được áp dụng trên GRU output trước Dense head.
        """
        super().__init__("GRU", task)

        # ── Extract và validate hyperparameters ───────────────────────────────
        _check_required_params(params, {"num_layers", "hidden_units", "dropout_rate"})

        num_layers  : int   = int(params["num_layers"])
        hidden_units: int   = int(params["hidden_units"])
        dropout_rate: float = float(params["dropout_rate"])

        if num_layers < 1:
            raise ValueError(f"GRU: num_layers={num_layers} phải >= 1.")
        if hidden_units < 1:
            raise ValueError(f"GRU: hidden_units={hidden_units} phải >= 1.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"GRU: dropout_rate={dropout_rate} phải trong [0, 1).")

        # Lưu để dùng trong logging
        self.n_features : int = n_features
        self.hidden_units: int = hidden_units

        # ── Recurrent layer ───────────────────────────────────────────────────
        # dropout của nn.GRU chỉ áp dụng giữa các layers khi num_layers > 1.
        # Với num_layers=1, PyTorch bỏ qua dropout (tránh UserWarning).
        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_units,
            num_layers  = num_layers,
            batch_first = True,          # input/output: (batch, seq, feature)
            dropout     = dropout_rate if num_layers > 1 else 0.0,
        )

        # ── Dense head: BN → fc1 → Dropout → ReLU → fc2 ──────────────────────
        # BatchNorm normalize GRU output trước khi vào Dense layers
        self.bn      = nn.BatchNorm1d(hidden_units)
        self.fc1     = nn.Linear(hidden_units, _FC_HIDDEN)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2     = nn.Linear(_FC_HIDDEN, 1)

        # Sigmoid chỉ tồn tại cho classification
        if task == "classification":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None  # type: ignore[assignment]

        logger.debug(
            f"[GRU] task={task} | n_features={n_features} | "
            f"hidden={hidden_units} | layers={num_layers} | "
            f"dropout={dropout_rate:.2f} | "
            f"fc: {hidden_units}→{_FC_HIDDEN}→1 | "
            f"params={self.get_param_count():,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Thứ tự xử lý:
          GRU → last timestep → BN → fc1 → Dropout → ReLU → fc2 → [Sigmoid]

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
                f"GRU.forward: expected 3-D input (batch, seq_len, n_features), "
                f"got {x.ndim}-D tensor shape={tuple(x.shape)}."
            )

        # ── GRU pass ──────────────────────────────────────────────────────────
        # out: (batch, seq_len, hidden_units) — output tại mỗi timestep
        # _:   (num_layers, batch, hidden_units) — hidden state cuối (không dùng)
        out, _ = self.gru(x)

        # Lấy output tại timestep cuối — tổng hợp toàn sequence
        # (batch, seq_len, hidden_units) → (batch, hidden_units)
        out = out[:, -1, :]

        # ── Dense head ────────────────────────────────────────────────────────
        # Theo spec bài báo: BN → fc1 → Dropout → ReLU → fc2
        out = self.bn(out)                           # BatchNorm normalize
        out = F.relu(self.dropout(self.fc1(out)))    # fc1 → Dropout → ReLU
        out = self.fc2(out)                          # Output: (batch, 1)

        # ── Output activation ─────────────────────────────────────────────────
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