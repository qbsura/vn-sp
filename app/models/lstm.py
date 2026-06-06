"""
app/models/lstm.py
===================
Long Short-Term Memory (LSTM) — standard baseline model.

Task 3.5 trong Phase 3 — Model Architecture.

LSTM là recurrent model mạnh với 3 gates (input/forget/output) và
cell state riêng biệt, giúp capture long-term dependencies hiệu quả.
Là baseline trực tiếp trước BiLSTM (main model).

Architecture:
  (batch, seq_len, n_features)
    └─ nn.LSTM(n_features → hidden_units, num_layers, batch_first)
    └─ Lấy timestep cuối: out[:, -1, :] → (batch, hidden_units)
    └─ BatchNorm1d(hidden_units)
    └─ Linear(hidden_units, 256) → Dropout → ReLU      [fc1]
    └─ Linear(256, 64)           → ReLU                 [fc2]
    └─ Linear(64, 1)                                    [fc3]
    └─ Sigmoid nếu task="classification"

Dense head LSTM dùng 3 layers (hidden→256→64→1) so với GRU chỉ 2 (hidden→64→1).
Lý do: LSTM có hidden state phong phú hơn, cần head sâu hơn để extract.

Hidden state initialization:
  h0, c0 = zeros — explicit init (thay vì để PyTorch dùng None mặc định).
  Điều này đảm bảo reproducibility trong mỗi batch khi SEED=42.
  Khởi tạo trên device của input x (device-agnostic pattern).

Output convention:
  Regression     : raw scalar — không activation — dùng với MSELoss
  Classification : probability ∈ [0,1] — sau Sigmoid — dùng với BCELoss

Training note: dùng Adam optimizer + ReduceLROnPlateau + early stopping
  (được set up ở training_service.py).

Class name: LSTM (không phải LSTMModel) để khớp factory map trong base.py:
  _MODEL_MODULE_MAP["LSTM"] = ("app.models.lstm", "LSTM")

Constructor interface (match build_model factory):
  LSTM(task, n_features, params)
  params keys cần có: num_layers, hidden_units, dropout_rate

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.4: Baseline models — LSTM.
  Hochreiter & Schmidhuber (1997): Long Short-Term Memory.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import DEVICE
from app.models.base import BaseModel

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# CPU only — KHÔNG dùng "mps" (bug BiLSTM bidirectional) hay "cuda"
_DEVICE_OBJ = torch.device(DEVICE)

# Kích thước các Dense layers sau LSTM (theo bài báo spec)
_FC1_HIDDEN: int = 256
_FC2_HIDDEN: int = 64


# =============================================================================
# LSTM MODEL
# =============================================================================

class LSTM(BaseModel):
    """
    Long Short-Term Memory — standard baseline temporal model.

    LSTM dùng cell state và 3 gates để kiểm soát luồng thông tin,
    giúp tránh vanishing/exploding gradient trong long sequences.

    Architecture:
      nn.LSTM → last timestep → BN → fc1 → Dropout → ReLU
             → fc2 → ReLU → fc3 → [Sigmoid]

    Attributes:
        lstm         (nn.LSTM)       : LSTM recurrent layer
        bn           (nn.BatchNorm1d): normalize LSTM output
        dropout      (nn.Dropout)    : dropout sau fc1
        fc1          (nn.Linear)     : hidden_units → 256
        fc2          (nn.Linear)     : 256 → 64
        fc3          (nn.Linear)     : 64 → 1
        sigmoid      (nn.Sigmoid)    : chỉ tồn tại nếu task="classification"
        num_layers   (int)           : số LSTM layers — cần cho h0/c0 init
        hidden_units (int)           : LSTM hidden size — cần cho h0/c0 init
        n_features   (int)           : số input features
    """

    def __init__(self, task: str, n_features: int, params: dict) -> None:
        """
        Khởi tạo LSTM.

        Args:
            task:       "regression" hoặc "classification".
            n_features: Số input features (wavelet: 8, no-wavelet: 5).
            params:     Hyperparameter dict từ Optuna. Keys cần có:
                          num_layers   (int)   : số LSTM layers [1, 2, 3]
                          hidden_units (int)   : hidden state size [32, 64, 128, 256]
                          dropout_rate (float) : dropout probability [0.1–0.5]

        Raises:
            ValueError: Nếu params thiếu required keys hoặc giá trị không hợp lệ.

        Notes:
            dropout trong nn.LSTM chỉ áp dụng giữa các layers khi num_layers > 1.
            num_layers và hidden_units được lưu để tạo h0/c0 trong forward().
        """
        super().__init__("LSTM", task)

        # ── Extract và validate hyperparameters ───────────────────────────────
        _check_required_params(params, {"num_layers", "hidden_units", "dropout_rate"})

        num_layers  : int   = int(params["num_layers"])
        hidden_units: int   = int(params["hidden_units"])
        dropout_rate: float = float(params["dropout_rate"])

        if num_layers < 1:
            raise ValueError(f"LSTM: num_layers={num_layers} phải >= 1.")
        if hidden_units < 1:
            raise ValueError(f"LSTM: hidden_units={hidden_units} phải >= 1.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"LSTM: dropout_rate={dropout_rate} phải trong [0, 1).")

        # Lưu để dùng trong forward() cho h0/c0 initialization
        self.n_features  : int = n_features
        self.num_layers  : int = num_layers
        self.hidden_units: int = hidden_units

        # ── LSTM recurrent layer ──────────────────────────────────────────────
        # dropout chỉ áp dụng giữa LSTM layers khi num_layers > 1.
        # Với num_layers=1, PyTorch bỏ qua dropout (tránh UserWarning).
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_units,
            num_layers  = num_layers,
            batch_first = True,          # input/output: (batch, seq, feature)
            dropout     = dropout_rate if num_layers > 1 else 0.0,
        )

        # ── Dense head: 3 layers — hidden_units → 256 → 64 → 1 ──────────────
        # LSTM dùng 3 Dense layers (deeper) vì hidden state phong phú hơn GRU/RNN.
        # BatchNorm normalize LSTM output trước Dense head.
        # Dropout chỉ sau fc1 (layer lớn nhất) — theo spec bài báo.
        self.bn      = nn.BatchNorm1d(hidden_units)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1     = nn.Linear(hidden_units,  _FC1_HIDDEN)   # hidden → 256
        self.fc2     = nn.Linear(_FC1_HIDDEN,   _FC2_HIDDEN)   # 256 → 64
        self.fc3     = nn.Linear(_FC2_HIDDEN,   1)             # 64 → 1

        # Sigmoid chỉ tồn tại cho classification
        if task == "classification":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None  # type: ignore[assignment]

        logger.debug(
            f"[LSTM] task={task} | n_features={n_features} | "
            f"hidden={hidden_units} | layers={num_layers} | "
            f"dropout={dropout_rate:.2f} | "
            f"fc: {hidden_units}→{_FC1_HIDDEN}→{_FC2_HIDDEN}→1 | "
            f"params={self.get_param_count():,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Thứ tự xử lý:
          LSTM(h0,c0) → last timestep → BN → fc1→Dropout→ReLU → fc2→ReLU → fc3 → [Sigmoid]

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
                f"LSTM.forward: expected 3-D input (batch, seq_len, n_features), "
                f"got {x.ndim}-D tensor shape={tuple(x.shape)}."
            )

        batch_size: int = x.size(0)

        # ── Explicit hidden + cell state initialization với zeros ─────────────
        # h0: hidden state  — shape (num_layers, batch, hidden_units)
        # c0: cell state    — shape (num_layers, batch, hidden_units)
        # Khởi tạo trên cùng device với input (device-agnostic, luôn là CPU ở đây)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units,
                         device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units,
                         device=x.device, dtype=x.dtype)

        # ── LSTM pass ─────────────────────────────────────────────────────────
        # out: (batch, seq_len, hidden_units) — output tại mỗi timestep
        # _:   ((num_layers, batch, hidden), (num_layers, batch, hidden)) — (h_n, c_n)
        out, _ = self.lstm(x, (h0, c0))

        # Lấy output tại timestep cuối — tổng hợp toàn sequence
        # (batch, seq_len, hidden_units) → (batch, hidden_units)
        out = out[:, -1, :]

        # ── Dense head ────────────────────────────────────────────────────────
        # BN → fc1 → Dropout → ReLU → fc2 → ReLU → fc3
        out = self.bn(out)                           # BatchNorm normalize
        out = F.relu(self.dropout(self.fc1(out)))    # fc1→Dropout→ReLU : hidden→256
        out = F.relu(self.fc2(out))                  # fc2→ReLU          : 256→64
        out = self.fc3(out)                          # fc3               : 64→1

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