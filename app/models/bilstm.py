"""
app/models/bilstm.py
====================
Bidirectional LSTM (BiLSTM) — main model của dự án VNSP.

Task 3.6 trong Phase 3 — Model Architecture.

BiLSTM là main model, kết hợp hai ý tưởng:
  1. Bidirectional LSTM: capture temporal dependencies từ cả 2 chiều (forward + backward)
  2. Dual-branch architecture: xử lý Approx (trend) và Detail (seasonal) riêng biệt
     → từ Stationary Wavelet Transform (SWT) db4 level-1

Architecture (wavelet mode — dual branch):
  Input (batch, seq_len, n_features)
    ├─ x[:,:,approx_indices] → bilstm_trend    → last timestep → (batch, hidden*2)  [Branch A]
    └─ x[:,:,detail_indices] → bilstm_seasonal → last timestep → (batch, hidden*2)  [Branch D]
  Concatenate [out_a, out_d] → (batch, hidden*4 = fused_size)
  Dense fusion head:
    Linear(fused_size, 256) → BN → Dropout → ReLU   [fc1]
    Linear(256, 64)         → BN → Dropout → ReLU   [fc2]
    Linear(64, 1)                                     [fc_out]
  [Sigmoid nếu classification]

Architecture (no-wavelet mode — single branch):
  Input (batch, seq_len, n_features)
    └─ bilstm_trend → last timestep → (batch, hidden*2)
  Dense fusion head: same (fused_size = hidden*2)

Weight initialization (bài báo section 4.5):
  - weight_ih, weight_hh : Orthogonal init (ổn định gradient ở đầu training)
  - bias                 : Zeros, sau đó set forget-gate = 1.0
    Forget-gate offset trong PyTorch LSTM bias [4H total]:
      [input=0:H | forget=H:2H | cell=2H:3H | output=3H:4H]
    → param.data[n//4 : n//2] = 1.0  (n = 4*H)

approx_indices / detail_indices:
  List[int] — indices vào feature dimension của input tensor.
  Truyền qua params dict: params["approx_indices"], params["detail_indices"].
  Ví dụ sau feature selection giữ 8 features (4 Approx, 4 Detail):
    approx_indices = [0, 1, 2, 3]   (vị trí _Approx features trong tensor)
    detail_indices = [4, 5, 6, 7]   (vị trí _Detail features trong tensor)

Class name: BiLSTM (match factory map trong base.py):
  _MODEL_MODULE_MAP["BiLSTM"] = ("app.models.bilstm", "BiLSTM")

Constructor interface (match build_model factory):
  BiLSTM(task, n_features, params)
  params keys bắt buộc: num_layers, hidden_units, dropout_rate, use_wavelet
  params keys khi use_wavelet=True: approx_indices, detail_indices

Output convention (nhất quán với các baseline models):
  Regression     : raw scaled Close(t+1), không activation → dùng với MSELoss
  Classification : probability ∈ [0,1] sau Sigmoid → dùng với BCELoss

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.4.1 — BiLSTM dual-branch architecture.
  Section 4.5   — Orthogonal init + positive forget-gate bias.
  Schuster & Paliwal (1997) — Bidirectional Recurrent Neural Networks.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import DEVICE
from app.models.base import BaseModel

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# CPU only — KHÔNG dùng "mps" (bug BiLSTM bidirectional) hay "cuda"
_DEVICE_OBJ = torch.device(DEVICE)

# Dense fusion layer sizes (theo bài báo section 3.4.1)
_FC1_HIDDEN: int = 256
_FC2_HIDDEN: int = 64

# BiLSTM có 2 directions (forward + backward)
_NUM_DIRECTIONS: int = 2


# =============================================================================
# BiLSTM MODEL
# =============================================================================

class BiLSTM(BaseModel):
    """
    Bidirectional LSTM với dual-branch architecture.

    Đây là main model của VNSP, kết hợp bidirectional temporal processing
    với wavelet-based feature decomposition:
      - Branch A (Trend)    : BiLSTM xử lý *_Approx features (A1 — low-freq)
      - Branch D (Seasonal) : BiLSTM xử lý *_Detail features (D1 — high-freq)
      - Fuse                : Concatenate last-timestep outputs
      - Dense head          : 2 fully-connected layers → prediction

    Attributes:
        bilstm_trend     (nn.LSTM)        : BiLSTM Branch A — Approx (trend)
        bilstm_seasonal  (nn.LSTM | None) : BiLSTM Branch D — Detail (seasonal),
                                            None khi use_wavelet=False
        fc1              (nn.Linear)      : fused_size → 256
        bn1              (nn.BatchNorm1d) : normalize sau fc1
        dropout1         (nn.Dropout)     : dropout sau bn1
        fc2              (nn.Linear)      : 256 → 64
        bn2              (nn.BatchNorm1d) : normalize sau fc2
        dropout2         (nn.Dropout)     : dropout sau bn2
        fc_out           (nn.Linear)      : 64 → 1
        sigmoid          (nn.Sigmoid | None) : chỉ tồn tại nếu task="classification"
        use_wavelet      (bool)           : True = dual-branch, False = single-branch
        approx_indices   (List[int])      : column indices cho Approx features
        detail_indices   (List[int])      : column indices cho Detail features
        num_layers       (int)            : số LSTM layers (mỗi direction)
        hidden_units     (int)            : LSTM hidden size (mỗi direction)
        n_features       (int)            : tổng số input features
    """

    def __init__(self, task: str, n_features: int, params: dict) -> None:
        """
        Khởi tạo BiLSTM.

        Args:
            task:       "regression" hoặc "classification".
            n_features: Tổng số input features sau feature selection.
                        Wavelet case: 8 | No-wavelet case: 5.
            params:     Hyperparameter dict. Keys bắt buộc:
                          num_layers     (int)       : số BiLSTM layers [1, 2, 3]
                          hidden_units   (int)       : hidden state size [32, 64, 128, 256]
                          dropout_rate   (float)     : dropout probability [0.1–0.5]
                          use_wavelet    (bool)      : True = dual-branch, False = single
                        Thêm khi use_wavelet=True:
                          approx_indices (List[int]) : feature column indices cho Branch A
                          detail_indices (List[int]) : feature column indices cho Branch D

        Raises:
            ValueError: Nếu params thiếu required keys, giá trị không hợp lệ,
                        hoặc indices nằm ngoài range n_features.

        Notes:
            - BiLSTM bidirectional: output size per timestep = hidden_units * 2.
            - Dual-branch fused_size = hidden_units * 4 (2 branches × 2 directions).
            - Single-branch fused_size = hidden_units * 2 (1 branch × 2 directions).
            - dropout trong nn.LSTM chỉ áp dụng giữa layers khi num_layers > 1.
        """
        super().__init__("BiLSTM", task)

        # ── Extract và validate hyperparameters ───────────────────────────────
        _check_required_params(params, {"num_layers", "hidden_units", "dropout_rate",
                                        "use_wavelet"})

        num_layers  : int   = int(params["num_layers"])
        hidden_units: int   = int(params["hidden_units"])
        dropout_rate: float = float(params["dropout_rate"])
        use_wavelet : bool  = bool(params["use_wavelet"])

        if num_layers < 1:
            raise ValueError(f"BiLSTM: num_layers={num_layers} phải >= 1.")
        if hidden_units < 1:
            raise ValueError(f"BiLSTM: hidden_units={hidden_units} phải >= 1.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"BiLSTM: dropout_rate={dropout_rate} phải trong [0, 1)."
            )

        # ── Feature indices — Approx vs Detail ───────────────────────────────
        if use_wavelet:
            # Dual-branch: cần biết vị trí Approx và Detail features trong input
            _check_required_params(params, {"approx_indices", "detail_indices"})
            approx_indices: List[int] = list(params["approx_indices"])
            detail_indices: List[int] = list(params["detail_indices"])

            if len(approx_indices) == 0:
                raise ValueError(
                    "BiLSTM: approx_indices không được rỗng khi use_wavelet=True."
                )
            if len(detail_indices) == 0:
                raise ValueError(
                    "BiLSTM: detail_indices không được rỗng khi use_wavelet=True."
                )

            # Validate tất cả indices nằm trong range [0, n_features)
            all_idx = set(approx_indices) | set(detail_indices)
            invalid = [i for i in all_idx if i < 0 or i >= n_features]
            if invalid:
                raise ValueError(
                    f"BiLSTM: indices {sorted(invalid)} nằm ngoài range "
                    f"[0, n_features={n_features})."
                )

        else:
            # Single-branch: sử dụng toàn bộ n_features (không split)
            approx_indices = list(range(n_features))
            detail_indices = []

        # Lưu attributes cần thiết trong forward()
        self.use_wavelet    : bool      = use_wavelet
        self.approx_indices : List[int] = approx_indices
        self.detail_indices : List[int] = detail_indices
        self.num_layers     : int       = num_layers
        self.hidden_units   : int       = hidden_units
        self.n_features     : int       = n_features

        # ── Branch A — Trend (Approximation features) ────────────────────────
        # Xử lý low-frequency components (A1 từ SWT) để capture long-term trend.
        # bidirectional=True → output size per timestep = hidden_units * 2.
        n_approx: int = len(approx_indices)
        self.bilstm_trend = nn.LSTM(
            input_size    = n_approx,
            hidden_size   = hidden_units,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,       # KEY: forward + backward temporal context
            dropout       = dropout_rate if num_layers > 1 else 0.0,
        )

        # ── Branch D — Seasonal (Detail features) — chỉ khi use_wavelet ──────
        # Xử lý high-frequency components (D1 từ SWT) để capture short-term fluctuations.
        if use_wavelet:
            n_detail: int = len(detail_indices)
            self.bilstm_seasonal: Optional[nn.LSTM] = nn.LSTM(
                input_size    = n_detail,
                hidden_size   = hidden_units,
                num_layers    = num_layers,
                batch_first   = True,
                bidirectional = True,
                dropout       = dropout_rate if num_layers > 1 else 0.0,
            )
            # Dual-branch: cat([out_a, out_d]) → hidden*2 + hidden*2 = hidden*4
            fused_size: int = hidden_units * _NUM_DIRECTIONS * 2
        else:
            self.bilstm_seasonal = None
            # Single-branch: out → hidden*2
            fused_size = hidden_units * _NUM_DIRECTIONS

        # ── Dense fusion head ─────────────────────────────────────────────────
        # 2 Dense layers sau fuse (bài báo section 3.4.1).
        # Thứ tự: Linear → BN → Dropout → ReLU (BN chuẩn hoá output của Linear).
        self.fc1      = nn.Linear(fused_size,   _FC1_HIDDEN)   # fused → 256
        self.bn1      = nn.BatchNorm1d(_FC1_HIDDEN)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.fc2      = nn.Linear(_FC1_HIDDEN, _FC2_HIDDEN)    # 256 → 64
        self.bn2      = nn.BatchNorm1d(_FC2_HIDDEN)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.fc_out   = nn.Linear(_FC2_HIDDEN, 1)              # 64 → 1

        # Sigmoid chỉ tạo cho classification task (nhất quán với LSTM baseline)
        if task == "classification":
            self.sigmoid: Optional[nn.Sigmoid] = nn.Sigmoid()
        else:
            self.sigmoid = None

        # ── Weight initialization (bài báo section 4.5) ───────────────────────
        self._init_weights()

        logger.debug(
            "[BiLSTM] task=%s | n_features=%d | use_wavelet=%s | "
            "n_approx=%d | n_detail=%d | hidden=%d | layers=%d | "
            "dropout=%.2f | fused=%d | params=%s",
            task, n_features, use_wavelet,
            n_approx, len(detail_indices),
            hidden_units, num_layers,
            dropout_rate, fused_size,
            f"{self.get_param_count():,}",
        )

    # ── Weight initialization ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Orthogonal initialization + positive forget-gate bias cho LSTM branches.

        Theo bài báo section 4.5 (Discussion):
          - Orthogonal init: weight_ih, weight_hh của mỗi layer và direction
            → giảm risk vanishing/exploding gradient khi training starts
          - Positive forget-gate bias = 1.0:
            → model "nhớ" nhiều hơn ở đầu training (tránh forgetting early)

        Bias layout trong PyTorch LSTM (kích thước 4*H):
          [0 : H]    input gate
          [H : 2H]   forget gate   ← set = 1.0
          [2H : 3H]  cell gate
          [3H : 4H]  output gate

        Với num_layers > 1 và bidirectional=True, PyTorch tạo params cho
        mỗi layer × direction, ví dụ:
          weight_ih_l0, weight_ih_l0_reverse, weight_ih_l1, weight_ih_l1_reverse, ...
        → named_parameters() tự bao gồm tất cả — không cần xử lý riêng.

        Note: Dense head (fc1, fc2, fc_out) giữ PyTorch default Kaiming init.
        """
        # Áp dụng cho cả 2 branches (bilstm_seasonal chỉ tồn tại khi wavelet)
        lstm_branches: list = [self.bilstm_trend]
        if self.bilstm_seasonal is not None:
            lstm_branches.append(self.bilstm_seasonal)

        for branch in lstm_branches:
            for name, param in branch.named_parameters():
                if "weight_ih" in name:
                    # Input-to-hidden weights: orthogonal init
                    nn.init.orthogonal_(param)
                elif "weight_hh" in name:
                    # Hidden-to-hidden weights: orthogonal init
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    # Bias vector: zeros, sau đó set forget gate = 1.0
                    nn.init.zeros_(param)
                    n: int = param.size(0)          # = 4 * hidden_units
                    # Forget gate nằm ở [H:2H] = [n//4 : n//2]
                    param.data[n // 4 : n // 2].fill_(1.0)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Wavelet mode (dual-branch):
          x[:,:,approx_indices] → bilstm_trend    → out_a[:,-1,:]   (batch, H*2)
          x[:,:,detail_indices] → bilstm_seasonal → out_d[:,-1,:]   (batch, H*2)
          cat([out_a, out_d])                                        (batch, H*4)
          → fc1→BN→Dropout→ReLU → fc2→BN→Dropout→ReLU → fc_out     (batch, 1)
          [→ Sigmoid nếu classification]

        No-wavelet mode (single-branch):
          x → bilstm_trend → out[:,-1,:]                            (batch, H*2)
          → fc1→BN→Dropout→ReLU → fc2→BN→Dropout→ReLU → fc_out     (batch, 1)

        Args:
            x: Tensor shape (batch_size, seq_len, n_features).
               Dtype: float32.

        Returns:
            Tensor shape (batch_size, 1):
              Regression     : raw scaled Close(t+1), dùng với MSELoss
              Classification : probability ∈ [0, 1] sau Sigmoid, dùng với BCELoss

        Raises:
            RuntimeError: Nếu x không phải 3-D tensor.
        """
        # ── Input validation ──────────────────────────────────────────────────
        if x.ndim != 3:
            raise RuntimeError(
                f"BiLSTM.forward: expected 3-D input (batch, seq_len, n_features), "
                f"got {x.ndim}-D tensor shape={tuple(x.shape)}."
            )

        batch_size: int = x.size(0)

        if self.use_wavelet:
            # ── Dual-branch (wavelet mode) ────────────────────────────────────

            # Tách input theo Approx và Detail feature columns
            x_approx = x[:, :, self.approx_indices]    # (batch, seq_len, n_approx)
            x_detail = x[:, :, self.detail_indices]    # (batch, seq_len, n_detail)

            # Branch A — Trend: bilstm_trend xử lý Approx features
            # h0/c0 shape: (num_layers * 2, batch, hidden) — *2 vì bidirectional
            h0_a = torch.zeros(
                self.num_layers * _NUM_DIRECTIONS, batch_size, self.hidden_units,
                device=x.device, dtype=x.dtype,
            )
            c0_a = torch.zeros(
                self.num_layers * _NUM_DIRECTIONS, batch_size, self.hidden_units,
                device=x.device, dtype=x.dtype,
            )
            # out_a: (batch, seq_len, hidden*2) — concat of forward + backward
            out_a, _ = self.bilstm_trend(x_approx, (h0_a, c0_a))
            out_a = out_a[:, -1, :]                    # last timestep → (batch, hidden*2)

            # Branch D — Seasonal: bilstm_seasonal xử lý Detail features
            h0_d = torch.zeros(
                self.num_layers * _NUM_DIRECTIONS, batch_size, self.hidden_units,
                device=x.device, dtype=x.dtype,
            )
            c0_d = torch.zeros(
                self.num_layers * _NUM_DIRECTIONS, batch_size, self.hidden_units,
                device=x.device, dtype=x.dtype,
            )
            # out_d: (batch, seq_len, hidden*2)
            out_d, _ = self.bilstm_seasonal(x_detail, (h0_d, c0_d))
            out_d = out_d[:, -1, :]                    # last timestep → (batch, hidden*2)

            # Fuse: concatenate theo feature dimension → (batch, hidden*4)
            out = torch.cat([out_a, out_d], dim=-1)

        else:
            # ── Single-branch (no-wavelet mode) ──────────────────────────────
            h0 = torch.zeros(
                self.num_layers * _NUM_DIRECTIONS, batch_size, self.hidden_units,
                device=x.device, dtype=x.dtype,
            )
            c0 = torch.zeros(
                self.num_layers * _NUM_DIRECTIONS, batch_size, self.hidden_units,
                device=x.device, dtype=x.dtype,
            )
            out, _ = self.bilstm_trend(x, (h0, c0))
            out = out[:, -1, :]                        # last timestep → (batch, hidden*2)

        # ── Dense fusion head ─────────────────────────────────────────────────
        # fc1 → BN → Dropout → ReLU  :  fused_size → 256
        out = self.dropout1(F.relu(self.bn1(self.fc1(out))))

        # fc2 → BN → Dropout → ReLU  :  256 → 64
        out = self.dropout2(F.relu(self.bn2(self.fc2(out))))

        # fc_out  :  64 → 1
        out = self.fc_out(out)

        # ── Output activation ─────────────────────────────────────────────────
        if self.sigmoid is not None:
            out = self.sigmoid(out)

        return out


# =============================================================================
# MODULE-LEVEL HELPERS (private)
# =============================================================================

def _check_required_params(params: dict, required: set) -> None:
    """
    Kiểm tra params dict có đủ required keys không.

    Args:
        params:   Hyperparameter dict từ Optuna (hoặc test params).
        required: Set tên keys cần có.

    Raises:
        ValueError: Nếu có key bị thiếu, kèm danh sách keys thiếu.
    """
    missing = required - set(params.keys())
    if missing:
        raise ValueError(
            f"params dict thiếu keys: {sorted(missing)}. "
            f"Required: {sorted(required)}"
        )