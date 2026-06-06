"""
app/models/base.py
===================
Abstract base class và factory function cho tất cả VNSP prediction models.

Task 3.1 trong Phase 3 — Model Architecture.

QUAN TRỌNG — Device:
  LUÔN dùng torch.device("cpu").
  TUYỆT ĐỐI KHÔNG dùng "mps" — có bug với BiLSTM bidirectional trên PyTorch MPS.
  Xem: https://github.com/pytorch/pytorch/issues/94691
  TUYỆT ĐỐI KHÔNG dùng "cuda" — MacBook M1 Pro không có NVIDIA GPU.

Models trong project:
  DNN    (Task 3.2) : Dense Neural Network — baseline
  RNN    (Task 3.3) : Recurrent Neural Network — baseline
  GRU    (Task 3.4) : Gated Recurrent Unit — baseline
  LSTM   (Task 3.5) : Long Short-Term Memory — baseline
  BiLSTM (Task 3.6) : Bidirectional LSTM, dual-branch A1+D1 — main model

Shape convention (forward I/O):
  Input  x : (batch_size, seq_len, n_features)
  Output   : (batch_size, 1)
    Regression     → raw scaled Close(t+1), no activation
    Classification → raw logit (no sigmoid), dùng với BCEWithLogitsLoss

Tham chiếu:
  Li et al., Engineering Applications of AI, 165 (2026) 113390.
  Section 3.4: BiLSTM dual-branch architecture.
  Section 4.3: Training — BCEWithLogitsLoss for classification.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from app.config import DEVICE, MODELS, TASKS

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Device — CPU ONLY ─────────────────────────────────────────────────────────
# TUYỆT ĐỐI KHÔNG đổi thành "mps" hoặc "cuda":
#   - "mps" : PyTorch issue #94691 — BiLSTM bidirectional incorrect results on MPS
#   - "cuda": MacBook M1 Pro không có NVIDIA GPU
DEVICE_OBJ = torch.device(DEVICE)   # DEVICE = "cpu" từ app/config.py

# Fail-fast nếu config bị thay đổi nhầm
if str(DEVICE_OBJ) != "cpu":
    raise RuntimeError(
        f"DEVICE='{DEVICE_OBJ}' không phải 'cpu'. "
        "Kiểm tra DEVICE trong app/config.py — phải là 'cpu'."
    )


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseModel(nn.Module, ABC):
    """
    Abstract base class cho tất cả VNSP prediction models.

    DNN, RNN, GRU, LSTM, BiLSTM đều kế thừa class này để đảm bảo
    interface nhất quán và giảm duplicate code.

    Shape convention (forward):
      Input  x : Tensor (batch_size, seq_len, n_features)
      Output   : Tensor (batch_size, 1)
        - Regression     : raw scaled Close(t+1) — không có activation
        - Classification : raw logit cho UP direction — không có sigmoid
                           Inference: sigmoid(output) > 0.5 → UP

    Lý do dùng raw logit cho classification (không sigmoid):
      BCEWithLogitsLoss = sigmoid + BCE trong một operation, numerically
      stable hơn BCELoss(sigmoid(output)) (tránh log(0) khi logit lớn).
      Khi inference hoặc tính AUC: dùng torch.sigmoid(output).

    Attributes:
        model_name (str): Tên model, ví dụ "BiLSTM".
        task (str):       "regression" hoặc "classification".
    """

    # Lấy từ config để luôn sync với experiment matrix
    _VALID_NAMES: frozenset[str] = frozenset(MODELS)
    _VALID_TASKS: frozenset[str] = frozenset(TASKS)

    def __init__(self, model_name: str, task: str) -> None:
        """
        Khởi tạo base attributes và validate inputs.

        Args:
            model_name: Tên model. Phải nằm trong MODELS (config.py):
                        "DNN", "RNN", "GRU", "LSTM", "BiLSTM".
            task:       "regression" hoặc "classification".

        Raises:
            ValueError: Nếu model_name hoặc task không hợp lệ.
        """
        super().__init__()

        if model_name not in self._VALID_NAMES:
            raise ValueError(
                f"model_name='{model_name}' không hợp lệ. "
                f"Phải là một trong: {sorted(self._VALID_NAMES)}"
            )
        if task not in self._VALID_TASKS:
            raise ValueError(
                f"task='{task}' không hợp lệ. "
                f"Phải là 'regression' hoặc 'classification'."
            )

        self.model_name: str = model_name
        self.task: str = task

    # ── Abstract methods (subclass PHẢI implement) ────────────────────────────

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — PHẢI được override bởi subclass.

        Args:
            x: Input tensor, shape (batch_size, seq_len, n_features).
               batch_size  = số samples trong mini-batch
               seq_len     = window size (sequence_length từ Optuna)
               n_features  = số input features sau feature selection
                             wavelet case: 8 | no-wavelet case: 5

        Returns:
            Tensor shape (batch_size, 1):
              Regression     : scaled Close(t+1), dùng với MSELoss
              Classification : raw logit, dùng với BCEWithLogitsLoss
                               Inference: torch.sigmoid(output) > 0.5 → UP

        Raises:
            NotImplementedError: Nếu subclass không override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() chưa được implement."
        )

    # ── Concrete helper methods ───────────────────────────────────────────────

    def get_param_count(self, trainable_only: bool = True) -> int:
        """
        Đếm số lượng tham số của model.

        Args:
            trainable_only: True (default) → chỉ đếm params có requires_grad.
                            False → đếm tất cả params (kể cả frozen).

        Returns:
            Tổng số scalar parameters (int).

        Example:
            >>> model = build_model("BiLSTM", "regression", 8, params)
            >>> print(f"Params: {model.get_param_count():,}")
            Params: 132,609
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        """
        Trả về string tóm tắt ngắn gọn: tên model, task, số params.

        Dùng để log sau khi tạo model.

        Returns:
            Formatted string, ví dụ:
            "BiLSTM | task=regression | trainable_params=132,609"
        """
        return (
            f"{self.model_name} | "
            f"task={self.task} | "
            f"trainable_params={self.get_param_count():,}"
        )

    def __repr__(self) -> str:
        """Representation ngắn gọn cho debugging."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"task='{self.task}', "
            f"trainable_params={self.get_param_count():,})"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def build_model(
    model_name: str,
    task: str,
    n_features: int,
    params: dict,
) -> BaseModel:
    """
    Factory function — tạo model instance từ tên và hyperparameters.

    Dispatch sang đúng subclass dựa vào model_name. Dùng lazy import
    để tránh circular imports và ImportError nếu model chưa được implement.

    Model-relevant params từ Optuna params dict:
      num_layers   (int)   : số recurrent layers (RNN/GRU/LSTM/BiLSTM)
      hidden_units (int)   : hidden size
      dropout_rate (float) : dropout probability
    Non-model params (learning_rate, batch_size, sequence_length) được
    bỏ qua bởi model constructors — chỉ dùng bởi training loop.

    Expected constructor signature của mỗi model class:
      ModelClass(task: str, n_features: int, params: dict)
    → Model class tự extract params cần thiết từ dict.

    Args:
        model_name: "DNN", "RNN", "GRU", "LSTM", hoặc "BiLSTM".
        task:       "regression" hoặc "classification".
        n_features: Số input features sau wavelet + feature selection.
                    Wavelet case: 8 | No-wavelet case: 5.
        params:     Hyperparameter dict từ Optuna. Keys:
                      num_layers   (int)
                      hidden_units (int)
                      dropout_rate (float)
                      learning_rate, batch_size, sequence_length (ignored)

    Returns:
        BaseModel instance đã được .to(cpu).

    Raises:
        ValueError:  Nếu model_name hoặc task không hợp lệ, hoặc n_features < 1.
        ImportError: Nếu model module chưa tồn tại (task chưa implement).

    Example:
        >>> params = {
        ...     "num_layers": 2, "hidden_units": 128, "dropout_rate": 0.2,
        ...     "learning_rate": 1e-3, "batch_size": 32, "sequence_length": 20
        ... }
        >>> model = build_model("BiLSTM", "regression", n_features=8, params=params)
        >>> print(model.summary())
        BiLSTM | task=regression | trainable_params=132,609
        >>> x = torch.randn(32, 20, 8)       # (batch, seq_len, features)
        >>> out = model(x)                   # → (32, 1)
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    valid_names = sorted(MODELS)
    if model_name not in valid_names:
        raise ValueError(
            f"build_model: model_name='{model_name}' không hợp lệ. "
            f"Phải là một trong: {valid_names}"
        )

    valid_tasks = list(TASKS)
    if task not in valid_tasks:
        raise ValueError(
            f"build_model: task='{task}' không hợp lệ. "
            f"Phải là: {valid_tasks}"
        )

    if n_features < 1:
        raise ValueError(
            f"build_model: n_features={n_features} không hợp lệ (phải >= 1)."
        )

    # ── Validate required hyperparams ─────────────────────────────────────────
    required_keys = {"num_layers", "hidden_units", "dropout_rate"}
    missing_keys = required_keys - set(params.keys())
    if missing_keys:
        raise ValueError(
            f"build_model: params dict thiếu keys: {sorted(missing_keys)}. "
            f"Required: {sorted(required_keys)}"
        )

    # ── Lazy import — mỗi model nằm ở file riêng ──────────────────────────────
    # Lazy import để:
    #   1. Tránh circular imports (model files cũng import base.py)
    #   2. build_model() vẫn import được ngay cả khi một model chưa implement
    #   3. ImportError rõ ràng chỉ xuất hiện khi gọi model cụ thể đó
    _MODEL_MODULE_MAP = {
        "DNN":    ("app.models.dnn",    "DNN"),
        "RNN":    ("app.models.rnn",    "RNN"),
        "GRU":    ("app.models.gru",    "GRU"),
        "LSTM":   ("app.models.lstm",   "LSTM"),
        "BiLSTM": ("app.models.bilstm", "BiLSTM"),
    }

    module_path, class_name = _MODEL_MODULE_MAP[model_name]

    try:
        import importlib
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
    except ImportError as exc:
        raise ImportError(
            f"build_model: Không thể import '{model_name}' từ '{module_path}'. "
            f"Kiểm tra file '{module_path.replace('.', '/')}.py' đã tồn tại chưa.\n"
            f"Original error: {exc}"
        ) from exc
    except AttributeError as exc:
        raise ImportError(
            f"build_model: Module '{module_path}' không có class '{class_name}'.\n"
            f"Original error: {exc}"
        ) from exc

    # ── Instantiate model ─────────────────────────────────────────────────────
    model: BaseModel = ModelClass(
        task=task,
        n_features=n_features,
        params=params,
    )

    # Đưa model về CPU — TUYỆT ĐỐI không dùng MPS hay CUDA
    model = model.to(DEVICE_OBJ)

    logger.info(
        f"[build_model] ✅ {model.summary()} | "
        f"n_features={n_features} | device={DEVICE_OBJ}"
    )

    return model