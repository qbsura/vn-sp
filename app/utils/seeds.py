"""
app/utils/seeds.py
==================
Quản lý random seeds để đảm bảo reproducibility hoàn toàn.

QUAN TRỌNG — Device policy:
    LUÔN dùng torch.device("cpu").
    KHÔNG dùng torch.device("mps"): có bug với nn.LSTM bidirectional=True
    trên PyTorch MPS (Apple Silicon).
    Xem: https://github.com/pytorch/pytorch/issues/94691
"""

import os
import random

import numpy as np
import torch

from app.config import SEED, DEVICE

# ── Device singleton ──────────────────────────────────────────────────────────
# Import device từ đây thay vì tự tạo ở mỗi file
device = torch.device(DEVICE)  # always "cpu"


def set_all_seeds(seed: int = SEED) -> None:
    """
    Set seed cho tất cả random number generators.
    Gọi hàm này ở đầu mỗi training script / experiment run.

    Args:
        seed: Integer seed value. Mặc định = 42 (từ config).
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU (không có MPS/CUDA)
    torch.manual_seed(seed)

    # Đảm bảo deterministic ops trên CPU
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Tắt benchmark để tránh non-determinism (không ảnh hưởng CPU)
    torch.backends.cudnn.benchmark = False

    # Env var cho một số ops
    os.environ["PYTHONHASHSEED"] = str(seed)
