"""
app/models/__init__.py
=======================
Model package cho VNSP — export public API.

Models available (implement dần qua các tasks):
  - DNN    (Task 3.2) : Dense Neural Network
  - RNN    (Task 3.3) : Recurrent Neural Network
  - GRU    (Task 3.4) : Gated Recurrent Unit
  - LSTM   (Task 3.5) : Long Short-Term Memory
  - BiLSTM (Task 3.6) : Bidirectional LSTM (main model)
"""

from app.models.base import BaseModel, build_model

__all__ = ["BaseModel", "build_model"]