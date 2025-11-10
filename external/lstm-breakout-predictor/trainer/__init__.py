"""
trainer package — model training utilities.

Keep the package __init__ lightweight to avoid importing heavy dependencies
at package import time. Consumers should import symbols from submodules, e.g.

from trainer.pipeline import LSTMModelTrainer
from trainer.train_model import prepare_sequences, train_model

"""

__all__ = [
    'pipeline',
    'train_model',
    'lstm_model',
    'evaluate_model',
]


