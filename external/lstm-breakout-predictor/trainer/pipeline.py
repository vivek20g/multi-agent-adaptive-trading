"""model/pipeline.py

Trainer pipeline that encapsulates preprocessing, training and saving.
"""
from typing import Optional, Dict
from trainer.train_model import prepare_sequences, compute_class_weights, train_model
from trainer.lstm_model import build_lstm_model
from trainer.evaluate_model import evaluate_model
import tensorflow as tf


class LSTMModelTrainer:
    """High-level trainer to run preprocessing -> train -> evaluate -> save."""

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        #self.model: Optional[tf.keras.Model] = None
        self.model = None

    def preprocess(self, df, price_features, indicator_features, time_features):
        """Scale and build sequences. Scalers are saved by prepare_sequences."""
        return prepare_sequences(df, price_features, indicator_features, time_features, self.sequence_length)

    def build_model(self, price_dim: int, indicator_dim: int, time_dim: int):
        self.model = build_lstm_model(self.sequence_length, price_dim, indicator_dim, time_dim)
        return self.model

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        if self.model is None:
            raise RuntimeError('Model not built. Call build_model first.')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, Xp, Xi, Xt, y, class_weights: Optional[Dict]=None, **fit_kwargs):
        if self.model is None:
            raise RuntimeError('Model not compiled/built.')
        if class_weights is None:
            class_weights = compute_class_weights(y)
        return train_model(self.model, Xp, Xi, Xt, y, class_weights)

    def evaluate(self, Xp_val, Xi_val, Xt_val, y_val):
        if self.model is None:
            raise RuntimeError('Model not available for evaluation.')
        evaluate_model(self.model, Xp_val, Xi_val, Xt_val, y_val)

    def save(self, path='model/daytrading_breakout_model.keras'):
        if self.model is None:
            raise RuntimeError('No model to save.')
        self.model.save(path)
        return path


