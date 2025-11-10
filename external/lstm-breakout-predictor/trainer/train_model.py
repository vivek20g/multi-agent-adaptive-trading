"""
model/train_trainer.py

Utilities to prepare sequences and train the trainer.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import os


def prepare_sequences(df, price_features, indicator_features, time_features, sequence_length):
    scaler_price = MinMaxScaler()
    scaler_indicators = MinMaxScaler()
    scaler_time = MinMaxScaler()

    X_price = scaler_price.fit_transform(df[price_features])
    X_indicators = scaler_indicators.fit_transform(df[indicator_features])
    X_time = scaler_time.fit_transform(df[time_features])

    os.makedirs('scalers', exist_ok=True)
    joblib.dump(scaler_price, "scalers/scaler_price.pkl")
    joblib.dump(scaler_indicators, "scalers/scaler_indicators.pkl")
    joblib.dump(scaler_time, "scalers/scaler_time.pkl")

    Xp, Xi, Xt, y = [], [], [], []
    for i in range(len(df) - sequence_length):
        Xp.append(X_price[i:i+sequence_length])
        Xi.append(X_indicators[i:i+sequence_length])
        Xt.append(X_time[i+sequence_length])
        y.append(df['IntradayTradeIndicator'].iloc[i+sequence_length])

    return np.array(Xp), np.array(Xi), np.array(Xt), tf.keras.utils.to_categorical(y, num_classes=3)

def compute_class_weights(y_train):
    y_labels = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
    return dict(enumerate(weights))


def train_model(model, Xp, Xi, Xt, y_train, class_weights):
    def scheduler(epoch, lr):
        return lr if epoch < 10 else lr * 0.5 if epoch < 30 else lr * 0.1

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        LearningRateScheduler(scheduler)
    ]

    history = model.fit(
        [Xp, Xi, Xt],
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.3,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history


