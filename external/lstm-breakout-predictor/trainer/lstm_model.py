"""model/lstm_trainer.py

LSTM model factory for the pipeline.
"""

from keras.layers import LSTM, Dense, Concatenate
from keras import Input, Model


def build_lstm_model(sequence_length, price_dim, indicator_dim, time_dim):
    input_price = Input(shape=(sequence_length, price_dim), name="price_input")
    input_indicators = Input(shape=(sequence_length, indicator_dim), name="indicator_input")
    input_time = Input(shape=(time_dim,), name="time_input")

    lstm_price = LSTM(32, return_sequences=False)(input_price)
    lstm_indicators = LSTM(32, return_sequences=False)(input_indicators)

    merged = Concatenate()([lstm_price, lstm_indicators, input_time])
    dense = Dense(64, activation='relu')(merged)
    output = Dense(3, activation='softmax')(dense)

    return Model(inputs=[input_price, input_indicators, input_time], outputs=output)


