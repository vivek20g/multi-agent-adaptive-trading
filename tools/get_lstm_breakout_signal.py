import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
import ta
from agents import function_tool
from utils import DataLoader
from keras.models import load_model

@function_tool
def get_lstm_breakout_signal(stockname: str):
    """
    Generate LSTM-based breakout trading signal for a given stock.
    """
    print('get_lstm_breakout_signal stockname:', stockname)

    # Load configuration and data
    data_loader = DataLoader()
    config = data_loader.load_config()
    stockfile_dir = config['DATA']['stockdata_dir']
    excel_file = f"{stockfile_dir}{stockname}.xlsx"
    sheet_name = 'price_history'

    try:
        data = load_and_prepare_data(data_loader, excel_file, sheet_name)
    except Exception as e:
        return {"error": f"Failed to load stock data for {stockname}: {str(e)}"}
    
    if len(data) < 50:
        return insufficient_data_response()
    
    # Add execution time and perform feature engineering
    add_execution_time(data)
    perform_feature_engineering(data)
    # Load model and scalers
    # Load base_path from filepath.ini
    
    model_base_path = config["MODEL_PATHS"]["model_base_path"]

    model_dir = os.path.join(model_base_path, "model")
    scalers_dir = os.path.join(model_base_path, "scalers")
    model, scalers = load_model_and_scalers(model_dir, scalers_dir)
    
    # Prepare input data for the model
    X_price, X_indicators, X_time, latest_row = prepare_model_inputs(data, scalers)
    if X_price is None:
        return {"signal": None, "reason": "Insufficient data"}

    # Generate predictions
    probs, signal, crossover = generate_predictions(model, X_price, X_indicators, X_time, latest_row)
    print(f"LSTM Breakout Signal for {stockname}: {signal} with confidence {max(probs)}")
    # Load classification report
    classification_report = load_classification_report(model_dir)
    # Extract classification metrics
    metrics = extract_classification_metrics(classification_report)
    # Format and return the result
    return format_response(signal, probs, latest_row, crossover, metrics)


# Helper Functions

def load_and_prepare_data(data_loader, excel_file, sheet_name):
    """Load and prepare stock price data."""
    data = data_loader.load_data(excel_file, sheet_name)
    data = data.sort_values(by='Date').tail(60).reset_index(drop=True)
    return data


def insufficient_data_response():
    """Return a response for insufficient data."""
    return {
        "lstm_breakout_signal": {
            "signal": "No Action",
            "confidence": 0.0,
            "RSI": None,
            "MACD_indicator": None,
            "reason": "Insufficient historical data for LSTM analysis"
        }
    }


def add_execution_time(data):
    """Add execution time column with random times."""
    execution_times = [
        f"{np.random.randint(9, 15):02d}:{np.random.randint(0, 59):02d}:{np.random.randint(0, 59):02d}"
        for _ in data['Date']
    ]
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    data['ExecutionTime'] = execution_times


def perform_feature_engineering(data):
    """Perform feature engineering on the data."""
    data['Entry_vs_PrevClose'] = data['Low'] - data['Close'].shift(1)
    data['EntryPriceChange'] = data['Open'].diff(periods=1)
    data["volatility"] = data["Close"].rolling(10).std()
    data['EMA_10'] = ta.trend.EMAIndicator(close=data['Close'], window=10).ema_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(close=data['Close'], window=20).ema_indicator()
    data['MA50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    bb = ta.volatility.BollingerBands(close=data['Close'], window=10, window_dev=2)
    data['BB_Width'] = bb.bollinger_wband()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=9).rsi()
    macd = ta.trend.MACD(data['Open'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['GoldenCrossover'] = np.where(data['MACD'] > data['MACD_Signal'], 1, 0)
    data['Momentum'] = ta.momentum.ROCIndicator(close=data['Open'], window=3).roc()
    data['TR'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(abs(data['High'] - data['Close'].shift(1)), abs(data['Low'] - data['Close'].shift(1)))
    )
    data['ATR'] = data['TR'].rolling(5).mean()
    data["OrderMonth"] = pd.to_datetime(data["Date"]).dt.month
    data["HourOfDay"] = int(f"{np.random.randint(9, 14):02d}")
    data.dropna(inplace=True)


def load_model_and_scalers(model_dir, scalers_dir):
    """Load the LSTM model and scalers."""
    price_pkl_path = f"{scalers_dir}/scaler_price.pkl"
    indicator_pkl_path = f"{scalers_dir}/scaler_indicators.pkl"
    time_pkl_path = f"{scalers_dir}/scaler_time.pkl"
    model_path = f"{model_dir}/daytrading_breakout_model.keras"
    #model_path = f"/Users/vivekgupta/assetnivesh/lstm-breakout-predictor/model/daytrading_breakout_model.keras"
    print('model_path:',model_path)
    # scaler_price = joblib.load(os.path.join(scalers_dir, "scaler_price.pkl"))
    # scaler_indicators = joblib.load(os.path.join(scalers_dir, "scaler_indicators.pkl"))
    # scaler_time = joblib.load(os.path.join(scalers_dir, "scaler_time.pkl"))
    try:
        scaler_price = joblib.load(price_pkl_path)
        scaler_indicators = joblib.load(indicator_pkl_path)
        scaler_time = joblib.load(time_pkl_path)
        #print('scalers loaded successfully')
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model or scalers: {str(e)}")
        custom_objects = {}
        model = load_model(model_path, custom_objects=custom_objects)
        print('model loaded without custom objects')
    return model, (scaler_price, scaler_indicators, scaler_time)


def prepare_model_inputs(data, scalers):
    """Prepare input data for the LSTM model."""
    scaler_price, scaler_indicators, scaler_time = scalers
    price_features = ['Entry_vs_PrevClose', 'EntryPriceChange', 'volatility']
    indicator_features = ['EMA_10', 'EMA_20', 'MA50', 'BB_Width', 'RSI', 'Momentum', 'ATR']
    time_features = ['HourOfDay', 'OrderMonth', 'GoldenCrossover']
    sequence_length = 9

    latest_price = data[price_features].tail(sequence_length)
    latest_indicators = data[indicator_features].tail(sequence_length)
    latest_time = data[time_features].iloc[-1]

    scaled_price = scaler_price.transform(pd.DataFrame(latest_price, columns=price_features))
    scaled_indicators = scaler_indicators.transform(pd.DataFrame(latest_indicators, columns=indicator_features))
    X_time = scaler_time.transform(pd.DataFrame([latest_time], columns=time_features))
    latest_row = data.iloc[-1]

    X_price = np.expand_dims(scaled_price, axis=0)
    X_indicators = np.expand_dims(scaled_indicators, axis=0)

    return X_price, X_indicators, X_time, latest_row


def generate_predictions(model, X_price, X_indicators, X_time, latest_row):
    """Generate predictions using the LSTM model."""
    probs = model.predict([X_price, X_indicators, X_time])[0]  # [BUY, SELL, NONE]
    signal_dict = {0: "No Action", 1: "LONG(BUY)", 2: "SHORT(SELL)"}
    signal = signal_dict[np.argmax(probs)]
    crossover = "Golden Crossover" if latest_row['GoldenCrossover'] == 1 else "Bearish Crossover"
    return probs, signal, crossover


def load_classification_report(model_dir):
    """Load the classification report from a file."""
    classification_report_path = os.path.join(model_dir, "classification_report.txt")
    with open(classification_report_path, "r") as f:
        return json.load(f)


def extract_classification_metrics(classification_report):
    """Extract classification metrics from the report."""
    return {
        "Long Buy": {
            "precision": classification_report.get("Long Buy", {}).get("precision", None),
            "recall": classification_report.get("Long Buy", {}).get("recall", None)
        },
        "Short Sell": {
            "precision": classification_report.get("Short Sell", {}).get("precision", None),
            "recall": classification_report.get("Short Sell", {}).get("recall", None)
        },
        "accuracy": classification_report.get("accuracy", None)
    }


def format_response(signal, probs, latest_row, crossover, metrics):
    """Format the response dictionary."""
    return {
        "lstm_breakout_signal": {
            "signal": signal,
            "confidence": round(max(probs), 3),
            "RSI": round(latest_row["RSI"], 2),
            "MACD Indicator": crossover,
            "model_classification_report": metrics
        }
    }
