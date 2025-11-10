"""
features/feature_engineering.py

Concise feature builders used by the training pipeline.
"""

import numpy as np
import ta


def add_price_dynamics(df):
    """Add simple price/volume delta features."""
    df['Entry_vs_PrevClose'] = df['EntryPrice'] - df['Close'].shift(1)
    df['Entry_vs_PrevOpen'] = df['EntryPrice'] - df['Open'].shift(1)
    df['EntryPriceChange'] = df['EntryPrice'].diff()
    df['ExitPriceChange'] = df['ExitPrice'].diff()
    df['VolumeChange'] = df['MarketVolume'].pct_change()
    return df


def add_technical_indicators(df):
    """Compute a compact set of technical indicators.

    Uses Close for trend indicators (typical practice).
    """
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=9).rsi()
    df['EMA_10'] = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    bb = ta.volatility.BollingerBands(df['Close'], window=10, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['GoldenCrossover'] = (df['MACD'] > df['MACD_Signal']).astype(int)
    df['MA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['Momentum'] = ta.momentum.ROCIndicator(df['Close'], window=3).roc()
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(5).mean()
    return df


def label_intraday_trade(df):
    """Create `IntradayTradeIndicator`: 0 none, 1 long, 2 short."""
    labels = []
    for profit, direction in zip(df.get('ProfitLoss', []), df.get('TradeDirection', [])):
        if profit > 0:
            labels.append(1 if direction == 'LONG' else 2)
        else:
            labels.append(0)
    df['IntradayTradeIndicator'] = labels
    return df.dropna()


