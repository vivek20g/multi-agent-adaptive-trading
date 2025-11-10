"""
simulator/trade_generator.py

Generate synthetic trade metadata and compute technical indicators.
"""

import pandas as pd
import numpy as np
import ta


def load_stock_data(filepath, sheet_name):
    df_stock_price = pd.read_excel(filepath, sheet_name=sheet_name)
    df_stock_price = df_stock_price.tail(500).reset_index(drop=True)
    return df_stock_price


def generate_trade_metadata(df_stock_price):
    order_qty = np.random.randint(1000, 2001, size=len(df_stock_price))
    trade_direction = np.random.choice(['LONG', 'SHORT'], size=len(df_stock_price), p=[0.65, 0.35])
    execution_times = [f"{np.random.randint(9, 14):02d}:{np.random.randint(0, 59):02d}:{np.random.randint(0, 59):02d}" for _ in df_stock_price['Date']]
    hour_of_day = [int(t.split(':')[0]) for t in execution_times]
    order_month = pd.to_datetime(df_stock_price['Date']).dt.month

    df_trades = pd.DataFrame({
        'TradeId': range(1, len(df_stock_price)+1),
        'Ticker': ['HDFCBANK'] * len(df_stock_price),
        'ExecutionDate': df_stock_price['Date'],
        'Open': df_stock_price['Open'],
        'High': df_stock_price['High'],
        'Low': df_stock_price['Low'],
        'Close': df_stock_price['Close'],
        'EntryPrice': df_stock_price['Open'],
        'ExitPrice': df_stock_price['Close'],
        'MarketVolume': df_stock_price['Volume'],
        'OrderMonth': order_month,
        'OrderQty': order_qty,
        'TradeDirection': trade_direction,
        'OrderSubType': ['MARKET'] * len(df_stock_price),
        'Exchange': ['NSE'] * len(df_stock_price),
        'Broker': ['ICICI'] * len(df_stock_price),
        'OrderStatus': ['Fulfilled'] * len(df_stock_price),
        'ExecutionTime': execution_times,
        'HourOfDay': hour_of_day,
        'ExecutedQty': order_qty,
        'AvgEntryExecutionPrice': [None] * len(df_stock_price),
        'AvgExitExecutionPrice': [None] * len(df_stock_price),
        'TotalEntryTradeValue': [None] * len(df_stock_price),
        'TotalExitTradeValue': [None] * len(df_stock_price),
        'EntryBrokerage': [None] * len(df_stock_price),
        'ExitBrokerage': [None] * len(df_stock_price),
        'NetEntryAmount': [None] * len(df_stock_price),
        'NetExitAmount': [None] * len(df_stock_price),
        'TotalTradeSlippageCost': [None] * len(df_stock_price),
        'ProfitLoss': [None] * len(df_stock_price),
        'ClientDematId': ['123'] * len(df_stock_price)
    })
    
    return df_trades


def apply_technical_indicators(df_trades):
    df_trades['RSI'] = ta.momentum.RSIIndicator(df_trades['Close'], window=9).rsi()
    macd = ta.trend.MACD(df_trades['Open'], window_slow=26, window_fast=12, window_sign=9)
    df_trades['MACD'] = macd.macd()
    df_trades['MACD_Signal'] = macd.macd_signal()
    df_trades['GoldenCrossover'] = np.where(df_trades['MACD'] > df_trades['MACD_Signal'], 1, 0)
    return df_trades


def re_assign_trade_directions(df_trades, df_stock_price):
    goldencrossover_oversold_indices = df_trades[(df_trades['GoldenCrossover']==1) | (df_trades['RSI']<30)].index
    num_long = int(0.68 * len(goldencrossover_oversold_indices))
    long_indices = np.random.choice(goldencrossover_oversold_indices, size=num_long, replace=False) if len(goldencrossover_oversold_indices)>0 else []
    df_trades.loc[long_indices, 'TradeDirection'] = "LONG"
    df_trades.loc[long_indices, 'ExitPrice'] = df_stock_price.loc[long_indices, 'High']
    df_trades.loc[long_indices, 'EntryPrice'] = df_stock_price.loc[long_indices, 'Low']

    bearcrossover_overbought_indices = df_trades[(df_trades['GoldenCrossover']==0) | (df_trades['RSI']>70)].index
    num_short = int(0.58 * len(bearcrossover_overbought_indices))
    short_indices = np.random.choice(bearcrossover_overbought_indices, size=num_short, replace=False) if len(bearcrossover_overbought_indices)>0 else []
    df_trades.loc[short_indices, 'TradeDirection'] = "SHORT"
    df_trades.loc[short_indices, 'ExitPrice'] = df_stock_price.loc[short_indices, 'Low']
    df_trades.loc[short_indices, 'EntryPrice'] = df_stock_price.loc[short_indices, 'High']

    return df_trades

