"""
simulator/core.py

Implementation for simulator functions. Kept separate from package __init__ to
avoid heavy import-time work and to make the package API tidy.
"""
from .trade_generator import load_stock_data, generate_trade_metadata, apply_technical_indicators, re_assign_trade_directions
from .execution_price_simulator import generate_sample_execution_prices, calculate_trade_metrics

import os
import json
from datetime import datetime
import pandas as pd


def generate_dataframe(excel_path, sheet_name=None):
    if sheet_name is None:
        sheet_name = 0
    df_stock = load_stock_data(excel_path, sheet_name)
    df_trades = generate_trade_metadata(df_stock)
    
    # The following will update the trade directions based on technical indicators RSI & GoldenCrossover- 
    # based on heuristics to make it more realistic and less random
    # will create profits. the rationale being that real traders wil look at
    # technical indicators before placing trades.
    df_trades = apply_technical_indicators(df_trades)
    df_trades = re_assign_trade_directions(df_trades, df_stock)
    df_trades.drop(columns=['RSI','MACD','MACD_Signal','GoldenCrossover'], inplace=True)
    
    # calculate volatility if not present
    if 'volatility' not in df_trades.columns:
        df_trades['volatility'] = round(df_trades['EntryPrice'].rolling(window=10).std(), 2)

    entry, exit = generate_sample_execution_prices(
        df_trades['EntryPrice'].tolist(),
        df_trades['ExitPrice'].tolist(),
        df_trades['MarketVolume'].tolist(),
        df_trades['volatility'].tolist(),
        df_trades['HourOfDay'].tolist(),
        df_trades['OrderMonth'].tolist(),
        df_trades['OrderQty'].tolist(),
        df_trades['TradeDirection'].tolist()
    )

    df_trades['AvgEntryExecutionPrice'] = entry
    df_trades['AvgExitExecutionPrice'] = exit

    df_trades = calculate_trade_metrics(df_trades)
    return df_trades


def generate_dataset(excel_path, out_path, sheet_name=None, seed=None):
    """Create dataset and write to Excel (.xlsx) at out_path. Also write metadata JSON alongside.

    Returns the path to the created Excel file.
    """
    df = generate_dataframe(excel_path, sheet_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # write to Excel
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='execution_log')
    except Exception:
        # fallback to CSV if Excel writer not available
        df.to_csv(out_path.replace('.xlsx', '.csv'), index=False)
        out_path = out_path.replace('.xlsx', '.csv')

    # write metadata
    meta = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'source_excel': excel_path,
        'sheet': sheet_name,
        'seed': seed,
        'rows': len(df)
    }
    meta_path = out_path + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return out_path
