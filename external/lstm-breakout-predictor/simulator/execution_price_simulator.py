"""
simulator/execution_price_simulator.py

Simulate execution (entry/exit) prices for synthetic trades and compute
trade-level metrics such as trade values, brokerage, slippage and P&L.
"""

import random
import math
import numpy as np
import pandas as pd
from .spread_utils import get_buy_sell_spread


def generate_random_number(volume, orderqty, volatility, lower, higher):
    """Return a deterministic random float within [lower, higher].

    The function builds a weighted log-scale combination of (volume,
    orderqty, volatility) to seed Python's RNG so that repeated calls with
    the same inputs produce consistent pseudo-random draws.
    """
    try:
        log_values = [math.log(volume), math.log(orderqty), math.log(volatility)]
        weights = [0.2, 0.35, 0.45]
        weighted_sum = sum(w * lv for w, lv in zip(weights, log_values))
        random.seed(weighted_sum)
        return round(random.uniform(lower, higher), 4)
    except ValueError as e:
        print(f"Error generating random number: {e}")
        raise


def generate_sample_execution_prices(open_prices, close_prices, volume, volatility, hour_of_day, order_month, orderqty, trade_direction):
    """Simulate average entry and exit execution prices for each row.

    Returns:
        tuple: (AvgEntryExecutionPrice, AvgExitExecutionPrice)
    """
    AvgEntryExecutionPrice = [None] * len(open_prices)
    AvgExitExecutionPrice = [None] * len(close_prices)

    for i in range(len(open_prices)):
        vol = volatility[i]
        if pd.isna(vol):
            continue

        entry_spread, exit_spread = get_buy_sell_spread(trade_direction[i], hour_of_day[i], order_month[i])

        AvgEntryExecutionPrice[i] = round(
            open_prices[i] * generate_random_number(volume[i], orderqty[i], abs(vol / 100), entry_spread[0], entry_spread[1]), 2
        )
        AvgExitExecutionPrice[i] = round(
            close_prices[i] * generate_random_number(volume[i], orderqty[i], abs(vol / 100), exit_spread[0], exit_spread[1]), 2
        )

    return AvgEntryExecutionPrice, AvgExitExecutionPrice


def calculate_trade_metrics(df):
    """Compute derived monetary metrics from simulated execution prices."""
    df['TotalEntryTradeValue'] = df['AvgEntryExecutionPrice'] * df['ExecutedQty']
    df['TotalExitTradeValue'] = df['AvgExitExecutionPrice'] * df['ExecutedQty']

    df['EntryBrokerage'] = df['TotalEntryTradeValue'] * 0.02
    df['ExitBrokerage'] = df['TotalExitTradeValue'] * 0.02

    df['NetEntryAmount'] = np.where(
        df['TradeDirection'] == "SHORT",
        df['TotalEntryTradeValue'] - df['EntryBrokerage'],
        df['TotalEntryTradeValue'] + df['EntryBrokerage']
    )

    df['NetExitAmount'] = np.where(
        df['TradeDirection'] == "LONG",
        df['TotalExitTradeValue'] + df['ExitBrokerage'],
        df['TotalExitTradeValue'] - df['ExitBrokerage']
    )

    df['TotalTradeSlippageCost'] = np.where(
        df['TradeDirection'] == "LONG",
        ((df['AvgEntryExecutionPrice'] - df['EntryPrice']) + (df['ExitPrice'] - df['AvgExitExecutionPrice'])) * df['OrderQty'],
        ((df['EntryPrice'] - df['AvgEntryExecutionPrice']) + (df['AvgExitExecutionPrice'] - df['ExitPrice'])) * df['OrderQty']
    )

    df['ProfitLoss'] = np.where(
        df['TradeDirection'] == "LONG",
        df['NetExitAmount'] - df['NetEntryAmount'],
        df['NetEntryAmount'] - df['NetExitAmount']
    )

    return df.round(2)
