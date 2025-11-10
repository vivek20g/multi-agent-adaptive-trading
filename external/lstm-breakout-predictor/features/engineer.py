"""features/engineer.py

Thin wrapper around feature-building functions to simplify callers.
"""
from typing import Optional


class FeatureEngineer:
    """Run the standard feature engineering sequence on a DataFrame.

    Usage:
        fe = FeatureEngineer()
        df = fe.run(df)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def run(self, df):
        # Local import to avoid circular imports at package import time
        from .feature_engineering import add_price_dynamics, add_technical_indicators, label_intraday_trade
        df = add_price_dynamics(df)
        df = add_technical_indicators(df)
        df = label_intraday_trade(df)
        return df

    # alias
    transform = run


