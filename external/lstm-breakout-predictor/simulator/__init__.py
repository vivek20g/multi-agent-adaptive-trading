"""
simulator package — public API for data generation.
"""
from .core import generate_dataframe, generate_dataset
from .cli import main as cli_main

__all__ = [
    'generate_dataframe',
    'generate_dataset',
    'cli_main'
]


