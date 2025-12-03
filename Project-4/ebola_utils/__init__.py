"""
Ebola data utilities module.
Contains functions for loading and analyzing Ebola epidemic data.
"""

from .ebola_analysis import (
    load_ebola_data,
    train_linear_regression,
    plot_linear_regression,
    plot_ebola_data,
    load_all_countries_data
)

__all__ = [
    'load_ebola_data',
    'train_linear_regression',
    'plot_linear_regression',
    'plot_ebola_data',
    'load_all_countries_data'
]

