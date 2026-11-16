"""Portfolio optimisation with noise-cleaned covariance matrices.

Quick start
-----------
>>> from portfolio import load_prices, to_returns, split_benchmark, covariance, optimizer
>>> returns, benchmark = split_benchmark(to_returns(load_prices("data/prices.csv")))  # doctest: +SKIP
>>> sigma = covariance.rmt_clipped(returns.iloc[-252:])                               # doctest: +SKIP
>>> weights = optimizer.min_variance(sigma)                                           # doctest: +SKIP
"""

from . import backtest, covariance, metrics, optimizer, rmt
from .backtest import BacktestResult, run_backtest
from .data import load_prices, sector_of, simulate_returns, split_benchmark, to_returns

__version__ = "1.0.0"

__all__ = [
    "BacktestResult", "backtest", "covariance", "load_prices", "metrics", "optimizer", "rmt",
    "run_backtest", "sector_of", "simulate_returns", "split_benchmark", "to_returns",
]
