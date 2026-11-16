"""Walk-forward backtest.

At each rebalancing date the strategy sees only past returns, chooses its
weights, and is then evaluated on the following period. Nothing from the
future ever enters an estimate, which is the only honest way to compare
estimators: in-sample, the noisiest estimator always looks best.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import covariance as cov
from . import optimizer
from .data import TRADING_DAYS

# A strategy maps a window of past returns (T, N) to N weights.
Strategy = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample results of several strategies on the same dates."""

    returns: pd.DataFrame          # daily net returns, one column per strategy
    weights: dict[str, pd.DataFrame]   # target weights at each rebalancing date
    turnover: pd.Series            # average one-way turnover per year
    predicted_vol: pd.DataFrame    # annualised volatility forecast at each rebalancing date
    realised_vol: pd.DataFrame     # annualised volatility realised over the following period

    @property
    def risk_underestimation(self) -> pd.Series:
        """Realised over predicted risk, as a ratio of variances pooled over time.

        1 means the estimator forecasts the risk of its own optimal portfolio
        correctly; 2 means the realised variance was twice the forecast.
        """
        return (self.realised_vol**2).mean() / (self.predicted_vol**2).mean()


def min_variance_strategy(estimator: Callable, long_only: bool = True, max_weight: float | None = None) -> Strategy:
    """Minimum-variance portfolio built on a given covariance estimator."""
    def strategy(window):
        return optimizer.min_variance(estimator(window), long_only, max_weight)

    strategy.estimator = estimator  # lets the backtest report this estimator's own risk forecast
    return strategy


def risk_parity_strategy(estimator: Callable) -> Strategy:
    def strategy(window):
        return optimizer.risk_parity(estimator(window))

    strategy.estimator = estimator
    return strategy


def max_sharpe_strategy(estimator: Callable, long_only: bool = True, max_weight: float | None = None) -> Strategy:
    """Maximum-Sharpe portfolio using historical mean returns as expected returns."""
    def strategy(window):
        return optimizer.max_sharpe(window.mean(axis=0), estimator(window), 0.0, long_only, max_weight)

    strategy.estimator = estimator
    return strategy


def equal_weight_strategy() -> Strategy:
    return lambda window: optimizer.equal_weight(window.shape[1])


def run_backtest(
    returns: pd.DataFrame,
    strategies: dict[str, Strategy],
    window: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
    risk_estimator: Callable = cov.sample_covariance,
) -> BacktestResult:
    """Walk-forward backtest of several strategies.

    Parameters
    ----------
    returns : DataFrame (dates x assets)
        Daily simple returns.
    strategies : dict
        Name -> function mapping a window of past returns to weights.
    window : int
        Number of past days each strategy is allowed to see.
    rebalance_every : int
        Days between two rebalancing dates (21 is about one month).
    cost_bps : float
        Proportional transaction cost, in basis points of the traded amount.
    risk_estimator : callable
        Only used for the *forecast* reported in ``predicted_vol`` when a
        strategy does not come with its own covariance (for example 1/N).

    Notes
    -----
    Between two rebalancing dates the positions are not touched, so the
    weights drift with prices. Turnover is measured between the drifted
    weights and the new targets.
    """
    values = returns.to_numpy()
    n_days, n_assets = values.shape
    if n_days <= window + rebalance_every:
        raise ValueError("not enough data for one estimation window and one holding period")
    rebalance_days = np.arange(window, n_days, rebalance_every)

    daily = {name: np.full(n_days, np.nan) for name in strategies}
    targets = {name: [] for name in strategies}
    traded = {name: 0.0 for name in strategies}
    predicted = {name: [] for name in strategies}
    realised = {name: [] for name in strategies}
    current = {name: np.zeros(n_assets) for name in strategies}

    for start in rebalance_days:
        end = min(start + rebalance_every, n_days)
        past, future = values[start - window:start], values[start:end]
        fallback_cov = risk_estimator(past)

        for name, strategy in strategies.items():
            weights = strategy(past)
            trade = np.abs(weights - current[name]).sum()
            traded[name] += trade
            targets[name].append(weights)

            # Buy-and-hold over the period: asset values grow, weights drift.
            growth = np.cumprod(1.0 + future, axis=0)
            portfolio_value = growth @ weights
            period_returns = np.diff(np.concatenate([[1.0], portfolio_value])) / np.concatenate([[1.0], portfolio_value[:-1]])
            period_returns[0] -= trade * cost_bps / 1e4
            daily[name][start:end] = period_returns
            current[name] = weights * growth[-1] / portfolio_value[-1]

            # Forecast made with the strategy's own estimator when it has one.
            estimator = getattr(strategy, "estimator", None)
            used_cov = estimator(past) if estimator is not None else fallback_cov
            predicted[name].append(np.sqrt(TRADING_DAYS * weights @ used_cov @ weights))
            realised[name].append(np.std(period_returns, ddof=1) * np.sqrt(TRADING_DAYS) if len(period_returns) > 1 else np.nan)

    index = returns.index[window:]
    rebalance_index = returns.index[rebalance_days]
    years = (n_days - window) / TRADING_DAYS
    return BacktestResult(
        returns=pd.DataFrame({name: series[window:] for name, series in daily.items()}, index=index),
        weights={name: pd.DataFrame(rows, index=rebalance_index, columns=returns.columns) for name, rows in targets.items()},
        # One-way turnover: half of the total traded amount, excluding the initial purchase.
        turnover=pd.Series({name: 0.5 * (amount - 1.0) / years for name, amount in traded.items()}),
        predicted_vol=pd.DataFrame(predicted, index=rebalance_index),
        realised_vol=pd.DataFrame(realised, index=rebalance_index),
    )
