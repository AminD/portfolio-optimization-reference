"""Performance statistics of a series of daily returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def annualised_return(returns: pd.Series) -> float:
    """Geometric average return per year."""
    growth = float((1.0 + returns).prod())
    return growth ** (TRADING_DAYS / len(returns)) - 1.0


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Mean excess return divided by volatility, annualised."""
    excess = returns - risk_free / TRADING_DAYS
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def drawdowns(returns: pd.Series) -> pd.Series:
    """Loss from the previous peak of the cumulative value, at each date (<= 0)."""
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss (a negative number)."""
    return float(drawdowns(returns).min())


def summary(returns: pd.DataFrame, turnover: pd.Series | None = None, risk_free: float = 0.0) -> pd.DataFrame:
    """One row of statistics per strategy (one column of ``returns`` each)."""
    table = pd.DataFrame(
        {
            "Annual return": returns.apply(annualised_return),
            "Volatility": returns.apply(annualised_volatility),
            "Sharpe ratio": returns.apply(sharpe_ratio, risk_free=risk_free),
            "Max drawdown": returns.apply(max_drawdown),
        }
    )
    if turnover is not None:
        table["Annual turnover"] = turnover.reindex(table.index)
    return table
