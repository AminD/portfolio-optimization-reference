"""Price data: loading, returns and sector classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
TRADING_DAYS = 252

# GICS sectors of the tickers downloaded by scripts/download_prices.py.
SECTORS = {
    "Technology": "AAPL MSFT CSCO INTC ADBE CRM ACN AVGO TXN QCOM ORCL IBM ADP",
    "Communication": "GOOGL DIS T VZ CMCSA NFLX",
    "Consumer Discretionary": "AMZN HD NKE MCD LOW SBUX BKNG TGT TJX F",
    "Consumer Staples": "PG KO PEP WMT COST MO PM CL KMB GIS MDLZ",
    "Health Care": "JNJ UNH PFE MRK ABT TMO LLY AMGN MDT GILD CVS BMY DHR ISRG SYK CI",
    "Financials": "JPM V MA BAC GS MS BLK AXP C WFC USB SCHW SPGI PNC MMC CB",
    "Industrials": "HON UNP UPS CAT BA MMM GE DE LMT RTX EMR ITW FDX NSC CSX WM",
    "Energy": "XOM CVX COP SLB EOG",
    "Utilities": "DUK SO NEE D AEP",
    "Materials": "ECL APD SHW NEM",
    "Real Estate": "PLD AMT SPG",
}
SECTOR_OF = {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers.split()}


def load_prices(path) -> pd.DataFrame:
    """Read the CSV written by ``scripts/download_prices.py`` (dates x tickers)."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if prices.empty or prices.shape[1] < 2:
        raise ValueError(f"{path} does not look like a price table (dates in rows, tickers in columns)")
    # A missing price in the middle of a series is a data glitch: carry the last one forward.
    return prices.ffill().dropna(axis=1, how="any")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, P_t / P_{t-1} - 1.

    Simple (not log) returns are used because the return of a portfolio is the
    weighted average of the simple returns of its assets.
    """
    return prices.pct_change().iloc[1:]


def split_benchmark(returns: pd.DataFrame, benchmark: str = BENCHMARK) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate the stock returns from the benchmark column, if present."""
    if benchmark in returns.columns:
        return returns.drop(columns=benchmark), returns[benchmark]
    return returns, None


def sector_of(tickers) -> pd.Series:
    """Sector of each ticker ("Other" when unknown)."""
    return pd.Series([SECTOR_OF.get(t, "Other") for t in tickers], index=list(tickers), name="sector")


def simulate_returns(
    n_assets: int = 60,
    n_days: int = 2520,
    n_sectors: int = 6,
    market_vol: float = 0.16,
    sector_vol: float = 0.10,
    specific_vol: float = 0.20,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns from a factor model with a known covariance matrix.

    r_i = beta_i * market + sector_{s(i)} + specific_i

    Because the true covariance is known, this is the test bench for the
    estimators: their error can be measured exactly, which is impossible with
    real data. Returns ``(returns, true_daily_covariance)``.
    """
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.6, 1.4, n_assets)
    sector = np.arange(n_assets) % n_sectors
    specific = specific_vol * rng.uniform(0.7, 1.3, n_assets)

    membership = (sector[:, None] == np.arange(n_sectors)[None, :]).astype(float)
    annual_cov = (
        market_vol**2 * np.outer(betas, betas)
        + sector_vol**2 * membership @ membership.T
        + np.diag(specific**2)
    )
    true_cov = annual_cov / TRADING_DAYS

    draws = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_days)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    columns = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(draws, index=dates, columns=columns), true_cov
