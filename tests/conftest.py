"""Shared fixtures: simulated returns with a known covariance matrix."""

import pytest

from portfolio import simulate_returns


@pytest.fixture(scope="session")
def simulated():
    """(returns, true daily covariance) for 40 assets over 6 years."""
    return simulate_returns(n_assets=40, n_days=1512, n_sectors=4, seed=0)


@pytest.fixture(scope="session")
def returns(simulated):
    return simulated[0]


@pytest.fixture(scope="session")
def true_cov(simulated):
    return simulated[1]
