"""Walk-forward backtest and performance metrics."""

import numpy as np
import pandas as pd
import pytest

from portfolio import covariance as cov
from portfolio import metrics, run_backtest
from portfolio.backtest import equal_weight_strategy, min_variance_strategy


@pytest.fixture(scope="module")
def result(returns):
    strategies = {
        "1/N": equal_weight_strategy(),
        "Sample": min_variance_strategy(cov.sample_covariance, long_only=False),
        "RMT": min_variance_strategy(cov.rmt_clipped, long_only=False),
    }
    return run_backtest(returns, strategies, window=120, rebalance_every=21)


def test_output_shapes(result, returns):
    assert len(result.returns) == len(returns) - 120
    assert not result.returns.isna().any().any()
    for weights in result.weights.values():
        assert np.allclose(weights.sum(axis=1), 1.0)


def test_no_look_ahead(returns):
    """Changing future returns must not change the weights chosen before them."""
    strategies = {"Sample": min_variance_strategy(cov.sample_covariance, long_only=False)}
    base = run_backtest(returns, strategies, window=120)
    altered = returns.copy()
    altered.iloc[600:] = altered.iloc[600:] * 3.0 + 0.01
    changed = run_backtest(altered, strategies, window=120)
    before = base.weights["Sample"].index < returns.index[600]
    assert np.allclose(base.weights["Sample"][before], changed.weights["Sample"][before])
    assert np.allclose(base.returns.loc[: returns.index[599], "Sample"], changed.returns.loc[: returns.index[599], "Sample"])


def test_equal_weight_matches_a_direct_computation(returns):
    """Daily rebalancing to 1/N is just the average return across assets."""
    result = run_backtest(returns, {"1/N": equal_weight_strategy()}, window=120, rebalance_every=1)
    assert result.returns["1/N"].to_numpy() == pytest.approx(returns.iloc[120:].mean(axis=1).to_numpy())


def test_transaction_costs_reduce_returns(returns):
    strategies = {"Sample": min_variance_strategy(cov.sample_covariance, long_only=False)}
    free = run_backtest(returns, strategies, window=120)
    costly = run_backtest(returns, strategies, window=120, cost_bps=20)
    assert costly.returns["Sample"].sum() < free.returns["Sample"].sum()
    assert costly.turnover["Sample"] == pytest.approx(free.turnover["Sample"])


def test_cleaning_lowers_realised_risk_and_turnover(result):
    """The central claim, on data where it must hold (few observations per asset)."""
    vol = result.returns.apply(metrics.annualised_volatility)
    assert vol["RMT"] < vol["Sample"] < vol["1/N"]
    assert result.turnover["RMT"] < result.turnover["Sample"]
    assert result.turnover["1/N"] < result.turnover["RMT"]


def test_sample_covariance_underestimates_its_own_risk(result):
    ratio = result.risk_underestimation
    assert ratio["Sample"] > 1.5              # realised variance far above the forecast
    assert ratio["RMT"] < ratio["Sample"]
    assert ratio["1/N"] == pytest.approx(1.0, abs=0.15)  # no optimisation, no bias


def test_metrics_on_a_known_series():
    daily = pd.Series([0.01, -0.01] * 126)   # one year
    assert metrics.annualised_return(daily) == pytest.approx((1.01 * 0.99) ** 126 - 1)
    assert metrics.annualised_volatility(daily) == pytest.approx(0.01 * np.sqrt(252), rel=1e-2)
    assert metrics.max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.5)
    assert metrics.drawdowns(pd.Series([0.1, 0.1])).max() == 0.0


def test_too_little_data_raises(returns):
    with pytest.raises(ValueError):
        run_backtest(returns.iloc[:100], {"1/N": equal_weight_strategy()}, window=120)


def test_known_data_error_is_corrected():
    from portfolio import to_returns

    dates = pd.to_datetime(["2016-06-30", "2016-07-01", "2016-07-05", "2016-07-06"])
    prices = pd.DataFrame({"DHR": [41.4, 41.9, 67.6, 67.8], "AAA": [10.0, 10.1, 10.0, 10.2],
                           "BBB": [20.0, 20.0, 20.2, 20.1]}, index=dates)
    raw = to_returns(prices, fix_known_errors=False)
    fixed = to_returns(prices)
    assert raw.loc["2016-07-05", "DHR"] > 0.6
    assert fixed.loc["2016-07-05", "DHR"] == pytest.approx(np.median([10.0 / 10.1 - 1, 20.2 / 20.0 - 1]))
    assert fixed.drop(index=pd.Timestamp("2016-07-05")).equals(raw.drop(index=pd.Timestamp("2016-07-05")))
