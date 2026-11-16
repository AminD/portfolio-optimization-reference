"""Covariance estimators."""

import numpy as np
import pytest

from portfolio import covariance as cov


def test_sample_covariance_matches_numpy(returns):
    window = returns.iloc[:300]
    assert cov.sample_covariance(window) == pytest.approx(np.cov(window.to_numpy(), rowvar=False))


def test_ledoit_wolf_is_a_mix_of_sample_and_identity(returns):
    window = returns.iloc[:100].to_numpy()
    delta = cov.ledoit_wolf_shrinkage(window)
    assert 0.0 < delta < 1.0
    centered = window - window.mean(axis=0)
    sample = centered.T @ centered / len(window)
    expected = delta * np.trace(sample) / sample.shape[0] * np.eye(sample.shape[0]) + (1 - delta) * sample
    assert cov.ledoit_wolf(window) == pytest.approx(expected)


def test_ledoit_wolf_matches_scikit_learn(returns):
    sklearn = pytest.importorskip("sklearn.covariance")
    window = returns.iloc[:150].to_numpy()
    reference = sklearn.LedoitWolf().fit(window)
    assert cov.ledoit_wolf_shrinkage(window) == pytest.approx(reference.shrinkage_)
    assert cov.ledoit_wolf(window) == pytest.approx(reference.covariance_)


def test_shrinkage_decreases_with_more_data(returns):
    assert cov.ledoit_wolf_shrinkage(returns.iloc[:60]) > cov.ledoit_wolf_shrinkage(returns.iloc[:1500])


def test_rmt_keeps_sample_volatilities(returns):
    window = returns.iloc[:200]
    cleaned = cov.rmt_clipped(window)
    assert np.sqrt(np.diag(cleaned)) == pytest.approx(window.std(ddof=1).to_numpy())


@pytest.mark.parametrize("name", list(cov.ESTIMATORS))
def test_estimators_are_symmetric_positive_definite(returns, name):
    # Fewer observations than assets: the sample matrix is singular, the cleaned ones are not.
    window = returns.iloc[:30]
    estimate = cov.ESTIMATORS[name](window)
    assert np.allclose(estimate, estimate.T)
    smallest = np.linalg.eigvalsh(estimate).min()
    if name == "Sample":
        assert smallest < 1e-10
    else:
        assert smallest > 0


def test_cleaning_reduces_the_estimation_error(returns, true_cov):
    """With few observations per asset, both cleaned estimators beat the sample matrix."""
    errors = {name: [] for name in cov.ESTIMATORS}
    for start in range(0, 1400, 100):
        window = returns.iloc[start:start + 80]
        for name, estimator in cov.ESTIMATORS.items():
            errors[name].append(cov.frobenius_error(estimator(window), true_cov))
    mean = {name: np.mean(values) for name, values in errors.items()}
    assert mean["Ledoit-Wolf"] < mean["Sample"]
    assert mean["RMT clipping"] < mean["Sample"]


def test_all_estimators_converge_with_many_observations(true_cov):
    from portfolio import simulate_returns

    returns, truth = simulate_returns(n_assets=20, n_days=50_000, n_sectors=4, seed=3)
    for estimator in cov.ESTIMATORS.values():
        assert cov.frobenius_error(estimator(returns), truth) < 0.05


def test_to_correlation(true_cov):
    correlation, vols = cov.to_correlation(true_cov)
    assert np.allclose(np.diag(correlation), 1.0)
    assert correlation * np.outer(vols, vols) == pytest.approx(true_cov)
