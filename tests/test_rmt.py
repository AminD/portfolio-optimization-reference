"""Random matrix theory: the Marchenko-Pastur law and eigenvalue clipping."""

import numpy as np
import pytest
from scipy.integrate import quad

from portfolio import rmt


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_marchenko_pastur_density_integrates_to_one(q):
    low, high = rmt.marchenko_pastur_bounds(q)
    total = quad(lambda x: rmt.marchenko_pastur_pdf(np.array([x]), q)[0], low, high)[0]
    mean = quad(lambda x: x * rmt.marchenko_pastur_pdf(np.array([x]), q)[0], low, high)[0]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert mean == pytest.approx(1.0, abs=1e-6)  # the trace of a correlation matrix is N


def test_pure_noise_eigenvalues_stay_inside_the_band():
    """Independent returns: (almost) every eigenvalue lies within the Marchenko-Pastur edges."""
    rng = np.random.default_rng(0)
    n_assets, n_obs = 200, 1000
    correlation = np.corrcoef(rng.standard_normal((n_obs, n_assets)), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)
    low, high = rmt.marchenko_pastur_bounds(n_assets / n_obs)
    assert eigenvalues.min() > 0.9 * low
    assert eigenvalues.max() < 1.1 * high
    assert rmt.analyse_spectrum(correlation, n_obs, adjust_for_market=False).n_signal <= 1


def test_empirical_spectrum_follows_the_density():
    rng = np.random.default_rng(1)
    n_assets, n_obs = 400, 1600
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n_obs, n_assets)), rowvar=False))
    median = np.median(eigenvalues)
    low, _ = rmt.marchenko_pastur_bounds(0.25)
    below = quad(lambda x: rmt.marchenko_pastur_pdf(np.array([x]), 0.25)[0], low, median)[0]
    assert below == pytest.approx(0.5, abs=0.03)


def test_factor_structure_is_detected(returns):
    """One market factor and four sectors were simulated: a handful of eigenvalues stand out."""
    window = returns.iloc[:500]
    spectrum = rmt.analyse_spectrum(np.corrcoef(window.to_numpy(), rowvar=False), len(window))
    assert 2 <= spectrum.n_signal <= 6
    assert spectrum.eigenvalues[0] > 5 * spectrum.noise_edge
    # The market eigenvector involves all assets with the same sign.
    market = spectrum.eigenvectors[:, 0]
    assert abs(np.sign(market).sum()) == len(market)
    assert 1 / rmt.inverse_participation_ratio(spectrum.eigenvectors)[0] > 0.8 * len(market)


def test_clipping_returns_a_valid_correlation_matrix(returns):
    window = returns.iloc[:120].to_numpy()
    raw = np.corrcoef(window, rowvar=False)
    cleaned = rmt.clip_eigenvalues(raw, len(window))
    assert np.allclose(np.diag(cleaned), 1.0)
    assert np.allclose(cleaned, cleaned.T)
    assert np.linalg.eigvalsh(cleaned).min() > 0
    # The smallest eigenvalues are raised: the matrix is better conditioned.
    assert np.linalg.cond(cleaned) < np.linalg.cond(raw)
    # Large eigenvalues (the signal) are essentially untouched.
    assert np.linalg.eigvalsh(cleaned)[-1] == pytest.approx(np.linalg.eigvalsh(raw)[-1], rel=0.05)
