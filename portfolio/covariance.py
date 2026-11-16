"""Covariance estimators.

All estimators take a (T, N) array or DataFrame of returns and return an
(N, N) NumPy covariance matrix in the same units (daily returns in, daily
covariance out).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .rmt import clip_eigenvalues


def _centered(returns) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    return x - x.mean(axis=0)


def sample_covariance(returns) -> np.ndarray:
    """The usual unbiased estimator. Optimal when T is much larger than N, noisy otherwise."""
    x = _centered(returns)
    return x.T @ x / (x.shape[0] - 1)


def ledoit_wolf_shrinkage(returns) -> float:
    """Optimal shrinkage intensity of Ledoit and Wolf (2004), between 0 and 1."""
    x = _centered(returns)
    n_obs, n_assets = x.shape
    sample = x.T @ x / n_obs
    mean_variance = np.trace(sample) / n_assets

    # d^2: how far the sample matrix is from the target (dispersion of the eigenvalues).
    distance = np.sum((sample - mean_variance * np.eye(n_assets)) ** 2) / n_assets
    # b^2: how noisy the sample matrix is, estimated from the data.
    # sum_t ||x_t x_t' - S||^2 = sum_t ||x_t||^4 - T ||S||^2
    noise = (np.sum(np.sum(x**2, axis=1) ** 2) - n_obs * np.sum(sample**2)) / (n_obs**2 * n_assets)
    return float(min(noise, distance) / distance)


def ledoit_wolf(returns) -> np.ndarray:
    """Linear shrinkage towards a multiple of the identity (Ledoit and Wolf, 2004).

    Sigma = delta * mean_variance * I + (1 - delta) * S

    The sample matrix S is unbiased but noisy; the identity target is biased
    but has no estimation error. The intensity delta minimises the expected
    squared error of the mix and is computed from the data. Shrinking pulls
    the extreme eigenvalues, which are the most over- and under-estimated,
    towards their mean.
    """
    x = _centered(returns)
    n_obs, n_assets = x.shape
    sample = x.T @ x / n_obs
    delta = ledoit_wolf_shrinkage(returns)
    return delta * np.trace(sample) / n_assets * np.eye(n_assets) + (1.0 - delta) * sample


def rmt_clipped(returns, adjust_for_market: bool = True) -> np.ndarray:
    """Sample volatilities with a correlation matrix denoised by eigenvalue clipping.

    Volatilities are estimated well even from short samples; correlations are
    not. So the volatilities are kept and only the correlation matrix is
    cleaned (see :func:`portfolio.rmt.clip_eigenvalues`).
    """
    x = _centered(returns)
    n_obs = x.shape[0]
    vols = x.std(axis=0, ddof=1)
    correlation = np.corrcoef(x, rowvar=False)
    cleaned = clip_eigenvalues(correlation, n_obs, adjust_for_market)
    return cleaned * np.outer(vols, vols)


def diagonal(returns) -> np.ndarray:
    """Sample variances with all correlations set to zero: the "no information" extreme."""
    return np.diag(_centered(returns).var(axis=0, ddof=1))


ESTIMATORS: dict[str, Callable] = {
    "Sample": sample_covariance,
    "Ledoit-Wolf": ledoit_wolf,
    "RMT clipping": rmt_clipped,
}


def to_correlation(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a covariance matrix into (correlation matrix, volatilities)."""
    vols = np.sqrt(np.diag(covariance))
    return covariance / np.outer(vols, vols), vols


def frobenius_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Relative error ||estimate - truth||_F / ||truth||_F."""
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
