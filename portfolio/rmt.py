"""Random matrix theory for correlation matrices.

With N assets and T observations, the sample correlation matrix has
N(N-1)/2 free entries estimated from N*T numbers. When N/T is not small,
most of what it contains is noise. Random matrix theory says exactly what
that noise looks like: if the true correlations were all zero, the
eigenvalues of the sample matrix would follow the Marchenko-Pastur law.
Eigenvalues inside that law cannot be told apart from noise; those above it
are genuine structure (the market, sectors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def marchenko_pastur_bounds(q: float, variance: float = 1.0) -> tuple[float, float]:
    """Edges of the Marchenko-Pastur spectrum for q = N / T.

    lambda_+- = variance * (1 +- sqrt(q))^2
    """
    return variance * (1.0 - np.sqrt(q)) ** 2, variance * (1.0 + np.sqrt(q)) ** 2


def marchenko_pastur_pdf(eigenvalues, q: float, variance: float = 1.0) -> np.ndarray:
    """Density of the eigenvalues of a pure-noise correlation matrix (q = N / T <= 1).

    rho(x) = sqrt((lambda_+ - x)(x - lambda_-)) / (2 pi q variance x)
    """
    x = np.asarray(eigenvalues, dtype=float)
    low, high = marchenko_pastur_bounds(q, variance)
    density = np.zeros_like(x)
    inside = (x > low) & (x < high)
    density[inside] = np.sqrt((high - x[inside]) * (x[inside] - low)) / (2.0 * np.pi * q * variance * x[inside])
    return density


@dataclass(frozen=True)
class Spectrum:
    """Eigen-decomposition of a correlation matrix, compared with the noise band."""

    eigenvalues: np.ndarray      # decreasing order
    eigenvectors: np.ndarray     # columns, same order
    q: float                     # N / T
    bulk_variance: float         # variance left once the market mode is removed
    noise_edge: float            # upper edge of the Marchenko-Pastur band

    @property
    def n_signal(self) -> int:
        """Number of eigenvalues above the noise band."""
        return int((self.eigenvalues > self.noise_edge).sum())

    @property
    def explained_by_signal(self) -> float:
        """Share of total variance carried by the eigenvalues above the band."""
        return float(self.eigenvalues[: self.n_signal].sum() / self.eigenvalues.sum())


def analyse_spectrum(correlation: np.ndarray, n_observations: int, adjust_for_market: bool = True) -> Spectrum:
    """Eigenvalues of a correlation matrix and the Marchenko-Pastur noise edge.

    Parameters
    ----------
    adjust_for_market : bool
        The largest eigenvalue (the market) carries a large share lambda_1 / N
        of the total variance. The remaining eigenvalues share only what is
        left, so the noise band is narrower: its variance is
        1 - lambda_1 / N (Laloux et al., 1999). Without the adjustment the
        band is too wide and hides genuine sector modes.
    """
    n_assets = correlation.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    q = n_assets / n_observations
    bulk_variance = 1.0 - eigenvalues[0] / n_assets if adjust_for_market else 1.0
    _, edge = marchenko_pastur_bounds(q, bulk_variance)
    return Spectrum(eigenvalues, eigenvectors, q, float(bulk_variance), float(edge))


def clip_eigenvalues(correlation: np.ndarray, n_observations: int, adjust_for_market: bool = True) -> np.ndarray:
    """Denoise a correlation matrix by eigenvalue clipping.

    Eigenvalues above the noise edge are kept. Those inside the band carry no
    reliable information about *which* directions are low-risk, so they are
    all replaced by their average, which preserves the trace. The diagonal is
    then rescaled to exactly one.
    """
    spectrum = analyse_spectrum(correlation, n_observations, adjust_for_market)
    eigenvalues = spectrum.eigenvalues.copy()
    noise = eigenvalues <= spectrum.noise_edge
    if noise.any():
        eigenvalues[noise] = eigenvalues[noise].mean()

    cleaned = (spectrum.eigenvectors * eigenvalues) @ spectrum.eigenvectors.T
    scale = 1.0 / np.sqrt(np.diag(cleaned))
    cleaned = cleaned * np.outer(scale, scale)
    return 0.5 * (cleaned + cleaned.T)


def inverse_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """Sum of the fourth powers of the components of each eigenvector.

    1 / IPR is the effective number of assets an eigenvector involves: N for a
    vector spread evenly over all assets (the market mode), about N / 3 for a
    random vector, 1 for a vector concentrated on one asset.
    """
    return (eigenvectors**4).sum(axis=0)
