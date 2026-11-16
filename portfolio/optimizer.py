"""Portfolio construction: from a covariance matrix (and expected returns) to weights.

Every function returns a NumPy vector of weights that sums to one.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def equal_weight(n_assets: int) -> np.ndarray:
    """1/N: no estimation at all, and a notoriously hard benchmark to beat."""
    return np.full(n_assets, 1.0 / n_assets)


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def _solve_qp(covariance, linear, long_only: bool, max_weight: float | None, start=None) -> np.ndarray:
    """Minimise 0.5 w' Sigma w - linear' w subject to sum(w) = 1 and box bounds."""
    n = covariance.shape[0]
    upper = 1.0 if max_weight is None else max_weight
    if upper * n < 1.0 - 1e-12:
        raise ValueError("max_weight is too small for the weights to sum to one")
    lower = 0.0 if long_only else -upper if max_weight is not None else None
    bounds = [(lower, upper if (long_only or max_weight is not None) else None)] * n

    # Rescale so that the objective is O(1): SLSQP's tolerances are absolute.
    scale = 1.0 / max(np.mean(np.diag(covariance)), np.abs(linear).max())
    result = minimize(
        lambda w: scale * (0.5 * w @ covariance @ w - linear @ w),
        equal_weight(n) if start is None else start,
        jac=lambda w: scale * (covariance @ w - linear),
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimisation failed: {result.message}")
    weights = result.x
    if long_only:
        weights = np.where(weights < 1e-10, 0.0, weights)
    return weights / weights.sum()


def _check_invertible(covariance: np.ndarray) -> None:
    """Closed-form solutions need the inverse covariance, which must exist."""
    if np.linalg.cond(covariance) > 1e12:
        raise ValueError(
            "the covariance matrix is singular (fewer observations than assets?): "
            "use a longer window, a cleaned estimator, or a long-only constraint"
        )


def min_variance(covariance: np.ndarray, long_only: bool = True, max_weight: float | None = None) -> np.ndarray:
    """Portfolio with the smallest variance.

    Without constraints the solution is w = Sigma^-1 1 / (1' Sigma^-1 1). It
    depends on the *inverse* covariance, which amplifies estimation noise:
    the optimiser loads on the directions that look least risky, and those
    are precisely the ones whose risk is most under-estimated.
    """
    n = covariance.shape[0]
    if not long_only and max_weight is None:
        _check_invertible(covariance)
        raw = np.linalg.solve(covariance, np.ones(n))
        return raw / raw.sum()
    return _solve_qp(covariance, np.zeros(n), long_only, max_weight)


def mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
    long_only: bool = True,
    max_weight: float | None = None,
) -> np.ndarray:
    """Markowitz portfolio: maximise mu' w - (risk_aversion / 2) w' Sigma w.

    Without constraints other than full investment, the solution is
    w = Sigma^-1 (mu - gamma 1) / risk_aversion, with gamma set so that the
    weights sum to one.
    """
    mu = np.asarray(expected_returns, float)
    if not long_only and max_weight is None:
        inv_mu = np.linalg.solve(covariance, mu)
        inv_one = np.linalg.solve(covariance, np.ones(len(mu)))
        gamma = (inv_mu.sum() - risk_aversion) / inv_one.sum()
        return (inv_mu - gamma * inv_one) / risk_aversion
    return _solve_qp(risk_aversion * covariance, np.asarray(expected_returns, float), long_only, max_weight)


def max_sharpe(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free: float = 0.0,
    long_only: bool = True,
    max_weight: float | None = None,
) -> np.ndarray:
    """Portfolio with the highest expected excess return per unit of risk (tangency portfolio)."""
    excess = np.asarray(expected_returns, float) - risk_free
    n = len(excess)
    if not long_only and max_weight is None:
        raw = np.linalg.solve(covariance, excess)
        if raw.sum() <= 0:
            raise ValueError("the tangency portfolio is not defined: expected excess returns are too low")
        return raw / raw.sum()
    if excess.max() <= 0:
        return min_variance(covariance, long_only, max_weight)

    def negative_sharpe(w):
        return -(excess @ w) / np.sqrt(w @ covariance @ w)

    upper = 1.0 if max_weight is None else max_weight
    best = None
    # The Sharpe ratio is not concave in w: try a few starting points.
    for start in (equal_weight(n), min_variance(covariance, long_only, max_weight)):
        result = minimize(negative_sharpe, start, bounds=[(0.0 if long_only else -upper, upper)] * n,
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                          method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("maximum-Sharpe optimisation failed")
    weights = np.where(np.abs(best.x) < 1e-10, 0.0, best.x)
    return weights / weights.sum()


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    n_points: int = 40,
    long_only: bool = True,
    max_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-variance portfolios for a range of target returns.

    Returns ``(volatilities, returns, weights)`` with one row of weights per
    point, from the global minimum-variance portfolio up to the highest
    achievable return.
    """
    mu = np.asarray(expected_returns, float)
    n = len(mu)
    start = min_variance(covariance, long_only, max_weight)
    upper = 1.0 if max_weight is None else max_weight
    bounds = [(0.0 if long_only else -upper, upper)] * n

    if long_only:
        # Highest return reachable: fill the best assets up to the weight cap.
        order = np.argsort(mu)[::-1]
        top = np.zeros(n)
        remaining = 1.0
        for i in order:
            top[i] = min(upper, remaining)
            remaining -= top[i]
        highest = mu @ top
    else:
        highest = mu @ start + 2.0 * (mu.max() - mu @ start)

    targets = np.linspace(mu @ start, highest, n_points)
    weights = [start]
    scale = 1.0 / np.mean(np.diag(covariance))
    for target in targets[1:]:
        result = minimize(
            lambda w: scale * 0.5 * w @ covariance @ w, weights[-1], jac=lambda w: scale * covariance @ w,
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
                         {"type": "eq", "fun": lambda w, t=target: mu @ w - t, "jac": lambda w: mu}],
            method="SLSQP", options={"maxiter": 500, "ftol": 1e-13},
        )
        weights.append(result.x if result.success else weights[-1])
    weights = np.array(weights)
    volatilities = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    return volatilities, weights @ mu, weights


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Share of the portfolio variance coming from each asset (sums to one)."""
    marginal = covariance @ weights
    return weights * marginal / (weights @ marginal)


def risk_parity(covariance: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution portfolio: every asset contributes the same share of risk.

    Solved through the convex problem  min 0.5 y' Sigma y - (1/N) sum(log y_i),
    whose optimum satisfies y_i (Sigma y)_i = 1/N for all i (Spinu, 2013).
    Normalising y gives the weights. Uses no expected returns.
    """
    n = covariance.shape[0]
    scale = 1.0 / np.mean(np.diag(covariance))
    scaled = covariance * scale
    budget = 1.0 / n

    result = minimize(
        lambda y: 0.5 * y @ scaled @ y - budget * np.log(y).sum(),
        np.full(n, 1.0 / np.sqrt(n)),
        jac=lambda y: scaled @ y - budget / y,
        bounds=[(1e-10, None)] * n,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return result.x / result.x.sum()


def black_litterman(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    views_matrix: np.ndarray | None = None,
    views: np.ndarray | None = None,
    view_confidence: np.ndarray | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> np.ndarray:
    """Expected returns that blend the market equilibrium with the investor's views.

    Step 1, reverse optimisation: if the market portfolio is optimal, the
    expected excess returns must be  pi = risk_aversion * Sigma * w_market.

    Step 2, Bayesian update with views  P mu = Q + noise(Omega):

        mu = pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P pi)

    With no views the result is the equilibrium ``pi``. Feeding ``pi`` to a
    mean-variance optimiser returns the market portfolio, so views tilt the
    portfolio smoothly instead of producing the extreme weights that raw
    historical means give.

    Parameters
    ----------
    views_matrix : (k, N) array, optional
        One row per view; e.g. [1, -1, 0, ...] for "asset 0 beats asset 1".
    views : (k,) array, optional
        Expected return of each view portfolio.
    view_confidence : (k,) array, optional
        Variance of each view (smaller = more confident). Defaults to the
        variance of the view portfolio under the prior, tau * P Sigma P'.
    """
    equilibrium = risk_aversion * covariance @ np.asarray(market_weights, float)
    if views_matrix is None or views is None:
        return equilibrium

    p = np.atleast_2d(np.asarray(views_matrix, float))
    q = np.atleast_1d(np.asarray(views, float))
    prior = tau * covariance
    omega = np.diag(np.diag(p @ prior @ p.T)) if view_confidence is None else np.diag(np.asarray(view_confidence, float))
    adjustment = prior @ p.T @ np.linalg.solve(p @ prior @ p.T + omega, q - p @ equilibrium)
    return equilibrium + adjustment
