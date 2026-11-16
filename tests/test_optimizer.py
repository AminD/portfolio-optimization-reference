"""Portfolio construction."""

import numpy as np
import pytest

from portfolio import optimizer as opt


@pytest.fixture(scope="module")
def mu(true_cov):
    rng = np.random.default_rng(5)
    return rng.uniform(0.0002, 0.0006, true_cov.shape[0])


def test_unconstrained_min_variance_closed_form(true_cov):
    weights = opt.min_variance(true_cov, long_only=False)
    assert weights.sum() == pytest.approx(1.0)
    # First-order condition: Sigma w is proportional to the vector of ones.
    marginal = true_cov @ weights
    assert marginal == pytest.approx(np.full_like(marginal, marginal.mean()), rel=1e-8)


def test_long_only_min_variance(true_cov):
    weights = opt.min_variance(true_cov)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.0
    variance = opt.portfolio_variance(weights, true_cov)
    # Adding a constraint cannot lower the variance; and it beats naive portfolios.
    assert variance >= opt.portfolio_variance(opt.min_variance(true_cov, long_only=False), true_cov) - 1e-12
    assert variance < opt.portfolio_variance(opt.equal_weight(len(weights)), true_cov)
    rng = np.random.default_rng(0)
    for _ in range(200):
        other = rng.dirichlet(np.ones(len(weights)))
        assert variance <= opt.portfolio_variance(other, true_cov) + 1e-12


def test_weight_cap(true_cov):
    weights = opt.min_variance(true_cov, max_weight=0.05)
    assert weights.max() <= 0.05 + 1e-9
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        opt.min_variance(true_cov, max_weight=0.01)  # 40 assets x 1% < 100%


def test_two_asset_min_variance_by_hand():
    """Uncorrelated assets with vols 10% and 20%: weights inversely proportional to variance."""
    covariance = np.diag([0.01, 0.04])
    assert opt.min_variance(covariance) == pytest.approx([0.8, 0.2], abs=1e-6)
    assert opt.risk_parity(covariance) == pytest.approx([2 / 3, 1 / 3], abs=1e-6)


def test_risk_parity_equalises_risk_contributions(true_cov):
    weights = opt.risk_parity(true_cov)
    contributions = opt.risk_contributions(weights, true_cov)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() > 0
    assert contributions == pytest.approx(np.full(len(weights), 1 / len(weights)), rel=1e-4)


def test_max_sharpe_unconstrained_closed_form(mu, true_cov):
    weights = opt.max_sharpe(mu, true_cov, long_only=False)
    expected = np.linalg.solve(true_cov, mu)
    assert weights == pytest.approx(expected / expected.sum())


def test_max_sharpe_beats_other_long_only_portfolios(mu, true_cov):
    def sharpe(w):
        return mu @ w / np.sqrt(w @ true_cov @ w)

    weights = opt.max_sharpe(mu, true_cov)
    assert weights.min() >= 0 and weights.sum() == pytest.approx(1.0)
    assert sharpe(weights) >= sharpe(opt.min_variance(true_cov)) - 1e-9
    assert sharpe(weights) >= sharpe(opt.equal_weight(len(mu)))
    rng = np.random.default_rng(1)
    assert all(sharpe(weights) >= sharpe(rng.dirichlet(np.ones(len(mu)))) for _ in range(200))


def test_efficient_frontier_shape(mu, true_cov):
    vols, rets, weights = opt.efficient_frontier(mu, true_cov, n_points=15)
    assert weights.shape == (15, len(mu))
    assert np.allclose(weights.sum(axis=1), 1.0) and weights.min() > -1e-8
    assert np.all(np.diff(rets) > 0)
    assert np.all(np.diff(vols) > -1e-10)      # more return requires more risk
    assert rets[-1] == pytest.approx(mu.max(), rel=1e-6)  # the top of the frontier is the best asset
    # The frontier starts at the minimum-variance portfolio.
    assert vols[0] ** 2 == pytest.approx(opt.portfolio_variance(opt.min_variance(true_cov), true_cov), rel=1e-6)


def test_mean_variance_limits(mu, true_cov):
    """Infinite risk aversion gives minimum variance; none gives the best asset."""
    cautious = opt.mean_variance(mu, true_cov, risk_aversion=1e6)
    assert cautious == pytest.approx(opt.min_variance(true_cov), abs=1e-3)
    bold = opt.mean_variance(mu, true_cov, risk_aversion=1e-6)
    assert bold[np.argmax(mu)] == pytest.approx(1.0, abs=1e-6)


def test_black_litterman_without_views_returns_the_market(true_cov):
    """Equilibrium returns fed back to the optimiser must give the market portfolio."""
    rng = np.random.default_rng(2)
    market = rng.dirichlet(np.full(true_cov.shape[0], 5.0))
    equilibrium = opt.black_litterman(true_cov, market, risk_aversion=3.0)
    assert opt.mean_variance(equilibrium, true_cov, risk_aversion=3.0, long_only=False) == pytest.approx(market, abs=1e-8)


def test_black_litterman_view_tilts_the_portfolio(true_cov):
    n = true_cov.shape[0]
    market = opt.equal_weight(n)
    equilibrium = opt.black_litterman(true_cov, market)
    view = np.zeros(n)
    view[0], view[1] = 1.0, -1.0                      # "asset 0 will beat asset 1 ..."
    target = view @ equilibrium + 0.02 / 252          # "... by 2% a year more than the market implies"
    posterior = opt.black_litterman(true_cov, market, views_matrix=view, views=[target])
    weights = opt.mean_variance(posterior, true_cov, risk_aversion=2.5, long_only=False)
    assert weights[0] > market[0] and weights[1] < market[1]
    # The posterior moves towards the view but, the view being uncertain, not all the way.
    assert view @ equilibrium < view @ posterior < target
    # Assets not involved in the view keep their market weight.
    assert weights[2:] == pytest.approx(market[2:], abs=1e-8)
