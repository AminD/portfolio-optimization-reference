"""Regenerate the figures and the headline numbers of the README.

Run from the repository root::

    python scripts/make_figures.py

Figures are written to ``docs/figures`` and the result tables to
``docs/results``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio import covariance as cov  # noqa: E402
from portfolio import load_prices, metrics, rmt, run_backtest, sector_of, simulate_returns, split_benchmark  # noqa: E402
from portfolio import to_returns  # noqa: E402
from portfolio.backtest import equal_weight_strategy, min_variance_strategy, risk_parity_strategy  # noqa: E402
from portfolio.plotting import AQUA, BLUE, GRID, INK_SOFT, MUTED, ORANGE, use_style  # noqa: E402

FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "results"
COLORS = {"Sample": ORANGE, "Ledoit-Wolf": AQUA, "RMT clipping": BLUE, "Equal weight": MUTED}
WINDOWS = [126, 189, 252, 378, 504, 756]
FIRST_TEST_DAY = 756  # every window is tested on the same dates, after three years of history


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


def spectrum_figure(returns: pd.DataFrame, window: int = 504) -> rmt.Spectrum:
    """Eigenvalues of the correlation matrix against the Marchenko-Pastur law."""
    recent = returns.iloc[-window:]
    spectrum = rmt.analyse_spectrum(np.corrcoef(recent.to_numpy(), rowvar=False), window)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    limit = 1.7 * spectrum.noise_edge
    bulk = spectrum.eigenvalues[spectrum.eigenvalues <= limit]
    grid = np.linspace(1e-4, limit, 500)
    left.hist(bulk, bins=30, density=True, color=BLUE, alpha=0.75, edgecolor="white", label="Observed eigenvalues")
    left.plot(grid, rmt.marchenko_pastur_pdf(grid, spectrum.q, spectrum.bulk_variance), color=ORANGE,
              label="Pure noise (Marchenko-Pastur)")
    left.axvline(spectrum.noise_edge, color=MUTED, linestyle=":", linewidth=1.2)
    left.set_xlabel(f"Eigenvalue ({(spectrum.eigenvalues > limit).sum()} larger ones not shown)")
    left.set_ylabel("Density")
    left.set_title("Most eigenvalues fall inside the noise band")
    left.legend()

    top = 15
    colors = [BLUE if value > spectrum.noise_edge else GRID for value in spectrum.eigenvalues[:top]]
    right.bar(np.arange(1, top + 1), spectrum.eigenvalues[:top], color=colors, width=0.7)
    right.axhline(spectrum.noise_edge, color=MUTED, linestyle=":", linewidth=1.2)
    right.text(top + 0.4, spectrum.noise_edge * 1.08, "noise edge", ha="right", color=INK_SOFT, fontsize=9)
    right.set_yscale("log")
    right.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    right.minorticks_off()
    right.set_xticks(np.arange(1, top + 1))
    right.set_xlabel("Rank")
    right.set_ylabel("Eigenvalue (log scale)")
    right.set_title(f"{spectrum.n_signal} eigenvalues stand above it")
    right.grid(axis="x", visible=False)
    save(fig, "spectrum.png")
    return spectrum


def eigenvector_figure(returns: pd.DataFrame, window: int = 504) -> None:
    """Components of the first three eigenvectors, with stocks grouped by sector."""
    sectors = sector_of(returns.columns).sort_values(kind="stable")
    recent = returns[sectors.index].iloc[-window:]
    spectrum = rmt.analyse_spectrum(np.corrcoef(recent.to_numpy(), rowvar=False), window)
    names = pd.unique(sectors.to_numpy())
    shade = {name: (BLUE if i % 2 == 0 else "#86b6ef") for i, name in enumerate(names)}

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
    for rank, ax in enumerate(axes):
        vector = spectrum.eigenvectors[:, rank]
        vector = vector * np.sign(vector.sum())
        ax.bar(np.arange(len(vector)), vector, color=[shade[s] for s in sectors], width=0.85)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        share = spectrum.eigenvalues[rank] / len(vector)
        ax.set_title(f"Eigenvector {rank + 1}: eigenvalue {spectrum.eigenvalues[rank]:.1f}, {share:.0%} of total variance",
                     fontsize=10.5)
        ax.set_ylabel("Component")
        ax.grid(axis="x", visible=False)
    position = 0
    for i, name in enumerate(names):
        size = int((sectors == name).sum())
        # Labels alternate between two lines so that neighbours never collide.
        axes[-1].text(position + (size - 1) / 2, -0.10 if i % 2 == 0 else -0.34, name.replace(" ", "\n"),
                      transform=axes[-1].get_xaxis_transform(), ha="center", va="top", fontsize=8, color=INK_SOFT)
        position += size
    axes[-1].set_xticks([])
    axes[-1].set_xlim(-1, len(sectors))
    fig.subplots_adjust(bottom=0.1, hspace=0.3)
    save(fig, "eigenvectors.png")


def window_sweep(returns: pd.DataFrame) -> pd.DataFrame:
    """Long-short minimum variance for every estimator and window, on identical test dates."""
    rows = []
    for window in WINDOWS:
        data = returns.iloc[FIRST_TEST_DAY - window:]
        strategies = {name: min_variance_strategy(estimator, long_only=False) for name, estimator in cov.ESTIMATORS.items()}
        result = run_backtest(data, strategies, window=window)
        for name in strategies:
            rows.append({
                "window": window, "q": returns.shape[1] / window, "estimator": name,
                "realised_vol": metrics.annualised_volatility(result.returns[name]),
                "variance_ratio": result.risk_underestimation[name],
                "turnover": result.turnover[name],
            })
    return pd.DataFrame(rows)


def window_figure(sweep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("realised_vol", "Realised volatility", "Out-of-sample risk", lambda v: 100 * v, "%"),
              ("variance_ratio", "Realised / predicted variance", "How wrong the risk forecast is", lambda v: v, "x"),
              ("turnover", "Turnover per year (times the portfolio)", "Trading required", lambda v: v, "x")]
    for ax, (column, ylabel, title, transform, _) in zip(axes, panels):
        for name, group in sweep.groupby("estimator", sort=False):
            ax.plot(group["window"], transform(group[column]), marker="o", color=COLORS[name], label=name)
        ax.set_xscale("log")
        ax.set_xticks(WINDOWS, [str(w) for w in WINDOWS])
        if column != "realised_vol":
            ax.set_yscale("log")
            ticks = [1, 2, 5, 10, 20, 50] if column == "variance_ratio" else [3, 5, 10, 20, 50]
            ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Estimation window (days)")
        ax.set_ylabel(ylabel + (" (%)" if column == "realised_vol" else ""))
        ax.set_title(title)
    axes[1].axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    axes[0].legend()
    fig.tight_layout()
    save(fig, "window_sweep.png")


def main_backtest(returns: pd.DataFrame, benchmark: pd.Series | None, window: int = 252):
    """Long-short and long-only minimum variance, risk parity and 1/N on the same dates."""
    data = returns.iloc[FIRST_TEST_DAY - window:]
    strategies = {}
    for name, estimator in cov.ESTIMATORS.items():
        strategies[f"Long-short min variance, {name}"] = min_variance_strategy(estimator, long_only=False)
    for name, estimator in cov.ESTIMATORS.items():
        strategies[f"Long-only min variance, {name}"] = min_variance_strategy(estimator, long_only=True)
    strategies["Risk parity, RMT clipping"] = risk_parity_strategy(cov.rmt_clipped)
    strategies["Equal weight"] = equal_weight_strategy()
    result = run_backtest(data, strategies, window=window, cost_bps=5)

    table = metrics.summary(result.returns, result.turnover)
    table["Realised / predicted variance"] = result.risk_underestimation
    if benchmark is not None:
        bench = benchmark.reindex(result.returns.index)
        table.loc["S&P 500 (SPY)"] = [metrics.annualised_return(bench), metrics.annualised_volatility(bench),
                                      metrics.sharpe_ratio(bench), metrics.max_drawdown(bench), np.nan, np.nan]
    return result, table


def backtest_figure(result) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, color in COLORS.items():
        column = name if name == "Equal weight" else f"Long-short min variance, {name}"
        label = name if name == "Equal weight" else f"Min variance, {name}"
        wealth = (1 + result.returns[column]).cumprod()
        left.plot(wealth.index, wealth, color=color, linewidth=1.5, label=label)
        rolling = result.returns[column].rolling(252).std() * np.sqrt(252)
        right.plot(rolling.index, 100 * rolling, color=color, linewidth=1.5, label=label)
    left.set_yscale("log")
    left.set_yticks([1, 2, 3, 4, 6, 8], ["1", "2", "3", "4", "6", "8"])
    left.minorticks_off()
    left.set_ylabel("Value of 1 invested (log scale)")
    left.set_title("Out-of-sample performance, after 5 bp costs")
    left.legend()
    right.set_ylabel("Volatility over the past year (%)")
    right.set_title("Realised volatility")
    save(fig, "backtest.png")


def simulation_figure() -> pd.DataFrame:
    """Estimation error against the true covariance, as a function of N / T."""
    n_assets = 100
    returns, truth = simulate_returns(n_assets=n_assets, n_days=30_000, n_sectors=8, seed=0)
    values = returns.to_numpy()
    rows = []
    for window in [110, 150, 200, 300, 500, 1000, 2000]:
        for name, estimator in cov.ESTIMATORS.items():
            errors, true_vols = [], []
            for start in range(0, 30_000 - window, max(window, 1500)):
                estimate = estimator(values[start:start + window])
                errors.append(cov.frobenius_error(estimate, truth))
                weights = np.linalg.solve(estimate, np.ones(n_assets))
                weights /= weights.sum()
                true_vols.append(np.sqrt(252 * weights @ truth @ weights))
            rows.append({"window": window, "q": n_assets / window, "estimator": name,
                         "error": np.mean(errors), "true_vol": np.mean(true_vols)})
    table = pd.DataFrame(rows)
    best = np.linalg.solve(truth, np.ones(n_assets))
    best /= best.sum()
    optimum = np.sqrt(252 * best @ truth @ best)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, group in table.groupby("estimator", sort=False):
        left.plot(group["q"], 100 * group["error"], marker="o", color=COLORS[name], label=name)
        right.plot(group["q"], 100 * group["true_vol"], marker="o", color=COLORS[name], label=name)
    right.axhline(100 * optimum, color=MUTED, linestyle="--", linewidth=1.2)
    right.text(0.9, 100 * optimum * 1.01, "true optimum", color=INK_SOFT, fontsize=9, va="bottom", ha="right")
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 0.9], ["0.05", "0.1", "0.2", "0.5", "0.9"])
        ax.minorticks_off()
        ax.set_xlabel("Assets per observation, N / T")
    left.set_ylabel("Relative error of the covariance matrix (%)")
    left.set_title("Simulated data: distance to the true matrix")
    right.set_ylabel("True volatility of the min-variance portfolio (%)")
    right.set_title("Simulated data: risk actually taken")
    left.legend()
    save(fig, "simulation.png")
    return table


def main() -> None:
    use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200

    simulation = simulation_figure()
    simulation.to_csv(RESULTS / "simulation.csv", index=False)
    print(simulation.pivot(index="q", columns="estimator", values=["error", "true_vol"]).round(3).to_string(), "\n")

    data = ROOT / "data" / "prices.csv"
    if not data.exists():
        print("data/prices.csv not found: market figures skipped")
        return
    returns, benchmark = split_benchmark(to_returns(load_prices(data)))
    print(f"{returns.shape[1]} stocks, {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}; "
          f"test period starts {returns.index[FIRST_TEST_DAY]:%Y-%m-%d}\n")

    spectrum = spectrum_figure(returns)
    print(f"spectrum: {spectrum.n_signal} above edge {spectrum.noise_edge:.2f}; top {spectrum.eigenvalues[:6].round(2)}; "
          f"signal share {spectrum.explained_by_signal:.2f}\n")
    eigenvector_figure(returns)

    sweep = window_sweep(returns)
    sweep.to_csv(RESULTS / "window_sweep.csv", index=False)
    window_figure(sweep)
    print(sweep.pivot(index="window", columns="estimator", values=["realised_vol", "variance_ratio", "turnover"]).round(3).to_string(), "\n")

    result, table = main_backtest(returns, benchmark)
    table.to_csv(RESULTS / "backtest_summary.csv")
    backtest_figure(result)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
