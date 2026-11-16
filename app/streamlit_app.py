"""Interactive dashboard: from noisy correlations to out-of-sample portfolios.

Run from the repository root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio import covariance as cov  # noqa: E402
from portfolio import load_prices, metrics, optimizer, rmt, run_backtest, sector_of, simulate_returns  # noqa: E402
from portfolio import split_benchmark, to_returns  # noqa: E402
from portfolio.backtest import (equal_weight_strategy, max_sharpe_strategy, min_variance_strategy,  # noqa: E402
                                risk_parity_strategy)
from portfolio.data import TRADING_DAYS  # noqa: E402
from portfolio.plotting import AQUA, BLUE, GRAY, GRID, INK, INK_SOFT, MAGENTA, ORANGE, YELLOW, plotly_layout  # noqa: E402

st.set_page_config(page_title="Portfolio Optimization", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
      h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem;}
      .subtitle {color: #52514e; font-size: 1.02rem; margin-bottom: 1.4rem;}
      .kpi-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 0.4rem 0 1.3rem 0;}
      .kpi {background: #ffffff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 14px 16px;}
      .kpi .label {color: #52514e; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;}
      .kpi .value {color: #0b0b0b; font-size: 1.65rem; font-weight: 650; line-height: 1.25; margin-top: 2px;}
      .kpi .note {color: #898781; font-size: 0.82rem; margin-top: 2px;}
      .kpi.accent {border-left: 4px solid #2a78d6;}
      .insight {background: #eef4fc; border-radius: 10px; padding: 12px 16px; color: #0b0b0b; margin: 0.2rem 0 1.2rem 0;}
      div[data-testid="stSidebarContent"] h2 {font-size: 1.0rem; margin-top: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_row(cards: list[tuple[str, str, str]], accent_first: bool = True) -> None:
    """A row of cards: (label, value, note)."""
    html = ""
    for i, (label, value, note) in enumerate(cards):
        css = "kpi accent" if accent_first and i == 0 else "kpi"
        html += f'<div class="{css}"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'
    st.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

DATA_FILE = ROOT / "data" / "prices.csv"


@st.cache_data(show_spinner=False)
def load_data(uploaded) -> tuple[pd.DataFrame, pd.Series | None, str]:
    if uploaded is not None:
        returns, benchmark = split_benchmark(to_returns(load_prices(uploaded)))
        return returns, benchmark, "uploaded file"
    if DATA_FILE.exists():
        returns, benchmark = split_benchmark(to_returns(load_prices(DATA_FILE)))
        return returns, benchmark, "US large caps, Yahoo Finance"
    simulated, _ = simulate_returns(n_assets=60, n_days=2520, n_sectors=6, seed=0)
    return simulated, None, "simulated factor model (no price file found)"


with st.sidebar:
    st.markdown("### 📊 Portfolio Optimization")
    page = st.radio("Section", ["Overview", "Correlations and noise", "Portfolio builder", "Backtest"],
                    label_visibility="collapsed")

    st.header("Universe")
    uploaded = st.file_uploader("Price file (optional)", type="csv",
                                help="CSV written by scripts/download_prices.py: dates in rows, tickers in columns.")
    all_returns, benchmark, source = load_data(uploaded)
    sectors = sector_of(all_returns.columns)
    available = sorted(sectors.unique())
    chosen_sectors = st.multiselect("Sectors", available, default=available)
    start, end = st.select_slider(
        "Period", options=list(all_returns.index.year.unique()),
        value=(int(all_returns.index.year.min()), int(all_returns.index.year.max())),
    )

    st.header("Estimation")
    window = st.select_slider("Estimation window (trading days)", [63, 126, 189, 252, 378, 504, 756], value=252,
                              help="Number of past days used to estimate the covariance matrix.")

tickers = [t for t in all_returns.columns if sectors[t] in chosen_sectors]
returns = all_returns.loc[str(start):str(end), tickers]
if benchmark is not None:
    benchmark = benchmark.loc[str(start):str(end)]
sectors = sectors[tickers]

if len(tickers) < 5 or len(returns) < window + 42:
    st.title("Portfolio Optimization")
    st.warning("Select at least 5 assets and a period longer than the estimation window plus two months.")
    st.stop()

n_assets = len(tickers)
q_ratio = n_assets / window
# Order assets by sector everywhere, so that block structure is visible.
order = sectors.sort_values(kind="stable").index.tolist()
recent = returns[order].iloc[-window:]

ESTIMATOR_COLORS = {"Sample": ORANGE, "Ledoit-Wolf": AQUA, "RMT clipping": BLUE}

# --------------------------------------------------------------------------
# Page 1: overview
# --------------------------------------------------------------------------

if page == "Overview":
    st.title("Portfolio Optimization")
    st.markdown('<div class="subtitle">How much of a correlation matrix is real, and what it costs to ignore the question.</div>',
                unsafe_allow_html=True)

    equal = returns.mean(axis=1)
    kpi_row([
        ("Assets", f"{n_assets}", f"{len(set(sectors))} sectors · {source}"),
        ("Period", f"{returns.index[0]:%Y} – {returns.index[-1]:%Y}", f"{len(returns):,} trading days"),
        ("Equal-weight return", f"{metrics.annualised_return(equal):.1%}", "per year, rebalanced daily"),
        ("Equal-weight volatility", f"{metrics.annualised_volatility(equal):.1%}", "per year"),
        ("Assets per observation", f"{q_ratio:.2f}", f"N / T with a {window}-day window"),
    ])
    insight("A covariance matrix of <b>N</b> assets has N(N+1)/2 entries. With "
            f"{n_assets} assets that is <b>{n_assets * (n_assets + 1) // 2:,}</b> numbers, estimated here from "
            f"<b>{n_assets * window:,}</b> returns. The sections on the left show how much of the result is noise, "
            "how to remove it, and whether that helps out of sample.")

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        wealth = (1 + equal).cumprod()
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth, name="Equal-weight portfolio", line=dict(color=BLUE, width=2)))
        if benchmark is not None:
            bench = (1 + benchmark).cumprod()
            fig.add_trace(go.Scatter(x=bench.index, y=bench, name="S&P 500 (SPY)", line=dict(color=GRAY, width=2)))
        fig.update_layout(**plotly_layout("Growth of 1 invested at the start", yaxis_type="log",
                                          yaxis_title="Value (log scale)"))
        show(fig)
    with right:
        stats = pd.DataFrame({"ret": returns.apply(metrics.annualised_return),
                              "vol": returns.apply(metrics.annualised_volatility), "sector": sectors})
        highlight = st.selectbox("Highlight a sector", ["None"] + sorted(set(sectors)))
        fig = go.Figure()
        for is_highlighted, color, name in [(False, BLUE, "Stocks"), (True, ORANGE, highlight)]:
            part = stats[(stats["sector"] == highlight) == is_highlighted]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part["vol"], y=part["ret"], mode="markers", name=name, text=part.index, customdata=part["sector"],
                marker=dict(color=color, size=9, line=dict(color="white", width=1), opacity=0.9),
                hovertemplate="<b>%{text}</b> · %{customdata}<br>Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>"))
        fig.update_layout(**plotly_layout("Risk and return of each stock", height=350, hovermode="closest",
                                          xaxis_title="Annual volatility", yaxis_title="Annual return",
                                          xaxis_tickformat=".0%", yaxis_tickformat=".0%"))
        show(fig)

    counts = sectors.value_counts().sort_values()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x} stocks<extra></extra>"))
    fig.update_layout(**plotly_layout("Stocks per sector", height=max(220, 30 * len(counts) + 70), hovermode="closest",
                                      xaxis_title="Number of stocks"))
    show(fig)

# --------------------------------------------------------------------------
# Page 2: correlations and random matrix theory
# --------------------------------------------------------------------------

elif page == "Correlations and noise":
    st.title("Correlations and noise")
    st.markdown('<div class="subtitle">Random matrix theory separates the eigenvalues that carry information '
                'from those that are indistinguishable from noise.</div>', unsafe_allow_html=True)

    correlation = np.corrcoef(recent.to_numpy(), rowvar=False)
    spectrum = rmt.analyse_spectrum(correlation, window)
    cleaned = rmt.clip_eigenvalues(correlation, window)
    off_diagonal = correlation[np.triu_indices(n_assets, 1)]

    kpi_row([
        ("Eigenvalues above the noise", f"{spectrum.n_signal} of {n_assets}",
         f"carry {spectrum.explained_by_signal:.0%} of the variance"),
        ("Market mode", f"{spectrum.eigenvalues[0] / n_assets:.0%}", "of total variance, largest eigenvalue"),
        ("Noise band upper edge", f"{spectrum.noise_edge:.2f}", f"Marchenko-Pastur, N/T = {q_ratio:.2f}"),
        ("Average correlation", f"{off_diagonal.mean():.2f}", f"last {window} days"),
    ])

    left, right = st.columns(2)
    with left:
        bulk = spectrum.eigenvalues[spectrum.eigenvalues <= 1.6 * spectrum.noise_edge]
        grid = np.linspace(1e-4, 1.6 * spectrum.noise_edge, 400)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=bulk, histnorm="probability density", nbinsx=40, name="Observed eigenvalues",
                                   marker=dict(color=BLUE, line=dict(color="white", width=1)), opacity=0.85))
        fig.add_trace(go.Scatter(x=grid, y=rmt.marchenko_pastur_pdf(grid, spectrum.q, spectrum.bulk_variance),
                                 name="Pure noise (Marchenko-Pastur)", line=dict(color=ORANGE, width=2.5)))
        fig.add_vline(x=spectrum.noise_edge, line=dict(color=GRAY, dash="dot"))
        fig.update_layout(**plotly_layout("Most eigenvalues fall inside the noise band", hovermode="closest",
                                          xaxis_title="Eigenvalue (the few largest are off the chart)",
                                          yaxis_title="Density", bargap=0.02))
        show(fig)
    with right:
        top = min(15, n_assets)
        colors = [BLUE if value > spectrum.noise_edge else GRID for value in spectrum.eigenvalues[:top]]
        fig = go.Figure(go.Bar(x=[f"{i + 1}" for i in range(top)], y=spectrum.eigenvalues[:top], marker_color=colors,
                               hovertemplate="Eigenvalue %{x}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=spectrum.noise_edge, line=dict(color=GRAY, dash="dot"))
        # On a log axis, annotation coordinates are given in log10 units.
        fig.add_annotation(x=top - 1, y=np.log10(spectrum.noise_edge), text="noise edge", showarrow=False,
                           yshift=10, xanchor="right", font=dict(size=11, color=INK_SOFT))
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50) if t <= 1.5 * spectrum.eigenvalues[0]]
        fig.update_layout(**plotly_layout("The largest eigenvalues stand far above it", hovermode="closest",
                                          xaxis_title="Rank", yaxis_title="Eigenvalue (log scale)",
                                          yaxis=dict(type="log", tickmode="array", tickvals=ticks,
                                                     ticktext=[f"{t:g}" for t in ticks], gridcolor=GRID)))
        show(fig)

    st.subheader("What the significant eigenvectors mean")
    mode = st.slider("Eigenvector", 1, max(2, min(spectrum.n_signal + 2, n_assets)), 1,
                     help="1 is the largest eigenvalue. The first is the market; the next ones oppose sectors.")
    vector = spectrum.eigenvectors[:, mode - 1]
    vector = vector * np.sign(vector.sum() or 1.0)
    sector_list = sectors[order].to_numpy()
    shade = {name: (BLUE if i % 2 == 0 else "#86b6ef") for i, name in enumerate(pd.unique(sector_list))}
    fig = go.Figure(go.Bar(x=order, y=vector, marker_color=[shade[s] for s in sector_list], customdata=sector_list,
                           hovertemplate="<b>%{x}</b> · %{customdata}<br>Component %{y:.3f}<extra></extra>"))
    position = 0
    for i, name in enumerate(pd.unique(sector_list)):
        size = int((sector_list == name).sum())
        # Sector names alternate between two lines so that neighbours never overlap.
        fig.add_annotation(x=position + (size - 1) / 2, y=1.05 if i % 2 == 0 else 1.14, yref="paper", text=name,
                           showarrow=False, font=dict(size=10, color=INK_SOFT))
        position += size
    status = "above the noise edge" if spectrum.eigenvalues[mode - 1] > spectrum.noise_edge else "inside the noise band"
    fig.update_layout(**plotly_layout(None, height=380, hovermode="closest", margin=dict(l=10, r=10, t=64, b=10),
                                      xaxis=dict(tickfont=dict(size=9), tickangle=-90), yaxis_title="Component"))
    show(fig)
    participation = 1 / rmt.inverse_participation_ratio(spectrum.eigenvectors)[mode - 1]
    st.caption(f"Eigenvalue {spectrum.eigenvalues[mode - 1]:.2f} ({status}). Stocks are grouped by sector, with "
               f"alternating shades. This vector involves the equivalent of {participation:.0f} of {n_assets} stocks.")

    st.subheader("Correlation matrix before and after cleaning")
    left, right = st.columns(2)
    for column, matrix, title in [(left, correlation, "Sample correlations"), (right, cleaned, "After eigenvalue clipping")]:
        with column:
            fig = go.Figure(go.Heatmap(z=matrix, x=order, y=order, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                                       colorbar=dict(thickness=10, len=0.8),
                                       hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
            fig.update_layout(**plotly_layout(title, height=470, hovermode="closest",
                                              xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed")))
            show(fig)
    insight("Cleaning keeps the large-scale structure (market and sector blocks) and smooths the pairwise detail, "
            "which is mostly sampling noise. Shorten the estimation window in the sidebar to see the noise grow.")

# --------------------------------------------------------------------------
# Page 3: portfolio builder
# --------------------------------------------------------------------------

elif page == "Portfolio builder":
    st.title("Portfolio builder")
    st.markdown(f'<div class="subtitle">Weights computed from the last {window} trading days '
                f'(to {recent.index[-1]:%d %B %Y}).</div>', unsafe_allow_html=True)

    one, two, three, four = st.columns(4)
    method = one.selectbox("Objective", ["Minimum variance", "Risk parity", "Maximum Sharpe", "Equal weight"])
    estimator_name = two.selectbox("Covariance estimator", list(cov.ESTIMATORS), index=2)
    long_only = three.toggle("Long only", value=True, disabled=method in ("Risk parity", "Equal weight"))
    cap = four.slider("Maximum weight per stock", 0.02, 1.0, 1.0, 0.01, format="%.2f",
                      disabled=method in ("Risk parity", "Equal weight"))
    max_weight = None if cap >= 1.0 else max(cap, 1.0 / n_assets + 1e-9)

    sigma = cov.ESTIMATORS[estimator_name](recent)
    mu = recent.mean().to_numpy()
    try:
        if method == "Minimum variance":
            weights = optimizer.min_variance(sigma, long_only, max_weight)
        elif method == "Risk parity":
            weights = optimizer.risk_parity(sigma)
        elif method == "Maximum Sharpe":
            weights = optimizer.max_sharpe(mu, sigma, 0.0, long_only, max_weight)
        else:
            weights = optimizer.equal_weight(n_assets)
    except (RuntimeError, ValueError) as error:
        st.error(f"The optimiser could not find a portfolio with these settings: {error}")
        st.stop()

    weights = pd.Series(weights, index=order)
    predicted_vol = np.sqrt(TRADING_DAYS * weights.to_numpy() @ sigma @ weights.to_numpy())
    equal_vol = np.sqrt(TRADING_DAYS * optimizer.portfolio_variance(optimizer.equal_weight(n_assets), sigma))
    held = int((weights.abs() > 1e-4).sum())
    effective = 1.0 / float((weights**2).sum())
    kpi_row([
        ("Predicted volatility", f"{predicted_vol:.1%}", f"equal weight: {equal_vol:.1%}"),
        ("Stocks held", f"{held} of {n_assets}", f"effective number {effective:.1f}"),
        ("Largest position", f"{weights.max():.1%}", weights.idxmax()),
        ("Short exposure", f"{max(0.0, -weights[weights < 0].sum()):.0%}", "sum of negative weights"),
    ])

    left, right = st.columns([3, 2])
    with left:
        shown = weights[weights.abs() > 1e-4].sort_values(ascending=False)
        shown = pd.concat([shown.head(20), shown.tail(10)]).loc[lambda s: ~s.index.duplicated()] if len(shown) > 30 else shown
        fig = go.Figure(go.Bar(x=shown.index, y=shown.values, marker_color=[BLUE if v >= 0 else ORANGE for v in shown.values],
                               customdata=sectors[shown.index], hovertemplate="<b>%{x}</b> · %{customdata}<br>Weight %{y:.2%}<extra></extra>"))
        fig.update_layout(**plotly_layout("Positions", hovermode="closest", yaxis_tickformat=".0%", yaxis_title="Weight"))
        show(fig)
    with right:
        by_sector = weights.groupby(sectors[order]).sum().sort_values()
        fig = go.Figure(go.Bar(x=by_sector.values, y=by_sector.index, orientation="h",
                               marker_color=[BLUE if v >= 0 else ORANGE for v in by_sector.values],
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_layout(**plotly_layout("Allocation by sector", hovermode="closest", xaxis_tickformat=".0%"))
        show(fig)

    st.subheader("Where the portfolio sits")
    frontier_vol, frontier_ret, _ = optimizer.efficient_frontier(mu, sigma, n_points=25, long_only=True, max_weight=max_weight)
    asset_vol = np.sqrt(TRADING_DAYS * np.diag(sigma))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=asset_vol, y=TRADING_DAYS * mu, mode="markers", name="Stocks", text=order,
                             marker=dict(color=GRID, size=8, line=dict(color=GRAY, width=1)),
                             hovertemplate="<b>%{text}</b><br>Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.sqrt(TRADING_DAYS) * frontier_vol, y=TRADING_DAYS * frontier_ret, mode="lines",
                             name="Efficient frontier (long only)", line=dict(color=BLUE, width=2.5),
                             hovertemplate="Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[predicted_vol], y=[TRADING_DAYS * float(mu @ weights.to_numpy())], mode="markers",
                             name="Selected portfolio", marker=dict(color=ORANGE, size=15, symbol="diamond", line=dict(color="white", width=2)),
                             hovertemplate="Selected portfolio<br>Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>"))
    fig.update_layout(**plotly_layout("In-sample risk and return, annualised", hovermode="closest",
                                      xaxis_title="Volatility", yaxis_title="Mean return over the window",
                                      xaxis_tickformat=".0%", yaxis_tickformat=".0%"))
    show(fig)
    insight("Everything on this page is <b>in sample</b>: risk and return are measured on the same days that were used "
            "to choose the weights. The frontier always looks attractive here. The Backtest section shows what "
            "happens on days the optimiser has not seen.")

# --------------------------------------------------------------------------
# Page 4: backtest
# --------------------------------------------------------------------------

else:
    st.title("Backtest")
    st.markdown('<div class="subtitle">Each month, weights are computed from past data only and held for the next month.</div>',
                unsafe_allow_html=True)

    one, two, three, four = st.columns(4)
    objective = one.selectbox("Objective", ["Minimum variance", "Maximum Sharpe", "Risk parity"])
    long_only = two.toggle("Long only", value=False, disabled=objective == "Risk parity",
                           help="Without this constraint the optimiser is free to exploit every estimated correlation, "
                                "which is where estimation noise hurts most.")
    cost = three.slider("Transaction cost (basis points)", 0, 30, 5)
    rebalance = four.select_slider("Rebalance every (days)", [5, 10, 21, 42, 63], value=21)

    @st.cache_data(show_spinner="Running the walk-forward backtest…")
    def backtest(data: pd.DataFrame, objective: str, long_only: bool, window: int, rebalance: int, cost: float):
        builders = {"Minimum variance": lambda e: min_variance_strategy(e, long_only),
                    "Maximum Sharpe": lambda e: max_sharpe_strategy(e, long_only),
                    "Risk parity": risk_parity_strategy}
        strategies = {name: builders[objective](estimator) for name, estimator in cov.ESTIMATORS.items()}
        strategies["Equal weight"] = equal_weight_strategy()
        return run_backtest(data, strategies, window=window, rebalance_every=rebalance, cost_bps=cost)

    try:
        result = backtest(returns, objective, long_only, window, rebalance, float(cost))
    except (RuntimeError, ValueError) as error:
        st.error(f"The backtest could not run with these settings: {error}")
        st.stop()

    table = metrics.summary(result.returns, result.turnover)
    table["Realised / predicted variance"] = result.risk_underestimation
    best = table["Volatility"].drop("Equal weight").idxmin()
    sample_vol, best_vol = table.loc["Sample", "Volatility"], table.loc[best, "Volatility"]
    kpi_row([
        ("Lowest realised volatility", f"{best_vol:.1%}", f"{best} · sample covariance: {sample_vol:.1%}"),
        ("Risk under-estimation, sample", f"{table.loc['Sample', 'Realised / predicted variance']:.2f}×",
         "realised over predicted variance"),
        ("Risk under-estimation, RMT", f"{table.loc['RMT clipping', 'Realised / predicted variance']:.2f}×",
         "1.00 would be a perfect forecast"),
        ("Turnover, sample vs RMT", f"{table.loc['Sample', 'Annual turnover']:.1f} vs {table.loc['RMT clipping', 'Annual turnover']:.1f}",
         "times the portfolio per year"),
    ])

    colors = {**ESTIMATOR_COLORS, "Equal weight": GRAY}
    fig = go.Figure()
    for name in result.returns.columns:
        wealth = (1 + result.returns[name]).cumprod()
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth, name=name, line=dict(color=colors[name], width=2)))
    fig.update_layout(**plotly_layout("Out-of-sample growth of 1, after costs", yaxis_type="log", yaxis_title="Value (log scale)"))
    show(fig)

    left, right = st.columns(2)
    with left:
        rolling = result.returns.rolling(126).std() * np.sqrt(TRADING_DAYS)
        fig = go.Figure()
        for name in rolling.columns:
            fig.add_trace(go.Scatter(x=rolling.index, y=rolling[name], name=name, line=dict(color=colors[name], width=1.8)))
        fig.update_layout(**plotly_layout("Realised volatility, rolling six months", yaxis_tickformat=".0%"))
        show(fig)
    with right:
        fig = go.Figure()
        for name in result.returns.columns:
            fig.add_trace(go.Scatter(x=result.returns.index, y=metrics.drawdowns(result.returns[name]), name=name,
                                     line=dict(color=colors[name], width=1.8)))
        fig.update_layout(**plotly_layout("Drawdown from the previous peak", yaxis_tickformat=".0%"))
        show(fig)

    st.subheader("Summary")
    st.dataframe(
        table.reset_index(names="Strategy"), hide_index=True, width="stretch",
        column_config={
            "Annual return": st.column_config.NumberColumn(format="percent"),
            "Volatility": st.column_config.NumberColumn(format="percent"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "Annual turnover": st.column_config.NumberColumn(format="%.2f"),
            "Realised / predicted variance": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    insight("Compare volatility, turnover and the realised-to-predicted ratio rather than returns: over a period this "
            "short, return differences between strategies are mostly luck, while risk differences are systematic. "
            "Turn <b>Long only</b> on to see the constraint do part of the cleaning by itself.")

    with st.expander("Weights through time"):
        strategy = st.selectbox("Strategy", list(result.weights))
        history = result.weights[strategy]
        top = history.abs().mean().sort_values(ascending=False).head(8).index
        fig = go.Figure()
        for ticker, color in zip(top, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY, INK, "#86b6ef"]):
            fig.add_trace(go.Scatter(x=history.index, y=history[ticker], name=ticker, line=dict(color=color, width=1.6)))
        fig.update_layout(**plotly_layout("Weights of the eight largest positions", yaxis_tickformat=".0%"))
        show(fig)
