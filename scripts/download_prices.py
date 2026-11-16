"""Download daily adjusted closing prices of large US stocks from Yahoo Finance.

Usage::

    python download_prices.py                       # writes prices.csv
    python download_prices.py --start 2008-01-01 --output prices.csv

The script only needs ``yfinance`` and ``pandas``::

    python -m pip install yfinance pandas

Output: one CSV row per trading day, one column per ticker. Prices are
adjusted for splits and dividends, so their ratios are total returns. The
last column, SPY, is an S&P 500 index fund used as the market benchmark.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd
import yfinance as yf

# About 100 of the largest US companies, all listed before 2010.
TICKERS = """
AAPL MSFT AMZN GOOGL JPM JNJ V PG XOM UNH HD MA BAC DIS CVX KO PFE PEP CSCO WMT
MRK ABT T VZ INTC CMCSA ADBE CRM NFLX NKE MCD COST TMO ACN AVGO TXN QCOM HON UNP LLY
ORCL IBM AMGN LOW UPS CAT GS MS BA MMM GE SBUX BLK AXP MDT GILD CVS C WFC USB
SCHW SPGI BKNG ADP DE LMT RTX COP SLB EOG DUK SO NEE D AEP MO PM CL KMB GIS
MDLZ TGT TJX BMY DHR ISRG SYK CI PNC MMC CB PLD AMT SPG EMR ITW FDX NSC CSX WM
ECL APD SHW NEM F
""".split()
BENCHMARK = "SPY"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily adjusted prices from Yahoo Finance.")
    parser.add_argument("--start", default="2010-01-01", help="first date (YYYY-MM-DD)")
    parser.add_argument("--output", default="prices.csv", help="output CSV file")
    args = parser.parse_args()

    symbols = TICKERS + [BENCHMARK]
    print(f"Downloading {len(symbols)} tickers from {args.start} (this takes about a minute)...")
    try:
        # ``end`` is exclusive: today's bar is left out because it is incomplete
        # while the market is open, and one partial day would distort the returns.
        data = yf.download(symbols, start=args.start, end=date.today().isoformat(), auto_adjust=True,
                           progress=False, threads=True)
    except Exception as error:
        sys.exit(f"Download failed: {error}")
    if data.empty:
        sys.exit("Download failed: no data returned. Check your internet connection and try again.")

    prices = data["Close"]
    # Keep tickers with a (nearly) complete history, in the original order.
    coverage = prices.notna().mean()
    kept = [s for s in symbols if s in prices.columns and coverage[s] > 0.99]
    dropped = [s for s in symbols if s not in kept]
    prices = prices[kept].dropna(how="all").round(4)
    prices.index = pd.to_datetime(prices.index).strftime("%Y-%m-%d")
    prices.index.name = "date"
    prices.to_csv(args.output)

    print(f"Saved {len(prices)} days x {len(kept)} tickers to {args.output}")
    print(f"From {prices.index[0]} to {prices.index[-1]}")
    if dropped:
        print(f"Dropped (incomplete history): {', '.join(dropped)}")


if __name__ == "__main__":
    main()
