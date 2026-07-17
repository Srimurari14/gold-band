# src/ingest.py
"""Pull raw series from yfinance and FRED, cache to parquet."""

import pandas as pd
import yfinance as yf
from fredapi import Fred

from config import RAW, START, YF_TICKERS, FRED_SERIES, FRED_API_KEY


def pull_yfinance(start=START):
    """Pull all yfinance tickers as a single dataframe of closes."""
    frames = {}
    for name, ticker in YF_TICKERS.items():
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        if df.empty:
            print(f"  WARNING: {name} ({ticker}) returned nothing")
            continue
        s = df["Close"]
        if isinstance(s, pd.DataFrame):      # yfinance sometimes returns MultiIndex
            s = s.iloc[:, 0]
        frames[name] = s
        print(f"  {name:10s} {ticker:10s} {len(s):5d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    out = pd.DataFrame(frames)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def pull_fred(start=START):
    """Pull FRED series. Returns empty frame if no API key."""
    if not FRED_API_KEY:
        print("  no FRED_API_KEY — skipping (add to .env later)")
        return pd.DataFrame()

    fred = Fred(api_key=FRED_API_KEY)
    frames = {}
    for name, series_id in FRED_SERIES.items():
        s = fred.get_series(series_id, observation_start=start)
        s = s.dropna()
        frames[name] = s
        print(f"  {name:10s} {series_id:10s} {len(s):5d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    out = pd.DataFrame(frames)
    out.index = pd.to_datetime(out.index).normalize()
    return out


def pull_all(start=START, save=True):
    print("yfinance:")
    yf_df = pull_yfinance(start)
    print("\nFRED:")
    fred_df = pull_fred(start)

    raw = yf_df.join(fred_df, how="outer") if not fred_df.empty else yf_df
    raw = raw.sort_index()

    if save:
        RAW.mkdir(parents=True, exist_ok=True)
        path = RAW / "raw.parquet"
        raw.to_parquet(path)
        print(f"\nsaved -> {path}")

    return raw


if __name__ == "__main__":
    raw = pull_all()

    print(f"\nshape: {raw.shape}")
    print(f"range: {raw.index.min().date()} -> {raw.index.max().date()}")

    print("\nmissing values per column:")
    print(raw.isna().sum().to_string())

    print("\nlast 5 rows:")
    print(raw.tail().to_string())