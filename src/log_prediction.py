# src/log_prediction.py
"""Append today's band to logs/predictions.csv.

This is the genuinely valuable artifact: a real out-of-sample track record.
The backtest says coverage is 80%. This proves it — or doesn't — with
predictions made before the outcome was known. No backtest can claim that.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import HORIZON, ROOT
import garch as G

LOG = ROOT / "logs" / "predictions.csv"


def todays_band(conf=0.80):
    returns, prices = G.load_returns()
    vol = G.vol_gjr(returns)
    q = G.empirical_quantiles(returns, vol, conf)
    drift = returns.mean() * HORIZON / G.SCALE
    spot = float(prices.iloc[-1])
    lo, hi = G.band(spot, q[0], q[1], drift=drift)
    return {
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "data_through": prices.index[-1].strftime("%Y-%m-%d"),
        "spot": round(spot, 2),
        "conf": conf,
        "lo": round(lo, 2),
        "hi": round(hi, 2),
        "h_vol_pct": round(vol, 3),
        "resolves_on": None,   # filled by resolve() once the outcome is known
        "actual": None,
        "inside": None,
    }


def resolve(df):
    """Fill in outcomes for any prediction whose horizon has now passed."""
    _, prices = G.load_returns()

    for i, row in df.iterrows():
        if pd.notna(row.get("actual")):
            continue

        start = pd.Timestamp(row["data_through"])
        future = prices.index[prices.index > start]
        if len(future) < HORIZON:
            continue

        settle = future[HORIZON - 1]
        actual = float(prices.loc[settle])
        df.at[i, "resolves_on"] = settle.strftime("%Y-%m-%d")
        df.at[i, "actual"] = round(actual, 2)
        df.at[i, "inside"] = bool(row["lo"] <= actual <= row["hi"])

    return df


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for conf in [0.80, 0.95]:
        rows.append(todays_band(conf))
    today = pd.DataFrame(rows)

    if LOG.exists():
        df = pd.read_csv(LOG)
        # don't double-log the same data date
        mask = ~(
            df["data_through"].isin(today["data_through"])
            & df["conf"].isin(today["conf"])
        )
        df = pd.concat([df[mask], today], ignore_index=True)
    else:
        df = today

    df = resolve(df)
    df = df.sort_values(["data_through", "conf"]).reset_index(drop=True)
    df.to_csv(LOG, index=False)

    print(f"logged {len(today)} rows -> {LOG}")

    done = df[df["inside"].notna()]
    if len(done) > 0:
        print("\nlive track record so far:")
        for conf in [0.80, 0.95]:
            sub = done[done["conf"] == conf]
            if len(sub) > 0:
                hit = sub["inside"].astype(bool).mean()
                print(f"  {conf:.0%} band: {hit:.1%} coverage over {len(sub)} resolved weeks")