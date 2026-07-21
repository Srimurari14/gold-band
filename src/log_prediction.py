# src/log_prediction.py
"""Append today's predictions to logs/. Three independent logs:

  predictions.csv        weekly GARCH band (validated by ten-year backtest,
                          now also tracked live)
  daily_predictions.csv  catboost next-day gold forecast (backtest edge was
                          thin and sign-flipped between halves; tracked live,
                          not yet trusted)
  fx_predictions.csv     random forest next-day USD/INR forecast (backtest
                          edge was real and consistent, survived dropping
                          calendar features and a split-half check, plausibly
                          explained by RBI currency management rather than
                          free-market inefficiency; tracked live before it
                          moves into the dashboard's trusted numbers)

None of the three point forecasts should move to a dashboard headline until
they've earned it here, with real future data nothing can be tuned against.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import HORIZON, ROOT
import garch as G
import daily_forecast as D
import fx_forecast as FX

WEEKLY_LOG = ROOT / "logs" / "predictions.csv"
DAILY_LOG = ROOT / "logs" / "daily_predictions.csv"
FX_LOG = ROOT / "logs" / "fx_predictions.csv"

DAILY_MODEL = "catboost"
FX_MODEL = "rf"


# ---------------------------------------------------------------- weekly band

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
        "resolves_on": None,
        "actual": None,
        "inside": None,
    }


def resolve_weekly(df):
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


def log_weekly():
    rows = [todays_band(c) for c in [0.80, 0.95]]
    today = pd.DataFrame(rows)

    if WEEKLY_LOG.exists():
        df = pd.read_csv(WEEKLY_LOG)
        mask = ~(df["data_through"].isin(today["data_through"]) & df["conf"].isin(today["conf"]))
        df = pd.concat([df[mask], today], ignore_index=True)
    else:
        df = today

    df = resolve_weekly(df).sort_values(["data_through", "conf"]).reset_index(drop=True)
    df.to_csv(WEEKLY_LOG, index=False)
    print(f"weekly:  logged {len(today)} rows -> {WEEKLY_LOG}")

    done = df[df["inside"].notna()]
    for conf in [0.80, 0.95]:
        sub = done[done["conf"] == conf]
        if len(sub) > 0:
            hit = sub["inside"].astype(bool).mean()
            print(f"         {conf:.0%} band: {hit:.1%} coverage over {len(sub)} resolved weeks")


# ---------------------------------------------------------------- daily gold point forecast

def todays_daily_forecast():
    ds = D.load_daily_dataset()
    cols = D.feature_cols(ds)

    model = D.MODELS[DAILY_MODEL]()
    model.fit(ds[cols], ds["target"])

    spot = float(ds["gold_price"].iloc[-1])
    pred_ret = float(model.predict(ds[cols].iloc[[-1]])[0])
    pred_price = spot * np.exp(pred_ret)

    return {
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "data_through": ds.index[-1].strftime("%Y-%m-%d"),
        "model": DAILY_MODEL,
        "spot": round(spot, 2),
        "pred_ret_pct": round(pred_ret * 100, 4),
        "pred_price": round(pred_price, 2),
        "resolves_on": None,
        "actual_price": None,
        "model_abs_err": None,
        "naive_abs_err": None,
        "model_beat_naive": None,
    }


def resolve_daily(df):
    _, prices = G.load_returns()
    for i, row in df.iterrows():
        if pd.notna(row.get("actual_price")):
            continue
        start = pd.Timestamp(row["data_through"])
        future = prices.index[prices.index > start]
        if len(future) < 1:
            continue

        settle = future[0]
        actual = float(prices.loc[settle])
        naive_pred = row["spot"]

        df.at[i, "resolves_on"] = settle.strftime("%Y-%m-%d")
        df.at[i, "actual_price"] = round(actual, 2)
        df.at[i, "model_abs_err"] = round(abs(actual - row["pred_price"]), 2)
        df.at[i, "naive_abs_err"] = round(abs(actual - naive_pred), 2)
        df.at[i, "model_beat_naive"] = bool(abs(actual - row["pred_price"]) < abs(actual - naive_pred))
    return df


def log_daily():
    today = pd.DataFrame([todays_daily_forecast()])

    if DAILY_LOG.exists():
        df = pd.read_csv(DAILY_LOG)
        mask = ~(df["data_through"].isin(today["data_through"]) & (df["model"] == DAILY_MODEL))
        df = pd.concat([df[mask], today], ignore_index=True)
    else:
        df = today

    df = resolve_daily(df).sort_values("data_through").reset_index(drop=True)
    df.to_csv(DAILY_LOG, index=False)
    print(f"daily:   logged 1 row -> {DAILY_LOG}")

    done = df[df["model_beat_naive"].notna()]
    if len(done) > 0:
        rate = done["model_beat_naive"].astype(bool).mean()
        print(f"         {DAILY_MODEL} beat naive on {rate:.1%} of {len(done)} resolved days "
              f"(50% = no edge)")
        if len(done) < 40:
            print(f"         n is still small, treat as provisional for a few more weeks")


# ---------------------------------------------------------------- FX point forecast

def todays_fx_forecast():
    ds = FX.build_fx_dataset(save=False)
    cols = FX.feature_cols(ds)

    model = FX.MODELS[FX_MODEL]()
    model.fit(ds[cols], ds["target"])

    spot = float(ds["usdinr_level"].iloc[-1])
    pred_ret = float(model.predict(ds[cols].iloc[[-1]])[0])
    pred_rate = spot * np.exp(pred_ret)

    return {
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "data_through": ds.index[-1].strftime("%Y-%m-%d"),
        "model": FX_MODEL,
        "spot": round(spot, 4),
        "pred_ret_pct": round(pred_ret * 100, 4),
        "pred_rate": round(pred_rate, 4),
        "resolves_on": None,
        "actual_rate": None,
        "model_abs_err": None,
        "naive_abs_err": None,
        "model_beat_naive": None,
    }


def resolve_fx(df):
    raw = FX.F.load_raw()
    aligned = FX.F.align(raw)
    rates = aligned["usdinr"]

    for i, row in df.iterrows():
        if pd.notna(row.get("actual_rate")):
            continue
        start = pd.Timestamp(row["data_through"])
        future = rates.index[rates.index > start]
        if len(future) < 1:
            continue

        settle = future[0]
        actual = float(rates.loc[settle])
        naive_pred = row["spot"]

        df.at[i, "resolves_on"] = settle.strftime("%Y-%m-%d")
        df.at[i, "actual_rate"] = round(actual, 4)
        df.at[i, "model_abs_err"] = round(abs(actual - row["pred_rate"]), 4)
        df.at[i, "naive_abs_err"] = round(abs(actual - naive_pred), 4)
        df.at[i, "model_beat_naive"] = bool(abs(actual - row["pred_rate"]) < abs(actual - naive_pred))
    return df


def log_fx():
    today = pd.DataFrame([todays_fx_forecast()])

    if FX_LOG.exists():
        df = pd.read_csv(FX_LOG)
        mask = ~(df["data_through"].isin(today["data_through"]) & (df["model"] == FX_MODEL))
        df = pd.concat([df[mask], today], ignore_index=True)
    else:
        df = today

    df = resolve_fx(df).sort_values("data_through").reset_index(drop=True)
    df.to_csv(FX_LOG, index=False)
    print(f"fx:      logged 1 row -> {FX_LOG}")

    done = df[df["model_beat_naive"].notna()]
    if len(done) > 0:
        rate = done["model_beat_naive"].astype(bool).mean()
        print(f"         {FX_MODEL} beat naive on {rate:.1%} of {len(done)} resolved days "
              f"(50% = no edge; backtest suggested a real edge here, unlike the gold one)")
        if len(done) < 40:
            print(f"         n is still small, treat as provisional for a few more weeks")


if __name__ == "__main__":
    WEEKLY_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_weekly()
    print()
    try:
        log_daily()
    except Exception as e:
        print(f"daily:   skipped ({e})")
    print()
    try:
        log_fx()
    except Exception as e:
        print(f"fx:      skipped ({e})")