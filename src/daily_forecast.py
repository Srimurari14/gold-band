# src/daily_forecast.py
"""Next-day gold forecast: five models, honest evaluation, GARCH interval.

Same discipline as the weekly analysis — walk-forward only, always scored
against naive and drift, winner shown as a headline number only if it earns
it. Shorter horizon than the weekly band means LESS signal relative to
noise, not more, so the honest expectation going in is that this is at
least as hard to beat as the weekly result already was.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    HAVE_CATBOOST = True
except Exception:
    HAVE_CATBOOST = False

from config import PROCESSED
import features as F
import garch as G

warnings.filterwarnings("ignore")

H = 1  # next trading day
NON_FEATURES = ("target", "gold_price")


def load_daily_dataset():
    path = PROCESSED / "dataset_h1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return F.build_dataset(save=True, horizon=H)


def feature_cols(ds):
    return [c for c in ds.columns if c not in NON_FEATURES]


# ---------------------------------------------------------------- models

def ridge_fn():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

def rf_fn():
    return RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=30,
                                 max_features=0.6, n_jobs=-1, random_state=0)

def lgbm_fn():
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15,
                             min_child_samples=30, subsample=0.8,
                             colsample_bytree=0.8, verbose=-1)

def xgb_fn():
    return xgb.XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=3,
                            min_child_weight=30, subsample=0.8,
                            colsample_bytree=0.8, verbosity=0)

def catboost_fn():
    return CatBoostRegressor(iterations=200, learning_rate=0.03, depth=4,
                             l2_leaf_reg=10, verbose=False)


MODELS = {"ridge": ridge_fn, "rf": rf_fn, "lightgbm": lgbm_fn, "xgboost": xgb_fn}
if HAVE_CATBOOST:
    MODELS["catboost"] = catboost_fn


# ---------------------------------------------------------------- backtest

def walk_forward(ds, model_fn, train_years=5, refit_every=21):
    """Expanding window, refit monthly. h=1 so every row is one independent
    trading day — no overlap correction needed, unlike the weekly case."""
    cols = feature_cols(ds)
    X, y = ds[cols], ds["target"]

    start_date = ds.index.min() + pd.DateOffset(years=train_years)
    start_i = int(ds.index.searchsorted(start_date))

    rows, model = [], None
    for i in range(start_i, len(ds)):
        if model is None or (i - start_i) % refit_every == 0:
            model = model_fn()
            model.fit(X.iloc[:i], y.iloc[:i])
        pred = model.predict(X.iloc[[i]])[0]
        rows.append({"date": ds.index[i], "actual": y.iloc[i], "pred": pred})

    return pd.DataFrame(rows).set_index("date")


def r2_vs(actual, pred, baseline):
    ss_model = ((actual - pred) ** 2).sum()
    ss_base = ((actual - baseline) ** 2).sum()
    return 1 - ss_model / ss_base


def evaluate(name, wf, ds):
    actual, pred = wf["actual"], wf["pred"]
    naive = pd.Series(0.0, index=wf.index)
    drift = ds["target"].expanding().mean().shift(1).reindex(wf.index).fillna(0)

    n = len(actual)
    dir_acc = (np.sign(actual) == np.sign(pred)).mean()
    z = (dir_acc - 0.5) / (0.5 / np.sqrt(n))

    return {
        "model": name,
        "rmse": np.sqrt(((actual - pred) ** 2).mean()),
        "mae": (actual - pred).abs().mean(),
        "vs_naive": r2_vs(actual, pred, naive),
        "vs_drift": r2_vs(actual, pred, drift),
        "dir_acc": dir_acc,
        "dir_significant": bool(abs(z) > 1.96),
        "n": n,
    }


if __name__ == "__main__":
    ds = load_daily_dataset()
    print(f"daily dataset: {len(ds)} rows  {ds.index.min().date()} -> {ds.index.max().date()}")
    print("(no decimation needed at h=1 — each row is one independent trading day)\n")

    results, forecasts = [], {}
    for name, fn in MODELS.items():
        print(f"walk-forward: {name}...")
        wf = walk_forward(ds, fn)
        forecasts[name] = wf
        results.append(evaluate(name, wf, ds))

    res = pd.DataFrame(results).set_index("model").sort_values("vs_naive", ascending=False)

    print("\n" + "=" * 78)
    print("NEXT-DAY MODEL COMPARISON — walk-forward, out-of-sample")
    print("=" * 78)
    print(res.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("\n  vs_naive / vs_drift : >0 beats that baseline, <0 loses to it")
    print("  dir_acc             : right-direction rate (50% = coin flip)")

    winner = res.index[0]
    beats_all = (res.loc[winner, "vs_naive"] > 0 and res.loc[winner, "vs_drift"] > 0
                and res.loc[winner, "dir_significant"])

    print("\n" + "-" * 78)
    if beats_all:
        print(f"VERDICT: {winner} beats both baselines with significant direction skill.")
    elif res.loc[winner, "vs_naive"] > 0:
        print(f"VERDICT: {winner} narrowly beats naive but fails another check.")
        print("With 5 models tested, one beating naive slightly is close to what")
        print("chance alone produces. Treated as noise, not signal.")
    else:
        print(f"VERDICT: no model beat naive. Best of {len(MODELS)} ({winner}) still lost.")
        print("Matches the weekly result. Daily direction isn't predictable from")
        print("these features either.")
    print("-" * 78)

    print(f"\nCaveat: {len(MODELS)} models tested, best one shown. Even with zero real")
    print(f"skill, the best of {len(MODELS)} tries beats naive more often than any single")
    print(f"model would by chance. Read the full table, not just the top row.")

    # ---- live output
    print("\n" + "=" * 78)
    print("LIVE — tomorrow")
    print("=" * 78)

    returns, prices = G.load_returns()
    spot = float(prices.iloc[-1])
    vol1 = G.vol_gjr(returns, horizon=1)

    for conf in [0.80, 0.95]:
        q = G.empirical_quantiles(returns, vol1, conf, horizon=1)
        if q is None:
            continue
        lo, hi = G.band(spot, q[0], q[1], drift=0.0)
        print(f"  naive-centred {conf:.0%} range: ${lo:,.0f} - ${hi:,.0f}")

    if beats_all:
        cols = feature_cols(ds)
        model = MODELS[winner]()
        model.fit(ds[cols], ds["target"])
        point_ret = float(model.predict(ds[cols].iloc[[-1]])[0])
        point_price = spot * np.exp(point_ret)
        print(f"\n  {winner} point forecast: ${point_price:,.0f} ({point_ret*100:+.2f}%)")
        print(f"  Shown because it passed all three checks. Still wrap it in the")
        print(f"  range above — treat the point as the centre, not the answer.")
    else:
        print(f"\n  No point forecast shown — nothing beat naive honestly.")
        print(f"  The range above, centred on today's price, is the whole answer.")