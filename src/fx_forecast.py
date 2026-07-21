# src/fx_forecast.py
"""Next-day USD/INR forecast test.

Converting tomorrow's predicted gold price to rupees using today's exchange
rate is slightly inconsistent if the rupee itself moves by tomorrow. This
tests whether USD/INR can be forecast well enough to close that gap.

Prior, worth stating before running this: exchange rates are among the most
studied "doesn't work" series in finance. Meese and Rogoff (1983) showed no
economic model beats a naive "tomorrow equals today" guess at short horizons,
and that result has held for over 40 years across currencies. Expect this to
fail the same way gold direction did. Run it anyway rather than assume it,
same rule as everything else in this project.
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

warnings.filterwarnings("ignore")

NON_FEATURES = ("target", "usdinr_level")


def build_fx_dataset(save=True):
    raw = F.load_raw()
    df = F.align(raw)  # gold's calendar, forward-filled; usdinr already present

    f = pd.DataFrame(index=df.index)
    fx_ret = np.log(df["usdinr"]).diff()

    f["fx_ret_1d"] = fx_ret
    f["fx_ret_5d"] = np.log(df["usdinr"]).diff(5)
    f["fx_ret_20d"] = np.log(df["usdinr"]).diff(20)
    f["fx_rv_5d"] = fx_ret.rolling(5).std()
    f["fx_rv_20d"] = fx_ret.rolling(20).std()

    for col in ["dxy", "vix", "tnx", "real_yield", "breakeven"]:
        if col in df:
            f[f"{col}_lvl"] = df[col]
            f[f"{col}_chg_5d"] = df[col].diff(5)

    f["gold_ret_5d"] = np.log(df["gold"]).diff(5)
    f["dow"] = f.index.dayofweek
    f["month"] = f.index.month

    f["target"] = np.log(df["usdinr"]).shift(-1) - np.log(df["usdinr"])
    f["usdinr_level"] = df["usdinr"]
    f = f.dropna()

    if save:
        path = PROCESSED / "fx_dataset_h1.parquet"
        f.to_parquet(path)
        print(f"saved -> {path}")
    return f


def feature_cols(ds):
    return [c for c in ds.columns if c not in NON_FEATURES]


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


def walk_forward(ds, model_fn, train_years=5, refit_every=21):
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
    return {"model": name, "rmse": np.sqrt(((actual - pred) ** 2).mean()),
            "vs_naive": r2_vs(actual, pred, naive), "vs_drift": r2_vs(actual, pred, drift),
            "dir_acc": dir_acc, "dir_significant": bool(abs(z) > 1.96), "n": n}


if __name__ == "__main__":
    ds = build_fx_dataset()
    print(f"FX dataset: {len(ds)} rows  {ds.index.min().date()} -> {ds.index.max().date()}\n")

    results = []
    for name, fn in MODELS.items():
        print(f"walk-forward: {name}...")
        wf = walk_forward(ds, fn)
        results.append(evaluate(name, wf, ds))

    res = pd.DataFrame(results).set_index("model").sort_values("vs_naive", ascending=False)
    print("\n" + "=" * 78)
    print("NEXT-DAY USD/INR — walk-forward, out-of-sample")
    print("=" * 78)
    print(res.to_string(float_format=lambda x: f"{x:8.4f}"))

    winner = res.index[0]
    beats_all = (res.loc[winner, "vs_naive"] > 0 and res.loc[winner, "vs_drift"] > 0
                and res.loc[winner, "dir_significant"])

    print("\n" + "-" * 78)
    if beats_all:
        print(f"VERDICT: {winner} beats both baselines with significant direction skill.")
        print("Worth a split-half consistency check before trusting, same as the gold test.")
    else:
        print(f"VERDICT: no model beat naive on USD/INR. Best of {len(MODELS)} ({winner}) still lost.")
        print("Matches Meese-Rogoff: short-horizon FX isn't predictable from these")
        print("features either. The rupee conversion should keep using today's rate,")
        print("with that assumption stated plainly rather than hidden.")
    print("-" * 78)