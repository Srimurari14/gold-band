# src/backtest.py
"""Walk-forward backtest: factor models vs naive. The experiment."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

from config import PROCESSED, HORIZON

NON_FEATURES = ("target", "gold_price")


def load_dataset():
    return pd.read_parquet(PROCESSED / "dataset.parquet")


def decimate(ds, horizon=HORIZON):
    """Take every Nth row so prediction windows don't overlap.

    Daily rows with a 5-day forward target overlap 4/5 days. Treating them as
    independent inflates apparent sample size ~5x and makes everything look
    more significant than it is.
    """
    return ds.iloc[::horizon].copy()


def feature_cols(ds):
    return [c for c in ds.columns if c not in NON_FEATURES]


def walk_forward(ds, model_fn, train_years=5, step_months=6):
    """Expanding-window walk-forward. Train on past only, predict forward.

    Returns dataframe of (date, actual, pred) for every out-of-sample point.
    """
    cols = feature_cols(ds)
    X = ds[cols]
    y = ds["target"]

    start = ds.index.min() + pd.DateOffset(years=train_years)
    edges = pd.date_range(start, ds.index.max(), freq=f"{step_months}MS")

    rows = []
    for i, edge in enumerate(edges):
        nxt = edges[i + 1] if i + 1 < len(edges) else ds.index.max() + pd.Timedelta(days=1)

        tr = ds.index < edge          # strictly past
        te = (ds.index >= edge) & (ds.index < nxt)

        if te.sum() == 0 or tr.sum() < 100:
            continue

        model = model_fn()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        rows.append(pd.DataFrame({
            "actual": y[te].values,
            "pred": pred,
        }, index=ds.index[te]))

    return pd.concat(rows)


def r2(actual, pred):
    ss_res = ((actual - pred) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def score(name, actual, pred):
    """R2 vs the mean, RMSE, directional accuracy."""
    rmse = np.sqrt(((actual - pred) ** 2).mean())
    dir_acc = (np.sign(actual) == np.sign(pred)).mean()
    return {
        "model": name,
        "rmse": rmse,
        "r2_vs_mean": r2(actual, pred),
        "dir_acc": dir_acc,
    }


def r2_vs_naive(actual, pred, naive_pred):
    """The number that matters: does the model beat naive?

    >0 means better than naive. <0 means worse. Almost always <0.
    """
    ss_model = ((actual - pred) ** 2).sum()
    ss_naive = ((actual - naive_pred) ** 2).sum()
    return 1 - ss_model / ss_naive


def ridge_fn():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def lgbm_fn():
    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )


def train_vs_test_gap(ds, model_fn, train_years=5):
    """Fit once on the first train_years, report train R2 vs test R2.

    This is the overfitting demonstration.
    """
    cols = feature_cols(ds)
    split = ds.index.min() + pd.DateOffset(years=train_years)
    tr, te = ds.index < split, ds.index >= split

    model = model_fn()
    model.fit(ds[cols][tr], ds["target"][tr])

    return (
        r2(ds["target"][tr], model.predict(ds[cols][tr])),
        r2(ds["target"][te], model.predict(ds[cols][te])),
    )


if __name__ == "__main__":
    ds_full = load_dataset()
    ds = decimate(ds_full)

    print(f"full dataset:    {len(ds_full)} rows (overlapping windows)")
    print(f"decimated:       {len(ds)} rows (independent, every {HORIZON}th)")
    print(f"range:           {ds.index.min().date()} -> {ds.index.max().date()}")

    # ---- naive baselines
    # naive 1: random walk. best guess for next week's return = 0.
    # naive 2: random walk with drift. = historical mean return.
    print("\n" + "=" * 62)
    print("WALK-FORWARD OUT-OF-SAMPLE")
    print("=" * 62)

    results = []
    preds = {}

    for name, fn in [("ridge", ridge_fn), ("lightgbm", lgbm_fn)]:
        wf = walk_forward(ds, fn)
        preds[name] = wf
        results.append(score(name, wf["actual"], wf["pred"]))

    # naive predictions over the same out-of-sample points
    oos_idx = preds["ridge"].index
    actual = preds["ridge"]["actual"]

    naive_zero = pd.Series(0.0, index=oos_idx)
    # drift computed expanding — no lookahead
    drift = ds["target"].expanding().mean().shift(1).reindex(oos_idx).fillna(0)

    results.append(score("naive (zero)", actual, naive_zero))
    results.append(score("naive (drift)", actual, drift))

    res = pd.DataFrame(results).set_index("model")
    print(res.to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n" + "-" * 62)
    print("SKILL vs NAIVE  (>0 = beats naive, <0 = loses to naive)")
    print("-" * 62)
    for name in ["ridge", "lightgbm"]:
        v = r2_vs_naive(actual, preds[name]["pred"], naive_zero)
        verdict = "BEATS naive" if v > 0 else "loses to naive"
        print(f"  {name:10s} {v:+8.4f}   {verdict}")

    print("\n" + "-" * 62)
    print("OVERFITTING: train R2 vs test R2")
    print("-" * 62)
    for name, fn in [("ridge", ridge_fn), ("lightgbm", lgbm_fn)]:
        tr_r2, te_r2 = train_vs_test_gap(ds, fn)
        print(f"  {name:10s} train {tr_r2:+7.4f}   test {te_r2:+7.4f}   gap {tr_r2 - te_r2:+7.4f}")

    print("\n" + "-" * 62)
    print("DIRECTIONAL ACCURACY vs coin flip (50%)")
    print("-" * 62)
    n = len(actual)
    se = 0.5 / np.sqrt(n)   # std error of a proportion under H0: p=0.5
    for name in ["ridge", "lightgbm"]:
        acc = (np.sign(actual) == np.sign(preds[name]["pred"])).mean()
        z = (acc - 0.5) / se
        sig = "significant" if abs(z) > 1.96 else "NOT significant"
        print(f"  {name:10s} {acc:.1%}   z={z:+.2f}   {sig}")
    print(f"\n  (n={n} independent predictions)")