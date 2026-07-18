# src/features.py
"""Align raw series onto gold's calendar, build lagged features and the target."""

import numpy as np
import pandas as pd

from config import RAW, PROCESSED, MASTER, HORIZON


def load_raw():
    return pd.read_parquet(RAW / "raw.parquet")


def align(raw):
    """Reindex everything onto gold's trading days. Forward-fill only."""
    raw = raw.sort_index()
    master_idx = raw[MASTER].dropna().index
    df = raw.reindex(master_idx)
    df = df.ffill()
    df = df.dropna()
    return df


def build_features(df):
    """All features are functions of data at time t or earlier."""
    f = pd.DataFrame(index=df.index)
    gold_ret = np.log(df["gold"]).diff()

    f["gold_ret_1d"] = gold_ret
    f["gold_ret_5d"] = np.log(df["gold"]).diff(5)
    f["gold_ret_20d"] = np.log(df["gold"]).diff(20)

    f["gold_rv_5d"] = gold_ret.rolling(5).std()
    f["gold_rv_20d"] = gold_ret.rolling(20).std()

    for col in ["dxy", "vix", "tnx", "real_yield", "breakeven"]:
        if col in df:
            f[f"{col}_lvl"] = df[col]
            f[f"{col}_chg_5d"] = df[col].diff(5)

    for col in ["silver"]:
        if col in df:
            f[f"{col}_ret_5d"] = np.log(df[col]).diff(5)

    f["dow"] = f.index.dayofweek
    f["month"] = f.index.month

    return f


def build_target(df, horizon=HORIZON):
    """Forward return: t -> t+horizon. This is the ONLY place the future appears."""
    return np.log(df["gold"]).shift(-horizon) - np.log(df["gold"])


def build_dataset(save=True, horizon=HORIZON):
    """horizon=5 (default) writes dataset.parquet — the weekly file everything
    else already depends on. Any other horizon writes dataset_h{horizon}.parquet
    so it never collides with that."""
    raw = load_raw()
    df = align(raw)

    X = build_features(df)
    y = build_target(df, horizon=horizon)

    out = X.copy()
    out["target"] = y
    out["gold_price"] = df["gold"]
    out = out.dropna()

    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        fname = "dataset.parquet" if horizon == HORIZON else f"dataset_h{horizon}.parquet"
        path = PROCESSED / fname
        out.to_parquet(path)
        print(f"saved -> {path}")

    return out


if __name__ == "__main__":
    ds = build_dataset()
    print(f"\nshape: {ds.shape}")
    print(f"range: {ds.index.min().date()} -> {ds.index.max().date()}")
    print(f"\nfeatures ({ds.shape[1] - 2}):")
    print("  " + ", ".join(c for c in ds.columns if c not in ("target", "gold_price")))

    print(f"\ntarget (5d fwd log return):")
    print(ds["target"].describe().to_string())

    ac = ds["target"].autocorr(lag=HORIZON)
    print(f"\ntarget autocorrelation at lag {HORIZON}: {ac:.4f}")

    abs_ret = ds["gold_ret_1d"].abs()
    print(f"\nvol clustering:")
    for lag in [1, 5, 10, 20, 40]:
        print(f"  |ret| autocorr lag {lag:2d}: {abs_ret.autocorr(lag=lag):.4f}")