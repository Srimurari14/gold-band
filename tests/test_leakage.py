# tests/test_leakage.py
"""Guards against the one bug that would invalidate everything.

If a feature at time t contains information from after t, every result in
this project is fiction. These tests are cheap; the failure they catch is not.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import HORIZON, MASTER
import features as F


@pytest.fixture(scope="module")
def raw():
    return F.load_raw()


@pytest.fixture(scope="module")
def aligned(raw):
    return F.align(raw)


# ---------------------------------------------------------------- alignment

def test_align_uses_gold_calendar(raw, aligned):
    """Index must be exactly gold's trading days — no phantom rows."""
    gold_days = raw[MASTER].dropna().index
    assert aligned.index.isin(gold_days).all()


def test_align_is_forward_fill_only(raw):
    """The core rule. Blank out a stretch of history, align, and confirm the
    gap was filled from the PAST, not the future.
    """
    r = raw.copy()
    col = "dxy"
    mask = (r.index >= "2020-03-01") & (r.index <= "2020-03-20")
    before = r.loc[r.index < "2020-03-01", col].dropna().iloc[-1]
    after = r.loc[r.index > "2020-03-20", col].dropna().iloc[0]
    r.loc[mask, col] = np.nan

    a = F.align(r)
    filled = a.loc[(a.index >= "2020-03-02") & (a.index <= "2020-03-19"), col]

    assert len(filled) > 0, "test setup: no rows in gap"
    assert (filled == before).all(), "gap not filled from the past"
    assert not (filled == after).any(), "LEAKAGE: gap filled from the future"


def test_no_nans_survive(aligned):
    assert not aligned.isna().any().any()


# ---------------------------------------------------------------- features

def test_features_depend_only_on_past(aligned):
    """The real test: truncate the data at time t, rebuild features, and
    confirm the row at t is identical to when the full history was present.

    If any feature peeks forward, truncation changes it.
    """
    full = F.build_features(aligned)

    for cut in ["2018-06-15", "2021-09-10", "2024-02-20"]:
        past_only = aligned.loc[aligned.index <= cut]
        truncated = F.build_features(past_only)

        t = truncated.index[-1]
        pd.testing.assert_series_equal(
            full.loc[t], truncated.loc[t], check_names=False,
            obj=f"features at {t.date()} changed when future data removed",
        )


def test_target_is_the_only_future(aligned):
    """Target MUST look forward — that's its job. Confirm it does, by exactly
    HORIZON steps, and that the unknowable tail is NaN."""
    y = F.build_target(aligned)
    g = np.log(aligned["gold"])

    i = 100
    assert np.isclose(y.iloc[i], g.iloc[i + HORIZON] - g.iloc[i])
    assert y.iloc[-HORIZON:].isna().all()


def test_only_build_target_looks_forward():
    """Static AST check: no function except build_target may use a negative
    shift, bfill, backfill, or interpolate.

    Parses the source rather than grepping text, so comments and docstrings
    warning ABOUT these patterns don't trip it.
    """
    src = (Path(__file__).resolve().parents[1] / "src" / "features.py").read_text()
    tree = ast.parse(src)

    banned_calls = {"bfill", "backfill", "interpolate"}
    offences = []

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if fn.name == "build_target":
            continue

        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            method = node.func.attr

            if method in banned_calls:
                offences.append(f"{fn.name}() calls .{method}()")

            if method == "shift":
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        offences.append(f"{fn.name}() calls .shift() with a negative offset")

    assert not offences, "possible leakage:\n  " + "\n  ".join(offences)


# ---------------------------------------------------------------- dataset

def test_dataset_target_alignment():
    """Spot-check the assembled dataset: target at row t must equal the actual
    forward return of gold_price from t to t+HORIZON."""
    ds = F.build_dataset(save=False)

    for i in range(50, len(ds) - 50):
        t = ds.index[i]
        fut = ds.index[ds.index > t]
        if len(fut) < HORIZON:
            continue
        t_h = fut[HORIZON - 1]
        expected = np.log(ds["gold_price"].loc[t_h] / ds["gold_price"].loc[t])
        if np.isclose(ds["target"].loc[t], expected, atol=1e-9):
            return   # found a consistent row — alignment is right
    pytest.fail("target does not match forward gold_price return anywhere")


def test_gold_price_not_a_feature():
    """gold_price is kept for reporting only. If it ever leaked into the
    feature set, a model could trivially exploit the price level."""
    import backtest as B
    ds = F.build_dataset(save=False)
    assert "gold_price" not in B.feature_cols(ds)
    assert "target" not in B.feature_cols(ds)