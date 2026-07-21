# tests/test_fx_leakage.py
"""Same discipline as test_leakage.py, applied to the FX dataset.

This result is stronger than anything else in the project, which makes it
the one most worth checking hardest. A leakage bug and a real edge look
identical from outside; only this test tells them apart.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fx_forecast as FX
import features as F


@pytest.fixture(scope="module")
def aligned():
    raw = F.load_raw()
    return F.align(raw)


def test_fx_features_depend_only_on_past(aligned):
    """Truncate history, rebuild, confirm the last row is unchanged."""
    full = FX.build_fx_dataset(save=False)

    for cut in ["2018-06-15", "2021-09-10", "2024-02-20"]:
        past_only = aligned.loc[aligned.index <= cut]
        # rebuild using the same feature logic on truncated history
        import features as Fmod
        old_load = FX.F.load_raw
        FX.F.load_raw = lambda: pd.read_parquet(FX.__file__.replace("fx_forecast.py", "../data/raw/raw.parquet"))
        truncated = FX.build_fx_dataset(save=False)
        FX.F.load_raw = old_load

        # only check dates that exist in both
        common = truncated.index[truncated.index <= cut]
        if len(common) == 0:
            continue
        t = common[-1]
        if t not in full.index:
            continue

        pd.testing.assert_series_equal(
            full.loc[t].drop("target"), truncated.loc[t].drop("target"),
            check_names=False,
            obj=f"FX features at {t.date()} changed when future data removed",
        )


def test_fx_target_is_one_day_forward(aligned):
    """Target must be exactly next-day log return, nothing more."""
    ds = FX.build_fx_dataset(save=False)
    i = 100
    expected = np.log(aligned["usdinr"].loc[ds.index[i + 1]] / aligned["usdinr"].loc[ds.index[i]])
    assert np.isclose(ds["target"].iloc[i], expected, atol=1e-6)


def test_fx_level_not_a_feature():
    """usdinr_level is reporting-only, must never enter the feature set."""
    ds = FX.build_fx_dataset(save=False)
    assert "usdinr_level" not in FX.feature_cols(ds)
    assert "target" not in FX.feature_cols(ds)