# src/daily_forecast_diagnostics.py
"""Robustness checks on daily_forecast.py's apparent winners.

A win this thin (vs_naive ~0.002) needs checking before it goes anywhere
near the dashboard:
  1. Does it survive removing day-of-week/month? Trees can carve calendar
     features into noise-fitting splits that look like skill in-sample.
  2. Is it consistent across time, or concentrated in one stretch?
If either check fails, the "win" was noise wearing a low p-value.
"""

import numpy as np
import pandas as pd

from daily_forecast import load_daily_dataset, walk_forward, MODELS

CANDIDATES = ["catboost", "rf", "xgboost"]

ds = load_daily_dataset()
ds_no_cal = ds.drop(columns=["dow", "month"])


def summarize(w):
    acc = (np.sign(w["actual"]) == np.sign(w["pred"])).mean()
    naive = pd.Series(0.0, index=w.index)
    ss_m = ((w["actual"] - w["pred"]) ** 2).sum()
    ss_n = ((w["actual"] - naive) ** 2).sum()
    return acc, 1 - ss_m / ss_n


print("=" * 78)
print("CHECK 1 — drop day-of-week / month, does the edge survive?")
print("=" * 78)

full_results = {}
for name in CANDIDATES:
    if name not in MODELS:
        continue
    wf_full = walk_forward(ds, MODELS[name])
    wf_nocal = walk_forward(ds_no_cal, MODELS[name])
    full_results[name] = wf_full

    a_full, r_full = summarize(wf_full)
    a_nocal, r_nocal = summarize(wf_nocal)

    print(f"\n{name}:")
    print(f"  with calendar features:    dir_acc {a_full:.4f}  vs_naive {r_full:+.4f}")
    print(f"  without calendar features: dir_acc {a_nocal:.4f}  vs_naive {r_nocal:+.4f}")

    drop = a_full - a_nocal
    if drop > 0.015:
        print(f"  -> edge shrinks {drop*100:.1f}pp without dow/month. Looks like")
        print(f"     a calendar artifact, not real signal.")
    else:
        print(f"  -> holds without dow/month ({drop*100:+.1f}pp change).")

print("\n" + "=" * 78)
print("CHECK 2 — stable across time, or concentrated in one stretch?")
print("=" * 78)

for name, wf in full_results.items():
    mid = len(wf) // 2
    a1, r1 = summarize(wf.iloc[:mid])
    a2, r2 = summarize(wf.iloc[mid:])

    print(f"\n{name}  (n={len(wf)}, split at {wf.index[mid].date()})")
    print(f"  first half:  dir_acc {a1:.4f}  vs_naive {r1:+.4f}")
    print(f"  second half: dir_acc {a2:.4f}  vs_naive {r2:+.4f}")

    if (a1 > 0.5) != (a2 > 0.5) or abs(a1 - a2) > 0.04:
        print(f"  -> inconsistent across halves — one regime, not a persistent edge.")
    else:
        print(f"  -> holds up in both halves.")

print("\n" + "-" * 78)
print("A model only earns the dashboard if it survives BOTH checks.")
print("If not, it's a documented negative result — same treatment as the")
print("weekly factor model.")
print("-" * 78)