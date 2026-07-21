# src/fx_forecast_diagnostics.py
"""Robustness checks on fx_forecast.py's apparent winners.

Four of five models beat naive by 3-5%, much stronger than anything seen in
the gold tests. Before trusting that, same two checks as the gold daily
forecast:
  1. Does it survive removing day-of-week/month?
  2. Is it consistent across time, or concentrated in one stretch?
A real hypothesis for why this might differ from gold: USD/INR isn't freely
floating, the RBI intervenes to smooth it, which could create genuine serial
correlation that free-floating currencies (the ones Meese-Rogoff studied)
don't have. That's a reason to take this seriously, not a reason to skip
checking it.
"""

import numpy as np
import pandas as pd

from fx_forecast import build_fx_dataset, walk_forward, MODELS, feature_cols

CANDIDATES = ["rf", "ridge", "catboost", "xgboost"]

ds = build_fx_dataset(save=False)
ds_no_cal = ds.drop(columns=["dow", "month"])


def summarize(w):
    acc = (np.sign(w["actual"]) == np.sign(w["pred"])).mean()
    naive = pd.Series(0.0, index=w.index)
    ss_m = ((w["actual"] - w["pred"]) ** 2).sum()
    ss_n = ((w["actual"] - naive) ** 2).sum()
    return acc, 1 - ss_m / ss_n


print("=" * 78)
print("CHECK 1 -- drop day-of-week / month, does the edge survive?")
print("=" * 78)

full_results = {}
for name in CANDIDATES:
    wf_full = walk_forward(ds, MODELS[name])
    wf_nocal = walk_forward(ds_no_cal, MODELS[name])
    full_results[name] = wf_full

    a_full, r_full = summarize(wf_full)
    a_nocal, r_nocal = summarize(wf_nocal)

    print(f"\n{name}:")
    print(f"  with calendar features:    dir_acc {a_full:.4f}  vs_naive {r_full:+.4f}")
    print(f"  without calendar features: dir_acc {a_nocal:.4f}  vs_naive {r_nocal:+.4f}")

    drop = r_full - r_nocal
    if drop > 0.015:
        print(f"  -> vs_naive drops {drop:+.4f} without dow/month. Partly a calendar artifact.")
    else:
        print(f"  -> holds without dow/month ({drop:+.4f} change). Not calendar-driven.")

print("\n" + "=" * 78)
print("CHECK 2 -- stable across time, or concentrated in one stretch?")
print("=" * 78)

for name, wf in full_results.items():
    mid = len(wf) // 2
    a1, r1 = summarize(wf.iloc[:mid])
    a2, r2 = summarize(wf.iloc[mid:])

    print(f"\n{name}  (n={len(wf)}, split at {wf.index[mid].date()})")
    print(f"  first half:  dir_acc {a1:.4f}  vs_naive {r1:+.4f}")
    print(f"  second half: dir_acc {a2:.4f}  vs_naive {r2:+.4f}")

    if (r1 > 0) != (r2 > 0):
        print(f"  -> vs_naive changes SIGN between halves. Not a persistent edge yet.")
    elif abs(r1 - r2) > 0.04:
        print(f"  -> same sign but sizes differ a lot ({r1:+.4f} vs {r2:+.4f}). Inconsistent.")
    else:
        print(f"  -> holds up in both halves, similar magnitude. Consistent.")

print("\n" + "-" * 78)
print("A model earns real trust only if vs_naive stays positive in BOTH halves,")
print("with similar magnitude, and survives dropping the calendar features.")
print("That is a higher bar than what gold's daily test cleared, and this")
print("result needs to clear it too before it goes anywhere near the dashboard.")
print("-" * 78)