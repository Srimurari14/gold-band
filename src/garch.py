# src/garch.py
"""Volatility models and the weekly band.

The band — not the point forecast — is the deliverable.

Distribution handling: we do NOT assume a parametric shape for returns.
Two prior attempts failed:
  1. Student-t with df=5 hardcoded  -> 80% coverage ~74%
  2. Student-t with df fitted (~4.4) -> 80% coverage ~74%
The fitted df was already right, so tails weren't the problem. And 95%
coverage passes while 80% fails — which means the extremes are fine and the
MIDDLE of the distribution is mis-shaped. Gold returns are peakier than any
t: too many small moves, too many huge ones, not enough medium ones.

So: filtered historical simulation. Standardise past returns by their vol
forecast, then read the quantiles straight off the empirical distribution.
No shape assumed. This is what practitioners use, for exactly this reason.
"""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from config import PROCESSED, HORIZON

warnings.filterwarnings("ignore")

SCALE = 100.0


def load_returns():
    """Daily gold log returns, in percent."""
    ds = pd.read_parquet(PROCESSED / "dataset.parquet")
    return (ds["gold_ret_1d"] * SCALE).dropna(), ds["gold_price"]


# ---------------------------------------------------------------- vol models
# Each returns the HORIZON-day vol forecast in percent, fitted on `past` only.


def _h_returns(past, horizon=HORIZON):
    """Non-overlapping h-day cumulative returns."""
    return past.rolling(horizon).sum().iloc[horizon - 1::horizon].dropna()


def vol_rolling(past, horizon=HORIZON, window=20):
    """Baseline: realised h-day vol over recent non-overlapping windows."""
    return _h_returns(past, horizon).iloc[-window:].std()


def vol_ewma(past, horizon=HORIZON, lam=0.94):
    """EWMA on h-day returns."""
    return _h_returns(past, horizon).ewm(alpha=1 - lam).std().iloc[-1]


def _garch_h_vol(res, horizon=HORIZON):
    """Sum the forecast variance path — accounts for clustering decay."""
    f = res.forecast(horizon=horizon, reindex=False)
    return float(np.sqrt(f.variance.values[-1, :].sum()))


def vol_garch(past, horizon=HORIZON):
    m = arch_model(past, vol="Garch", p=1, q=1, dist="t", mean="Zero")
    res = m.fit(disp="off", show_warning=False)
    return _garch_h_vol(res, horizon)


def vol_gjr(past, horizon=HORIZON):
    m = arch_model(past, vol="Garch", p=1, o=1, q=1, dist="t", mean="Zero")
    res = m.fit(disp="off", show_warning=False)
    return _garch_h_vol(res, horizon)


VOL_MODELS = {
    "rolling20": vol_rolling,
    "ewma":      vol_ewma,
    "garch":     vol_garch,
    "gjr":       vol_gjr,
}


# ---------------------------------------------------------------- quantiles

def empirical_quantiles(past, h_vol, conf, horizon=HORIZON):
    """Filtered historical simulation.

    Standardise past h-day returns by a trailing vol estimate, take the
    empirical quantiles of those standardised shocks, rescale by the CURRENT
    vol forecast. Shape comes from history; scale comes from the model.
    """
    hr = _h_returns(past, horizon)
    if len(hr) < 40:
        return None

    # trailing vol of h-day returns, causal (shifted)
    trail = hr.rolling(20).std().shift(1)
    z = (hr / trail).dropna()
    if len(z) < 30:
        return None

    lo_q = np.quantile(z, (1 - conf) / 2)
    hi_q = np.quantile(z, 1 - (1 - conf) / 2)
    return lo_q * h_vol, hi_q * h_vol


def parametric_quantiles(past, h_vol, conf, horizon=HORIZON):
    """Student-t with df fitted from kurtosis. The approach that failed at 80%.
    Kept as a comparison."""
    z = past / past.std()
    k = stats.kurtosis(z, fisher=False, bias=False)
    df = float(np.clip(4 + 6 / max(k - 3, 0.1), 2.5, 30.0)) if k > 3.1 else 30.0
    df = max(df, 2.1)
    q = stats.t.ppf(0.5 + conf / 2, df) / np.sqrt(df / (df - 2))
    return -q * h_vol, q * h_vol


QUANTILE_METHODS = {
    "empirical":  empirical_quantiles,
    "parametric": parametric_quantiles,
}


def band(last_price, lo_shock, hi_shock, drift=0.0):
    """Price band from lower/upper shock quantiles (in percent)."""
    center = np.log(last_price) + drift
    return (
        float(np.exp(center + lo_shock / SCALE)),
        float(np.exp(center + hi_shock / SCALE)),
    )


# ---------------------------------------------------------------- backtest

def backtest_coverage(returns, prices, vol_fn, q_fn, min_train=500,
                      step=HORIZON, conf=0.80, refit_every=20):
    """Walk-forward: fit on past only, forecast band, check if price landed in."""
    rows = []
    last_vol = None

    for i in range(min_train, len(returns) - step, step):
        past = returns.iloc[:i]

        if last_vol is None or i % refit_every < step:
            try:
                last_vol = vol_fn(past)
            except Exception:
                continue

        q = q_fn(past, last_vol, conf, step)
        if q is None:
            continue
        lo_shock, hi_shock = q

        drift = past.mean() * step / SCALE
        p0 = prices.iloc[i - 1]
        p1 = prices.iloc[i - 1 + step]
        lo, hi = band(p0, lo_shock, hi_shock, drift=drift)

        rows.append({
            "date": returns.index[i - 1],
            "p0": p0, "p1": p1, "lo": lo, "hi": hi,
            "inside": lo <= p1 <= hi,
            "width_pct": (hi - lo) / p0 * 100,
            "h_vol": last_vol,
        })

    return pd.DataFrame(rows).set_index("date")


def kupiec_pof(n_inside, n_total, conf=0.80):
    """Kupiec POF test. p<0.05 => coverage significantly wrong."""
    x, n = n_total - n_inside, n_total
    p = 1 - conf
    if x == 0:
        return n_inside / n, 0.0, 1.0
    pi = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
               - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return n_inside / n, float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_independence(inside):
    """Are the breaches CLUSTERED or independent?

    A band can have perfect average coverage and still be useless if all its
    failures happen in one bad month. Tests whether a breach today predicts a
    breach next period. p<0.05 => breaches cluster => band doesn't adapt.
    """
    b = (~inside.astype(bool)).astype(int).values
    if len(b) < 10 or b.sum() < 2:
        return np.nan, np.nan

    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan, np.nan

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi in (0, 1) or pi01 in (0, 1) or pi11 in (0, 1):
        return np.nan, np.nan

    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
              + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


if __name__ == "__main__":
    returns, prices = load_returns()
    prices = prices.reindex(returns.index)

    print(f"returns: {len(returns)} days  {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"daily vol: {returns.std():.3f}%   annualised: {returns.std() * np.sqrt(252):.1f}%")
    print(f"kurtosis:  {stats.kurtosis(returns / returns.std(), fisher=False, bias=False):.2f}  (3.0 = normal)")

    # --- why parametric fails: compare empirical vs t quantiles of h-day shocks
    hr = _h_returns(returns)
    z = (hr / hr.rolling(20).std().shift(1)).dropna()
    print(f"\nstandardised {HORIZON}d shocks — empirical vs Student-t(4.4) quantiles")
    print(f"{'conf':>6s} {'empirical':>11s} {'t(4.4)':>9s} {'ratio':>7s}")
    for c in [0.80, 0.90, 0.95, 0.99]:
        e = np.quantile(np.abs(z), c)
        t = stats.t.ppf(0.5 + c / 2, 4.4) / np.sqrt(4.4 / 2.4)
        print(f"{c:6.0%} {e:11.3f} {t:9.3f} {e / t:7.2f}")
    print("(ratio > 1 => t is too narrow at that level)")

    for conf in [0.80, 0.95]:
        print("\n" + "=" * 86)
        print(f"COVERAGE — claimed {conf:.0%}, {HORIZON}-day horizon")
        print("=" * 86)
        print(f"{'vol model':12s} {'quantiles':12s} {'coverage':>9s} {'n':>5s} "
              f"{'width':>8s} {'kupiec p':>9s} {'indep p':>8s} {'verdict':>16s}")
        print("-" * 86)

        for q_name, q_fn in QUANTILE_METHODS.items():
            for v_name, v_fn in VOL_MODELS.items():
                bt = backtest_coverage(returns, prices, v_fn, q_fn, conf=conf)
                if len(bt) == 0:
                    continue
                cov, _, p = kupiec_pof(bt["inside"].sum(), len(bt), conf=conf)
                _, ip = christoffersen_independence(bt["inside"])

                if p < 0.05:
                    verdict = "MISCALIBRATED"
                elif abs(cov - conf) < 0.03:
                    verdict = "well calibrated"
                else:
                    verdict = "ok"

                ip_s = f"{ip:8.3f}" if not np.isnan(ip) else "     n/a"
                print(f"{v_name:12s} {q_name:12s} {cov:8.1%} {len(bt):5d} "
                      f"{bt['width_pct'].mean():7.2f}% {p:9.3f} {ip_s} {verdict:>16s}")
            print()

    print("-" * 86)
    print("  kupiec p  — is average coverage right?  p<0.05 = no")
    print("  indep p   — are breaches spread out, or clustered in bad months?")
    print("              p<0.05 = clustered = band doesn't adapt fast enough")

    # ---- live band
    print("\n" + "=" * 86)
    print("LIVE BAND — empirical quantiles")
    print("=" * 86)
    p_now = prices.iloc[-1]
    drift_now = returns.mean() * HORIZON / SCALE
    print(f"gold now: ${p_now:,.2f}   as of {prices.index[-1].date()}\n")

    for v_name, v_fn in VOL_MODELS.items():
        v = v_fn(returns)
        for conf in [0.80, 0.95]:
            q = empirical_quantiles(returns, v, conf)
            if q is None:
                continue
            lo, hi = band(p_now, q[0], q[1], drift=drift_now)
            print(f"  {v_name:12s} {conf:.0%}  ${lo:,.0f} – ${hi:,.0f}   "
                  f"(-{(1 - lo / p_now) * 100:.1f}% / +{(hi / p_now - 1) * 100:.1f}%)")
        print()