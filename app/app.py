# app/app.py
"""Gold band dashboard.

USD throughout. The model is USD-native — converting to INR would add
exchange-rate error and a hardcoded duty assumption without adding
information. Local rates are a lookup; the range is the thing this provides.

Price refreshes every 30s. The band does not — GARCH is fitted on daily
closes and does not meaningfully move intraday.

Two audiences: a jeweller who needs the range in plain words, and anyone
checking whether the range is trustworthy. The first gets the page; the
second gets the expanders.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import HORIZON, PROCESSED, RAW, ROOT
import garch as G

st.set_page_config(
    page_title="Gold Weekly Band",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=30_000, key="price_refresh")

# ---------------------------------------------------------------- constants

OZ_TO_GRAM = 31.1035

GOLD = "#c9a227"
GOLD_DIM = "rgba(201, 162, 39, 0.16)"
INK = "#1c1917"
MUTED = "#78716c"
LINE = "#e7e5e4"
RED = "#b91c1c"
GREEN = "#15803d"

BACKTEST = {
    0.80: {"coverage": "80.0%", "kupiec": "0.98", "indep": "0.41"},
    0.95: {"coverage": "94.3%", "kupiec": "0.50", "indep": "0.07"},
}

PRED_LOG = ROOT / "logs" / "predictions.csv"

# ---------------------------------------------------------------- style

st.markdown(f"""
<style>
  #MainMenu, footer, header {{visibility: hidden;}}

  .block-container {{
      padding-top: 2.5rem;
      padding-bottom: 4rem;
      max-width: 1180px;
  }}

  html, body, [class*="css"] {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }}

  .eyebrow {{
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: {MUTED};
      margin-bottom: 0.35rem;
  }}

  .display {{
      font-size: 2.8rem;
      font-weight: 650;
      letter-spacing: -0.03em;
      color: {INK};
      line-height: 1.05;
      font-variant-numeric: tabular-nums;
  }}

  .band-figure {{
      font-size: 2.4rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: {INK};
      font-variant-numeric: tabular-nums;
  }}

  .sub {{
      font-size: 0.82rem;
      color: {MUTED};
      margin-top: 0.2rem;
  }}

  .delta-up {{ color: {GREEN}; font-weight: 600; font-size: 0.9rem; }}
  .delta-down {{ color: {RED}; font-weight: 600; font-size: 0.9rem; }}

  .card {{
      border: 1px solid {LINE};
      border-radius: 10px;
      padding: 1.15rem 1.3rem;
      background: #fff;
      height: 100%;
  }}

  .band-card {{
      border: 1px solid {LINE};
      border-left: 3px solid {GOLD};
      border-radius: 10px;
      padding: 1.4rem 1.5rem;
      background: linear-gradient(180deg, rgba(201,162,39,0.04), transparent);
  }}

  .pill {{
      display: inline-block;
      padding: 0.3rem 0.75rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.02em;
  }}

  .live-dot {{
      display: inline-block;
      width: 6px; height: 6px;
      border-radius: 50%;
      background: {GREEN};
      margin-right: 5px;
      animation: pulse 2s ease-in-out infinite;
  }}

  @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.35; }}
  }}

  .rule {{
      border: none;
      border-top: 1px solid {LINE};
      margin: 2.5rem 0 1.75rem 0;
  }}

  .trust {{
      text-align: center;
      font-size: 0.88rem;
      color: {MUTED};
      padding: 0.9rem 1rem;
      background: #fafaf9;
      border-radius: 8px;
      border: 1px solid {LINE};
  }}

  .unit-row {{
      display: flex;
      gap: 1.5rem;
      margin-top: 0.55rem;
  }}

  .unit {{
      font-size: 0.95rem;
      font-weight: 600;
      color: {INK};
      font-variant-numeric: tabular-nums;
  }}

  .unit-label {{
      font-size: 0.72rem;
      color: {MUTED};
  }}

  table {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data

@st.cache_data(ttl=30)
def get_live_price():
    import yfinance as yf
    gold = yf.Ticker("GC=F").history(period="5d")["Close"]
    return float(gold.iloc[-1]), gold.index[-1]


@st.cache_data(ttl=3600)
def get_history():
    """Feature dataset. Ends ~HORIZON days behind today by design — the
    forward target isn't knowable yet for the most recent rows."""
    return pd.read_parquet(PROCESSED / "dataset.parquet")


@st.cache_data(ttl=3600)
def get_raw_asof():
    """Date of the newest RAW observation — the real freshness measure."""
    try:
        raw = pd.read_parquet(RAW / "raw.parquet")
        return raw["gold"].dropna().index.max()
    except Exception:
        return None


@st.cache_data(ttl=86400)
def fit_band(conf):
    returns, _ = G.load_returns()
    vol = G.vol_gjr(returns)
    q = G.empirical_quantiles(returns, vol, conf)
    drift = returns.mean() * HORIZON / G.SCALE
    return vol, q, drift


@st.cache_data(ttl=86400)
def vol_context():
    """Current vol, its decade percentile, and whether it's rising or easing."""
    returns, _ = G.load_returns()
    hist = returns.rolling(20).std().dropna()
    current = float(hist.iloc[-1])
    pctile = float((hist < current).mean())
    month_ago = float(hist.iloc[-21]) if len(hist) > 21 else current
    trend = (current / month_ago - 1) if month_ago > 0 else 0.0
    return current, pctile, trend


@st.cache_data(ttl=3600)
def live_track_record(conf):
    """Resolved out-of-sample predictions from the nightly log, if any.

    Returns (n_resolved, coverage, n_breaches_last10) or None. This is the
    strongest evidence on the page once it accumulates — predictions made
    before the outcome existed, unlike the backtest.
    """
    try:
        df = pd.read_csv(PRED_LOG)
    except Exception:
        return None
    sub = df[(df["conf"] == conf) & df["inside"].notna()]
    if len(sub) == 0:
        return None
    inside = sub["inside"].astype(bool)
    last10 = inside.iloc[-10:]
    return len(sub), float(inside.mean()), int((~last10).sum()), len(last10)


@st.cache_resource
def build_chart(hist_idx, hist_vals, spot, lo, hi, centre_end, last_date, conf):
    future = pd.bdate_range(last_date, periods=HORIZON + 1)[1:]
    fig = go.Figure()

    cone_x = [last_date] + list(future)
    cone_hi = [spot] + list(np.linspace(spot, hi, HORIZON + 1)[1:])
    cone_lo = [spot] + list(np.linspace(spot, lo, HORIZON + 1)[1:])

    fig.add_trace(go.Scatter(
        x=cone_x + cone_x[::-1],
        y=cone_hi + cone_lo[::-1],
        fill="toself", fillcolor=GOLD_DIM,
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=hist_idx, y=hist_vals,
        mode="lines", line=dict(color=GOLD, width=2),
        name="Gold",
        hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[last_date, future[-1]], y=[spot, centre_end],
        mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
        name="Centre", hoverinfo="skip",
    ))

    for y, lbl in [(hi, f"${hi:,.0f}"), (lo, f"${lo:,.0f}")]:
        fig.add_annotation(x=future[-1], y=y, text=f"<b>{lbl}</b>",
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(size=12, color=GOLD))

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=72, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(showgrid=False, showline=True, linecolor=LINE,
                   tickfont=dict(size=11, color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False,
                   tickfont=dict(size=11, color=MUTED),
                   tickprefix="$", side="right"),
        font=dict(family="Inter, sans-serif"),
        transition=dict(duration=0),
    )
    return fig


# ---------------------------------------------------------------- load

try:
    spot, asof = get_live_price()
    is_live = True
except Exception as e:
    ds_t = get_history()
    spot, asof = ds_t["gold_price"].iloc[-1], ds_t.index[-1]
    is_live = False
    st.error(f"Live price unavailable — showing last cached close. ({e})")

ds = get_history()
raw_asof = get_raw_asof()
prev = ds["gold_price"].iloc[-1]
week_ago = ds["gold_price"].iloc[-6] if len(ds) > 6 else prev

d_day = (spot / prev - 1) * 100
d_week = (spot / week_ago - 1) * 100

per_g = spot / OZ_TO_GRAM
per_10g = per_g * 10


# ---------------------------------------------------------------- header

st.markdown(
    f"<div class='eyebrow'>Gold · next {HORIZON} trading days</div>"
    f"<div style='font-size:1.9rem;font-weight:680;letter-spacing:-0.035em;"
    f"color:{INK};margin-bottom:0.15rem;'>What it might cost next week</div>"
    f"<div class='sub'>A range, not a prediction. Nobody knows which way gold "
    f"will move — but how far it can move is measurable.</div>",
    unsafe_allow_html=True,
)

st.write("")

def delta_html(v, label):
    cls = "delta-up" if v >= 0 else "delta-down"
    arrow = "▲" if v >= 0 else "▼"
    return f"<span class='{cls}'>{arrow} {abs(v):.1f}%</span> <span class='sub'>{label}</span>"

h1, h2 = st.columns([1.15, 1])

with h1:
    live_badge = (f"<span class='live-dot'></span><span class='sub'>live</span>"
                  if is_live else "<span class='sub'>cached</span>")
    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div class='eyebrow' style='margin-bottom:0'>Price now · USD</div>{live_badge}</div>"
        f"<div class='display' style='margin-top:0.4rem'>${spot:,.2f}"
        f"<span style='font-size:1rem;font-weight:500;color:{MUTED}'> /oz</span></div>"
        f"<div class='unit-row'>"
        f"<div><div class='unit'>${per_g:,.2f}</div><div class='unit-label'>per gram</div></div>"
        f"<div><div class='unit'>${per_10g:,.2f}</div><div class='unit-label'>per 10g</div></div>"
        f"</div>"
        f"<div style='margin-top:0.55rem'>{delta_html(d_day, 'vs last close')} "
        f"&nbsp;&nbsp;·&nbsp;&nbsp; {delta_html(d_week, 'this week')}</div></div>",
        unsafe_allow_html=True,
    )

with h2:
    lo_120 = float(ds["gold_price"].iloc[-120:].min())
    hi_120 = float(ds["gold_price"].iloc[-120:].max())
    lo_ext, hi_ext = min(lo_120, spot), max(hi_120, spot)
    pos = (spot - lo_ext) / (hi_ext - lo_ext) * 100 if hi_ext > lo_ext else 50

    if pos <= 2:
        ctx = "Today's price is at the bottom of that range."
    elif pos >= 98:
        ctx = "Today's price is at the top of that range."
    else:
        ctx = f"Today's price sits {pos:.0f}% of the way up that range."

    dd = (spot / hi_ext - 1) * 100

    st.markdown(
        f"<div class='card'><div class='eyebrow'>Where prices have been · past 6 months</div>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"margin-top:0.5rem'>"
        f"<span style='font-size:1.15rem;font-weight:600;color:{MUTED};"
        f"font-variant-numeric:tabular-nums'>${lo_ext:,.0f}</span>"
        f"<span style='font-size:1.15rem;font-weight:600;color:{MUTED};"
        f"font-variant-numeric:tabular-nums'>${hi_ext:,.0f}</span></div>"
        f"<div style='position:relative;height:6px;background:{LINE};border-radius:3px;"
        f"margin:0.7rem 0 0.5rem'>"
        f"<div style='position:absolute;left:calc({pos:.1f}% - 1.5px);top:-4px;"
        f"width:3px;height:14px;background:{GOLD};border-radius:2px;'></div></div>"
        f"<div class='sub'>{ctx} Down {abs(dd):.0f}% from the peak. This looks "
        f"backward — it says nothing about where prices go next.</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='rule'>", unsafe_allow_html=True)


# ---------------------------------------------------------------- band

bl, br = st.columns([2.2, 1])

with br:
    conf = st.radio("How sure do you want to be?", options=[0.80, 0.95],
                    format_func=lambda x: f"{x:.0%}", horizontal=True)

vol, q, drift = fit_band(conf)
lo, hi = G.band(spot, q[0], q[1], drift=drift)
cur_vol, pctile, vol_trend = vol_context()

weeks = "4 weeks out of 5" if conf == 0.80 else "19 weeks out of 20"

with bl:
    st.markdown(
        f"<div class='band-card'>"
        f"<div class='eyebrow'>Next week, gold will most likely be between</div>"
        f"<div class='band-figure'>${lo:,.0f} &nbsp;—&nbsp; ${hi:,.0f}"
        f"<span style='font-size:1rem;font-weight:500;color:{MUTED}'> /oz</span></div>"
        f"<div class='unit-row'>"
        f"<div><div class='unit'>${lo / OZ_TO_GRAM * 10:,.0f} — "
        f"${hi / OZ_TO_GRAM * 10:,.0f}</div>"
        f"<div class='unit-label'>per 10g</div></div>"
        f"<div><div class='unit'>−{(1 - lo / spot) * 100:.1f}% / "
        f"+{(hi / spot - 1) * 100:.1f}%</div>"
        f"<div class='unit-label'>from today's price</div></div>"
        f"</div>"
        f"<div class='sub' style='margin-top:0.55rem;color:{INK}'>"
        f"Right about <b>{weeks}</b>. The rest of the time it goes outside."
        f"</div></div>",
        unsafe_allow_html=True,
    )

with br:
    if pctile > 0.7:
        bg, fg, word = "rgba(185,28,28,0.09)", RED, "Choppy"
        note = ("Prices are jumping around more than usual. The range is wide. "
                "Think about buying in smaller lots.")
    elif pctile < 0.3:
        bg, fg, word = "rgba(21,128,61,0.09)", GREEN, "Calm"
        note = ("Prices have been steady. The range is tight. A safer window to "
                "commit to a bigger purchase.")
    else:
        bg, fg, word = "rgba(120,113,108,0.09)", MUTED, "Normal"
        note = "Prices are moving about as much as they usually do."

    if vol_trend > 0.15:
        trend_txt = "Choppiness is <b>rising</b> vs a month ago."
    elif vol_trend < -0.15:
        trend_txt = "Choppiness is <b>easing</b> vs a month ago."
    else:
        trend_txt = "Choppiness is steady vs a month ago."

    st.markdown(
        f"<div style='margin-top:0.2rem'>"
        f"<span class='pill' style='background:{bg};color:{fg}'>{word}</span>"
        f"<div class='sub' style='margin-top:0.6rem;line-height:1.5'>{note}<br>"
        f"{trend_txt}</div></div>",
        unsafe_allow_html=True,
    )

if raw_asof is not None:
    stale = (pd.Timestamp.now().normalize() - raw_asof.tz_localize(None)).days
    if stale > 4:
        st.caption(f"⚠️ Market data last updated {raw_asof:%d %b %Y} "
                   f"({stale} days ago). Something may have broken.")

st.write("")


# ---------------------------------------------------------------- chart

hist = ds["gold_price"].iloc[-130:]
centre_end = spot * np.exp(drift)

fig = build_chart(
    tuple(hist.index), tuple(hist.values),
    round(spot, 2), round(lo, 2), round(hi, 2), round(centre_end, 2),
    hist.index[-1], conf,
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown(
    f"<div class='sub' style='text-align:center'>The gold line is the last six "
    f"months. The shaded cone on the right is next week's range — it widens because "
    f"the further ahead you look, the less certain things get.</div>",
    unsafe_allow_html=True,
)

st.write("")
st.write("")


# ---------------------------------------------------------------- trust line

track = live_track_record(conf)

if track is not None:
    n, cov, breaches10, n10 = track
    st.markdown(
        f"<div class='trust'>Backtested over ten years: claimed <b>{conf:.0%}</b>, "
        f"delivered <b>{BACKTEST[conf]['coverage']}</b>. &nbsp;·&nbsp; "
        f"Running live since Jul 2026: <b>{cov:.0%}</b> over {n} resolved "
        f"week{'s' if n != 1 else ''}"
        + (f", with {breaches10} miss{'es' if breaches10 != 1 else ''} in the "
           f"last {n10}" if n10 > 0 else "")
        + ".</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='trust'>Checked against ten years of history: when this said "
        f"<b>{conf:.0%}</b>, it was right <b>{BACKTEST[conf]['coverage']}</b> of "
        f"the time. A live track record is accumulating — each day's range is "
        f"logged before the outcome is known, and scored a week later.</div>",
        unsafe_allow_html=True,
    )

st.write("")


# ---------------------------------------------------------------- explainers

with st.expander("How to read this"):
    st.markdown(f"""
**The price at the top** is what gold costs right now in US dollars — shown per
ounce, per gram, and per 10 grams (they're the same price in different units:
one troy ounce is 31.1 grams). It updates every 30 seconds while markets are
open. Your local rate will be higher — duties, taxes and dealer margins add on
top — but when this moves, yours moves with it.

**"Where prices have been"** shows the lowest and highest gold has traded over
the past six months, and where today sits between them. It looks *backward*.
A price at the bottom of its range is not a signal it will bounce — that's the
single most tempting mistake this page can invite, so it's worth saying twice.

**The big range is the point of this page.** Next week, gold will most likely
end up somewhere between those two numbers. Not exactly where — somewhere in
there. It's shown per ounce and per 10g.

**The 80% / 95% choice.** At 80%, gold stays inside the range about 4 weeks
out of 5. One week in five it escapes — that's not the tool failing, that's
what 80% means. At 95% it's almost always right, but the range gets so wide it
stops being useful. Use **80%** day to day. Check **95%** before committing to
something big, to see the worst case.

**Calm / Normal / Choppy** is how jumpy prices have been lately, plus whether
that's rising or easing. Calm and easing → safer to commit to a large purchase.
Choppy and rising → smaller lots, or wait.

**The chart.** The dotted centre line is **not** a prediction — it's today's
price drawn forward, because nothing predicts better than that.

---

**The one thing to remember:** this will never tell you gold is going up or
down. Nobody can tell you that honestly. It tells you **how much it could
move** — which is what you need when deciding whether to buy now, buy small,
or wait.

Anyone who says they know where gold is going next week is guessing. This
tells you how big the guess is.
""")

with st.expander("Is this trustworthy? — the testing behind it"):
    bt = BACKTEST[conf]
    st.markdown(f"""
Two things were built. One works. One doesn't, and that's worth saying out loud.

**The range works.** Tested on 474 separate weeks between 2017 and 2026 — each
time using only the data that existed at that moment, then checking what gold
actually did afterwards.

| Claimed | Actually landed inside | Verdict |
|---|---|---|
| 80% | 80.0% | matches |
| 95% | 94.3% | matches |

The statistical test for miscalibration (Kupiec) returns p = {bt['kupiec']} —
no evidence the range lies about its own confidence. A second test
(Christoffersen, p = {bt['indep']}) confirms misses are spread over time rather
than bunched into a few bad months.

On top of the backtest, a **live log** runs nightly: each day's range is
recorded *before* the outcome exists, then scored a week later. That record
appears above as it accumulates — it's the strongest evidence possible, because
unlike a backtest, nothing about it can be tuned after the fact.

**Predicting the direction does not work.** A model was trained on 18 things
that supposedly drive gold — the dollar index, real interest rates, the VIX
fear gauge, inflation expectations, silver, momentum, seasonality — and tested
the same honest way.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Guessing the direction right**")
        st.table(pd.DataFrame({
            "Accuracy": ["49.7%", "47.5%", "50.0%"],
        }, index=pd.Index(["Ridge model", "LightGBM model", "A coin flip"], name="")))
    with c2:
        st.markdown("**vs. just assuming no change**")
        st.table(pd.DataFrame({
            "Result": ["6.7% worse", "12.3% worse"],
        }, index=pd.Index(["Ridge model", "LightGBM model"], name="")))

    st.markdown("""
Both lost to a coin flip. The more sophisticated model lost by more — it fit
the past beautifully (R² = 0.60) and the future not at all (R² = −0.06):
textbook overfitting, memorising history instead of learning a rule.

That negative result is why the centre line on the chart is just today's
price. Nothing beat it, so nothing replaced it.

**Why direction fails but range works:** which way gold moves next week is
close to random — the past genuinely doesn't tell you. But *how much* it moves
is persistent. Turbulent weeks follow turbulent weeks; calm follows calm. That
pattern is real and measurable, and it's the only thing this page claims to
know.
""")

with st.expander("Limitations"):
    st.markdown(
        f"- **The range is wide.** Currently ±{(hi - lo) / 2 / spot * 100:.1f}%, "
        f"about \\${(hi - lo) / 2:,.0f} an ounce. That's the honest width of a week "
        f"of gold uncertainty. A narrower range would be a lie.\n"
        f"- **It's weakest exactly when it matters most.** In a real crisis, prices "
        f"jump faster than the model adapts, and misses bunch together.\n"
        f"- **80% is an average over ten years**, not a promise about any given "
        f"month. Some stretches are worse.\n"
        f"- **This is the futures price, not your local rate.** It runs \\$20–30 "
        f"above quoted spot, and local retail adds duty, tax and dealer margin on "
        f"top. The *movement* carries across; the *level* doesn't.\n"
        f"- **The price ticks every 30 seconds; the range updates once a day.** "
        f"Intraday changes to the range would be noise, not information.\n"
        f"- **The model is fitted through {ds.index.max():%d %b %Y}**, five trading "
        f"days behind today — it can only learn from weeks that have finished.\n"
        f"- **Ten years of testing was mostly a rising market.** How this behaves "
        f"in a sustained crash is untested.\n"
        f"- **Not investment advice.** A range is a statement about risk, not a "
        f"forecast."
    )

st.markdown(
    f"<div class='sub' style='text-align:center;margin-top:2rem'>"
    f"Data: Yahoo Finance, FRED · Model: GJR-GARCH with empirical quantiles</div>",
    unsafe_allow_html=True,
)