# app/app.py
"""Gold band dashboard.

Shows the honest weekly range, not a direction call. The point forecast is
naive (last price + drift) because nothing beats it — see the scorecard.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import HORIZON, PROCESSED
import garch as G

st.set_page_config(
    page_title="Gold Weekly Band",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- constants

IMPORT_DUTY = 0.06
GST = 0.03
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
      font-size: 2.6rem;
      font-weight: 650;
      letter-spacing: -0.03em;
      color: {INK};
      line-height: 1.05;
      font-variant-numeric: tabular-nums;
  }}

  .band-figure {{
      font-size: 2.1rem;
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
      padding: 1.3rem 1.4rem;
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

  .rule {{
      border: none;
      border-top: 1px solid {LINE};
      margin: 2.5rem 0 1.75rem 0;
  }}

  .verdict-yes {{ color: {GREEN}; font-weight: 650; }}
  .verdict-no {{ color: {RED}; font-weight: 650; }}

  table {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data

@st.cache_data(ttl=900)
def get_live_price():
    import yfinance as yf
    gold = yf.Ticker("GC=F").history(period="5d")["Close"]
    inr = yf.Ticker("USDINR=X").history(period="5d")["Close"]
    return float(gold.iloc[-1]), float(inr.iloc[-1]), gold.index[-1]


@st.cache_data(ttl=3600)
def get_history():
    return pd.read_parquet(PROCESSED / "dataset.parquet")


@st.cache_data(ttl=86400)
def fit_band(conf):
    returns, _ = G.load_returns()
    vol = G.vol_gjr(returns)
    q = G.empirical_quantiles(returns, vol, conf)
    drift = returns.mean() * HORIZON / G.SCALE
    return vol, q, drift


@st.cache_data(ttl=86400)
def vol_context():
    returns, _ = G.load_returns()
    hist = returns.rolling(20).std().dropna()
    return float(hist.iloc[-1]), float((hist < hist.iloc[-1]).mean()), returns.index[-1]


def to_inr(usd_per_oz, usdinr):
    return (usd_per_oz / OZ_TO_GRAM) * usdinr * (1 + IMPORT_DUTY) * (1 + GST) * 10


# ---------------------------------------------------------------- load

try:
    spot, usdinr, asof = get_live_price()
except Exception as e:
    ds_t = get_history()
    spot, usdinr, asof = ds_t["gold_price"].iloc[-1], 96.5, ds_t.index[-1]
    st.error(f"Live price unavailable — showing last cached close. ({e})")

ds = get_history()
prev = ds["gold_price"].iloc[-1]
week_ago = ds["gold_price"].iloc[-6] if len(ds) > 6 else prev

d_day = (spot / prev - 1) * 100
d_week = (spot / week_ago - 1) * 100


# ---------------------------------------------------------------- header

st.markdown(
    f"<div class='eyebrow'>Gold · next {HORIZON} trading days</div>"
    f"<div style='font-size:1.9rem;font-weight:680;letter-spacing:-0.035em;"
    f"color:{INK};margin-bottom:0.15rem;'>What it might cost next week</div>"
    f"<div class='sub'>A range, not a prediction. The direction is a coin flip — "
    f"the size of the swing is not.</div>",
    unsafe_allow_html=True,
)

st.write("")

h1, h2, h3 = st.columns([1.1, 1.1, 0.8])

def delta_html(v, label):
    cls = "delta-up" if v >= 0 else "delta-down"
    arrow = "▲" if v >= 0 else "▼"
    return f"<span class='{cls}'>{arrow} {abs(v):.1f}%</span> <span class='sub'>{label}</span>"

with h1:
    st.markdown(
        f"<div class='card'><div class='eyebrow'>Spot</div>"
        f"<div class='display'>${spot:,.0f}</div>"
        f"<div style='margin-top:0.4rem'>{delta_html(d_day, 'vs last close')}</div>"
        f"<div class='sub' style='margin-top:0.5rem'>per troy ounce</div></div>",
        unsafe_allow_html=True,
    )

with h2:
    st.markdown(
        f"<div class='card'><div class='eyebrow'>Landed, India</div>"
        f"<div class='display'>₹{to_inr(spot, usdinr):,.0f}</div>"
        f"<div style='margin-top:0.4rem'>{delta_html(d_week, 'this week')}</div>"
        f"<div class='sub' style='margin-top:0.5rem'>per 10g · incl. 6% duty, 3% GST</div></div>",
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        f"<div class='card'><div class='eyebrow'>USD / INR</div>"
        f"<div class='display' style='font-size:2rem'>{usdinr:.2f}</div>"
        f"<div class='sub' style='margin-top:0.9rem'>as of {asof:%d %b}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='rule'>", unsafe_allow_html=True)


# ---------------------------------------------------------------- band

bl, br = st.columns([2.2, 1])

with br:
    conf = st.radio("Confidence level", options=[0.80, 0.95],
                    format_func=lambda x: f"{x:.0%}", horizontal=True)

vol, q, drift = fit_band(conf)
lo, hi = G.band(spot, q[0], q[1], drift=drift)
lo_inr, hi_inr = to_inr(lo, usdinr), to_inr(hi, usdinr)
cur_vol, pctile, vol_asof = vol_context()

with bl:
    st.markdown(
        f"<div class='band-card'>"
        f"<div class='eyebrow'>Expected range · {conf:.0%} confidence</div>"
        f"<div class='band-figure'>₹{lo_inr:,.0f} &nbsp;—&nbsp; ₹{hi_inr:,.0f}</div>"
        f"<div class='sub' style='margin-top:0.35rem'>per 10g &nbsp;·&nbsp; "
        f"${lo:,.0f} – ${hi:,.0f} per oz &nbsp;·&nbsp; "
        f"±{(hi - lo) / 2 / spot * 100:.1f}%</div></div>",
        unsafe_allow_html=True,
    )

with br:
    if pctile > 0.7:
        bg, fg, word = "rgba(185,28,28,0.09)", RED, "Choppy"
        note = "Wider range than usual. Consider smaller commitments."
    elif pctile < 0.3:
        bg, fg, word = "rgba(21,128,61,0.09)", GREEN, "Calm"
        note = "Prices steady. Safer window for larger purchases."
    else:
        bg, fg, word = "rgba(120,113,108,0.09)", MUTED, "Normal"
        note = "Volatility around its typical level."

    st.markdown(
        f"<div style='margin-top:0.2rem'>"
        f"<span class='pill' style='background:{bg};color:{fg}'>{word}</span>"
        f"<div class='sub' style='margin-top:0.6rem;line-height:1.45'>"
        f"20-day volatility <b>{cur_vol:.2f}%</b> daily — higher than "
        f"<b>{pctile:.0%}</b> of the last decade.<br>{note}</div></div>",
        unsafe_allow_html=True,
    )

stale_days = (pd.Timestamp.now().normalize() - vol_asof.tz_localize(None)).days
if stale_days > 3:
    st.caption(f"⚠️ Band fitted on history through {vol_asof:%d %b %Y} "
               f"({stale_days} days old). Re-run the data pull to refresh.")

st.write("")


# ---------------------------------------------------------------- chart

hist = ds["gold_price"].iloc[-130:]
future = pd.bdate_range(hist.index[-1], periods=HORIZON + 1)[1:]
centre_end = spot * np.exp(drift)

fig = go.Figure()

# cone — fans out from today rather than a flat rectangle
cone_x = [hist.index[-1]] + list(future)
cone_hi = [spot] + list(np.linspace(spot, hi, HORIZON + 1)[1:])
cone_lo = [spot] + list(np.linspace(spot, lo, HORIZON + 1)[1:])

fig.add_trace(go.Scatter(
    x=cone_x + cone_x[::-1],
    y=cone_hi + cone_lo[::-1],
    fill="toself", fillcolor=GOLD_DIM,
    line=dict(width=0), hoverinfo="skip", showlegend=False,
))

fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values,
    mode="lines", line=dict(color=GOLD, width=2),
    name="Gold", hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.0f}</b><extra></extra>",
))

fig.add_trace(go.Scatter(
    x=[hist.index[-1], future[-1]], y=[spot, centre_end],
    mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
    name="Centre (today's price)", hoverinfo="skip",
))

for y, lbl, col in [(hi, f"${hi:,.0f}", GOLD), (lo, f"${lo:,.0f}", GOLD)]:
    fig.add_annotation(x=future[-1], y=y, text=f"<b>{lbl}</b>",
                       showarrow=False, xanchor="left", xshift=8,
                       font=dict(size=12, color=col))

fig.update_layout(
    height=380,
    margin=dict(l=0, r=68, t=10, b=0),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    showlegend=False,
    xaxis=dict(showgrid=False, showline=True, linecolor=LINE,
               tickfont=dict(size=11, color=MUTED)),
    yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False,
               tickfont=dict(size=11, color=MUTED),
               tickprefix="$", side="right"),
    font=dict(family="Inter, sans-serif"),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown(
    f"<div class='sub' style='text-align:center'>Last 6 months. The cone is where "
    f"gold lands {conf:.0%} of the time. The dotted line is just today's price "
    f"carried forward — not a forecast.</div>",
    unsafe_allow_html=True,
)

st.markdown("<hr class='rule'>", unsafe_allow_html=True)


# ---------------------------------------------------------------- scorecard

st.markdown(
    f"<div class='eyebrow'>Evidence</div>"
    f"<div style='font-size:1.35rem;font-weight:650;letter-spacing:-0.02em;"
    f"color:{INK};margin-bottom:1rem'>Does this actually work?</div>",
    unsafe_allow_html=True,
)

bt = BACKTEST[conf]
e1, e2 = st.columns(2)

with e1:
    st.markdown(
        f"<div class='card'>"
        f"<div class='eyebrow'>The range</div>"
        f"<div style='font-size:1.5rem;margin:0.3rem 0 0.6rem'>"
        f"<span class='verdict-yes'>Yes</span></div>"
        f"<div class='sub' style='line-height:1.55'>"
        f"Tested on 474 non-overlapping weeks, 2017–2026. Claimed {conf:.0%} "
        f"confidence; the price actually landed inside <b>{bt['coverage']}</b> "
        f"of the time.<br><br>"
        f"Kupiec test p = {bt['kupiec']} — no significant miscalibration.<br>"
        f"Independence p = {bt['indep']} — breaches spread out, not clustered."
        f"</div></div>",
        unsafe_allow_html=True,
    )

with e2:
    st.markdown(
        f"<div class='card'>"
        f"<div class='eyebrow'>The direction</div>"
        f"<div style='font-size:1.5rem;margin:0.3rem 0 0.6rem'>"
        f"<span class='verdict-no'>No</span></div>"
        f"<div class='sub' style='line-height:1.55'>"
        f"A model trained on 18 factors — dollar index, real yields, VIX, "
        f"inflation breakevens, silver, momentum, seasonality — to predict "
        f"which way gold moves.<br><br>"
        f"It lost to a coin flip. Both models scored worse than assuming no "
        f"change at all."
        f"</div></div>",
        unsafe_allow_html=True,
    )

st.write("")
t1, t2 = st.columns(2)

with t1:
    st.markdown("<div class='eyebrow'>Directional accuracy · 320 weeks</div>",
                unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "Accuracy": ["49.7%", "47.5%", "50.0%"],
        "Better than chance?": ["no", "no", "—"],
    }, index=pd.Index(["Ridge", "LightGBM", "Coin flip"], name="")))

with t2:
    st.markdown("<div class='eyebrow'>Skill vs doing nothing</div>",
                unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "Score": ["−6.7%", "−12.3%"],
        "Verdict": ["loses", "loses"],
    }, index=pd.Index(["Ridge", "LightGBM"], name="")))

st.markdown(
    f"<div class='sub' style='line-height:1.6;margin-top:0.5rem'>"
    f"LightGBM fit the training data well (R² = 0.60) and the future not at all "
    f"(R² = −0.06). That gap is overfitting — the model memorised history instead "
    f"of learning a rule. It's why the centre line above is just today's price.<br><br>"
    f"<b>Why direction fails but range works:</b> which way gold moves next week is "
    f"close to random. But <i>how much</i> it moves is persistent — turbulent weeks "
    f"follow turbulent weeks. That's what the band measures."
    f"</div>",
    unsafe_allow_html=True,
)

st.write("")

with st.expander("Limitations — read before relying on this"):
    st.markdown(
        f"- **The range is wide.** Currently ±{(hi - lo) / 2 / spot * 100:.0f}%, "
        f"about ₹{(hi_inr - lo_inr) / 2:,.0f} per 10g. That is the honest width of a "
        f"week of gold uncertainty. A narrower band would be a lie.\n"
        f"- **It fails when it matters most.** At 95%, breaches cluster "
        f"(independence p = 0.002–0.07 across vol models). In genuine crises, "
        f"volatility jumps faster than the model adapts.\n"
        f"- **Coverage is an average, not a promise.** 80% across a decade — some "
        f"months are worse.\n"
        f"- **Two currencies of risk.** The INR figure carries USD/INR uncertainty "
        f"on top of gold's. The band does not model that.\n"
        f"- **Duty and GST are hardcoded** at 6% and 3%. Policy changes; this will "
        f"not notice. Making charges excluded entirely.\n"
        f"- **Backtest is not live.** 2017–2026 was mostly a bull market. Regimes "
        f"shift, and this has not been tested through one.\n"
        f"- **Not investment advice.** A range is a risk statement, not a forecast."
    )

st.markdown(
    f"<div class='sub' style='text-align:center;margin-top:2rem'>"
    f"GJR-GARCH(1,1,1) · empirical quantiles via filtered historical simulation · "
    f"Yahoo Finance, FRED</div>",
    unsafe_allow_html=True,
)