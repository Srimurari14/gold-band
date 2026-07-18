# app/app.py
"""Gold band dashboard.

USD is the model's native currency. INR figures are a convenience estimate:
today's dollar price, converted at today's exchange rate, with an estimate
for import duty, GST, and a typical dealer margin added on. It is not a
quote, and it does not account for the rupee itself moving, only for gold's
dollar price moving. That gap is stated plainly in the limitations.

Price refreshes every 30 seconds. The band does not, because it is built
from full days of price history, not seconds.
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

OZ_TO_GRAM = 31.1034768

# Confirmed current as of May 2026: duty raised from 6% to 15%.
# GST is 3% on bullion. Dealer premium varies by shop; 100 INR per 10g is a
# rough midpoint, not a quote. Making charges are excluded entirely since
# they vary too much shop to shop to estimate honestly.
IMPORT_DUTY = 0.15
GST = 0.03
DEALER_PREMIUM_10G = 100

GOLD = "#c9a227"
GOLD_DIM = "rgba(201, 162, 39, 0.16)"
AMBER = "#b45309"
AMBER_BG = "rgba(180, 83, 9, 0.06)"
INK = "#1c1917"
MUTED = "#78716c"
LINE = "#e7e5e4"
RED = "#b91c1c"
GREEN = "#15803d"

BACKTEST = {
    0.80: {"coverage": "80.0%"},
    0.95: {"coverage": "94.3%"},
}

WEEKLY_LOG = ROOT / "logs" / "predictions.csv"
DAILY_LOG = ROOT / "logs" / "daily_predictions.csv"

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
      font-size: 2.2rem;
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

  .inr-row {{
      margin-top: 0.9rem;
      padding-top: 0.8rem;
      border-top: 1px dashed {LINE};
  }}

  .experimental-card {{
      border: 1px dashed {AMBER};
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      background: {AMBER_BG};
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
      flex-wrap: wrap;
  }}

  .unit {{
      font-size: 0.95rem;
      font-weight: 600;
      color: {INK};
      font-variant-numeric: tabular-nums;
  }}

  .unit-inr {{
      font-size: 0.95rem;
      font-weight: 600;
      color: {AMBER};
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
def get_live_gold():
    import yfinance as yf
    gold = yf.Ticker("GC=F").history(period="5d")["Close"]
    return float(gold.iloc[-1]), gold.index[-1]


@st.cache_data(ttl=300)
def get_live_usdinr():
    import yfinance as yf
    inr = yf.Ticker("USDINR=X").history(period="5d")["Close"]
    return float(inr.iloc[-1])


@st.cache_data(ttl=3600)
def get_history():
    return pd.read_parquet(PROCESSED / "dataset.parquet")


@st.cache_data(ttl=3600)
def get_raw_asof():
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
    returns, _ = G.load_returns()
    hist = returns.rolling(20).std().dropna()
    current = float(hist.iloc[-1])
    pctile = float((hist < current).mean())
    month_ago = float(hist.iloc[-21]) if len(hist) > 21 else current
    trend = (current / month_ago - 1) if month_ago > 0 else 0.0
    return current, pctile, trend


@st.cache_data(ttl=3600)
def live_track_record(conf):
    try:
        df = pd.read_csv(WEEKLY_LOG)
    except Exception:
        return None
    sub = df[(df["conf"] == conf) & df["inside"].notna()]
    if len(sub) == 0:
        return None
    inside = sub["inside"].astype(bool)
    last10 = inside.iloc[-10:]
    return len(sub), float(inside.mean()), int((~last10).sum()), len(last10)


@st.cache_data(ttl=3600)
def daily_experiment():
    try:
        df = pd.read_csv(DAILY_LOG)
    except Exception:
        return None
    if len(df) == 0:
        return None

    latest = df.iloc[-1]
    resolved = df[df["model_beat_naive"].notna()]
    n = len(resolved)
    rate = float(resolved["model_beat_naive"].astype(bool).mean()) if n > 0 else None

    return {
        "spot": float(latest["spot"]),
        "pred_price": float(latest["pred_price"]),
        "data_through": latest["data_through"],
        "n_resolved": n,
        "beat_rate": rate,
    }


def inr_per_gram(usd_per_oz, usdinr):
    base = (usd_per_oz / OZ_TO_GRAM) * usdinr
    return base * (1 + IMPORT_DUTY) * (1 + GST)


def inr_per_10g(usd_per_oz, usdinr):
    return inr_per_gram(usd_per_oz, usdinr) * 10 + DEALER_PREMIUM_10G


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
    spot, asof = get_live_gold()
    is_live = True
except Exception as e:
    ds_t = get_history()
    spot, asof = ds_t["gold_price"].iloc[-1], ds_t.index[-1]
    is_live = False
    st.error(f"Live price is not available right now. Showing the last saved price. ({e})")

try:
    usdinr = get_live_usdinr()
except Exception:
    usdinr = None

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
    f"<div class='eyebrow'>Gold, next {HORIZON} trading days</div>"
    f"<div style='font-size:1.9rem;font-weight:680;letter-spacing:-0.035em;"
    f"color:{INK};margin-bottom:0.15rem;'>What it might cost next week</div>"
    f"<div class='sub'>A range, not a prediction. Nobody knows which way gold "
    f"will move. How far it can move, though, can be measured.</div>",
    unsafe_allow_html=True,
)

st.write("")

def delta_html(v, label):
    cls = "delta-up" if v >= 0 else "delta-down"
    arrow = "▲" if v >= 0 else "▼"
    return f"<span class='{cls}'>{arrow} {abs(v):.1f}%</span> <span class='sub'>{label}</span>"

h1, h2 = st.columns([1.25, 1])

with h1:
    live_badge = (f"<span class='live-dot'></span><span class='sub'>live</span>"
                  if is_live else "<span class='sub'>last saved price</span>")

    if usdinr:
        inr_g = inr_per_gram(spot, usdinr)
        inr_10 = inr_per_10g(spot, usdinr)
        inr_block = (
            f"<div class='inr-row'>"
            f"<div class='unit-label' style='margin-bottom:0.3rem'>Estimated price in India</div>"
            f"<div class='unit-row'>"
            f"<div><div class='unit-inr'>₹{inr_g:,.0f}</div>"
            f"<div class='unit-label'>per gram</div></div>"
            f"<div><div class='unit-inr'>₹{inr_10:,.0f}</div>"
            f"<div class='unit-label'>per 10 grams</div></div>"
            f"</div></div>"
        )
    else:
        inr_block = (
            f"<div class='inr-row'><div class='sub'>Rupee rate is not available "
            f"right now.</div></div>"
        )

    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div class='eyebrow' style='margin-bottom:0'>Price now, in dollars</div>{live_badge}</div>"
        f"<div class='display' style='margin-top:0.4rem'>${spot:,.2f}"
        f"<span style='font-size:1rem;font-weight:500;color:{MUTED}'> per ounce</span></div>"
        f"<div class='unit-row'>"
        f"<div><div class='unit'>${per_g:,.2f}</div><div class='unit-label'>per gram</div></div>"
        f"<div><div class='unit'>${per_10g:,.2f}</div><div class='unit-label'>per 10 grams</div></div>"
        f"</div>"
        f"<div style='margin-top:0.55rem'>{delta_html(d_day, 'vs yesterday')} "
        f"&nbsp;&nbsp; {delta_html(d_week, 'this week')}</div>"
        f"{inr_block}</div>",
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
        ctx = f"Today's price sits {pos:.0f} percent of the way up that range."

    dd = (spot / hi_ext - 1) * 100

    st.markdown(
        f"<div class='card'><div class='eyebrow'>Where prices have been, past 6 months</div>"
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
        f"<div class='sub'>{ctx} Down {abs(dd):.0f} percent from the highest point. "
        f"This looks backward. It does not tell you what happens next.</div></div>",
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
    if usdinr:
        inr_lo = inr_per_10g(lo, usdinr)
        inr_hi = inr_per_10g(hi, usdinr)
        inr_band = (
            f"<div class='inr-row'>"
            f"<div class='unit-label' style='margin-bottom:0.3rem'>Estimated in rupees, per 10 grams</div>"
            f"<div class='unit-inr' style='font-size:1.3rem'>₹{inr_lo:,.0f} to ₹{inr_hi:,.0f}</div>"
            f"</div>"
        )
    else:
        inr_band = ""

    st.markdown(
        f"<div class='band-card'>"
        f"<div class='eyebrow'>Next week, gold will most likely be between</div>"
        f"<div class='band-figure'>${lo:,.0f} to ${hi:,.0f}"
        f"<span style='font-size:1rem;font-weight:500;color:{MUTED}'> per ounce</span></div>"
        f"<div class='unit-row'>"
        f"<div><div class='unit'>${lo / OZ_TO_GRAM * 10:,.0f} to "
        f"${hi / OZ_TO_GRAM * 10:,.0f}</div>"
        f"<div class='unit-label'>per 10 grams</div></div>"
        f"<div><div class='unit'>{(1 - lo / spot) * 100:.1f}% below to "
        f"{(hi / spot - 1) * 100:.1f}% above</div>"
        f"<div class='unit-label'>compared to today's price</div></div>"
        f"</div>"
        f"<div class='sub' style='margin-top:0.55rem;color:{INK}'>"
        f"This is right about <b>{weeks}</b>. The rest of the time, the real "
        f"price ends up outside it."
        f"</div>"
        f"{inr_band}"
        f"</div>",
        unsafe_allow_html=True,
    )

with br:
    if pctile > 0.7:
        bg, fg, word = "rgba(185,28,28,0.09)", RED, "Choppy"
        note = ("Prices are jumping around more than usual. The range is wide. "
                "Think about buying in smaller amounts.")
    elif pctile < 0.3:
        bg, fg, word = "rgba(21,128,61,0.09)", GREEN, "Calm"
        note = ("Prices have been steady. The range is narrow. This may be a "
                "safer time to make a bigger purchase.")
    else:
        bg, fg, word = "rgba(120,113,108,0.09)", MUTED, "Normal"
        note = "Prices are moving about as much as they usually do."

    if vol_trend > 0.15:
        trend_txt = "Prices are getting <b>choppier</b> compared to a month ago."
    elif vol_trend < -0.15:
        trend_txt = "Prices are getting <b>calmer</b> compared to a month ago."
    else:
        trend_txt = "This has not changed much in the last month."

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
        st.caption(f"The market data was last updated on {raw_asof:%d %b %Y}, "
                   f"{stale} days ago. Something may have stopped working.")

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
    f"<div class='sub' style='text-align:center'>The gold line shows the last "
    f"six months. The shaded area on the right is next week's range. It gets "
    f"wider because the further ahead you look, the less certain things get.</div>",
    unsafe_allow_html=True,
)

st.write("")
st.write("")


# ---------------------------------------------------------------- trust line

track = live_track_record(conf)

if track is not None:
    n, cov, breaches10, n10 = track
    st.markdown(
        f"<div class='trust'>We checked this against ten years of history. "
        f"When we said <b>{conf:.0%}</b>, it was right <b>{BACKTEST[conf]['coverage']}</b> "
        f"of the time. &nbsp;·&nbsp; "
        f"We have also been running it live since July 2026, and so far it has "
        f"been right <b>{cov:.0%}</b> of the time over {n} finished "
        f"week{'s' if n != 1 else ''}"
        + (f", missing {breaches10} time{'s' if breaches10 != 1 else ''} in the "
           f"last {n10}" if n10 > 0 else "")
        + ".</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='trust'>We checked this against ten years of history. "
        f"When we said <b>{conf:.0%}</b>, it was right <b>{BACKTEST[conf]['coverage']}</b> "
        f"of the time. We are also now tracking it live, each day's range is "
        f"written down before we know the outcome, and checked a week later.</div>",
        unsafe_allow_html=True,
    )

st.write("")


# ---------------------------------------------------------------- experiment: tomorrow

exp = daily_experiment()

if exp is not None:
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='eyebrow'>Not part of the main tool. Watch it, do not rely on it</div>"
        f"<div style='font-size:1.25rem;font-weight:650;letter-spacing:-0.02em;"
        f"color:{INK};margin-bottom:0.6rem'>An experiment: guessing tomorrow's price</div>",
        unsafe_allow_html=True,
    )

    n = exp["n_resolved"]
    rate = exp["beat_rate"]

    if n < 20:
        record_txt = (f"Only <b>{n}</b> day{'s' if n != 1 else ''} have finished so far. "
                      f"That is too few to judge. Check back once there are around 40.")
    elif rate is None:
        record_txt = "No finished days yet."
    else:
        verdict = ("doing better than" if rate > 0.55 else
                  "doing worse than" if rate < 0.45 else "about the same as")
        record_txt = (f"Out of the last <b>{n}</b> finished days, this guess has beaten "
                      f"a simple 'tomorrow will be the same as today' guess "
                      f"<b>{rate:.0%}</b> of the time. Right now it is {verdict} "
                      f"that simple guess.")

    c1, c2 = st.columns([1.4, 1])

    with c1:
        pred_delta = (exp["pred_price"] / exp["spot"] - 1) * 100
        st.markdown(
            f"<div class='experimental-card'>"
            f"<div class='sub' style='margin-bottom:0.5rem'>A computer model was "
            f"trained on 18 pieces of market information and is being tested "
            f"against what actually happens the next day. For {exp['data_through']}, "
            f"it guessed:</div>"
            f"<div style='display:flex;gap:2rem;align-items:baseline'>"
            f"<div><div class='unit' style='font-size:1.3rem'>${exp['spot']:,.0f}</div>"
            f"<div class='unit-label'>simple guess: no change</div></div>"
            f"<div><div class='unit' style='font-size:1.3rem;color:{AMBER}'>"
            f"${exp['pred_price']:,.0f}</div>"
            f"<div class='unit-label'>model's guess ({pred_delta:+.2f}%)</div></div>"
            f"</div>"
            f"<div class='sub' style='margin-top:0.8rem'>{record_txt}</div></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div class='sub' style='line-height:1.55'>"
            f"When we checked this model against ten years of history, it barely "
            f"beat the simple guess, by less than one fifth of one percent. It also "
            f"did worse than the simple guess for the first half of that history, "
            f"and only pulled ahead in the second half. That is not a strong enough "
            f"result to trust yet.<br><br>"
            f"So instead of using it, we are watching it run for real, every day, "
            f"and writing down whether it wins or loses. If you are curious, you "
            f"can compare its guess to what actually happens tomorrow. That is the "
            f"only way it earns a real place in this tool."
            f"</div>",
            unsafe_allow_html=True,
        )

st.write("")


# ---------------------------------------------------------------- explainers

with st.expander("How to read this"):
    st.markdown(f"""
**The price at the top** is what gold costs right now, in US dollars, shown
per ounce, per gram, and per 10 grams. One troy ounce equals about 31.1
grams, so these are all the same price, just measured differently. There is
also a rough rupee price, which adds an estimate for import duty (15
percent), tax (3 percent), and a typical dealer margin. Your own shop's price
will still be different, this is a guide, not a quote, and it does not
include making charges.

**"Where prices have been"** shows the lowest and highest price over the
past six months, and where today sits between them. This only looks
backward. A price near the bottom does not mean it is about to go up, that
is the easiest mistake to make when looking at this box.

**The big range in the middle is the whole point of this page.** It says
that next week, gold will most likely end up somewhere between two numbers.
Not exactly where, just somewhere in that range.

**The choice between 80% and 95%** changes how sure you want to be. At 80
percent, the real price stays inside the range about 4 weeks out of 5. One
week in five, it goes outside, and that is not a mistake, that is what 80
percent means. At 95 percent, it is almost always right, but the range gets
so wide that it stops being very useful. Use 80 percent most of the time.
Check 95 percent only when you are about to make a big purchase and want to
know the worst case.

**Calm, Normal, or Choppy** tells you how much prices have been jumping
around lately, and whether that is getting better or worse. Calm and getting
calmer means it may be a safer time for a bigger purchase. Choppy and
getting choppier means smaller purchases, or waiting, may be wiser.

**The chart** shows the last six months as a line, with next week's range
shown as a shaded area on the right. The dotted line running through the
middle is not a prediction, it is simply today's price drawn forward,
because nothing does better than that.

**The experiment section**, if you see it, is something different being
tested live: a model trying to guess tomorrow's exact price. It has not
earned trust yet. It is there to watch, not to act on.

The one thing worth remembering through all of this: nothing here will ever
tell you gold is about to go up or down. Nobody can honestly tell you that.
What this tells you is how much it could move, which is what actually
matters when you are deciding whether to buy now, buy a little, or wait.
""")

with st.expander("Can this be trusted? Here is how we checked"):
    bt = BACKTEST[conf]
    st.markdown(f"""
Two different things were built here. One of them works well. One of them
does not. Both results are shown honestly below.

**The range works.** We tested it on 474 separate weeks between 2017 and
2026. Each time, we only used information that would have actually been
available at that moment, made the same kind of guess this page makes today,
and then checked what really happened.

When we said the range would be right 80 times out of 100, it actually
landed inside the range about 80 times out of 100. When we said 95 times out
of 100, it landed inside about 94 times out of 100. Both numbers matched
almost exactly what we claimed.

We also checked whether the misses happened evenly over time, or whether
they piled up in a few bad stretches. They were spread out evenly, not
clumped together. That matters, because a tool that is right on average but
badly wrong for one whole bad month would not be something you could
actually rely on week to week.

On top of that ten year check, we are also now tracking this live. Each day,
before anyone knows what will happen, the range gets written down. A week
later, we check whether the real price landed inside it. That live result is
shown near the top of this page, and it grows more meaningful every week,
because nothing about it can be adjusted after the fact.

**Guessing the exact price does not work, at least not yet.** Several
different computer programs were trained on 18 pieces of information that
are supposed to affect gold prices: the strength of the US dollar, interest
rates, how nervous investors are feeling, expectations about inflation,
silver prices, and more.

None of them could reliably guess which way gold would move next week. The
best of them was right about direction only half the time, which is no
better than flipping a coin. One of the more complex programs matched old
data almost perfectly, but when tested on new data it had never seen, it did
worse than simply assuming nothing would change. In plain terms, it had
memorized the past so well that it stopped being useful for the future.

A separate attempt tried to guess tomorrow's exact price specifically, using
five different programs. The best one barely beat a simple "tomorrow will be
the same as today" guess, by less than one fifth of one percent, and even
that thin edge came from doing badly for the first few years and then doing
well for the rest. That is not solid enough to trust, so instead of using
it, it is being tracked live (see the experiment box above, if it is
showing). If it earns a real edge over the next couple of months, it will be
added properly. If it does not, that will be reported honestly too.

This is why the dotted line down the middle of the chart is simply today's
price drawn forward. Nothing has beaten that fairly yet, so nothing has
replaced it.

**Why does the range work when exact guessing does not?** Because which way
gold moves is close to random, the past truly does not tell you that. But
how much gold tends to move stays fairly steady over time. Calm weeks tend
to be followed by more calm weeks, and choppy weeks tend to be followed by
more choppy weeks. That pattern is real, it can be measured, and it is the
only thing this tool actually depends on.
""")

with st.expander("What this does not do, and what to be careful about"):
    st.markdown(
        f"- **The range is wide.** Right now it is about "
        f"plus or minus {(hi - lo) / 2 / spot * 100:.1f} percent, or roughly "
        f"${(hi - lo) / 2:,.0f} an ounce. That is the honest size of how much "
        f"gold can move in a week. Making it look narrower would be lying to you.\n"
        f"- **It is weakest exactly when it matters most.** During a real crisis, "
        f"prices can jump faster than the model can keep up, and mistakes can "
        f"happen close together instead of being spread out.\n"
        f"- **The 80 percent figure is an average over ten years**, not a promise "
        f"for any single month. Some months will be worse than others.\n"
        f"- **This uses the international futures price, not your local shop's "
        f"price.** It usually runs 20 to 30 dollars above the quoted spot price. "
        f"Your local price adds duty, tax, and dealer margin on top of that. "
        f"When this price moves, yours moves too, but the exact level will "
        f"always be different.\n"
        f"- **The rupee price is only an estimate.** It uses the 15 percent duty "
        f"rate in effect since May 2026, 3 percent GST, and a rough guess at "
        f"dealer margin. It does not include making charges, which vary too much "
        f"shop to shop to estimate fairly. It also does not account for the "
        f"rupee to dollar exchange rate itself moving, only for gold's dollar "
        f"price moving, so your actual rupee cost could shift by more than shown "
        f"here.\n"
        f"- **The price updates every 30 seconds. The range only updates once a "
        f"day.** That is because the range is built from full days of price "
        f"history, and updating it every few seconds would add noise, not useful "
        f"information.\n"
        f"- **The model behind the range only knows data up through "
        f"{ds.index.max():%d %b %Y}**, about five trading days behind today. It "
        f"can only learn from weeks that have already fully played out.\n"
        f"- **Ten years of testing happened mostly while gold prices were rising.** "
        f"Nobody knows for certain how well this would hold up during a long, "
        f"severe crash.\n"
        f"- **The experiment section, if you see it, has not proven itself yet.** "
        f"That is exactly why it is being watched instead of trusted.\n"
        f"- **None of this is investment advice.** It tells you how much risk "
        f"there is. It does not tell you what to do about it."
    )

st.markdown(
    f"<div class='sub' style='text-align:center;margin-top:2rem'>"
    f"Data from Yahoo Finance and FRED.</div>",
    unsafe_allow_html=True,
)