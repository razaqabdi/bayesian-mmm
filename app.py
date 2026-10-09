"""MMM Studio: Streamlit front end for the Bayesian marketing mix model.

Run:  streamlit run app.py   (after src/simulate.py and src/fit_mmm.py)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import analytics as A  # noqa: E402
from optimiser import PARAMS_PATH, load_params, optimise, summarise, total_response, weekly_response  # noqa: E402

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK, INK_2, INK_3 = "#1b1b1a", "#52514e", "#8a8984"
SURFACE, CARD, LINE, GRID = "#f7f6f3", "#ffffff", "#e7e5df", "#ecebe6"
BASE_GREY = "#c9c7c0"
NAVY = "#1c3d6e"
RED = "#d03b3b"
AMBER = "#b7791f"
COL = {c: m["colour"] for c, m in A.CHANNEL_META.items()}
MINUS = "−"

st.set_page_config(page_title="MMM Studio", layout="wide", initial_sidebar_state="expanded")

# Newer Streamlit replaces use_container_width with width="stretch"
_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {SURFACE}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(120deg, #13294b 0%, #1c3d6e 55%, #2a78d6 100%); border-radius: 18px;
        padding: 26px 30px; color: #fff; margin-bottom: 18px; }}
.hero h1 {{ color: #fff; font-size: 30px; font-weight: 700; margin: 0 0 6px 0; }}
.hero p {{ color: #dbe6f6; font-size: 15px; margin: 0 0 14px 0; max-width: 860px; }}
.chip {{ display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
        border-radius: 999px; padding: 4px 12px; font-size: 12.5px; margin: 0 6px 6px 0; color: #fff; }}
.kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px; height: 100%;
       container-type: inline-size; min-width: 0; overflow: hidden; }}
.kpi .label {{ font-size: 11.5px; color: {INK_3}; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
              line-height: 1.3; }}
.kpi .value {{ font-size: clamp(17px, 17cqw, 26px); font-weight: 700; color: {INK}; margin-top: 4px; line-height: 1.15;
              white-space: nowrap; }}
.kpi .sub {{ font-size: 12.5px; color: {INK_2}; margin-top: 4px; line-height: 1.4; }}
.insight {{ background: {CARD}; border: 1px solid {LINE}; border-left: 4px solid {NAVY}; border-radius: 12px;
           padding: 14px 16px; height: 100%; }}
.insight .tag {{ font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: {NAVY}; }}
.insight .body {{ font-size: 14.5px; color: {INK}; margin-top: 6px; line-height: 1.45; }}
.insight.warn {{ border-left-color: {AMBER}; margin: 6px 0 4px 0; }}
.insight.warn .tag {{ color: {AMBER}; }}
.section {{ font-size: 18px; font-weight: 650; color: {INK}; margin: 18px 0 2px 0; }}
.section-sub {{ font-size: 13.5px; color: {INK_2}; margin-bottom: 8px; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ height: 44px; padding: 0 16px; font-weight: 600; color: {INK_2}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
[data-testid="stSidebar"] {{ background: #fbfaf8; border-right: 1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gbp(x, dp=0):
    """Compact money: £58.4m, £71k, £3.6k, £950. Negative values get a proper minus sign before the £."""
    sign, a = (MINUS if x < 0 else ""), abs(x)
    if a >= 1e6:
        return f"{sign}£{a / 1e6:,.1f}m"
    if a >= 1e4:
        return f"{sign}£{a / 1e3:,.0f}k"
    if a >= 1e3:
        return f"{sign}£{a / 1e3:,.1f}k"
    return f"{sign}£{a:,.{dp}f}"


def gbp_full(x):
    return f"{MINUS if round(x) < 0 else ''}£{abs(x):,.0f}"


def money(x, dp=2):
    """Per-£1 amounts: £1.39, −£0.31."""
    return f"{MINUS if x < 0 else ''}£{abs(x):.{dp}f}"


def pct(x, dp=0, signed=False):
    if round(x * 100, dp) == 0:
        x = 0.0
    s = f"{x:+.{dp}%}" if signed else f"{x:.{dp}%}"
    return s.replace("-", MINUS)


def prob(x):
    """A posterior share of draws. Never claim certainty: 400 draws can't show 100%."""
    if x >= 0.995:
        return ">99%"
    if x <= 0.005:
        return "<1%"
    return f"{x:.0%}"


def names(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(kpi(*item), unsafe_allow_html=True)


def insight(tag, body, kind=""):
    return f'<div class="insight {kind}"><div class="tag">{tag}</div><div class="body">{body}</div></div>'


def section(title, sub=""):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def styled(fig, height=380, legend=True, top=None):
    fig.update_layout(
        height=height, template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12.5, color=INK_2),
        margin=dict(l=10, r=10, t=top if top is not None else (36 if legend else 16), b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12, color=INK_2)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LINE, font=dict(family="Inter, sans-serif", color=INK)),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE, ticks="outside", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    try:
        fig.update_layout(barcornerradius=4)
    except ValueError:
        pass
    return fig


def show(fig, height=380, legend=True, top=None):
    st.plotly_chart(styled(fig, height, legend, top), **STRETCH, config={"displayModeBar": False})


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def table(df, fmt, height=None):
    kw = {"height": height} if height else {}
    st.dataframe(df.style.format(fmt), hide_index=True, **STRETCH, **kw)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
if not PARAMS_PATH.exists():
    st.error("No fitted model found. Run `python src/simulate.py` then `python src/fit_mmm.py` first.")
    st.stop()


@st.cache_resource
def get_params():
    p = load_params()
    p["roi_draws"] = np.asarray(p.get("roi_draws", []))
    return p


p = get_params()
C = p["channels"]
L = {c: A.label(c) for c in C}
current = {c: float(p["avg_weekly_spend"][c]) for c in C}
current_total = sum(current.values())
decomp = A.load_table("decomposition.csv")
periods = A.load_table("roi_by_period.csv")
holdout_df = A.load_table("holdout.csv")
if holdout_df is not None:
    holdout_df["date"] = pd.to_datetime(holdout_df["date"])
holdout = p.get("holdout") or {}
errors = A.roi_errors(p)
errors_uncal = A.roi_errors(p, "roi_uncalibrated")
calib = p.get("calibration") or {}
lifts = A.lift_check(p) if calib.get("used") else pd.DataFrame()
n_weeks = p.get("n_weeks", 156)

# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------
chips = [f"{n_weeks} weeks of data", f"{len(C)} media channels", "PyMC-Marketing", "Adstock + saturation",
         "Checked against known truth"] + ([f"Calibrated with {len(lifts)} lift tests"] if not lifts.empty else []) \
        + ([f"{holdout['weeks']}-week holdout test"] if holdout else [])
st.markdown(f"""
<div class="hero">
  <h1>MMM Studio</h1>
  <p>A Bayesian marketing mix model for a fictional UK ecommerce brand. It measures what each channel really
  returns, how long the effect lasts and where spend stops paying, then recommends a better budget split,
  with uncertainty on every number.</p>
  {"".join(f'<span class="chip">{c}</span>' for c in chips)}
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Budget scenario")
    budget_box = st.container()
    min_mult = st.slider("Minimum per channel (× today's spend)", 0.0, 0.95, 0.7, 0.05)
    max_mult = st.slider("Maximum per channel (× today's spend)", 1.05, 3.0, 1.3, 0.05)
    st.caption("Limits keep each channel near spend levels the model has seen. "
               "Wider limits rely on extrapolating the response curves.")
    lo_b = int(math.ceil(current_total * min_mult / 1000) * 1000)
    hi_b = int(math.floor(current_total * max_mult / 1000) * 1000)
    with budget_box:
        total_budget = st.slider("Total weekly media budget (£)", lo_b, hi_b,
                                 int(min(max(round(current_total, -3), lo_b), hi_b)), step=1000)
    st.markdown("---")
    st.markdown("### Profit assumption")
    margin = st.slider("Gross margin (% of revenue)", 20, 80, 50, 5, format="%d%%") / 100
    breakeven = 1 / margin
    A_M = f"{'an' if str(round(margin * 100)).startswith('8') else 'a'} {margin:.0%} margin"  # "an 80% margin"
    st.caption(f"At {A_M}, £1 of media pays for itself only if it brings in "
               f"£{breakeven:.2f} of revenue. Set this to the brand's real margin.")
    st.markdown("---")
    st.markdown("### Channels")
    for c in C:
        st.markdown(f'<span class="swatch" style="background:{COL[c]}"></span>{L[c]} '
                    f'<span style="color:{INK_3};font-size:12px">· {A.CHANNEL_META[c]["group"]}</span>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.caption("ROI = revenue per £1 of media. All intervals are 90% Bayesian credible intervals from the posterior.")

lower = {c: current[c] * min_mult for c in C}
upper = {c: current[c] * max_mult for c in C}
try:
    optimal = optimise(p, total_budget, lower, upper)
except ValueError:
    st.error("That budget can't be split within the min/max limits. Loosen the limits or change the budget.")
    st.stop()
current_scaled = {c: current[c] * total_budget / current_total for c in C}
rev_cur, rev_opt = total_response(p, current_scaled), total_response(p, optimal)
uplift = rev_opt - rev_cur
score = A.channel_scorecard(p, margin)
BE_LABEL = f"Break-even at {margin:.0%} margin"

tabs = st.tabs(["Overview", "Channels", "Segments", "Revenue drivers", "Budget optimiser", "Model validation"])

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    spend_tot = sum(p["total_spend"].values())
    draws = p["roi_draws"]
    media_rev_draws = (draws * np.array([p["total_spend"][c] for c in C])).sum(1) if draws.size else np.array([np.nan])
    blended = media_rev_draws / spend_tot
    revenue_total = float(decomp["actual"].sum()) if decomp is not None else np.nan
    fit = p.get("fit", {})
    if holdout:
        fit_card = ("Forecast error", pct(holdout["mape"], 1), f"MAPE on {holdout['weeks']} weeks the model never saw")
    elif fit:
        fit_card = ("Model fit", f"R² {fit['r2']:.2f}", f"in-sample MAPE {pct(fit['mape'], 1)}")
    else:
        fit_card = ("Model fit", "n/a", "")
    kpi_row([
        ("Revenue", gbp(revenue_total) if decomp is not None else "n/a", f"{n_weeks} weeks"),
        ("Media revenue", gbp(media_rev_draws.mean()),
         f"{media_rev_draws.mean() / revenue_total:.0%} of total" if decomp is not None else ""),
        ("Media spend", gbp(spend_tot), f"{gbp(current_total)} a week"),
        ("Blended ROI", money(blended.mean()), f"90%: {money(np.percentile(blended, 5))} to {money(np.percentile(blended, 95))}"),
        fit_card,
    ])

    st.write("")
    pays = score[score["ROI"] >= breakeven].sort_values("ROI", ascending=False)
    loses = score[score["ROI"] < breakeven].sort_values("ROI", ascending=False)
    fmt_ch = lambda d: names(f"<b>{r['Channel']}</b> ({money(r['ROI'])})" for _, r in d.iterrows())  # noqa: E731
    if pays.empty:
        prof_text = f"No channel clears the <b>£{breakeven:.2f}</b> break-even on short-term revenue."
    elif loses.empty:
        prof_text = f"Every channel clears the <b>£{breakeven:.2f}</b> break-even."
    else:
        prof_text = (f"{fmt_ch(pays)} clear{'s' if len(pays) == 1 else ''} the <b>£{breakeven:.2f}</b> break-even. "
                     f"{fmt_ch(loses)} {'doesn’t' if len(loses) == 1 else 'don’t'} on short-term revenue.")
    sat = score.sort_values("Saturation now", ascending=False).iloc[0]
    m_up, m_lo, m_hi = summarise(uplift)
    c1, c2, c3 = st.columns(3)
    c1.markdown(insight(f"Profitability at {margin:.0%} margin", prof_text), unsafe_allow_html=True)
    c2.markdown(insight("Most saturated", f"<b>{sat['Channel']}</b> is at <b>{sat['Saturation now']:.0%}</b> of its "
                        f"maximum effect. The next £1 returns <b>{money(sat['Marginal ROI'])}</b>."),
                unsafe_allow_html=True)
    c3.markdown(insight("Opportunity", f"Splitting {gbp(total_budget)}/week better adds <b>{gbp(m_up)}</b> of weekly "
                        f"revenue ({pct(m_up / rev_cur.mean(), 1, True)}), or {gbp(m_up * margin)} of gross profit. "
                        f"Chance it beats the current mix: <b>{prob((uplift > 0).mean())}</b>."),
                unsafe_allow_html=True)

    section("Where weekly revenue comes from", "Baseline demand (grey) plus the revenue each channel drives, against actual revenue.")
    if decomp is not None:
        base_cols = [c for c in ["intercept", "trend", "seasonality", "price_index", "promo"] if c in decomp]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decomp["date"], y=decomp[base_cols].sum(axis=1), name="Baseline", stackgroup="one",
                                 line=dict(width=0), fillcolor=rgba(BASE_GREY, 0.7),
                                 hovertemplate="Baseline: £%{y:,.0f}<extra></extra>"))
        for c in C:
            fig.add_trace(go.Scatter(x=decomp["date"], y=decomp[c], name=L[c], stackgroup="one",
                                     line=dict(width=0.5, color=COL[c]), fillcolor=rgba(COL[c], 0.85),
                                     hovertemplate=f"{L[c]}: £%{{y:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=decomp["date"], y=decomp["actual"], name="Actual revenue", mode="lines",
                                 line=dict(color=INK, width=1.6), hovertemplate="Actual: £%{y:,.0f}<extra></extra>"))
        fig.update_layout(hovermode="x unified", yaxis_title="Weekly revenue (£)")
        fig.update_yaxes(tickprefix="£", tickformat="~s")
        show(fig, 420)
    else:
        st.info("Re-run fit_mmm.py to generate the revenue decomposition.")

    section("Spend share vs revenue share", "Channels whose coloured dot sits right of the grey dot earn more than their share of budget.")
    sh = A.share_table(p).sort_values("Efficiency index")
    fig = go.Figure()
    for _, r in sh.iterrows():
        fig.add_trace(go.Scatter(x=[r["Spend share"], r["Revenue share"]], y=[r["Channel"]] * 2, mode="lines",
                                 line=dict(color=LINE, width=6), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sh["Spend share"], y=sh["Channel"], mode="markers", name="Share of spend",
                             marker=dict(size=13, color=INK_3, line=dict(color="#fff", width=2)),
                             hovertemplate="%{y}: %{x:.0%} of spend<extra></extra>"))
    fig.add_trace(go.Scatter(x=sh["Revenue share"], y=sh["Channel"], mode="markers", name="Share of media revenue",
                             marker=dict(size=13, color=[COL[c] for c in sh["channel"]], line=dict(color="#fff", width=2)),
                             hovertemplate="%{y}: %{x:.0%} of media revenue<extra></extra>"))
    # label sits right of whichever dot is further right, so it never covers a dot
    fig.add_trace(go.Scatter(x=sh[["Spend share", "Revenue share"]].max(axis=1), y=sh["Channel"], mode="text",
                             text=[f"     {e:.2f}× efficiency" for e in sh["Efficiency index"]],
                             textposition="middle right", textfont=dict(color=INK_2, size=11.5),
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(sh["Spend share"].max(), sh["Revenue share"].max()) * 1.45],
                     showgrid=True, gridcolor=GRID)
    show(fig, 300)

# ---------------------------------------------------------------------------
# 2. Channels
# ---------------------------------------------------------------------------
with tabs[1]:
    section("Channel scorecard",
            f"ROI is revenue per £1. Profit per £1 is ROI × {margin:.0%} margin − £1. "
            "Next £1 is the revenue from one more £1 at today's spend.")
    tbl = pd.DataFrame({
        "Channel": score["Channel"],
        "Weekly spend": score["Weekly spend"],
        "ROI": score["ROI"],
        "90% interval": [f"{money(a)}–{money(b)}" for a, b in zip(score["ROI low"], score["ROI high"])],
        "Profit per £1": score["Profit per £1"],
        "Next £1": score["Marginal ROI"],
        "Saturation": score["Saturation now"],
        "Carry-over": score["Weeks to 90% effect"],
    })
    table(tbl, {"Weekly spend": gbp_full, "ROI": money, "Profit per £1": money, "Next £1": money,
                "Saturation": lambda v: f"{v:.0%}", "Carry-over": lambda v: f"{v:.0f} wk{'' if round(v) == 1 else 's'}"})

    section("How certain is each ROI?",
            f"Full posterior distribution of ROI per channel. The red cross marks the true ROI used to simulate the data. "
            f"The dotted line is break-even at {A_M}.")
    if p["roi_draws"].size:
        fig = go.Figure()
        for j, c in enumerate(C):
            fig.add_trace(go.Violin(x=p["roi_draws"][:, j], y=[L[c]] * len(p["roi_draws"]), orientation="h",
                                    name=L[c], line_color=COL[c], fillcolor=rgba(COL[c], 0.35), side="positive",
                                    width=1.6, points=False, meanline_visible=True, spanmode="hard",
                                    hoverinfo="x", showlegend=False))
        true = {r["channel"]: r["true_roi"] for r in p["roi"]}
        fig.add_trace(go.Scatter(x=[true[c] for c in C], y=[L[c] for c in C], mode="markers", name="True ROI",
                                 marker=dict(symbol="x-thin", size=14, color=RED, line=dict(width=3, color=RED)),
                                 hovertemplate="True ROI %{y}: £%{x:.2f}<extra></extra>"))
        fig.add_vline(x=breakeven, line_dash="dot", line_color=INK_3, annotation_text=BE_LABEL,
                      annotation_font_color=INK_3, annotation_font_size=11.5)
        fig.update_xaxes(title="Revenue per £1 spent", tickprefix="£", showgrid=True, gridcolor=GRID,
                         range=[0, max(float(p["roi_draws"].max()), breakeven) * 1.08])
        show(fig, 360)

    section("Channel deep dive")
    ch = st.selectbox("Channel", C, format_func=lambda c: L[c], label_visibility="collapsed")
    row = score.set_index("channel").loc[ch]
    co = A.carryover_summary(p, ch)
    s80, s80_lo, s80_hi = A.saturation_point(p, ch)
    x_seen = float(p["x_max"][C.index(ch)])
    kpi_row([
        ("ROI", money(row["ROI"]), f"90%: {money(row['ROI low'])} to {money(row['ROI high'])}"),
        ("Profit per £1", money(row["Profit per £1"]), f"90%: {money(row['Profit low'])} to {money(row['Profit high'])}"),
        ("Next £1 returns", money(row["Marginal ROI"]), f"{prob(row['P(next £1 profitable)'])} chance it's profitable"),
        ("Carry-over", f"{co['weeks_to_90']:.0f} week{'' if round(co['weeks_to_90']) == 1 else 's'}",
         f"{co['first_week_share']:.0%} of the effect lands in week 1"),
        ("Saturation now", f"{row['Saturation now']:.0%}", f"of max effect · 80% at {gbp(s80)}/week"),
    ])
    st.write("")
    a, b, c_ = st.columns(3)
    s = np.linspace(0, max(current[ch] * 3, s80 * 1.3, x_seen * 1.15), 160)

    def shade_unseen(fig):
        if s.max() > x_seen:
            fig.add_vrect(x0=x_seen, x1=s.max(), fillcolor=INK_3, opacity=0.1, line_width=0)

    with a:
        st.markdown(f"**Carry-over profile**  \n<span style='color:{INK_2};font-size:13px'>Share of a week's spend effect landing in each following week</span>", unsafe_allow_html=True)
        wk = np.arange(len(co["weights_mean"]))
        fig = go.Figure(go.Bar(x=[f"W{w + 1}" for w in wk], y=co["weights_mean"], marker_color=COL[ch],
                               error_y=dict(type="data", symmetric=False, array=co["weights_hi"] - co["weights_mean"],
                                            arrayminus=co["weights_mean"] - co["weights_lo"], color=INK_3, thickness=1.2, width=0),
                               hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        show(fig, 300, legend=False)
    with b:
        st.markdown(f"**Response curve**  \n<span style='color:{INK_2};font-size:13px'>Weekly revenue at a steady weekly spend</span>", unsafe_allow_html=True)
        curves = weekly_response(p, ch, s)
        lo, hi = np.percentile(curves, [5, 95], axis=0)
        fig = go.Figure()
        shade_unseen(fig)
        fig.add_trace(go.Scatter(x=np.r_[s, s[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0),
                                 fillcolor=rgba(COL[ch], 0.18), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=s, y=curves.mean(0), line=dict(color=COL[ch], width=2.5), showlegend=False,
                                 hovertemplate="Spend £%{x:,.0f} → £%{y:,.0f}<extra></extra>"))
        y_now = weekly_response(p, ch, [current[ch]], use_draws=False)[0]
        fig.add_trace(go.Scatter(x=[current[ch]], y=[y_now], mode="markers", marker=dict(size=11, color=INK, line=dict(color="#fff", width=2)),
                                 showlegend=False, hovertemplate="Current: £%{x:,.0f} → £%{y:,.0f}<extra></extra>"))
        fig.add_vline(x=s80, line_dash="dot", line_color=INK_3, annotation_text="80% saturation",
                      annotation_font_color=INK_3, annotation_position="bottom right", annotation_font_size=11)
        fig.update_xaxes(tickprefix="£", tickformat="~s", nticks=4)
        fig.update_yaxes(tickprefix="£", tickformat="~s")
        show(fig, 300, legend=False)
    with c_:
        st.markdown(f"**Marginal ROI**  \n<span style='color:{INK_2};font-size:13px'>Revenue from the next £1 at each spend level</span>", unsafe_allow_html=True)
        m = A.marginal_roi(p, ch, s)
        mlo, mhi = np.percentile(m, [5, 95], axis=0)
        fig = go.Figure()
        shade_unseen(fig)
        fig.add_trace(go.Scatter(x=np.r_[s, s[::-1]], y=np.r_[mhi, mlo[::-1]], fill="toself", line=dict(width=0),
                                 fillcolor=rgba(COL[ch], 0.18), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=s, y=m.mean(0), line=dict(color=COL[ch], width=2.5), showlegend=False,
                                 hovertemplate="Spend £%{x:,.0f}: next £1 returns £%{y:.2f}<extra></extra>"))
        fig.add_hline(y=breakeven, line_dash="dot", line_color=RED, annotation_text="Break-even",
                      annotation_font_color=RED, annotation_position="top left", annotation_font_size=11)
        fig.add_vline(x=current[ch], line_color=INK, line_width=1, annotation_text="Current",
                      annotation_position="bottom right", annotation_font_color=INK_2, annotation_font_size=11)
        fig.update_xaxes(tickprefix="£", tickformat="~s", nticks=4)
        fig.update_yaxes(tickprefix="£", rangemode="tozero")
        show(fig, 300, legend=False)
    if s.max() > x_seen:
        st.caption(f"Grey shading marks weekly spend above the highest week in the data ({gbp(x_seen)}). "
                   "The curves there are extrapolated.")

# ---------------------------------------------------------------------------
# 3. Segments
# ---------------------------------------------------------------------------
with tabs[2]:
    section("Brand vs performance",
            "Short-term ROI by channel type. MMMs tend to undervalue brand channels: their effect on long-term "
            "demand outlasts the 8-week carry-over this model measures.")
    g = A.group_roi(p)
    if not g.empty:
        cols = st.columns(len(g))
        for col, (_, r) in zip(cols, g.iterrows()):
            col.markdown(kpi(f"{r['Group']} · {r['Channels']}", f"{money(r['ROI'])} ROI",
                             f"90%: {money(r['ROI low'])}–{money(r['ROI high'])} · {gbp(r['Spend'])} spend → "
                             f"{gbp(r['Revenue'])} revenue · profit per £1: {money(r['ROI'] * margin - 1)}"),
                         unsafe_allow_html=True)

    if periods is None:
        st.info("Re-run fit_mmm.py to generate the period-level segments.")
    else:
        left, right = st.columns(2)
        with left:
            section("ROI by year", f"Is each channel getting more or less efficient? Dotted line: break-even at {margin:.0%} margin.")
            yr = periods[(periods["period_type"] == "year") & periods["roi"].notna()]
            fig = go.Figure()
            for c in C:
                d = yr[yr["channel"] == c]
                fig.add_trace(go.Bar(x=d["period"], y=d["roi"], name=L[c], marker_color=COL[c],
                                     error_y=dict(type="data", symmetric=False, array=d["roi_95%"] - d["roi"],
                                                  arrayminus=d["roi"] - d["roi_5%"], color=INK_3, thickness=1.2, width=0),
                                     hovertemplate=f"{L[c]} %{{x}}: £%{{y:.2f}}<extra></extra>"))
            fig.add_hline(y=breakeven, line_dash="dot", line_color=INK_3)
            fig.update_layout(barmode="group", bargroupgap=0.08)
            fig.update_yaxes(tickprefix="£", title="Revenue per £1")
            show(fig, 360)
        with right:
            section("Media revenue by season", "Which quarters each channel earns its money in (all years combined).")
            qy = periods[periods["period_type"] == "quarter_of_year"]
            fig = go.Figure()
            for c in C:
                d = qy[qy["channel"] == c]
                fig.add_trace(go.Bar(x=d["period"], y=d["contribution"], name=L[c], marker_color=COL[c],
                                     marker_line=dict(color="#fff", width=2),
                                     hovertemplate=f"{L[c]} %{{x}}: £%{{y:,.0f}}<extra></extra>"))
            fig.update_layout(barmode="stack")
            fig.update_yaxes(tickprefix="£", tickformat="~s", title="Media-driven revenue")
            show(fig, 360)

        q = periods[periods["period_type"] == "quarter"]
        piv = q.pivot(index="channel", columns="period", values="roi").reindex(C)
        z = piv.values
        section("Quarterly ROI heatmap", "Darker cells mean a higher return per £1."
                + (" Blank cells are quarters with no spend." if np.isnan(z).any() else ""))
        text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"£{v:.2f}")(np.nan_to_num(z)))
        fig = go.Figure(go.Heatmap(z=z, x=list(piv.columns), y=[L[c] for c in C], text=text, texttemplate="%{text}",
                                   textfont=dict(size=11), xgap=3, ygap=3,
                                   colorscale=[[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]],
                                   colorbar=dict(title="ROI", thickness=10, outlinewidth=0, tickprefix="£"),
                                   hovertemplate="%{y} · %{x}: £%{z:.2f} per £1<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig, 300, legend=False)

# ---------------------------------------------------------------------------
# 4. Revenue drivers
# ---------------------------------------------------------------------------
with tabs[3]:
    if decomp is None:
        st.info("Re-run fit_mmm.py to generate the revenue decomposition.")
    else:
        drv = A.driver_totals(decomp, C)
        total = drv["Revenue"].sum()
        section("What built total revenue", f"{gbp(total)} of modelled revenue over {n_weeks} weeks, split by driver. "
                "Base demand is weekly revenue at average price, with no media and no promotion.")
        by_label = {L[c]: c for c in C}
        colours = [COL[by_label[d]] if t == "media" else BASE_GREY for d, t in zip(drv["Driver"], drv["Type"])]
        two_way = (drv["Weekly low"] < 0) & (drv["Weekly high"] > 0)
        near_zero = (drv["Type"] == "base") & two_way & (drv["Revenue"].abs() < 0.01 * total)
        labels = ["≈ £0 net" if nz else gbp(v) for v, nz in zip(drv["Revenue"], near_zero)]
        running, bases, peak = 0.0, [], 0.0
        for v in drv["Revenue"]:
            bases.append(running if v >= 0 else running + v)
            running += v
            peak = max(peak, running)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=drv["Driver"], y=drv["Revenue"].abs(), base=bases, marker_color=colours,
                             customdata=drv["Revenue"], text=labels, textposition="outside",
                             textfont=dict(color=INK_2, size=11.5), hovertemplate="%{x}: £%{customdata:,.0f}<extra></extra>",
                             showlegend=False))
        fig.add_trace(go.Bar(x=["Total"], y=[total], marker_color=NAVY, text=[gbp(total)], textposition="outside",
                             textfont=dict(color=INK, size=12), hovertemplate="Total: £%{y:,.0f}<extra></extra>", showlegend=False))
        fig.update_yaxes(tickprefix="£", tickformat="~s", range=[0, max(peak, total) * 1.1])
        show(fig, 440, legend=False)

        swing = drv[near_zero]
        if not swing.empty:
            parts = [f"{r['Driver'].lower()} from {gbp(r['Weekly low'])} to {gbp(r['Weekly high'])} a week"
                     for _, r in swing.iterrows()]
            drivers = [d if i == 0 else d.lower() for i, d in enumerate(swing["Driver"])]
            st.caption(f"{names(drivers)} {'adds' if len(drivers) == 1 else 'add'} almost nothing over the full period "
                       f"because {'it moves' if len(drivers) == 1 else 'they move'} revenue between weeks rather than "
                       f"adding to it: {names(parts)}.")

        section("Driver table", "Total over the period and the range of weekly contributions.")
        t = drv.copy()
        t["Share"] = t["Revenue"] / total
        t["Weekly range"] = [f"{gbp(lo)} every week" if abs(hi - lo) < 1 else f"{gbp(lo)} to {gbp(hi)}"
                             for lo, hi in zip(t["Weekly low"], t["Weekly high"])]
        table(t[["Driver", "Revenue", "Share", "Weekly range"]].rename(columns={"Revenue": "Total"}),
              {"Total": gbp_full, "Share": lambda v: pct(v, 1)})

        section("Media share of revenue over time", "Rolling 8-week share of revenue driven by paid media.")
        share = decomp[C].sum(axis=1).rolling(8, min_periods=1).sum() / decomp["actual"].rolling(8, min_periods=1).sum()
        fig = go.Figure(go.Scatter(x=decomp["date"], y=share, line=dict(color=NAVY, width=2.2),
                                   fill="tozeroy", fillcolor=rgba(NAVY, 0.08), hovertemplate="%{x|%d %b %Y}: %{y:.0%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        show(fig, 300, legend=False)

# ---------------------------------------------------------------------------
# 5. Budget optimiser
# ---------------------------------------------------------------------------
with tabs[4]:
    m_cur, _, _ = summarise(rev_cur)
    m_opt, lo_opt, hi_opt = summarise(rev_opt)
    m_up, lo_up, hi_up = summarise(uplift)
    kpi_row([
        ("Current mix", gbp(m_cur), f"weekly media revenue at {gbp(total_budget)}"),
        ("Optimised mix", gbp(m_opt), f"90%: {gbp(lo_opt)} to {gbp(hi_opt)}"),
        ("Uplift per week", gbp(m_up), f"{pct(m_up / m_cur, 1, True)} revenue · {gbp(m_up * margin)} gross profit"),
        ("Chance it beats current", prob((uplift > 0).mean()), "share of posterior draws with a gain"),
    ])

    # Validation shows which ROIs are overstated; flag any the optimiser wants to grow
    grown = pd.DataFrame()
    if not errors.empty:
        over = errors[(errors["Error"] > 0.05) & (~errors["Inside"] | (errors["Error"] > 0.15))]
        grown = over[[optimal[c] > current_scaled[c] * 1.01 for c in over["channel"]]] if not over.empty else over
    if not grown.empty:
        listed = names(f"<b>{r['Channel']}</b> (estimated {money(r['Estimated'])} vs true {money(r['True ROI'])})"
                       for _, r in grown.iterrows())
        st.markdown(insight("Check before acting",
                            f"Validation against the known truth shows the model overstates {listed}. The optimiser "
                            f"moves budget towards {'it' if len(grown) == 1 else 'them'}, so treat "
                            f"{'that increase' if len(grown) == 1 else 'those increases'} as an upper bound. "
                            "With real data, test it first with a geo or spend-cut experiment.", kind="warn"),
                    unsafe_allow_html=True)

    section("Recommended split", "Current mix (scaled to this budget) vs the optimised mix, within the per-channel limits.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[L[c] for c in C], y=[current_scaled[c] for c in C], name="Current mix",
                         marker_color=BASE_GREY, hovertemplate="%{x}: £%{y:,.0f}/week<extra></extra>"))
    fig.add_trace(go.Bar(x=[L[c] for c in C], y=[optimal[c] for c in C], name="Optimised mix",
                         marker_color=[COL[c] for c in C], hovertemplate="%{x}: £%{y:,.0f}/week<extra></extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.1)
    fig.update_yaxes(tickprefix="£", tickformat="~s", title="Weekly spend")
    show(fig, 340)

    rows = []
    for c in C:
        mroi = A.marginal_roi(p, c, [optimal[c]], use_draws=False)[0]
        rows.append({"Channel": L[c], "Current": current_scaled[c], "Optimised": optimal[c],
                     "Change": optimal[c] / current_scaled[c] - 1 if current_scaled[c] else np.nan, "Next £1": mroi,
                     "Limit": "at minimum" if optimal[c] <= lower[c] * 1.005 else
                              ("at maximum" if optimal[c] >= upper[c] * 0.995 else "")})
    table(pd.DataFrame(rows), {"Current": gbp_full, "Optimised": gbp_full,
                               "Change": lambda v: pct(v, 0, True), "Next £1": money})
    st.caption("At the optimum, the next £1 returns about the same in every channel that isn't at a limit.")

    section("Budget frontier",
            "Best achievable weekly media revenue at each total budget within the per-channel limits, and the gross "
            f"profit left after paying for the media at {A_M}.")

    @st.cache_data(show_spinner="Optimising across budgets…")
    def frontier(min_m, max_m, lo, hi):
        budgets = np.linspace(lo, hi, 15)
        return A.budget_frontier(p, budgets, min_m, max_m)

    fr = frontier(min_mult, max_mult, lo_b, hi_b)
    if not fr.empty:
        fr = fr.assign(Profit=fr["Revenue"] * margin - fr["Budget"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=np.r_[fr["Budget"], fr["Budget"][::-1]], y=np.r_[fr["Revenue high"], fr["Revenue low"][::-1]],
                                 fill="toself", fillcolor=rgba(NAVY, 0.12), line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=fr["Budget"], y=fr["Revenue"], name="Optimised revenue", line=dict(color=NAVY, width=2.5),
                                 customdata=fr["Marginal ROI"],
                                 hovertemplate="Budget £%{x:,.0f}: £%{y:,.0f} revenue · next £1 → £%{customdata:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[current_total], y=[total_response(p, current, use_draws=False)], mode="markers",
                                 name="Today (current mix)", marker=dict(size=12, color=INK_3, line=dict(color="#fff", width=2)),
                                 hovertemplate="Today: £%{x:,.0f} → £%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[total_budget], y=[m_opt], mode="markers", name="This scenario (optimised)",
                                 marker=dict(size=12, color=NAVY, line=dict(color="#fff", width=2)),
                                 hovertemplate="Scenario: £%{x:,.0f} → £%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fr["Budget"], y=fr["Profit"], name="Gross profit after media (right axis)",
                                 line=dict(color=AMBER, width=2, dash="dash"),
                                 hovertemplate="Budget £%{x:,.0f}: £%{y:,.0f} gross profit after media<extra></extra>"),
                      secondary_y=True)
        best = fr.loc[fr["Profit"].idxmax()]
        if best["Budget"] <= fr["Budget"].min():
            note = (f"every extra £1 in this range returns less than £{breakeven:.2f}, so profit is highest at the "
                    f"lowest budget these limits allow ({gbp(best['Budget'])}/week).")
        elif best["Budget"] >= fr["Budget"].max():
            note = (f"every extra £1 in this range returns more than £{breakeven:.2f}, so profit is still rising at the "
                    f"highest budget these limits allow ({gbp(best['Budget'])}/week).")
        else:
            note = f"profit after media peaks at about {gbp(best['Budget'])}/week."
            fig.add_vline(x=best["Budget"], line_dash="dot", line_color=AMBER,
                          annotation_text=f"Profit peaks ~{gbp(best['Budget'])}/week", annotation_font_color=AMBER)
        fig.update_xaxes(tickprefix="£", tickformat="~s", title="Total weekly media budget")
        fig.update_yaxes(tickprefix="£", tickformat="~s", title="Weekly media-driven revenue", secondary_y=False)
        fig.update_yaxes(tickprefix="£", tickformat="~s", title="Gross profit after media", showgrid=False,
                         tickmode="auto", nticks=6, secondary_y=True)
        show(fig, 420, top=60)
        st.caption(f"At {A_M}, {note} This counts short-term revenue only; brand effects beyond the "
                   "8-week carry-over window are not included.")

# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------
with tabs[5]:
    d = p.get("diagnostics", {})
    kpi_row([
        ("Max R-hat", f"{d.get('max_r_hat', np.nan):.3f}", "chains agree below 1.01"),
        ("Min effective samples", f"{d.get('min_ess_bulk', np.nan):,.0f}", "above 400 is reliable"),
        ("Divergences", f"{d.get('divergences', 0)}", "0 means a clean sampler run"),
        ("Truth recovered", f"{int(errors['Inside'].sum())} of {len(errors)}" if not errors.empty else "n/a",
         "channels with true ROI inside the 90% interval"),
        ("Forecast error", pct(holdout["mape"], 1) if holdout else "n/a",
         f"{holdout['coverage_90']:.0%} of {holdout['weeks']} unseen weeks inside the 90% interval" if holdout
         else "re-run fit_mmm.py for the holdout test"),
    ])
    left, right = st.columns(2)
    with left:
        section("Estimated vs true ROI", "The data is simulated, so the right answer is known.")
        if not errors.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=errors["Estimated"], y=errors["Channel"], mode="markers", name="Model estimate (90% interval)",
                                     marker=dict(color=[COL[c] for c in errors["channel"]], size=12, line=dict(color="#fff", width=2)),
                                     error_x=dict(type="data", symmetric=False, array=errors["High"] - errors["Estimated"],
                                                  arrayminus=errors["Estimated"] - errors["Low"], color=INK_3, thickness=2.5, width=0),
                                     hovertemplate="%{y}: £%{x:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=errors["True ROI"], y=errors["Channel"], mode="markers", name="True ROI",
                                     marker=dict(symbol="x-thin", size=14, line=dict(width=3, color=RED), color=RED),
                                     hovertemplate="True %{y}: £%{x:.2f}<extra></extra>"))
            fig.update_xaxes(tickprefix="£", showgrid=True, gridcolor=GRID, title="Revenue per £1")
            show(fig, 340)
    with right:
        fit = p.get("fit", {})
        sub = (f"In-sample R² {fit['r2']:.2f}, MAPE {pct(fit['mape'], 1)}. " if fit else "")
        if holdout and holdout_df is not None:
            sub += f"Shaded: the last {holdout['weeks']} weeks, forecast by a model fitted without them."
        section("Actual vs modelled revenue", sub)
        if decomp is not None:
            fig = go.Figure()
            if holdout and holdout_df is not None:
                h0, h1 = holdout_df["date"].min(), holdout_df["date"].max()
                fig.add_vrect(x0=h0 - pd.Timedelta(days=3), x1=h1 + pd.Timedelta(days=3), fillcolor=INK_3, opacity=0.08,
                              line_width=0)
                fig.add_trace(go.Scatter(x=np.r_[holdout_df["date"], holdout_df["date"][::-1]],
                                         y=np.r_[holdout_df["forecast_95%"], holdout_df["forecast_5%"][::-1]],
                                         fill="toself", fillcolor=rgba(RED, 0.12), line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=decomp["date"], y=decomp["actual"], name="Actual", line=dict(color=INK, width=1.6)))
            fig.add_trace(go.Scatter(x=decomp["date"], y=decomp["fitted"], name="Model (in-sample)",
                                     line=dict(color=COL["tv"], width=2, dash="dot")))
            if holdout and holdout_df is not None:
                fig.add_trace(go.Scatter(x=holdout_df["date"], y=holdout_df["forecast"], name="Forecast (unseen weeks)",
                                         line=dict(color=RED, width=2.2)))
            fig.update_layout(hovermode="x unified")
            fig.update_yaxes(tickprefix="£", tickformat="~s")
            show(fig, 340)

    if not errors.empty:
        section("How close is each estimate?")
        e = errors.assign(Interval=[f"{money(a)}–{money(b)}" for a, b in zip(errors["Low"], errors["High"])],
                          Inside=np.where(errors["Inside"], "Yes", "No"))
        table(e[["Channel", "True ROI", "Estimated", "Interval", "Error", "Inside"]]
              .rename(columns={"Interval": "90% interval", "Inside": "Truth inside interval"}),
              {"True ROI": money, "Estimated": money, "Error": lambda v: pct(v, 0, True)})

    if not errors_uncal.empty and not errors.empty:
        section("What the lift tests changed",
                f"The same model fitted without and with spend-cut experiments on {names(lifts['Channel'])}. "
                "Experiments pin down channels the observational data can't separate from demand."
                if not lifts.empty else "The same model fitted without and with lift tests.")
        u = errors_uncal.set_index("channel")
        fig = go.Figure()
        for _, r in errors.iterrows():
            fig.add_trace(go.Scatter(x=[u.loc[r["channel"], "Estimated"], r["Estimated"]], y=[r["Channel"]] * 2, mode="lines",
                                     line=dict(color=LINE, width=5), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[u.loc[c, "Estimated"] for c in errors["channel"]], y=errors["Channel"], mode="markers",
                                 name="Without lift tests", marker=dict(size=12, color="#fff", line=dict(color=INK_3, width=2)),
                                 hovertemplate="%{y} without lift tests: £%{x:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=errors["Estimated"], y=errors["Channel"], mode="markers", name="With lift tests",
                                 marker=dict(size=12, color=[COL[c] for c in errors["channel"]], line=dict(color="#fff", width=2)),
                                 hovertemplate="%{y} with lift tests: £%{x:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=errors["True ROI"], y=errors["Channel"], mode="markers", name="True ROI",
                                 marker=dict(symbol="x-thin", size=14, line=dict(width=3, color=RED), color=RED),
                                 hovertemplate="True %{y}: £%{x:.2f}<extra></extra>"))
        fig.update_xaxes(tickprefix="£", showgrid=True, gridcolor=GRID, title="Revenue per £1", rangemode="tozero")
        show(fig, 300)
        cmp_ = pd.DataFrame({
            "Channel": errors["Channel"], "True ROI": errors["True ROI"],
            "Without lift tests": [u.loc[c, "Estimated"] for c in errors["channel"]],
            "Error before": [u.loc[c, "Error"] for c in errors["channel"]],
            "With lift tests": errors["Estimated"], "Error after": errors["Error"],
            "Tested": ["Yes" if c in set(lifts.get("channel", [])) else "No" for c in errors["channel"]],
        })
        table(cmp_, {"True ROI": money, "Without lift tests": money, "With lift tests": money,
                     "Error before": lambda v: pct(v, 0, True), "Error after": lambda v: pct(v, 0, True)})

    if not lifts.empty:
        section("The lift tests",
                "Each test cut a channel's weekly spend and measured the weekly revenue lost. Real tests have error too; "
                "the true loss is shown because the data is simulated.")
        lt = pd.DataFrame({
            "Channel": lifts["Channel"],
            "Spend cut": [f"{gbp(a)} → {gbp(b)} a week" for a, b in zip(lifts["Spend from"], lifts["Spend to"])],
            "Measured change": [f"{gbp(m)} ± {gbp(e)}" for m, e in zip(lifts["Measured"], lifts["Test error"])],
            "True change": [gbp(v) for v in lifts["True"]],
            "Model now implies": [f"{gbp(m)} ({gbp(lo)} to {gbp(hi)})"
                                  for m, lo, hi in zip(lifts["Model"], lifts["Model low"], lifts["Model high"])],
        })
        table(lt, {})

    if periods is not None and "true_contribution" in periods:
        section("Quarterly contributions vs truth", "Each dot is one channel in one quarter. Dots on the diagonal are exactly right; "
                "dots above it mean the model over-credits that channel.")
        q = periods[periods["period_type"] == "quarter"]
        fig = make_subplots(rows=1, cols=len(C), subplot_titles=[L[c] for c in C], horizontal_spacing=0.06)
        for j, c in enumerate(C, start=1):
            dq = q[q["channel"] == c]
            mx = max(dq["true_contribution"].max(), dq["contribution"].max()) * 1.05
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=LINE, width=1.5, dash="dot"),
                                     hoverinfo="skip", showlegend=False), row=1, col=j)
            fig.add_trace(go.Scatter(x=dq["true_contribution"], y=dq["contribution"], mode="markers", name=L[c],
                                     marker=dict(color=COL[c], size=9, line=dict(color="#fff", width=1.5)), showlegend=False,
                                     text=dq["period"], hovertemplate="%{text}<br>True £%{x:,.0f} · Model £%{y:,.0f}<extra></extra>"),
                          row=1, col=j)
        fig.update_xaxes(tickprefix="£", tickformat="~s", title_text="True", nticks=3, tickangle=0)
        fig.update_yaxes(tickprefix="£", tickformat="~s", nticks=5)
        fig.update_yaxes(title_text="Model", row=1, col=1)
        fig.update_annotations(font=dict(size=13, color=INK))
        show(fig, 330, legend=False, top=44)

    section("Known limitations")
    notes = []
    over = errors[(errors["Error"] > 0.1) | ((errors["Error"] > 0) & ~errors["Inside"])] if not errors.empty else errors
    under = errors[(errors["Error"] < -0.1) | ((errors["Error"] < 0) & ~errors["Inside"])] if not errors.empty else errors
    if not over.empty:
        notes.append("The model **overstates** " + names(f"{r['Channel']} ({pct(r['Error'], 0, True)})" for _, r in over.iterrows())
                     + ". Small channels, and channels whose spend moves with demand, are hard to separate from the baseline.")
    if not under.empty:
        notes.append("It **understates** " + names(f"{r['Channel']} ({pct(r['Error'], 0, True)})" for _, r in under.iterrows()) + ".")
    if lifts.empty:
        notes.append("Paid search spend rises in Q4 when demand rises anyway, so the model gives search some credit that "
                     "belongs to seasonality. Real MMMs fix this with spend-cut experiments or informative priors.")
    else:
        untested = [L[c] for c in C if c not in set(lifts["channel"])]
        e0 = errors_uncal.set_index("channel")["Error"].get("paid_search") if not errors_uncal.empty else None
        e1 = errors.set_index("channel")["Error"].get("paid_search") if not errors.empty else None
        if e0 is not None and e1 is not None:
            notes.append(f"Paid search spend rises in Q4 when demand rises anyway, so the observational data alone "
                         f"misjudges search ROI by {pct(e0, 0, True)}. With its lift test the error is {pct(e1, 0, True)}.")
        else:
            notes.append("Paid search spend rises in Q4 when demand rises anyway, which the observational data alone "
                         "can't separate from seasonality.")
        if untested:
            notes.append(f"{names(untested)} {'has' if len(untested) == 1 else 'have'} no lift test, so "
                         f"{'its estimate rests' if len(untested) == 1 else 'their estimates rest'} on the observational "
                         "data alone.")
    notes.append("ROI here is short-term revenue. Brand effects that build over months are not captured.")
    st.markdown("\n".join(f"- {n}" for n in notes))
