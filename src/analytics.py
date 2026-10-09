"""Deeper analytics on top of the fitted model: segments, carry-over, saturation,
marginal ROI, budget frontier and revenue drivers. Pure numpy/pandas, no PyMC."""
from pathlib import Path

import numpy as np
import pandas as pd

from optimiser import optimise, total_response, weekly_response

OUT = Path(__file__).resolve().parents[1] / "outputs"

# Fixed colour per channel (validated categorical order); never re-assigned by rank
CHANNEL_META = {
    "tv":          {"label": "TV",          "colour": "#2a78d6", "group": "Brand"},
    "paid_search": {"label": "Paid Search", "colour": "#eb6834", "group": "Performance"},
    "paid_social": {"label": "Paid Social", "colour": "#1baf7a", "group": "Performance"},
    "display":     {"label": "Display",     "colour": "#eda100", "group": "Brand"},
}
DRIVER_LABELS = {"intercept": "Base demand", "trend": "Growth trend", "seasonality": "Seasonality",
                 "price_index": "Pricing", "promo": "Promotions"}


def label(c):
    return CHANNEL_META.get(c, {}).get("label", c.replace("_", " ").title())


def load_table(name):
    path = OUT / name
    return pd.read_csv(path, parse_dates=["date"] if name == "decomposition.csv" else None) if path.exists() else None


# ---- Carry-over ------------------------------------------------------------
def adstock_weights(alpha, l_max):
    w = alpha ** np.arange(l_max)
    return w / w.sum()


def carryover_summary(p, channel):
    """Share of effect in the week of spend, and weeks needed to deliver 90% of it (posterior mean + 90% CI)."""
    i = p["channels"].index(channel)
    alphas = p["draws"]["alpha"][:, i]
    first_week, weeks90, weights = [], [], []
    for a in alphas:
        w = adstock_weights(a, p["l_max"])
        weights.append(w)
        first_week.append(w[0])
        weeks90.append(int(np.searchsorted(np.cumsum(w), 0.9) + 1))
    weights = np.array(weights)
    return {
        "alpha": float(alphas.mean()),
        "first_week_share": float(np.mean(first_week)),
        "weeks_to_90": float(np.median(weeks90)),
        "weights_mean": weights.mean(0), "weights_lo": np.percentile(weights, 5, axis=0),
        "weights_hi": np.percentile(weights, 95, axis=0),
    }


# ---- Saturation and marginal returns ---------------------------------------
def saturation_point(p, channel, level=0.8):
    """Weekly spend at which a channel reaches `level` of its maximum possible effect."""
    i = p["channels"].index(channel)
    z = 2 * np.arctanh(level)  # logistic saturation equals tanh(z/2)
    s = z * p["x_max"][i] / p["draws"]["lam"][:, i]
    return float(np.median(s)), float(np.percentile(s, 5)), float(np.percentile(s, 95))


def saturation_level(p, channel, spend):
    """Share of a channel's maximum possible effect reached at a steady weekly spend (posterior draws).

    Logistic saturation (1 - e^-x) / (1 + e^-x) equals tanh(x / 2).
    """
    i = p["channels"].index(channel)
    s = np.atleast_1d(np.asarray(spend, dtype=float)) / p["x_max"][i]
    lam = p["draws"]["lam"][:, i][:, None]
    return np.tanh(lam * s[None, :] / 2)


def marginal_roi(p, channel, spend, step=100.0, use_draws=True):
    """Revenue from the next £1 at a given weekly spend."""
    s = np.atleast_1d(np.asarray(spend, dtype=float))
    hi = weekly_response(p, channel, s + step, use_draws)
    lo = weekly_response(p, channel, s, use_draws)
    return (hi - lo) / step


def channel_scorecard(p, margin=0.5):
    """One row per channel: spend, ROI and profit per £1 with intervals, marginal ROI now, saturation, carry-over.

    margin: gross margin on revenue. £1 of media breaks even when it drives £1 / margin of revenue.
    """
    roi = {r["channel"]: r for r in p.get("roi", [])}
    draws = np.asarray(p.get("roi_draws", []))
    rows = []
    for j, c in enumerate(p["channels"]):
        cur = float(p["avg_weekly_spend"][c])
        m = marginal_roi(p, c, [cur])[:, 0]
        sat, _, _ = saturation_point(p, c)
        sat_now = saturation_level(p, c, [cur])[:, 0]
        co = carryover_summary(p, c)
        r = roi.get(c, {})
        est = r.get("est_roi", np.nan)
        if draws.size:
            profit = draws[:, j] * margin - 1
            p_mean, p_lo, p_hi = float(profit.mean()), float(np.percentile(profit, 5)), float(np.percentile(profit, 95))
            p_win = float((profit > 0).mean())
        else:
            p_mean = est * margin - 1
            p_lo, p_hi = r.get("roi_5%", np.nan) * margin - 1, r.get("roi_95%", np.nan) * margin - 1
            p_win = np.nan
        rows.append({
            "channel": c, "Channel": label(c), "Group": CHANNEL_META.get(c, {}).get("group", ""),
            "Weekly spend": cur, "ROI": est, "ROI low": r.get("roi_5%", np.nan),
            "ROI high": r.get("roi_95%", np.nan), "True ROI": r.get("true_roi", np.nan),
            "Profit per £1": p_mean, "Profit low": p_lo, "Profit high": p_hi, "P(profitable)": p_win,
            "Marginal ROI": float(m.mean()), "P(next £1 profitable)": float((m * margin > 1).mean()),
            "Saturation now": float(np.median(sat_now)),
            "80% saturation spend": sat, "Headroom": sat / cur - 1 if cur else np.nan,
            "Weeks to 90% effect": co["weeks_to_90"], "Same-week share": co["first_week_share"],
        })
    return pd.DataFrame(rows)


def lift_check(p):
    """Each lift test next to what the fitted model now implies for the same spend change."""
    rows = []
    for t in (p.get("calibration") or {}).get("tests", []):
        c = t["channel"]
        if c not in p["channels"]:
            continue
        implied = (weekly_response(p, c, [t["x"] + t["delta_x"]]) - weekly_response(p, c, [t["x"]]))[:, 0]
        rows.append({"channel": c, "Channel": label(c), "Spend from": t["x"], "Spend to": t["x"] + t["delta_x"],
                     "Measured": t["delta_y"], "Test error": t["sigma"], "True": t.get("true_delta_y", np.nan),
                     "Model": float(implied.mean()), "Model low": float(np.percentile(implied, 5)),
                     "Model high": float(np.percentile(implied, 95))})
    return pd.DataFrame(rows)


def roi_errors(p, key="roi"):
    """Estimated vs true ROI per channel (only possible because the data is simulated)."""
    rows = []
    for r in p.get(key, []):
        rows.append({"channel": r["channel"], "Channel": label(r["channel"]), "True ROI": r["true_roi"],
                     "Estimated": r["est_roi"], "Low": r["roi_5%"], "High": r["roi_95%"],
                     "Error": r["est_roi"] / r["true_roi"] - 1, "Inside": bool(r["truth_in_90%_interval"])})
    return pd.DataFrame(rows)


# ---- Segments --------------------------------------------------------------
def group_roi(p):
    """ROI for Brand vs Performance groups with 90% intervals, using joint posterior draws."""
    draws = np.asarray(p.get("roi_draws", []))
    if draws.size == 0:
        return pd.DataFrame()
    spend = np.array([p["total_spend"][c] for c in p["channels"]])
    contrib = draws * spend  # (samples, channels) total revenue per channel
    rows = []
    for g in ["Brand", "Performance"]:
        idx = [i for i, c in enumerate(p["channels"]) if CHANNEL_META.get(c, {}).get("group") == g]
        if not idx:
            continue
        r = contrib[:, idx].sum(1) / spend[idx].sum()
        rows.append({"Group": g, "Channels": ", ".join(label(p["channels"][i]) for i in idx),
                     "Spend": float(spend[idx].sum()), "Revenue": float(contrib[:, idx].sum(1).mean()),
                     "ROI": float(r.mean()), "ROI low": float(np.percentile(r, 5)),
                     "ROI high": float(np.percentile(r, 95))})
    return pd.DataFrame(rows)


def share_table(p):
    """Spend share vs revenue share per channel; efficiency index = revenue share / spend share."""
    draws = np.asarray(p.get("roi_draws", []))
    spend = pd.Series(p["total_spend"])
    if draws.size:
        rev = pd.Series((draws * spend.values).mean(0), index=p["channels"])
    else:
        rev = pd.Series({r["channel"]: r["est_roi"] * r["spend_gbp"] for r in p["roi"]})
    t = pd.DataFrame({"channel": p["channels"], "Channel": [label(c) for c in p["channels"]],
                      "Spend share": (spend / spend.sum()).values,
                      "Revenue share": (rev / rev.sum()).reindex(p["channels"]).values})
    t["Efficiency index"] = t["Revenue share"] / t["Spend share"]
    return t


# ---- Drivers ---------------------------------------------------------------
def driver_totals(decomp, channels):
    """Total revenue by driver over the whole period, in reading order for a waterfall,
    with each driver's lowest and highest weekly contribution."""
    order = [c for c in ["intercept", "trend", "seasonality", "price_index", "promo"] if c in decomp]
    rows = [(DRIVER_LABELS[c], float(decomp[c].sum()), "base", float(decomp[c].min()), float(decomp[c].max()))
            for c in order]
    rows += [(label(c), float(decomp[c].sum()), "media", float(decomp[c].min()), float(decomp[c].max()))
             for c in channels if c in decomp]
    return pd.DataFrame(rows, columns=["Driver", "Revenue", "Type", "Weekly low", "Weekly high"])


# ---- Budget frontier -------------------------------------------------------
def budget_frontier(p, budgets, min_mult=0.0, max_mult=3.0):
    """Best achievable weekly media revenue at each total budget, with 90% interval and marginal ROI."""
    cur = {c: float(p["avg_weekly_spend"][c]) for c in p["channels"]}
    lower = {c: cur[c] * min_mult for c in cur}
    upper = {c: cur[c] * max_mult for c in cur}
    rows = []
    for b in budgets:
        try:
            alloc = optimise(p, b, lower, upper)
        except ValueError:
            continue
        rev = total_response(p, alloc)
        rev_mean = total_response(p, alloc, use_draws=False)
        step = 1000.0
        try:  # forward difference; fall back to backward at the top of the feasible range
            mroi = (total_response(p, optimise(p, b + step, lower, upper), use_draws=False) - rev_mean) / step
        except ValueError:
            mroi = (rev_mean - total_response(p, optimise(p, b - step, lower, upper), use_draws=False)) / step
        rows.append({"Budget": b, "Revenue": float(rev.mean()), "Revenue low": float(np.percentile(rev, 5)),
                     "Revenue high": float(np.percentile(rev, 95)), "Marginal ROI": float(mroi),
                     **{f"alloc_{c}": v for c, v in alloc.items()}})
    return pd.DataFrame(rows)
