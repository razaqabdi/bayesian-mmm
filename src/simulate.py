"""Simulate 3 years of weekly revenue for a fictional UK ecommerce brand.

Every channel effect is known, so we can check whether the Bayesian MMM
recovers the truth. Outputs:
  data/mmm_data.csv            model input (spend, controls, revenue)
  data/ground_truth.json       true parameters and true ROI per channel
  data/true_contributions.csv  true weekly revenue driven by each channel
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transforms import geometric_adstock, logistic_saturation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = np.random.default_rng(7)
N_WEEKS = 156
L_MAX = 8
NOISE_SD = 15_000  # weekly revenue noise (£), roughly 4% of average weekly revenue

CHANNELS = ["tv", "paid_search", "paid_social", "display"]
# True media parameters (lam acts on spend scaled to its max, as in PyMC-Marketing)
TRUE_ALPHA = {"tv": 0.65, "paid_search": 0.10, "paid_social": 0.35, "display": 0.20}
TRUE_LAM = {"tv": 2.0, "paid_search": 3.0, "paid_social": 2.5, "display": 4.0}
TARGET_ROI = {"tv": 1.4, "paid_search": 3.2, "paid_social": 2.1, "display": 0.7}


def simulate_spend(dates):
    q4 = ((dates.month >= 10) | (dates.month == 1)).astype(float)

    # TV: 4-week flights, roughly 6 per year, off otherwise
    tv = np.zeros(N_WEEKS)
    starts = np.sort(RNG.choice(np.arange(0, N_WEEKS - 4, 6), size=18, replace=False))
    for s in starts:
        tv[s : s + 4] = RNG.uniform(45_000, 90_000)

    # Search: always on, a little higher in Q4, with quarterly budget changes
    # and two planned pull-back tests (these give the model variation to learn from)
    quarterly = np.repeat(RNG.uniform(10_000, 30_000, N_WEEKS // 13 + 1), 13)[:N_WEEKS]
    search = quarterly + 2_000 * q4 + RNG.normal(0, 2_000, N_WEEKS)
    search[60:66] *= 0.3
    search[130:136] *= 0.3

    # Social: always on with slow-moving test budgets
    walk = np.cumsum(RNG.normal(0, 1_200, N_WEEKS))
    social = 14_000 + walk - walk.mean() + RNG.normal(0, 1_500, N_WEEKS)

    # Display: on/off bursts with a couple of dark periods
    display = RNG.uniform(5_000, 14_000, N_WEEKS)
    display[40:52] = 0
    display[110:118] = 0

    spend = pd.DataFrame({"tv": tv, "paid_search": search, "paid_social": social, "display": display})
    return spend.clip(lower=0).round(0)


def main():
    DATA.mkdir(exist_ok=True)
    dates = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")
    spend = simulate_spend(dates)

    # Controls
    promo = np.zeros(N_WEEKS)
    promo[RNG.choice(N_WEEKS, size=24, replace=False)] = 1
    price_index = 1 + 0.03 * np.sin(np.arange(N_WEEKS) / 9) + RNG.normal(0, 0.01, N_WEEKS) - 0.08 * promo

    # Baseline demand: level + trend + yearly seasonality (Fourier order 2, as in the model)
    t = np.arange(N_WEEKS)
    doy = dates.dayofyear.values / 365.25
    seasonality = (
        9_000 * np.sin(2 * np.pi * doy) - 14_000 * np.cos(2 * np.pi * doy)
        + 4_000 * np.sin(4 * np.pi * doy) + 2_500 * np.cos(4 * np.pi * doy)
    )
    baseline = 220_000 + 150 * t + seasonality
    price_effect = -150_000 * (price_index - 1)
    promo_effect = 14_000 * promo

    # Channel effects, scaled so true ROI hits the targets exactly
    contributions, truth = {}, {}
    for c in CHANNELS:
        x_scaled = spend[c].values / spend[c].max()
        sat = logistic_saturation(geometric_adstock(x_scaled, TRUE_ALPHA[c], L_MAX), TRUE_LAM[c])
        beta = TARGET_ROI[c] * spend[c].sum() / sat.sum()  # revenue (£) per week at full saturation
        contributions[c] = beta * sat
        truth[c] = {
            "alpha": TRUE_ALPHA[c], "lam": TRUE_LAM[c], "beta_gbp": round(float(beta), 2),
            "total_spend": float(spend[c].sum()), "total_contribution": float(contributions[c].sum()),
            "roi": TARGET_ROI[c],
        }

    # Unexplained week-to-week noise (weather, competitors, stock-outs...). About 4% of weekly
    # revenue, so the model can't fit perfectly - real weekly MMMs typically miss by 3-10%.
    noise = RNG.normal(0, NOISE_SD, N_WEEKS)
    revenue = baseline + price_effect + promo_effect + sum(contributions.values()) + noise

    df = pd.DataFrame({"date": dates, **spend, "price_index": price_index.round(4),
                       "promo": promo.astype(int), "revenue": revenue.round(0)})
    df.to_csv(DATA / "mmm_data.csv", index=False)
    pd.DataFrame({"date": dates, **{c: v.round(0) for c, v in contributions.items()}}).to_csv(DATA / "true_contributions.csv", index=False)
    with open(DATA / "ground_truth.json", "w") as f:
        json.dump({"l_max": L_MAX, "channels": truth}, f, indent=2)

    lifts = simulate_lift_tests(spend, truth)
    lifts.to_csv(DATA / "lift_tests.csv", index=False)

    media_share = sum(c.sum() for c in contributions.values()) / revenue.sum()
    print(f"Simulated {N_WEEKS} weeks: revenue £{revenue.sum()/1e6:.1f}m, media-driven share {media_share:.0%}")
    for c in CHANNELS:
        print(f"  {c:<12} spend £{spend[c].sum()/1e6:5.2f}m  true ROI {TARGET_ROI[c]:.1f}")
    print("Simulated lift tests (weekly revenue change when spend is cut):")
    for _, r in lifts.iterrows():
        print(f"  {r['channel']:<12} spend £{r['x']:,.0f} -> £{r['x'] + r['delta_x']:,.0f}/week: "
              f"measured £{r['delta_y']:,.0f} ± £{r['sigma']:,.0f} (true £{r['true_delta_y']:,.0f})")


# Spend-cut experiments (e.g. geo holdouts) on the digital channels, which are easy to switch off.
# TV is left untested: TV geo tests are expensive, so many brands don't run them.
# Each test cuts a channel's steady weekly spend and measures the weekly revenue lost, with the
# measurement error a real test would have.
LIFT_TESTS = {  # channel: (share of spend cut, test standard error as a share of the true effect)
    "paid_search": (0.7, 0.15),
    "paid_social": (0.5, 0.20),
    "display": (1.0, 0.25),
}


def simulate_lift_tests(spend, truth, seed=11):
    rng = np.random.default_rng(seed)  # separate stream, so the main data doesn't change
    rows = []
    for c, (cut, rel_se) in LIFT_TESTS.items():
        x = float(spend[c].mean())
        dx = -cut * x
        x_max = float(spend[c].max())
        t = truth[c]
        # Steady weekly spend s gives adstocked spend s (normalised adstock), so the long-run
        # weekly effect is beta * saturation(s / x_max)
        true_dy = t["beta_gbp"] * (logistic_saturation((x + dx) / x_max, t["lam"]) - logistic_saturation(x / x_max, t["lam"]))
        sigma = rel_se * abs(true_dy)
        rows.append({"channel": c, "x": round(x, 0), "delta_x": round(dx, 0),
                     "delta_y": round(float(true_dy + rng.normal(0, sigma)), 0), "sigma": round(float(sigma), 0),
                     "true_delta_y": round(float(true_dy), 0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
