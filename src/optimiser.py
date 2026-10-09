"""Response curves and budget optimisation from the fitted model's posterior.

Uses outputs/model_params.json written by fit_mmm.py, so the app and this module
need only numpy/scipy, not PyMC.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from transforms import logistic_saturation

PARAMS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model_params.json"


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        p = json.load(f)
    for k in ("alpha", "lam", "beta"):
        p["draws"][k] = np.asarray(p["draws"][k])  # shape (n_draws, n_channels)
    return p


def weekly_response(p, channel, spend, use_draws=True):
    """Weekly revenue (£) from a steady weekly spend in one channel.

    With normalised adstock, steady spend s gives adstocked spend s, so the
    long-run weekly effect is beta * y_max * saturation(lam * s / x_max).
    Returns shape (n_draws, len(spend)) or (len(spend),) for the posterior mean.
    """
    i = p["channels"].index(channel)
    s = np.atleast_1d(np.asarray(spend, dtype=float)) / p["x_max"][i]
    lam = p["draws"]["lam"][:, i][:, None]
    beta = p["draws"]["beta"][:, i][:, None]
    curves = beta * p["y_max"] * logistic_saturation(lam * s[None, :], 1.0) * p["scale_correction"][i]
    return curves if use_draws else curves.mean(axis=0)


def total_response(p, allocation, use_draws=True):
    """Total weekly media revenue for an allocation {channel: weekly spend}."""
    parts = [weekly_response(p, c, [allocation[c]], use_draws)[..., 0] for c in p["channels"]]
    return np.sum(parts, axis=0)


def optimise(p, total_budget, lower=None, upper=None):
    """Split a weekly budget across channels to maximise expected revenue.

    lower/upper: dicts of per-channel weekly spend bounds.
    """
    ch = p["channels"]
    lower = lower or {c: 0.0 for c in ch}
    upper = upper or {c: total_budget for c in ch}
    bounds = [(lower[c], upper[c]) for c in ch]
    if sum(b[0] for b in bounds) > total_budget or sum(b[1] for b in bounds) < total_budget:
        raise ValueError("Budget is outside what the channel bounds allow.")

    def neg_rev(x):
        return -total_response(p, dict(zip(ch, x)), use_draws=False)

    # start from a feasible split proportional to the bound range
    x0 = np.array([lo + (hi - lo) * 0.5 for lo, hi in bounds])
    x0 = np.clip(x0 * total_budget / x0.sum(), [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(neg_rev, x0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - total_budget}],
                   options={"maxiter": 500, "ftol": 1e-9})
    return dict(zip(ch, res.x))


def summarise(draws, lo=5, hi=95):
    return float(np.mean(draws)), float(np.percentile(draws, lo)), float(np.percentile(draws, hi))
