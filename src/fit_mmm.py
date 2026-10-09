"""Fit a Bayesian marketing mix model with PyMC-Marketing and check it against the truth.

Run after simulate.py. Writes to outputs/:
  model_params.json     posterior draws + scaling used by the optimiser and app
  roi_validation.csv    estimated vs true ROI with 90% credible intervals
  roi_validation.png    the same as a chart
  decomposition.csv     weekly revenue split by driver, plus in-sample fit
  holdout.csv           out-of-sample forecast for the last 13 weeks (model fitted without them)
If data/lift_tests.csv exists, the model is calibrated with those experiments, and an uncalibrated
fit is run alongside so the app can show what the experiments changed.
  contributions.png     revenue decomposition over time
  response_curves.png   diminishing-returns curves with uncertainty bands
and refreshes the results table in README.md.
"""
import json
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from transforms import geometric_adstock, logistic_saturation
from optimiser import weekly_response

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
CHANNELS = ["tv", "paid_search", "paid_social", "display"]
CONTROLS = ["price_index", "promo", "trend"]
L_MAX = 8
N_KEEP = 400  # posterior draws saved for the app
HOLDOUT_WEEKS = 13  # last quarter held back to test out-of-sample accuracy
SAMPLER = dict(chains=4, draws=1000, tune=1500, target_accept=0.95, random_seed=42, progressbar=False)
HOLDOUT_SAMPLER = dict(chains=4, draws=500, tune=1000, target_accept=0.95, random_seed=7, progressbar=False)
LIFT_PATH = DATA / "lift_tests.csv"
CALIBRATE = True  # calibrate with the lift tests in data/lift_tests.csv when the file exists
LIFT_COLS = ["channel", "x", "delta_x", "delta_y", "sigma"]
NAMES = {"tv": "TV", "paid_search": "Paid Search", "paid_social": "Paid Social", "display": "Display"}
COLOURS = {"tv": "#2a6fdb", "paid_search": "#e07a1f", "paid_social": "#2f9e6e", "display": "#b4447c"}


def posterior_var(post, suffix):
    """Find a channel-level parameter by name suffix (robust to prefix changes across versions)."""
    for name in post.data_vars:
        if name.endswith(suffix) and "channel" in post[name].dims:
            return post[name]
    raise KeyError(f"No posterior variable ending in '{suffix}' with a channel dimension")


def _has(node, name):
    """True if a variable exists in an InferenceData / DataTree group."""
    try:
        return name in node.data_vars
    except Exception:  # noqa: BLE001
        return False


def model_scales(idata, df, y):
    """The divisors PyMC-Marketing used to scale spend and revenue (falls back to column maxima)."""
    x_max = df[CHANNELS].abs().max().values.astype(float)
    y_max = float(y.abs().max())
    try:
        cd = idata.constant_data
        if _has(cd, "channel_scale"):
            x_max = cd["channel_scale"].sel(channel=CHANNELS).values.astype(float)
        if _has(cd, "target_scale"):
            y_max = float(np.asarray(cd["target_scale"].values).ravel()[0])
    except Exception:  # noqa: BLE001
        pass
    return x_max, y_max


def channel_contributions(idata, df, alpha, lam, beta, x_max, y_max):
    """Channel contributions in GBP, dims (chain, draw, date, channel), whatever the library version stores."""
    post = idata.posterior
    if _has(post, "channel_contribution_original_scale"):
        print("Contributions: taken from channel_contribution_original_scale")
        return post["channel_contribution_original_scale"].sel(channel=CHANNELS)
    if _has(post, "channel_contribution"):
        print("Contributions: channel_contribution x target scale")
        return post["channel_contribution"].sel(channel=CHANNELS) * y_max
    print("Contributions: rebuilt from posterior parameters")
    dims = ("chain", "draw", "channel")
    a = alpha.sel(channel=CHANNELS).transpose(*dims).values
    lm = lam.sel(channel=CHANNELS).transpose(*dims).values
    b = beta.sel(channel=CHANNELS).transpose(*dims).values
    n_chain, n_draw, _ = a.shape
    out = np.empty((n_chain, n_draw, len(df), len(CHANNELS)))
    for j, c in enumerate(CHANNELS):
        xs = df[c].values / x_max[j]
        for ci in range(n_chain):
            for di in range(n_draw):
                out[ci, di, :, j] = b[ci, di, j] * y_max * logistic_saturation(
                    geometric_adstock(xs, a[ci, di, j], L_MAX), lm[ci, di, j])
    return xr.DataArray(out, dims=("chain", "draw", "date", "channel"),
                        coords={"chain": alpha["chain"].values, "draw": alpha["draw"].values,
                                "date": df["date"].values, "channel": CHANNELS})


def export_decomposition(idata, df, contrib, y_max):
    """Weekly revenue split into baseline drivers and channels (posterior means), plus fit metrics."""
    post = idata.posterior
    out = pd.DataFrame({"date": df["date"].values, "actual": df["revenue"].values})
    mean_contrib = contrib.mean(dim=("chain", "draw")).transpose("date", "channel")
    for c in CHANNELS:
        out[c] = mean_contrib.sel(channel=c).values

    names = list(post.data_vars)
    found = []
    for name in names:
        if "channel" in name or "original_scale" in name or "total" in name or "contribution" not in name:
            continue
        v = post[name].mean(dim=("chain", "draw"))
        scale = y_max
        if "date" not in v.dims:  # e.g. a scalar intercept
            if v.ndim == 0 and "intercept" in name:
                out["intercept"] = float(v) * scale
                found.append(name)
            continue
        if "control" in v.dims:
            for ctl in v["control"].values:
                out[str(ctl)] = v.sel(control=ctl).transpose("date").values * scale
            found.append(name)
        elif "seasonality" in name:
            out["seasonality"] = v.transpose("date", ...).values.reshape(len(df), -1).sum(axis=1) * scale
            found.append(name)
        elif "fourier" in name and "seasonality" not in out:
            other = [d for d in v.dims if d != "date"]
            out["seasonality"] = (v.sum(dim=other) if other else v).transpose("date").values * scale
            found.append(name)
        elif "intercept" in name:
            out["intercept"] = v.transpose("date", ...).values.reshape(len(df), -1).sum(axis=1) * scale
            found.append(name)

    driver_cols = [c for c in ["intercept", "trend", "seasonality", "price_index", "promo"] if c in out]
    out["fitted"] = out[driver_cols + CHANNELS].sum(axis=1)
    if "intercept" not in out:  # fall back: absorb the level into an intercept
        out["intercept"] = (out["actual"] - out["fitted"]).mean()
        out["fitted"] += out["intercept"]
        driver_cols = ["intercept"] + driver_cols
    resid = out["actual"] - out["fitted"]
    r2 = 1 - (resid ** 2).sum() / ((out["actual"] - out["actual"].mean()) ** 2).sum()
    mape = (resid.abs() / out["actual"]).mean()
    out.round(2).to_csv(OUT / "decomposition.csv", index=False)
    print(f"Decomposition from: {', '.join(found) or 'channels only'}  |  R2 {r2:.3f}  MAPE {mape:.2%}")
    return {"r2": float(r2), "mape": float(mape)}


def build_mmm():
    return MMM(
        date_column="date",
        channel_columns=CHANNELS,
        control_columns=CONTROLS,
        adstock=GeometricAdstock(l_max=L_MAX),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )


def load_lift_tests():
    if not CALIBRATE or not LIFT_PATH.exists():
        return None
    lift = pd.read_csv(LIFT_PATH)
    return lift if len(lift) else None


def fit_model(X, y, lift=None, sampler=SAMPLER):
    """Fit the MMM. With lift tests, each test's measured revenue change becomes extra data the
    channel's response curve has to match. Returns (mmm, calibrated)."""
    mmm = build_mmm()
    calibrated = False
    if lift is not None:
        try:
            mmm.build_model(X, y)  # the lift tests attach to a built model
            mmm.add_lift_test_measurements(lift[LIFT_COLS].reset_index(drop=True))
            calibrated = True
        except Exception as e:  # noqa: BLE001
            print(f"Could not add lift tests ({type(e).__name__}: {e}); fitting without them.")
            mmm = build_mmm()
    mmm.fit(X, y, **sampler)
    return mmm, calibrated


def roi_summary(idata, df, y, truth):
    """Posterior ROI per channel against the true ROI, plus the pieces the rest of the script reuses."""
    post = idata.posterior
    alpha, lam, beta = (posterior_var(post, s) for s in ("alpha", "lam", "beta"))
    x_max, y_max = model_scales(idata, df, y)
    contrib = channel_contributions(idata, df, alpha, lam, beta, x_max, y_max)
    total = contrib.sum(dim="date")
    spend = df[CHANNELS].sum()
    roi = total.sel(channel=CHANNELS) / xr.DataArray(spend.values, dims="channel", coords={"channel": CHANNELS})
    rows = []
    for c in CHANNELS:
        d = roi.sel(channel=c).values.ravel()
        lo, hi = np.percentile(d, [5, 95])
        rows.append({"channel": c, "true_roi": truth[c]["roi"], "est_roi": float(d.mean()),
                     "roi_5%": float(lo), "roi_95%": float(hi),
                     "truth_in_90%_interval": bool(lo <= truth[c]["roi"] <= hi),
                     "spend_gbp": float(spend[c])})
    return pd.DataFrame(rows), dict(alpha=alpha, lam=lam, beta=beta, x_max=x_max, y_max=y_max,
                                    contrib=contrib, total=total, spend=spend, roi=roi)


def holdout_check(df, n_test=HOLDOUT_WEEKS, lift=None):
    """Refit without the last n_test weeks, forecast them, and score the forecast.

    In-sample fit flatters any model; this is the honest accuracy number.
    """
    train, test = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)
    cols = ["date"] + CHANNELS + CONTROLS
    mmm, _ = fit_model(train[cols], train["revenue"].rename("y"), lift, HOLDOUT_SAMPLER)
    _, y_scale = model_scales(mmm.idata, train, train["revenue"])
    # include_last_observations carries the last weeks of training spend into the forecast,
    # so adstock (carry-over) from earlier spend is not lost at the split
    pp = mmm.sample_posterior_predictive(test[cols], extend_idata=False, include_last_observations=True,
                                         random_seed=7, progressbar=False)
    draws = np.asarray(pp["y"].transpose("date", ...).values, dtype=float).reshape(n_test, -1)
    if np.nanmean(draws) < 100:  # predictions come back on the model's scaled target
        draws = draws * y_scale
    pred = draws.mean(axis=1)
    lo, hi = np.percentile(draws, [5, 95], axis=1)
    actual = test["revenue"].values
    out = pd.DataFrame({"date": test["date"].dt.strftime("%Y-%m-%d"), "actual": actual,
                        "forecast": pred.round(0), "forecast_5%": lo.round(0), "forecast_95%": hi.round(0)})
    out.to_csv(OUT / "holdout.csv", index=False)
    stats = {
        "weeks": int(n_test), "start": str(test["date"].iloc[0].date()),
        "mape": float(np.mean(np.abs(actual - pred) / actual)),
        "coverage_90": float(np.mean((actual >= lo) & (actual <= hi))),
        "bias": float(np.mean(pred - actual) / np.mean(actual)),
    }
    print(f"Holdout ({n_test} weeks from {stats['start']}): MAPE {stats['mape']:.1%}, "
          f"{stats['coverage_90']:.0%} of weeks inside the 90% interval, bias {stats['bias']:+.1%}")
    return stats


def export_period_roi(df, contrib, truth_contrib=None):
    """ROI by year and by quarter, with 90% intervals, for the segment views."""
    arr = contrib.transpose("chain", "draw", "date", "channel").sel(channel=CHANNELS).values
    arr = arr.reshape(-1, arr.shape[2], arr.shape[3])  # (samples, date, channel)
    dates = pd.to_datetime(df["date"])
    periods = {
        "year": dates.dt.year.astype(str).values,
        "quarter": (dates.dt.year.astype(str) + " Q" + dates.dt.quarter.astype(str)).values,
        "quarter_of_year": ("Q" + dates.dt.quarter.astype(str)).values,
    }
    rows = []
    for ptype, labels in periods.items():
        for lab in sorted(set(labels)):
            mask = labels == lab
            for j, c in enumerate(CHANNELS):
                spend = float(df.loc[mask, c].sum())
                tot = arr[:, mask, j].sum(axis=1)
                row = {"period_type": ptype, "period": lab, "channel": c, "spend": spend,
                       "contribution": float(tot.mean()), "contribution_5%": float(np.percentile(tot, 5)),
                       "contribution_95%": float(np.percentile(tot, 95))}
                if spend > 0:
                    r = tot / spend
                    row.update({"roi": float(r.mean()), "roi_5%": float(np.percentile(r, 5)),
                                "roi_95%": float(np.percentile(r, 95))})
                if truth_contrib is not None:
                    row["true_contribution"] = float(truth_contrib.loc[mask, c].sum())
                rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(OUT / "roi_by_period.csv", index=False)
    print(f"Saved ROI by period ({len(rows)} rows)")


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "mmm_data.csv", parse_dates=["date"])
    df["trend"] = np.arange(len(df)) / len(df)
    # Price as % above/below its average. Centring means the pricing effect measures price
    # changes, not "price vs a price of zero" (which made pricing look like -£21m of revenue).
    df["price_index"] = df["price_index"] / df["price_index"].mean() - 1
    truth = json.load(open(DATA / "ground_truth.json"))["channels"]

    X = df[["date"] + CHANNELS + CONTROLS]
    y = df["revenue"].rename("y")  # PyMC-Marketing expects the target to be called "y"

    import pymc_marketing
    print(f"PyMC-Marketing {pymc_marketing.__version__}")

    lift = load_lift_tests()
    if lift is not None:
        print(f"Calibrating with {len(lift)} lift tests: {', '.join(lift['channel'])}")
    mmm, calibrated = fit_model(X, y, lift)
    idata = mmm.idata

    # ---- Diagnostics -------------------------------------------------------
    val, parts = roi_summary(idata, df, y, truth)
    alpha, lam, beta = parts["alpha"], parts["lam"], parts["beta"]
    summ = az.summary(idata, var_names=[alpha.name, lam.name, beta.name])
    divergences = int(idata.sample_stats["diverging"].sum())
    print(f"\nMax r_hat: {summ['r_hat'].max():.3f} (want < 1.01)   "
          f"Min ESS: {summ['ess_bulk'].min():.0f} (want > 400)   Divergences: {divergences} (want 0)")

    # ---- ROI vs truth ------------------------------------------------------
    x_max, y_max, contrib, total, spend, roi = (parts[k] for k in ("x_max", "y_max", "contrib", "total", "spend", "roi"))
    val.round(3).to_csv(OUT / "roi_validation.csv", index=False)
    print("\n" + val.round(2).to_string(index=False))

    # ---- Deeper analytics exports (never block the core outputs) ---------
    fit_stats = {}
    try:
        fit_stats = export_decomposition(idata, df, contrib, y_max)
    except Exception as e:  # noqa: BLE001
        print(f"Skipped decomposition export: {e}")
    try:
        true_c = pd.read_csv(DATA / "true_contributions.csv")
        export_period_roi(df, contrib, true_c)
    except Exception as e:  # noqa: BLE001
        print(f"Skipped period ROI export: {e}")

    # ---- What did the lift tests change? -----------------------------------
    val_uncal = None
    if calibrated:
        print("\nRefitting without the lift tests for comparison...")
        try:
            mmm0, _ = fit_model(X, y, None, HOLDOUT_SAMPLER)
            val_uncal, _ = roi_summary(mmm0.idata, df, y, truth)
            print(val_uncal[["channel", "true_roi", "est_roi", "roi_5%", "roi_95%"]].round(2).to_string(index=False))
        except Exception as e:  # noqa: BLE001
            print(f"Skipped uncalibrated comparison: {type(e).__name__}: {e}")

    # ---- Out-of-sample check -----------------------------------------------
    holdout = {}
    print(f"\nRefitting without the last {HOLDOUT_WEEKS} weeks to test the forecast...")
    try:
        holdout = holdout_check(df, lift=lift if calibrated else None)
    except Exception as e:  # noqa: BLE001
        print(f"Skipped holdout check: {type(e).__name__}: {e}")

    # ---- Save posterior draws for the optimiser ---------------------------
    rng = np.random.default_rng(0)
    def stacked(v):
        return v.stack(sample=("chain", "draw")).transpose("sample", "channel").sel(channel=CHANNELS).values

    a_all, l_all, b_all = stacked(alpha), stacked(lam), stacked(beta)
    keep = rng.choice(a_all.shape[0], size=min(N_KEEP, a_all.shape[0]), replace=False)
    # Self-check: rebuild contributions from the saved parameters and compare to PyMC-Marketing's
    # own numbers. A small correction keeps the app consistent if internal scaling ever differs.
    correction = []
    for i, c in enumerate(CHANNELS):
        xs = df[c].values / x_max[i]
        recon = np.mean([
            (b_all[k, i] * y_max * logistic_saturation(geometric_adstock(xs, a_all[k, i], L_MAX), l_all[k, i])).sum()
            for k in keep
        ])
        model_total = float(total.sel(channel=c).mean())
        ratio = model_total / recon
        correction.append(float(ratio))
        flag = "" if 0.95 <= ratio <= 1.05 else "  <-- check scaling"
        print(f"  scaling check {c:<12} {ratio:.3f}{flag}")

    params = {
        "channels": CHANNELS, "l_max": L_MAX, "x_max": x_max.tolist(), "y_max": y_max,
        "scale_correction": correction,
        "avg_weekly_spend": df[CHANNELS].mean().round(0).to_dict(),
        "draws": {"alpha": a_all[keep].tolist(), "lam": l_all[keep].tolist(), "beta": b_all[keep].tolist()},
        "roi": val.round(3).to_dict(orient="records"),
        "roi_draws": roi.transpose("chain", "draw", "channel").values.reshape(-1, len(CHANNELS))[keep].tolist(),
        "total_spend": {c: float(spend[c]) for c in CHANNELS},
        "fit": fit_stats,
        "holdout": holdout,
        "calibration": {"used": calibrated,
                        "tests": lift.round(0).to_dict(orient="records") if calibrated else []},
        "roi_uncalibrated": val_uncal.round(3).to_dict(orient="records") if val_uncal is not None else [],
        "n_weeks": int(len(df)),
        "diagnostics": {"max_r_hat": float(summ["r_hat"].max()), "min_ess_bulk": float(summ["ess_bulk"].min()),
                        "divergences": divergences},
    }
    with open(OUT / "model_params.json", "w") as f:
        json.dump(params, f)

    # ---- Charts ------------------------------------------------------------
    plot_roi(val)
    plot_contributions(df, contrib)
    plot_response_curves(params, df)
    update_readme(val, params["diagnostics"], fit_stats, holdout, val_uncal,
                  tested=set(lift["channel"]) if calibrated else set())
    print("\nSaved outputs to", OUT)


def plot_roi(val):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    yy = np.arange(len(val))[::-1]
    ax.hlines(yy, val["roi_5%"], val["roi_95%"], color="#888", lw=3, label="90% credible interval")
    ax.plot(val["est_roi"], yy, "o", color="#1f3864", ms=8, label="Model estimate")
    ax.plot(val["true_roi"], yy, "x", color="#d62728", ms=10, mew=2.5, label="True ROI")
    ax.set_yticks(yy, val["channel"].map(NAMES))
    ax.set_xlabel("Revenue per £1 spent")
    ax.set_title("Estimated vs true ROI", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "roi_validation.png", dpi=160)
    plt.close(fig)


def plot_contributions(df, contrib):
    mean = contrib.mean(dim=("chain", "draw")).sel(channel=CHANNELS).transpose("date", "channel").values
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.stackplot(df["date"], mean.T, labels=[NAMES[c] for c in CHANNELS],
                 colors=[COLOURS[c] for c in CHANNELS], alpha=0.85)
    ax.set_ylabel("Weekly revenue from media (£)")
    ax.set_title("Revenue driven by each channel (posterior mean)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "contributions.png", dpi=160)
    plt.close(fig)


def plot_response_curves(params, df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in CHANNELS:
        s = np.linspace(0, df[c].max() * 1.5, 120)
        curves = weekly_response(params_np(params), c, s)
        lo, hi = np.percentile(curves, [5, 95], axis=0)
        ax.fill_between(s / 1000, lo / 1000, hi / 1000, color=COLOURS[c], alpha=0.18)
        ax.plot(s / 1000, curves.mean(0) / 1000, color=COLOURS[c], lw=2, label=NAMES[c])
    ax.set_xlabel("Weekly spend (£k)")
    ax.set_ylabel("Weekly revenue (£k)")
    ax.set_title("Response curves with 90% uncertainty", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "response_curves.png", dpi=160)
    plt.close(fig)


def params_np(p):
    q = dict(p)
    q["draws"] = {k: np.asarray(v) for k, v in p["draws"].items()}
    return q


def update_readme(val, diag, fit=None, holdout=None, val_uncal=None, tested=()):
    """Refresh the results block between the RESULTS markers in README.md."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return

    def gbp(x):
        return f"£{x:.2f}"

    def err(est, true):
        return f"{est / true - 1:+.0%}".replace("-", "\u2212")

    uncal = {} if val_uncal is None else val_uncal.set_index("channel")["est_roi"].to_dict()
    cols = ["Channel", "True ROI"] + (["Model alone", "With lift tests"] if uncal else ["Estimated ROI"]) + \
           ["90% interval", "Truth inside interval"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["--:"] * (len(cols) - 2) + [":-:"]) + "|"]
    for _, r in val.iterrows():
        name = NAMES[r["channel"]] + (" (no lift test)" if uncal and r["channel"] not in tested else "")
        cells = [name, gbp(r["true_roi"])]
        if uncal:
            u = uncal[r["channel"]]
            cells.append(f"{gbp(u)} ({err(u, r['true_roi'])})")
        cells.append(f"**{gbp(r['est_roi'])} ({err(r['est_roi'], r['true_roi'])})**")
        cells += [f"{gbp(r['roi_5%'])}–{gbp(r['roi_95%'])}", "Yes" if r["truth_in_90%_interval"] else "No"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    facts = []
    if holdout:
        inside = round(holdout["coverage_90"] * holdout["weeks"])
        facts.append(f"**Forecast on {holdout['weeks']} unseen weeks:** {holdout['mape']:.1%} average error (MAPE), "
                     f"{inside} of {holdout['weeks']} weeks inside the 90% interval")
    if fit:
        facts.append(f"**In-sample fit:** R² {fit['r2']:.2f}, MAPE {fit['mape']:.1%}")
    facts.append(f"**Sampler:** max R-hat {diag['max_r_hat']:.3f}, min bulk ESS {diag['min_ess_bulk']:,.0f}, "
                 f"{diag['divergences']} divergences")
    lines += [f"- {x}" for x in facts]

    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        readme.write_text(head + start + "\n" + "\n".join(lines) + "\n" + end + tail, encoding="utf-8")


if __name__ == "__main__":
    main()
