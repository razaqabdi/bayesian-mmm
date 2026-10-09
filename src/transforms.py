"""Media transformations, written to match PyMC-Marketing's GeometricAdstock
(normalised weights) and LogisticSaturation exactly, so the simulated "truth"
and the fitted model share the same functional form."""
import numpy as np


def geometric_adstock(x, alpha, l_max=8, normalize=True):
    """Carry-over: this week's effective spend is a decaying weighted sum of the last l_max weeks."""
    x = np.asarray(x, dtype=float)
    w = alpha ** np.arange(l_max)
    if normalize:
        w = w / w.sum()
    out = np.zeros_like(x)
    for lag, wl in enumerate(w):
        out[lag:] += wl * x[: len(x) - lag]
    return out


def logistic_saturation(x, lam):
    """Diminishing returns: maps scaled spend into (0, 1); larger lam saturates faster."""
    x = np.asarray(x, dtype=float)
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
