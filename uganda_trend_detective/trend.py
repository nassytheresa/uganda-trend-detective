"""Trend statistics: Mann-Kendall test (with tie correction) and Sen's slope."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class TrendResult:
    n: int
    slope_per_year: float      # Sen's slope, units per year
    intercept: float           # for drawing the trend line
    p_value: float             # two-sided Mann-Kendall
    tau: float                 # Kendall's tau (-1..1)
    significant: bool          # p < alpha
    lag1_autocorr: float       # of detrended residuals (caveat flag)


def mann_kendall(years, values, alpha=0.05) -> TrendResult:
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    n = len(y)
    if n < 8:
        raise ValueError("Need at least 8 years of data for a meaningful trend test.")

    # Mann-Kendall S statistic
    s = 0
    for i in range(n - 1):
        s += np.sign(y[i + 1:] - y[i]).sum()

    # Variance with tie correction
    _, counts = np.unique(y, return_counts=True)
    tie_term = (counts * (counts - 1) * (2 * counts + 5)).sum()
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)

    # Sen's slope: median of all pairwise slopes
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        slopes.append((y[i + 1:] - y[i]) / dx)
    slope = float(np.median(np.concatenate(slopes)))
    intercept = float(np.median(y - slope * x))

    # Lag-1 autocorrelation of residuals (flag if serial dependence may inflate significance)
    resid = y - (intercept + slope * x)
    resid = resid - resid.mean()
    denom = (resid ** 2).sum()
    lag1 = float((resid[:-1] * resid[1:]).sum() / denom) if denom > 0 else 0.0

    return TrendResult(n, slope, intercept, float(p), float(tau), bool(p < alpha), lag1)
