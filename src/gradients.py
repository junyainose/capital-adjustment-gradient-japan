"""Investment gradients and their drift over time.

The paper's main object is the cross-sectional gradient of investment with
respect to a firm characteristic, and how that gradient moves over the sample.

Two estimators are provided.

`fit_trend` runs the pooled specification

    y_{i,t+h} = sum_a [ delta_a * a_{i,t-1} + tau_a * a_{i,t-1} * (t - tbar) ]
                + phi' x_{i,t-1} + mu_i + lambda_t + u_{i,t}

where `a` indexes firm characteristics.  `delta_a` is the average gradient and
`tau_a` its linear drift per year; the tables report `10 * tau_a`, the drift per
decade.  Firm and year fixed effects are absorbed by alternating projection and
standard errors are clustered on the firm.

`fit_year_gradients` replaces the linear drift with a full set of
characteristic-by-year interactions, giving one gradient `delta_{a,t}` per year.
That version is used for the figures and is the non-parametric counterpart of
`fit_trend`.

Units matter and are not innocuous.  A characteristic whose cross-sectional
dispersion changes over the sample will show a drift in its gradient even if
the relation per standard deviation is constant.  `standardize` therefore
offers three conventions and the robustness tables report all of them:

    "industry_year"  demean and rescale within industry x year (contemporaneous
                     standard deviation units; the paper's default)
    "industry_fixed" demean within industry x year, rescale by the industry's
                     full-sample standard deviation (fixed units)
    "industry_mean"  demean by the industry's full-sample mean and rescale over
                     the pooled sample (the convention used by Albrizio,
                     Gonzalez and Khametshin, 2026)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

FIRM = "stkno"
YEAR = "fy"
INDUSTRY = "nkilm"


# --------------------------------------------------------------------------
# variable construction
# --------------------------------------------------------------------------
def standardize(df: pd.DataFrame, col: str, how: str = "industry_year",
                industry: str = INDUSTRY, year: str = YEAR) -> pd.Series:
    """Normalise a characteristic; see the module docstring for the options."""
    x = df[col]
    if how == "industry_year":
        g = x.groupby([df[industry], df[year]])
        return (x - g.transform("mean")) / g.transform("std")
    if how == "industry_fixed":
        g = x.groupby([df[industry], df[year]])
        scale = x.groupby(df[industry]).transform("std")
        return (x - g.transform("mean")) / scale
    if how == "industry_mean":
        d = x - x.groupby(df[industry]).transform("mean")
        return (d - d.mean()) / d.std()
    raise ValueError(f"unknown standardisation: {how}")


def log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """log(num/den), defined only where both are strictly positive."""
    return np.log((num / den).where((num > 0) & (den > 0)))


def add_leads(df: pd.DataFrame, col: str, horizons=(0, 1, 2, 3),
              prefix: str | None = None, firm: str = FIRM) -> pd.DataFrame:
    """Add h-period leads of `col`.

    Leads must be built on the full panel *before* any period restriction.
    Building them inside a sub-period silently drops the last h years of that
    period, which for five-year windows leaves two or three usable years.
    """
    d = df.copy()
    prefix = prefix or col
    for h in horizons:
        d[f"{prefix}_h{h}"] = d.groupby(firm)[col].shift(-h)
    return d


def add_capital_growth(df: pd.DataFrame, log_capital: str = "log_ppe",
                       horizons=(0, 1, 2, 3), firm: str = FIRM) -> pd.DataFrame:
    """Cumulative log change in the capital stock, 100 x (log k_{t+h} - log k_{t-1}).

    This is the dependent variable of Albrizio, Gonzalez and Khametshin (2026).
    """
    d = df.copy()
    g = d.groupby(firm)[log_capital]
    base = g.shift(1)
    for h in horizons:
        d[f"dlogk_h{h}"] = 100.0 * (g.shift(-h) - base)
    return d


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------
def _absorb(frame: pd.DataFrame, cols: list[str], groups: list[str],
            n_iter: int = 15) -> pd.DataFrame:
    """Within transformation for several fixed effects (alternating projection)."""
    x = frame[cols].astype(float).copy()
    for _ in range(n_iter):
        for g in groups:
            x = x - x.groupby(frame[g].values).transform("mean")
    return x


def _absorb_firm_trends(frame: pd.DataFrame, cols: list[str], time: str,
                        firm: str = FIRM, year: str = YEAR,
                        n_iter: int = 12) -> pd.DataFrame:
    """Absorb firm-specific linear trends together with year fixed effects.

    Used as a robustness check.  Note that a characteristic which is itself
    close to a firm-specific trend -- log capital is -- loses most of its
    variation here, so its coefficient is not interpretable under this option.
    """
    x = frame[cols].astype(float).copy()
    t = frame[time].to_numpy(float)
    fid = frame[firm].to_numpy()
    for _ in range(n_iter):
        for c in cols:
            block = pd.DataFrame({"f": fid, "t": t, "y": x[c].to_numpy(float)})
            m = block.groupby("f")[["t", "y"]].transform("mean")
            dt, dy = block["t"] - m["t"], block["y"] - m["y"]
            num = (dt * dy).groupby(fid).transform("sum")
            den = (dt * dt).groupby(fid).transform("sum")
            slope = (num / den.replace(0, np.nan)).fillna(0.0)
            x[c] = (dy - slope * dt).to_numpy()
        x = x - x.groupby(frame[year].values).transform("mean")
    return x


def _cluster_vcov(z: np.ndarray, resid: np.ndarray, groups: np.ndarray,
                  bread: np.ndarray) -> np.ndarray:
    meat = np.zeros((z.shape[1], z.shape[1]))
    order = np.argsort(groups, kind="stable")
    zs, rs, gs = z[order], resid[order], groups[order]
    cuts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        s = zs[lo:hi].T @ rs[lo:hi]
        meat += np.outer(s, s)
    n_g = len(cuts) - 1
    adj = n_g / max(n_g - 1, 1)
    return adj * bread @ meat @ bread


def _multiway_vcov(z: np.ndarray, resid: np.ndarray, bread: np.ndarray,
                   groups: list[np.ndarray]) -> np.ndarray:
    """Cameron-Gelbach-Miller variance for one or two clustering dimensions.

    With two dimensions the variance is the sum of the two one-way variances
    less the variance clustered on their intersection, which corrects the
    double counting of observations that share both.  The result is not
    guaranteed to be positive semi-definite in finite samples; a negative
    diagonal entry is reported as a missing standard error rather than
    silently clipped to zero.
    """
    if len(groups) == 1:
        return _cluster_vcov(z, resid, groups[0], bread)
    a, b = groups[0], groups[1]
    both = np.array([f"{x}|{y}" for x, y in zip(a, b)])
    return (_cluster_vcov(z, resid, a, bread)
            + _cluster_vcov(z, resid, b, bread)
            - _cluster_vcov(z, resid, both, bread))


@dataclass
class Fit:
    """Coefficients, standard errors and p-values, keyed by regressor name."""
    coef: dict[str, float]
    se: dict[str, float]
    pval: dict[str, float]
    nobs: int
    nfirms: int
    meta: dict = field(default_factory=dict)

    def get(self, name: str) -> tuple[float, float, float]:
        return self.coef[name], self.se[name], self.pval[name]


def _ols(y: np.ndarray, z: np.ndarray, names: list[str],
         cluster: np.ndarray | list[np.ndarray]) -> tuple[dict, dict, dict]:
    """Cluster-robust least squares, with identically zero columns removed first.

    Some columns carry no information at all: in the year-by-year design the
    first year has no lagged regressor, so ``mrpk@<first year>`` is zero for
    every row once the fixed effects are absorbed.  Leaving such a column in
    makes ``z'z`` exactly singular, and then whether the pseudo-inverse keeps or
    discards the corresponding singular value turns on whether the SVD returns
    it as exactly zero or as something just above the rcond cutoff.  That is a
    property of the LAPACK build, and under threaded MKL it can differ between
    runs on one machine.  Keeping one blows the inverse up and every standard
    error comes back non-positive, which reads downstream as "no year is
    identified".  Dropping the columns first makes the solve deterministic and
    leaves every other coefficient unchanged.  The dropped names are reported
    with a zero coefficient and a NaN standard error, which is what callers
    already treat as unidentified.
    """
    groups = [cluster] if isinstance(cluster, np.ndarray) else list(cluster)
    keep = np.asarray(np.abs(z).max(axis=0) > 0)
    if not keep.all():
        kept = [n for n, k in zip(names, keep) if k]
        coef, stderr, pval = _ols(y, z[:, keep], kept, cluster)
        for n, k in zip(names, keep):
            if not k:
                coef[n], stderr[n], pval[n] = 0.0, float("nan"), float("nan")
        return coef, stderr, pval
    bread = np.linalg.pinv(z.T @ z)
    beta = bread @ (z.T @ y)
    resid = y - z @ beta
    vcov = _multiway_vcov(z, resid, bread, groups)
    diag = np.diag(vcov)
    se = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
    # Degrees of freedom follow the coarsest dimension, which is the binding
    # one for the t distribution when clusters are few.
    dof = max(min(len(np.unique(g)) for g in groups) - 1, 1)
    coef = {n: float(beta[i]) for i, n in enumerate(names)}
    stderr = {n: float(se[i]) for i, n in enumerate(names)}
    pval = {n: float(2 * (1 - stats.t.cdf(abs(beta[i] / max(se[i], 1e-12)), dof)))
            for i, n in enumerate(names)}
    return coef, stderr, pval


def fit_trend(df: pd.DataFrame, dep: str, axes: dict[str, str],
              controls: tuple[str, ...] = ("log_assets_lag",),
              time: str = "tc", firm_trends: bool = False,
              absorb: tuple[str, ...] | None = None,
              cluster: tuple[str, ...] | None = None,
              firm: str = FIRM, year: str = YEAR) -> Fit | None:
    """Average gradient and linear drift for each characteristic.

    Parameters
    ----------
    axes : mapping from a short label to the column holding the characteristic,
           e.g. ``{"mrpk": "mrpk_s_lag", "lev": "lev_s_lag", "logk": "logk_s_lag"}``.
           All characteristics are entered jointly, so the coefficients are
           partial with respect to one another.
    time : column holding the centred year, t - mean(t).  Centring makes the
           level coefficient the gradient at the middle of the sample.
    firm_trends : absorb firm-specific linear trends instead of firm fixed
           effects.  See `_absorb_firm_trends` for the caveat.
    absorb : fixed effects to absorb, defaulting to firm and year.  Passing an
           industry-year identifier instead of the firm identifier estimates the
           gradient from differences between firms rather than within them.
           The two answer different questions.  With firm fixed effects the
           coefficient is the response of a firm whose own characteristic has
           risen relative to its peers, and any permanent difference between
           firms -- management, vintage, group affiliation -- is absorbed.
           Without them it is the ordinary cross-sectional gradient, which is
           the quantity the misallocation literature reports, and which carries
           whatever unobserved permanent heterogeneity happens to correlate with
           the characteristic.

    Returns None when fewer than 500 usable observations remain.
    """
    d = df.copy()
    names: list[str] = []
    for label, col in axes.items():
        d[f"level_{label}"] = d[col]
        d[f"drift_{label}"] = d[col] * d[time]
        names += [f"level_{label}", f"drift_{label}"]
    names += list(controls)

    groups = list(absorb) if absorb is not None else [firm, year]
    cluster_on = list(cluster) if cluster is not None else [firm]
    need = [dep] + names + [firm, year, time] + groups + cluster_on
    d = d.dropna(subset=[c for c in dict.fromkeys(need) if c in d.columns])
    if len(d) < 500:
        return None

    if firm_trends:
        x = _absorb_firm_trends(d, [dep] + names, time=time, firm=firm, year=year)
    else:
        x = _absorb(d, [dep] + names, groups)

    coef, se, pval = _ols(x[dep].to_numpy(float), x[names].to_numpy(float),
                          names, [d[c].to_numpy() for c in cluster_on])
    return Fit(coef, se, pval, len(d), d[firm].nunique(),
               meta={"dep": dep, "axes": dict(axes), "controls": list(controls),
                     "firm_trends": firm_trends, "absorb": groups,
                     "cluster": cluster_on})


def fit_year_gradients(df: pd.DataFrame, dep: str, axes: dict[str, str],
                       years: list[int],
                       controls: tuple[str, ...] = ("log_assets_lag",),
                       firm: str = FIRM, year: str = YEAR) -> Fit | None:
    """One gradient per characteristic per year.

    The design matrix is rank deficient whenever a year contributes no usable
    observations -- the first year has no lagged regressors and the last h
    years have no lead of the dependent variable.  Those columns come back as
    exact zeros; callers should drop them before plotting.
    """
    d = df.copy()
    names: list[str] = []
    for label, col in axes.items():
        for t in years:
            nm = f"{label}@{t}"
            d[nm] = (d[col] * (d[year] == t)).astype(float)
            names.append(nm)
    names += list(controls)

    d = d.dropna(subset=[dep] + [axes[a] for a in axes] + list(controls)
                 + [firm, year])
    if len(d) < 500:
        return None
    x = _absorb(d, [dep] + names, [firm, year])
    coef, se, pval = _ols(x[dep].to_numpy(float), x[names].to_numpy(float),
                          names, d[firm].to_numpy())
    return Fit(coef, se, pval, len(d), d[firm].nunique(),
               meta={"dep": dep, "years": list(years), "axes": dict(axes)})


def gradient_series(fit: Fit, label: str, years: list[int]) -> pd.DataFrame:
    """Extract one characteristic's yearly gradients as a tidy frame.

    Years that contribute no usable observations come back from the least
    squares solve as exact zeros with a zero standard error, and are dropped
    here.  A zero standard error is the reliable marker; a zero coefficient on
    its own is not, since a genuinely estimated gradient can be small.

    The standard error can also come back as NaN rather than zero, which is
    what the pseudo-inverse returns when the column is collinear rather than
    identically zero.  ``nan <= 0`` is False, so a test written only against
    zero lets such a year through with a coefficient of exactly zero, and any
    average taken over a window containing it is pulled towards zero.  The test
    below is therefore on the standard error being a positive finite number.
    """
    rows = []
    for t in years:
        key = f"{label}@{t}"
        if key not in fit.coef or not np.isfinite(fit.se[key]) or fit.se[key] <= 0:
            continue
        rows.append({"year": t, "coef": fit.coef[key], "se": fit.se[key],
                     "pval": fit.pval[key]})
    if not rows:
        available = sorted({k.split("@")[0] for k in fit.coef if "@" in k})
        raise ValueError(
            f"no identified yearly gradient for '{label}'. "
            f"Characteristics present in the fit: {available}. "
            "If the label is right, every year was absorbed, which points to "
            "the characteristic being missing for the whole sample.")
    return pd.DataFrame(rows, columns=["year", "coef", "se", "pval"])


def dispersion(df: pd.DataFrame, cols: list[str], industry: str = INDUSTRY,
               year: str = YEAR) -> pd.DataFrame:
    """Mean within-industry-year standard deviation of each column, by year.

    Reported alongside the gradients: a drift in a gradient measured in fixed
    units is observationally equivalent to a compression of the characteristic's
    distribution, and the two are told apart by this series.
    """
    out = {}
    for c in cols:
        out[c] = df.groupby([industry, year])[c].std().groupby(year).mean()
    return pd.DataFrame(out)
