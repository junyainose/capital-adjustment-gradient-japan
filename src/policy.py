"""The response of the investment gradient to monetary policy shocks.

Section 6 asks a narrower question than Section 5.  There the object is the slow
movement of the gradient over two decades; here it is whether the gradient also
moves with monetary policy surprises from one year to the next.

The estimand is the coefficient on the interaction of the shock with the
characteristic,

    y_{i,t+h} = beta2 * x_{i,t-1} * eps_t + beta3 * x_{i,t-1}
                + phi' controls + mu_i + lambda_t + u_{i,t},

signed so that a positive shock is an easing.  Year fixed effects absorb the
shock's own level, so beta1, the average response, is not identified here and
the paper makes no claim about it.  What is identified is beta2: whether easing
raises investment more at firms with a higher characteristic.

The design has a property that governs how far the results can be pushed.  With
year effects, beta2 is identified from the year-to-year movement of a single
annual series, so the effective number of observations is the number of years,
not the number of firm-years.  `leave_one_year_out` reports how the estimate
moves when each year is dropped in turn, and `shocks.identifying_variance`
reports how concentrated that variation is.  In this sample four of twenty-one
years carry about four fifths of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.gradients import FIRM, YEAR, Fit, _absorb, _cluster_vcov, _ols


def beta2(df: pd.DataFrame, dep: str, mrpk: str, shock: str, h: int = 0,
          controls: tuple[str, ...] = ("log_assets_lag",),
          absorb: tuple[str, ...] = (FIRM, YEAR),
          firm: str = FIRM) -> Fit | None:
    """Interaction of the shock with one characteristic.

    `dep` is a column that already holds the h-period lead; building leads
    inside a sub-period would drop its final years.
    """
    d = df.copy()
    d["_int"] = d[mrpk] * d[shock]
    names = ["_int", mrpk] + list(controls)
    d = d.dropna(subset=[dep] + names + list(absorb))
    if len(d) < 500 or d[YEAR].nunique() < 3:
        return None
    x = _absorb(d, [dep] + names, list(absorb))
    try:
        coef, se, pval = _ols(x[dep].to_numpy(float), x[names].to_numpy(float),
                              names, d[firm].to_numpy())
    except np.linalg.LinAlgError:
        return None
    return Fit(coef, se, pval, len(d), d[firm].nunique(),
               meta={"dep": dep, "shock": shock, "h": h})


def leave_one_year_out(df: pd.DataFrame, dep: str, mrpk: str, shock: str,
                       years: list[int], **kwargs) -> pd.DataFrame:
    """Re-estimate with each year dropped in turn.

    A coefficient that survives this is identified by more than one episode.
    One that does not is reported as such rather than as a finding: in this
    sample the early rolling windows turn out to rest on a single year.
    """
    rows = []
    for y in years:
        fit = beta2(df[df[YEAR] != y], dep, mrpk, shock, **kwargs)
        if fit is None:
            continue
        b, se, p = fit.get("_int")
        rows.append({"dropped": y, "coef": b, "se": se, "pval": p})
    return pd.DataFrame(rows)


def by_series(df: pd.DataFrame, dep_prefix: str, mrpk: str,
              series: dict[str, str], horizons=(0, 1, 2, 3),
              years: list[int] | None = None, **kwargs) -> pd.DataFrame:
    """One row per shock series and horizon, with the leave-one-year-out range."""
    rows = []
    for label, col in series.items():
        for h in horizons:
            dep = f"{dep_prefix}_h{h}"
            fit = beta2(df, dep, mrpk, col, h=h, **kwargs)
            if fit is None:
                continue
            b, se, p = fit.get("_int")
            row = {"series": label, "h": h, "coef": b, "se": se, "pval": p,
                   "nobs": fit.nobs}
            if years is not None:
                loyo = leave_one_year_out(df, dep, mrpk, col, years, h=h,
                                          **kwargs)
                row["loyo_min"] = loyo["coef"].min()
                row["loyo_max"] = loyo["coef"].max()
                row["loyo_sig"] = int((loyo["pval"] < 0.05).sum())
                row["loyo_n"] = len(loyo)
            rows.append(row)
    return pd.DataFrame(rows)


def conditional_beta2(df: pd.DataFrame, dep: str, mrpk: str, shock: str,
                      proxy: str, n_groups: int = 3, **kwargs) -> pd.DataFrame:
    """beta2 within each group of a constraint proxy, and the difference.

    The groups are formed within year so that a trend in the proxy does not
    reshuffle firms across them.  The difference is estimated jointly rather
    than by comparing the separate estimates, so that it comes with a standard
    error.
    """
    d = df.dropna(subset=[proxy]).copy()
    d["_grp"] = d.groupby(YEAR)[proxy].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups,
                          labels=range(1, n_groups + 1)))
    rows = []
    for g in range(1, n_groups + 1):
        fit = beta2(d[d["_grp"] == g], dep, mrpk, shock, **kwargs)
        if fit is None:
            continue
        b, se, p = fit.get("_int")
        rows.append({"group": g, "coef": b, "se": se, "pval": p,
                     "nobs": fit.nobs})
    return pd.DataFrame(rows)


def difference_in_beta2(df: pd.DataFrame, dep: str, mrpk: str, shock: str,
                        flag: str, controls: tuple[str, ...] = ("log_assets_lag",),
                        absorb: tuple[str, ...] = (FIRM, YEAR),
                        firm: str = FIRM) -> Fit | None:
    """Triple interaction: does beta2 differ where `flag` is one?

    Reporting the difference with its own standard error is the point.  Two
    subsample estimates that straddle zero are not evidence that they differ,
    and in this sample the difference is never estimated precisely enough to
    settle the question either way.
    """
    d = df.copy()
    d["_x"] = d[mrpk] * d[shock]
    d["_xf"] = d["_x"] * d[flag]
    d["_mf"] = d[mrpk] * d[flag]
    names = ["_x", "_xf", mrpk, "_mf"] + list(controls)
    d = d.dropna(subset=[dep] + names + list(absorb))
    if len(d) < 500:
        return None
    x = _absorb(d, [dep] + names, list(absorb))
    coef, se, pval = _ols(x[dep].to_numpy(float), x[names].to_numpy(float),
                          names, d[firm].to_numpy())
    return Fit(coef, se, pval, len(d), d[firm].nunique(),
               meta={"dep": dep, "shock": shock, "flag": flag})
