"""Annualising the Kubota-Shintani (2022) high-frequency monetary policy shocks.

Source
------
Kubota, H., and M. Shintani (2022), "High-frequency identification of monetary
policy shocks in Japan", The Japanese Economic Review 73(3), 483-513.

The raw file is a monthly panel of surprise measures:

    TARGET      factor loading on short-horizon (three-month) rate expectations
    PATH        factor moving expectations beyond six months only, identified
                by the restriction that it has no effect on EYF3
    EYF3/6/9/12 euroyen futures surprises at the stated horizon
    JGBF        ten-year JGB futures surprise
    PC1, PC2    the underlying principal components
    flag_30min  1 for a thirty-minute window, 0 for a daily window.  Daily
                windows are used before September 1999, so within this sample
                only the first half of FY1999 is affected.

Sign convention: positive is a tightening surprise.  The paper signs easing as
positive, which is done at the point of use rather than here, so that the
series retains the orientation of the original.

TARGET is the baseline series.  Its effect on asset prices is about twice that
of PATH (their Table 3); the negative interest rate announcement of 29 January
2016, the most consequential policy change in the sample, appears as a large
negative TARGET (-0.059) against a PATH near zero; and TARGET has the largest
annual variation.  PATH and JGBF are nonetheless reported alongside it, since a
claim about transmission should not rest on a single factor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHOCK_COLS = ["TARGET", "PATH", "PC1", "PC2",
              "EYF3", "EYF6", "EYF9", "EYF12", "JGBF"]


def load_ks(path: str | Path, fiscal_start_month: int = 4) -> pd.DataFrame:
    """Read the monthly shock file and attach a Japanese fiscal year.

    The fiscal year starts in April, so January to March belong to the previous
    fiscal year.  This must match the convention used for the firm panel: a firm
    closing its books in March 2000 is a FY1999 observation.
    """
    ks = pd.read_csv(path)
    ks["DATE"] = pd.to_datetime(ks["DATE"])
    ks["year"] = ks["DATE"].dt.year
    ks["month"] = ks["DATE"].dt.month
    ks["fy"] = np.where(ks["month"] < fiscal_start_month,
                        ks["year"] - 1, ks["year"])
    return ks


def annualize(ks: pd.DataFrame, col: str = "TARGET",
              method: str = "sum") -> pd.DataFrame:
    """Aggregate the monthly surprises to fiscal years.

    Parameters
    ----------
    method : "sum"   cumulate within the year (default).  The natural choice
                     when the object of interest is the total policy news a firm
                     faced before setting investment.
             "mean"  average instead of cumulating.
             "last3" keep only January to March, the three months closest to the
                     balance sheet date for March-closing firms.  Trades sample
                     variation for timing precision.

    Returns
    -------
    DataFrame with columns [fy, shock].
    """
    if method == "sum":
        out = ks.groupby("fy")[col].sum()
    elif method == "mean":
        out = ks.groupby("fy")[col].mean()
    elif method == "last3":
        out = ks[ks["month"].isin([1, 2, 3])].groupby("fy")[col].sum()
    else:
        raise ValueError(f"unknown method: {method}")
    return out.rename("shock").reset_index()


def identifying_variance(ks: pd.DataFrame, col: str = "TARGET",
                         years: tuple[int, int] | None = None) -> pd.Series:
    """Share of the annual shock variance contributed by each fiscal year.

    Under year fixed effects an interaction coefficient is identified from the
    year-to-year movement of the shock, so the effective number of observations
    is the number of years rather than the number of firm-years.  Where a few
    years dominate this series, the coefficient is fragile to dropping them.
    """
    a = ks.groupby("fy")[col].sum()
    if years is not None:
        a = a.loc[(a.index >= years[0]) & (a.index <= years[1])]
    d = (a - a.mean()) ** 2
    return (d / d.sum()).sort_values(ascending=False)


def diagnostics(ks: pd.DataFrame, periods: list[tuple[int, int, str]],
                col: str = "TARGET") -> pd.DataFrame:
    """Dispersion of the shock by sub-period.

    When an interaction coefficient weakens in a later sub-sample, the cause may
    be that the shock became less variable rather than that firms became less
    responsive.  This separates the two.
    """
    rows = []
    annual = ks.groupby("fy")[col].sum()
    for lo, hi, label in periods:
        monthly = ks.loc[(ks["fy"] >= lo) & (ks["fy"] <= hi), col]
        a = annual.loc[(annual.index >= lo) & (annual.index <= hi)]
        rows.append({
            "period": label,
            "monthly_sd": monthly.std(),
            "monthly_abs_mean": monthly.abs().mean(),
            "annual_sd": a.std(), "annual_mean": a.mean(),
            "annual_min": a.min(), "annual_max": a.max(), "n_years": len(a),
        })
    return pd.DataFrame(rows).set_index("period")


def merge_shock(panel: pd.DataFrame, shock: pd.DataFrame,
                on: str = "fy") -> pd.DataFrame:
    """Attach an annual shock series to the firm panel, replacing any existing one."""
    out = panel.drop(columns=[c for c in ["shock"] if c in panel.columns])
    return out.merge(shock, on=on, how="left")
