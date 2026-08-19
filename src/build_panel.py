"""Building the analysis panel from the tidied NEEDS extract.

This module produces the ratios and transformations the paper uses -- investment
rates, leverage, MRPK, productivity residuals, firm age, bond holdings -- and
nothing else.  Winsorising, standardising and sample restrictions are left to
the analysis so that the panel on disk holds the raw ratios and every trimming
decision is visible in the script that makes it.

Ordering matters in three places, and getting any of them wrong changes results
silently rather than raising:

1. Lags and leads are taken on the full panel, before any period restriction.
   Restricting first drops the boundary years: taking a three-period lead inside
   a five-year window leaves two usable years.
2. Winsorising comes after lags and leads are constructed but before estimation,
   and it must cover the dependent variable.  The raw investment rate reaches
   1,985 in this panel, so a lead of an untrimmed investment rate carries the
   tail into the left-hand side.
3. Industry standardisation is applied to the level, not to an already
   winsorised series, since the two trims interact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Production function parameters used to form the productivity residual,
# y = z * k^THETA_K * l^GAMMA_L, with decreasing returns THETA_K + GAMMA_L < 1.
THETA_K = 0.30
GAMMA_L = 0.60


def filter_sample(df: pd.DataFrame,
                  fy_min: int = 1999, fy_max: int = 2025,
                  month: int | None = 3,
                  cons_only: bool = False) -> pd.DataFrame:
    """Restrict the sample.

    Parameters
    ----------
    month : keep only firms closing their books in this month; 3 by default,
            which covers the large majority of listed Japanese firms and keeps
            the mapping from accounting period to fiscal year unambiguous.
            Pass None to keep every closing month.
    cons_only : keep only firms reporting on a consolidated basis (cons_flag==2).
    """
    out = df.copy()
    if month is not None:
        out = out[out["month"] == month]
    out = out[(out["fy"] >= fy_min) & (out["fy"] <= fy_max)]
    if cons_only and "cons_flag" in out.columns:
        out = out[out["cons_flag"] == 2]
    return out.sort_values(["stkno", "fy"]).reset_index(drop=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ratios and transformations used throughout the paper.

    Notes
    -----
    Two investment rates are constructed and then combined:

        inv_rate     capex / k_{t-1}, from the cash flow statement.  Available
                     from FY1999 only, since the statement itself is.
        inv_rate_bs  (k_t - k_{t-1} + depreciation) / k_{t-1}, from the balance
                     sheet.  Available throughout but noisier, since it absorbs
                     revaluations, impairments and disposals.
        inv_rate_comb  the cash flow measure where available, filled with the
                     balance sheet measure elsewhere.  This is the baseline.

    MRPK is formed as log(sales / k_{t-1}) here, which is proportional to the
    marginal product under Cobb-Douglas.  The paper's preferred numerator is
    operating surplus, operating profit plus depreciation, which is what the
    model's profit function implies and which needs no outside data; it is built
    in the analysis rather than here so that the choice is explicit.  Value
    added, the convention in the misallocation literature, requires imputed
    labour costs and is handled in `labor_cost`.

    The productivity residual `log_z` uses sales as the output measure, so it
    absorbs markups as well as technology.  It is retained for comparison but is
    not used for the paper's claims.
    """
    d = df.sort_values(["stkno", "fy"]).copy()
    g = d.groupby("stkno")

    d["ppe_lag"] = g["ppe"].shift(1)
    d["ta_lag"] = g["total_assets"].shift(1)

    # --- capital and investment ---
    d["inv_rate"] = d["capex"] / d["ppe_lag"]
    d["inv_rate_bs"] = (d["ppe"] - d["ppe_lag"] + d["depreciation"]) / d["ppe_lag"]
    d["inv_rate_comb"] = d["inv_rate"].fillna(d["inv_rate_bs"])
    d["dep_rate"] = d["depreciation"] / d["ppe_lag"]
    d["log_k"] = np.log(d["ppe"].where(d["ppe"] > 0))

    # --- debt and liquidity ---
    d["debt"] = d["st_debt"].fillna(0) + d["lt_debt"].fillna(0)
    d["leverage"] = d["debt"] / d["total_assets"]
    d["lev_alt"] = d["total_debt"] / d["total_assets"]   # NEEDS' own debt total
    d["cash_ratio"] = d["cash"] / d["total_assets"]
    d["net_debt"] = (d["debt"] - d["cash"]) / d["total_assets"]
    d["eff_rate"] = d["interest_exp"] / d["debt"].replace(0, np.nan)

    # --- profitability ---
    d["roa"] = d["op_profit"] / d["total_assets"]
    d["margin"] = d["op_profit"] / d["sales"]
    d["turnover"] = d["sales"] / d["total_assets"]
    d["icr"] = d["op_profit"] / d["interest_exp"].replace(0, np.nan)

    # --- size ---
    d["log_assets"] = np.log(d["total_assets"].where(d["total_assets"] > 0))

    # --- payout: total dividends are rarely available, so only a zero-dividend
    #     indicator is formed, and only where dividends per share are reported
    if "dps" in d.columns:
        d["nodiv"] = np.where(d["dps"].notna(), (d["dps"] <= 0).astype(float),
                              np.nan)

    # --- MRPK and productivity ---
    d["mrpk"] = np.log((d["sales"] / d["ppe_lag"]).where(
        (d["sales"] > 0) & (d["ppe_lag"] > 0)))
    ok = (d["sales"] > 0) & (d["ppe"] > 0) & (d["employees"] > 0)
    d["log_z"] = np.where(
        ok, np.log(d["sales"]) - THETA_K * np.log(d["ppe"])
            - GAMMA_L * np.log(d["employees"]), np.nan)

    # --- flows, scaled by lagged total assets ---
    for source, name in [("debt", "d_debt"), ("cash", "d_cash"),
                         ("total_assets", "d_ta"), ("ppe", "d_ppe"),
                         ("retained_earnings", "d_re"),
                         ("fixed_assets", "d_fa")]:
        if source in d.columns:
            d[name] = d.groupby("stkno")[source].diff() / d["ta_lag"]
    d["capex_ta"] = d["capex"] / d["ta_lag"]
    d["dep_ta"] = d["depreciation"] / d["ta_lag"]
    if "d_fa" in d.columns and "d_ppe" in d.columns:
        d["d_other_fa"] = d["d_fa"] - d["d_ppe"]

    return d


def add_industry_standardized(df: pd.DataFrame,
                              cols: list[str] | None = None,
                              sector_col: str = "nkilm",
                              year_col: str = "fy",
                              suffix: str = "_s") -> pd.DataFrame:
    """Standardise columns within industry and year.

    This removes level differences in technology and in capital turnover across
    industries, and it puts a coefficient in units of one contemporaneous
    cross-sectional standard deviation.

    That last property is not innocuous.  A characteristic whose dispersion
    shrinks over the sample will show a rising gradient in fixed units even when
    the relation per standard deviation is unchanged; leverage in this panel
    compresses by about a fifth.  Use this function when the object of interest
    is the relation, `add_industry_demeaned` when it is the dispersion, and
    report both when the distinction could matter.
    """
    cols = cols or ["mrpk", "log_z", "log_k"]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            continue
        g = d.groupby([sector_col, year_col])[c]
        d[c + suffix] = (d[c] - g.transform("mean")) / g.transform("std")
    return d


def add_industry_demeaned(df: pd.DataFrame,
                          cols: list[str] | None = None,
                          sector_col: str = "nkilm",
                          year_col: str = "fy",
                          suffix: str = "_d") -> pd.DataFrame:
    """Subtract the industry-year mean, preserving the dispersion of the series."""
    cols = cols or ["mrpk", "log_z", "log_k"]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            continue
        g = d.groupby([sector_col, year_col])[c]
        d[c + suffix] = d[c] - g.transform("mean")
    return d


def winsorize(df: pd.DataFrame, cols: list[str],
              lower: float = 0.01, upper: float = 0.99,
              by_year: bool = False, year_col: str = "fy") -> pd.DataFrame:
    """Trim the tails of the named columns.

    Call this on the level of a variable before taking its leads, not after: a
    lead built from an untrimmed series carries the tail into the dependent
    variable even when the trimmed level is what appears on the right-hand side.

    `by_year=True` trims within each year, which is appropriate when the tails
    themselves move over the sample.
    """
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            continue
        if by_year:
            d[c] = d.groupby(year_col)[c].transform(
                lambda s: s.clip(s.quantile(lower), s.quantile(upper)))
        else:
            lo, hi = d[c].quantile([lower, upper])
            d[c] = d[c].clip(lo, hi)
    return d


def add_quantiles(df: pd.DataFrame, col: str, n: int = 5,
                  year_col: str = "fy", out_col: str | None = None) -> pd.DataFrame:
    """Assign within-year quantiles, breaking ties by order of appearance."""
    out_col = out_col or f"{col}_q"
    d = df.copy()
    d[out_col] = d.groupby(year_col)[col].transform(
        lambda x: pd.qcut(x.rank(method="first"), n, labels=range(1, n + 1)))
    return d


def balanced_firms(df: pd.DataFrame, fy_min: int, fy_max: int,
                   min_years: int | None = None) -> set:
    """Firms observed at least `min_years` times within the window."""
    span = fy_max - fy_min + 1
    min_years = min_years or span
    sub = df[(df["fy"] >= fy_min) & (df["fy"] <= fy_max)]
    count = sub.groupby("stkno")["fy"].nunique()
    return set(count[count >= min_years].index)


def coverage_report(df: pd.DataFrame, cols: list[str] | None = None,
                    fy_min: int = 1995, fy_max: int = 2025) -> pd.DataFrame:
    """Firms and non-missing counts by year, for checking a fresh extract."""
    cols = cols or ["ppe", "sales", "capex", "depreciation", "cash",
                    "employees", "dps"]
    sub = df[(df["fy"] >= fy_min) & (df["fy"] <= fy_max)]
    rows = []
    for fy, s in sub.groupby("fy"):
        row = {"fy": fy, "n_firms": s["stkno"].nunique(), "n_obs": len(s)}
        for c in cols:
            row[c] = int(s[c].notna().sum()) if c in s.columns else 0
        rows.append(row)
    return pd.DataFrame(rows).set_index("fy")


# --------------------------------------------------------------------------
# variables from the later extract: cash flow, bonds, construction in progress,
# incorporation date
# --------------------------------------------------------------------------
def add_needs_extra_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Derive variables from the later NEEDS extract.

    The source items are read by `io_needs.load_needs_extra` and joined on
    (stkno, fy) before this is called.

    Variables created
    -----------------
    cf_k, cf_ta       operating cash flow over lagged capital and over lagged
                      total assets
    cfi_k, cff_k      investing and financing cash flow over lagged capital
    age, log_age      firm age from the substantive incorporation date
    hold_flag         substantive and formal incorporation dates differ, which
                      indicates a holding company reorganisation or similar
    bond, bond_ta     bonds outstanding, long term plus due within one year
    has_bond          an indicator for positive bonds outstanding, the
                      Kashyap-Lamont-Stein style proxy for access to public debt
    ever_bond         whether the firm ever held bonds, at the firm level
    bond_share        bonds as a share of interest-bearing debt
    cip_k, d_cip      construction in progress and its change
    inv_completed     (capex - change in construction in progress) / k_{t-1}

    Three points that are easy to get wrong
    ---------------------------------------
    **A missing bond balance means zero, not undisclosed.**  Non-missing rates
    fall from about 0.43 in 1999 to 0.26 in 2019.  That is not deteriorating
    disclosure: NEEDS drops the line item for a firm with no bonds outstanding,
    and the decline reflects Japanese firms moving away from public debt.
    Treating these as missing would drop precisely the firms without bond market
    access, which is the variation the proxy is meant to capture.

    **`inv_completed` must not have its missing values filled with zero.**  The
    first implementation did, which set the correction to zero for seventy per
    cent of firms and produced a series correlated 1.000 with the uncorrected
    investment rate.  It is defined only for firms whose construction in
    progress is observed in two consecutive periods (about 65 per cent of the
    sample, correlation 0.862 with the uncorrected rate), and it belongs in the
    robustness tables rather than the main specification.

    **Use `cf_ta`, not `cf_k`, when testing cash flow sensitivity.**  The two
    differ only in the denominator, but `cf_k` shares its denominator with the
    investment rate, so measurement error in capital inflates both in the same
    direction: corr(cf_k, i/k) is +0.245 against +0.026 for corr(cf_ta, i/k).

    Firm age uses the substantive incorporation date.  The formal date is reset
    when a holding company is created, which affects about twelve per cent of
    firms here; because those firms are identified by `hold_flag`, the concern
    that Japanese incorporation dates are unusable turns out to be overstated.
    """
    d = df.sort_values(["stkno", "fy"]).copy()

    # --- cash flow
    d["cf_k"] = (d["cfo"] / d["ppe_lag"]).replace([np.inf, -np.inf], np.nan)
    d["cf_ta"] = (d["cfo"] / d["ta_lag"]).replace([np.inf, -np.inf], np.nan)
    d["cfi_k"] = (d["cfi"] / d["ppe_lag"]).replace([np.inf, -np.inf], np.nan)
    d["cff_k"] = (d["cff"] / d["ppe_lag"]).replace([np.inf, -np.inf], np.nan)

    # --- firm age, measured at the end of the fiscal year
    if "founded_real" in d.columns:
        fy_end = (pd.to_datetime(d["fy"].astype(str) + "-04-01")
                  + pd.DateOffset(years=1))
        d["age"] = (fy_end - pd.to_datetime(d["founded_real"])).dt.days / 365.25
        d["age"] = d["age"].where(d["age"] > 0)
        d["log_age"] = np.log(d["age"])
        d["hold_flag"] = (pd.to_datetime(d["founded_real"])
                          != pd.to_datetime(d["founded_form"])).astype(float)

    # --- bonds; a missing balance is a zero balance, see the docstring
    d["bond"] = d[["bond_lt", "bond_st"]].fillna(0).sum(axis=1)
    d["bond_ta"] = (d["bond"] / d["total_assets"]).replace([np.inf, -np.inf],
                                                           np.nan)
    d["has_bond"] = (d["bond"] > 0).astype(float)
    d["ever_bond"] = d.groupby("stkno")["has_bond"].transform("max")
    if "debt" in d.columns:
        d["bond_share"] = ((d["bond"] / d["debt"].where(d["debt"] > 0))
                           .replace([np.inf, -np.inf], np.nan).clip(0, 1))

    # --- construction in progress and completion-based investment
    d["cip_k"] = (d["cip"] / d["ppe_lag"]).replace([np.inf, -np.inf], np.nan)
    d["d_cip"] = d.groupby("stkno")["cip"].diff()
    d["has_cip"] = d["cip"].notna() & d.groupby("stkno")["cip"].shift(1).notna()
    d["inv_completed"] = np.where(
        d["has_cip"], (d["capex"] - d["d_cip"]) / d["ppe_lag"], np.nan)
    d["inv_completed"] = (pd.Series(d["inv_completed"], index=d.index)
                          .replace([np.inf, -np.inf], np.nan))
    return d
