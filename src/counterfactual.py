"""What the rise in the gradient did to the allocation of capital.

Section 7 asks what the fact of Section 5 is worth.  The answer is built the way
Decker, Haltiwanger, Jarmin and Miranda (2020) and Hambur and Andrews (2023)
build theirs, with the sign reversed: they construct the path the economy would
have taken had responsiveness not fallen, and this constructs the path it would
have taken had responsiveness not risen.

The construction has three steps.

1.  Estimate the gradient year by year on one-period capital growth.
2.  Strip from each firm's growth the part attributable to the gradient
    exceeding its base-period level, and cumulate what is left into a
    counterfactual capital stock.
3.  Recompute MRPK from the counterfactual capital and compare the dispersion
    of MRPK under the two paths.

Two properties of the construction matter for reading the result.

The adjustment is demeaned within industry and year, so the counterfactual moves
capital between firms without changing how much there is.  What is measured is
therefore allocation, not accumulation; the total capital stock differs by about
a tenth of a percent at the end of the sample.

MRPK is recomputed under the counterfactual capital.  Capital is the denominator
of MRPK, so a counterfactual that moves capital and leaves MRPK alone would not
be internally consistent: the whole point is that moving capital toward
high-MRPK firms pulls their MRPK down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import gradients as gr

# Elasticity of output with respect to capital, used only to convert a variance
# into an output equivalent.  The conversion is a first-order approximation that
# assumes log normality and a constant elasticity, so it fixes an order of
# magnitude rather than a number.
THETA = 0.30


def year_gradients(p: pd.DataFrame, axes: dict[str, str],
                   growth: str = "g1", trim: float = 0.01) -> pd.DataFrame:
    """Gradient of one-period capital growth on the characteristics, by year."""
    d = p.copy()
    d["logk"] = np.log(d["ppe"].where(d["ppe"] > 0))
    d[growth] = d["logk"] - d.groupby(gr.FIRM)["logk"].shift(1)
    lo, hi = d[growth].quantile([trim, 1 - trim])
    d[growth] = d[growth].clip(lo, hi)
    years = sorted(d[gr.YEAR].unique())
    fit = gr.fit_year_gradients(d, growth, axes, years)
    return gr.gradient_series(fit, "mrpk", years)


def counterfactual_capital(p: pd.DataFrame, deltas: pd.DataFrame,
                           mrpk: str, base: tuple[int, int]) -> pd.DataFrame:
    """Capital path with the gradient held at its base-period level.

    Only the excess of the gradient over the base level is removed, so a year in
    which the gradient happened to fall below that level is left alone; the
    counterfactual is 'no rise', not 'the base level exactly'.
    """
    base_level = deltas.loc[deltas["year"].between(*base), "coef"].mean()
    d = p.merge(deltas[["year", "coef"]].rename(columns={"year": gr.YEAR,
                                                         "coef": "delta"}),
                on=gr.YEAR, how="left")
    d["logk"] = np.log(d["ppe"].where(d["ppe"] > 0))
    d["gap"] = (d["delta"] - base_level).clip(lower=0) * d[mrpk]
    d["gap"] = d["gap"] - d.groupby([gr.INDUSTRY, gr.YEAR])["gap"].transform("mean")
    d = d.dropna(subset=["gap", "logk", "operating_surplus"])
    d = d.sort_values([gr.FIRM, gr.YEAR])
    d["logk_cf"] = d["logk"] - d.groupby(gr.FIRM)["gap"].cumsum()

    profit = np.log(d["operating_surplus"].where(d["operating_surplus"] > 0))
    d["mrpk_actual"] = profit - d.groupby(gr.FIRM)["logk"].shift(1)
    d["mrpk_cf"] = profit - d.groupby(gr.FIRM)["logk_cf"].shift(1)
    d.attrs["base_level"] = base_level
    return d


def weighted_dispersion(d: pd.DataFrame, mrpk: str, logk: str,
                        min_cell: int = 5) -> pd.Series:
    """Capital-weighted variance of log MRPK within industry-year, by year.

    Weighting by capital is what makes the number an allocation measure rather
    than a description of the firm distribution: a gap at a firm holding a
    hundredth of the industry's capital costs a hundredth as much output.
    """
    f = d.dropna(subset=[mrpk, logk]).copy()
    f["k"] = np.exp(f[logk])
    rows = []
    for (industry, year), cell in f.groupby([gr.INDUSTRY, gr.YEAR]):
        if len(cell) < min_cell:
            continue
        w = cell["k"] / cell["k"].sum()
        m = cell[mrpk]
        mu = float((w * m).sum())
        rows.append({gr.YEAR: year, "var": float((w * (m - mu) ** 2).sum()),
                     "weight": float(cell["k"].sum())})
    out = pd.DataFrame(rows)
    return out.groupby(gr.YEAR).apply(
        lambda x: np.average(x["var"], weights=x["weight"]))


def run(p: pd.DataFrame, axes: dict[str, str], base: tuple[int, int] = (1999, 2004),
        theta: float = THETA, window: int = 5) -> tuple[pd.DataFrame, dict]:
    """The whole exercise.  Returns the yearly comparison and a summary.

    The headline figure is the average over the last ``window`` years, not the
    value in the final year.  Capital-weighted dispersion is far noisier from
    year to year than the unweighted kind, because within an industry-year the
    largest tenth of firms hold about sixty per cent of the capital, so where
    those few firms happen to sit moves the whole number.  Over FY2019--2024 it
    runs 0.27, 0.33, 0.25, 0.32, 0.23, 0.19 with no trend inside the window.
    Reading the exercise off whichever year happens to be last would make the
    answer depend on that draw.
    """
    deltas = year_gradients(p, axes)
    d = counterfactual_capital(p, deltas, axes["mrpk"], base)
    actual = weighted_dispersion(d, "mrpk_actual", "logk")
    counter = weighted_dispersion(d, "mrpk_cf", "logk_cf")
    table = pd.DataFrame({"actual": actual, "counterfactual": counter})
    table["difference"] = table["counterfactual"] - table["actual"]
    table["output_equivalent"] = 100 * theta / 2 * table["difference"]

    last = int(table.index.max())
    lo = last - window + 1
    tail = table.loc[lo:last]
    k = d.loc[d[gr.YEAR].between(lo, last)]
    summary = {
        "base_level": d.attrs["base_level"],
        "final_year": last,
        "final_window": (lo, last),
        "difference": float(tail["difference"].mean()),
        "relative": float(tail["difference"].mean() / tail["actual"].mean()),
        "output_equivalent": float(tail["output_equivalent"].mean()),
        "capital_drift": float(np.exp(k["logk_cf"]).sum()
                               / np.exp(k["logk"]).sum() - 1),
    }
    return table, summary
