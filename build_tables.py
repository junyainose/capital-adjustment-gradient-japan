"""Build the paper's tables.

Run after the panel has been constructed:

    python build_tables.py

Writes LaTeX to output/tables/.  The panel itself is built from Nikkei NEEDS
Financial QUEST, which cannot be redistributed; see README for how to obtain it.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src import build_panel as bp, gradients as gr, shocks as sh, tables as tb  # noqa: E402
from src import counterfactual as cf  # noqa: E402
from src import policy as pol  # noqa: E402

PANEL = "data/processed/panel_annual.parquet"
FIRST_YEAR, LAST_YEAR = 1999, 2024
# Section 6 uses high-frequency monetary policy shocks that end in February
# 2020, so it stops where the shock series does.
POLICY_LAST_YEAR = 2019

MANUFACTURING = [f"{i:02d}" for i in range(1, 40)]
REGULATED = ["67", "69", "55", "57", "63"]      # power, gas, rail, ..., telecom


# --------------------------------------------------------------------------
def attach_value_added(p: pd.DataFrame,
                       hojin: str = "data/raw/hojin_kigyo.csv") -> pd.DataFrame:
    """Value added, imputed from industry wages in the Financial Statements
    Statistics of Corporations by Industry.

    Used only for the measurement robustness row that reproduces the exact
    numerator of Albrizio, Gonzalez and Khametshin (2026).  The main results do
    not depend on this file: operating surplus needs no outside source.
    """
    try:
        from src import labor_cost as lc
        wage = lc.wage_by_industry(lc.load_hojin(hojin))
        return lc.attach_value_added(p, lc.build_wage_map(wage, verbose=False),
                                     wage_all=wage)
    except Exception as exc:                      # file absent or unreadable
        print(f"  value added unavailable ({exc}); skipping that robustness row")
        return p


SYNTHETIC = "data/processed/panel_synthetic.parquet"


def prepare(path: str = PANEL) -> pd.DataFrame:
    """Panel with every variable the tables need.

    The default panel is built from Nikkei NEEDS Financial QUEST, which may not
    be redistributed, so it is absent from the public package.  Failing here
    with a bare FileNotFoundError would leave a reader without a licence with no
    idea what to do next, so the absence is caught and answered.
    """
    if not Path(path).exists():
        hint = (f"panel not found: {path}\n")
        if path == PANEL:
            hint += (
                "\nThe NEEDS-derived panel cannot be redistributed and is not "
                "part of the public package.\n"
                "Without a NEEDS licence, build the synthetic panel and pass it "
                "explicitly on every call:\n\n"
                "    python -m src.synthetic\n"
                f"    python build_tables.py  --panel {SYNTHETIC}\n"
                f"    python build_figures.py --panel {SYNTHETIC}\n\n"
                "The synthetic panel checks that the code runs and that the "
                "estimator recovers a planted gradient.  It reproduces none of "
                "the numbers in the paper; see Appendix D.\n"
                "With a NEEDS licence, run `python build_data.py` first, which "
                "writes the panel to the default path.")
        else:
            hint += ("\nCheck the path passed to --panel.  To build the "
                     "synthetic panel, run `python -m src.synthetic`.")
        raise SystemExit(hint)
    p = pd.read_parquet(path).sort_values([gr.FIRM, gr.YEAR])
    p = attach_value_added(p)
    p["log_ppe"] = np.log(p["ppe"].where(p["ppe"] > 0))
    p["operating_surplus"] = p["op_profit"] + p["depreciation"]
    p["log_emp"] = np.log(p["employees"].where(p["employees"] > 0))

    # marginal revenue product of capital, three numerators and three
    # standardisations (see gradients.standardize)
    numerators = {"os": p["operating_surplus"], "va": p.get("value_added"),
                  "sa": p["sales"]}
    for tag, num in numerators.items():
        if num is None:
            continue
        p[f"mrpk_{tag}_raw"] = gr.log_ratio(num, p["ppe_lag"])
        for how, suffix in [("industry_year", "s"), ("industry_mean", "a"),
                            ("industry_fixed", "f")]:
            p[f"mrpk_{tag}_{suffix}"] = gr.standardize(p, f"mrpk_{tag}_raw", how)

    for how, suffix in [("industry_year", "s"), ("industry_fixed", "f")]:
        p[f"lev_{suffix}"] = gr.standardize(p, "leverage", how)
        p[f"nd_{suffix}"] = gr.standardize(p, "net_debt", how)

    lagged = ([f"mrpk_{t}_{s}" for t in numerators for s in "saf"
               if f"mrpk_{t}_{s}" in p.columns]
              + ["lev_s", "lev_f", "nd_s", "nd_f", "leverage", "net_debt",
                 "log_k_s", "log_assets", "log_emp", "cash_ratio", "log_age",
                 "cf_ta"])
    for c in lagged:
        if c in p.columns:
            p[c + "_lag"] = p.groupby(gr.FIRM)[c].shift(1)

    # Order matters: winsorise first, then take leads, then restrict the period.
    # Taking leads of a raw investment rate leaves the tails in the dependent
    # variable -- inv_rate_comb reaches 1,985 in the raw panel -- and taking
    # leads after restricting the period silently drops the last h years.
    p = bp.winsorize(p, [c for c in p.columns if c.endswith("_lag")]
                     + ["inv_rate_comb", "inv_completed"])
    p = gr.add_leads(p, "inv_rate_comb", prefix="ik")
    p = gr.add_leads(p, "inv_completed", prefix="ic")
    p = gr.add_capital_growth(p)

    p = p[p[gr.YEAR].between(FIRST_YEAR, LAST_YEAR)].copy()
    p["tc"] = p[gr.YEAR] - p[gr.YEAR].mean()

    # capital intensity, fixed at the firm's first observation
    p["industry_year"] = (p[gr.INDUSTRY].astype(str) + "_"
                          + p[gr.YEAR].astype(str))
    # Accounting standard.  NEEDS codes 1 as Japanese GAAP, 2 as US GAAP and 3
    # as IFRS; 0 marks parent-only reporting.  Impairment accounting became
    # mandatory in FY2005 and IFRS could be adopted voluntarily from FY2010, and
    # both change how the book value of tangible assets is written down.  Since
    # that book value is the denominator of MRPK and the base of the dependent
    # variable, a drift that coincided with either change would be suspect.
    if "std_flag" in p.columns:
        p["ever_ifrs"] = p.groupby(gr.FIRM)["std_flag"].transform(
            lambda s: (s == 3).any())
        p["ever_usgaap"] = p.groupby(gr.FIRM)["std_flag"].transform(
            lambda s: (s == 2).any())
    # Alternative denominators.  The dependent variable is defined relative to
    # capital and MRPK is measured over capital, so a mismeasured capital stock
    # enters both.  Replacing capital with total assets on either side breaks
    # that link, at the cost of a denominator that includes cash.
    p["mrpk_ta_raw"] = gr.log_ratio(p["operating_surplus"], p["ta_lag"])
    p["mrpk_ta_s"] = gr.standardize(p, "mrpk_ta_raw", "industry_year")
    p["mrpk_ta_s_lag"] = p.groupby(gr.FIRM)["mrpk_ta_s"].shift(1)
    p["log_ta"] = np.log(p["total_assets"].where(p["total_assets"] > 0))
    p["capex_ta"] = p["capex"] / p["ta_lag"]
    p = bp.winsorize(p, ["mrpk_ta_s_lag", "capex_ta"])
    g_ta = p.groupby(gr.FIRM)["log_ta"]
    for h in (0, 1, 2, 3):
        p[f"dlogta_h{h}"] = 100.0 * (g_ta.shift(-h) - g_ta.shift(1))
        p[f"capexta_h{h}"] = p.groupby(gr.FIRM)["capex_ta"].shift(-h)

    p["k_per_worker"] = gr.log_ratio(p["ppe_lag"], p["employees"])
    first = p.sort_values(gr.YEAR).groupby(gr.FIRM)["k_per_worker"].first()
    p["k_intensity_0"] = p[gr.FIRM].map(first)

    # MRPK averaged over the three preceding years.  MRPK is a flow scaled by a
    # stock and moves far more from year to year than the other two axes:
    # within a firm, its autocorrelation is 0.55 against 0.79 for leverage and
    # 0.84 for log capital.  Leverage needs no such treatment -- it is a balance
    # sheet ratio, and new borrowing enters on top of an existing stock, so it
    # is smoothed by construction.  Averaging MRPK over three years raises its
    # autocorrelation to 0.83, which puts the three axes on the same footing.
    # Three is not chosen to obtain a result; it is the number of years at which
    # the persistence of MRPK matches that of the other regressors.
    #
    # The average is taken over whichever of the three lags exist, rather than
    # requiring all three.  Requiring all three would drop each firm's first
    # three years, which costs FY1999--2001 entirely and makes the leverage
    # drift turn on the handful of early observations that survive.  The strict
    # version is reported in the robustness table.
    lags = pd.concat([p.groupby(gr.FIRM)["mrpk_os_raw"].shift(k)
                      for k in (1, 2, 3)], axis=1)
    have = lags.notna().sum(axis=1)
    p["mrpk_a3_raw"] = lags.mean(axis=1).where(have >= 1)
    p["mrpk_a3_lag"] = gr.standardize(p, "mrpk_a3_raw", "industry_year")
    p["mrpk_a3_strict_raw"] = lags.mean(axis=1).where(have == 3)
    p["mrpk_a3_strict_lag"] = gr.standardize(p, "mrpk_a3_strict_raw",
                                             "industry_year")
    p = bp.winsorize(p, ["mrpk_a3_lag", "mrpk_a3_strict_lag"])
    return p


AXES = {"mrpk": "mrpk_os_s_lag", "lev": "lev_s_lag", "logk": "log_k_s_lag"}
# The same three axes with MRPK averaged over the three preceding years, so
# that all three regressors have comparable persistence.  Reported alongside
# the single-year measure rather than instead of it: averaging isolates the
# persistent part of MRPK, and whether firms also respond to the transitory
# part is a question the paper should not settle by construction.
AXES_AVG = {"mrpk": "mrpk_a3_lag", "lev": "lev_s_lag", "logk": "log_k_s_lag"}
AXES_AVG_STRICT = {"mrpk": "mrpk_a3_strict_lag", "lev": "lev_s_lag",
                   "logk": "log_k_s_lag"}
AXES_RENORMALISED = {"mrpk": "mrpk_renorm_lag", "lev": "lev_renorm_lag",
                     "logk": "logk_renorm_lag"}

# Section 6 follows Albrizio, Gonzalez and Khametshin in normalising MRPK over
# the pooled sample rather than within industry-year, so that the interaction is
# comparable with theirs.
SHOCK_MRPK = "mrpk_os_a_lag"
SHOCK_FILE = "data/raw/KSdata_VAR_IV.csv"
SHOCK_SERIES = {"TARGET": "e_TARGET", "PATH": "e_PATH", "JGBF": "e_JGBF"}
DRIFTS = [("drift_mrpk", "MRPK"), ("drift_lev", "Leverage"),
          ("drift_logk", r"log $k$")]


# --------------------------------------------------------------------------
def attach_shocks(p: pd.DataFrame, path: str = SHOCK_FILE) -> pd.DataFrame:
    """Attach the annual Kubota-Shintani series, signed so that easing is positive."""
    if not Path(path).exists():
        print(f"  {path} not found; Section 6 tables are skipped")
        return p
    ks = sh.load_ks(path)
    annual = pd.DataFrame(
        {c: sh.annualize(ks, col=c).set_index("fy")["shock"]
         for c in ["TARGET", "PATH", "JGBF"]})
    annual = annual.loc[FIRST_YEAR:LAST_YEAR]
    for c in list(annual.columns):
        annual["e_" + c] = -annual[c]
    return p.merge(annual[list(SHOCK_SERIES.values())],
                   left_on=gr.YEAR, right_index=True, how="left")


def table_policy(p: pd.DataFrame) -> str | None:
    """Interaction of the shock with MRPK, by series, horizon and dependent variable.

    No significance markers.  With year fixed effects the coefficient is
    identified from the year-to-year movement of a single annual series, so an
    asterisk based on a firm-clustered standard error would overstate what
    twenty-one years of variation can support.  The leave-one-year-out range is
    reported instead, which shows directly how much the estimate depends on any
    one year.
    """
    if SHOCK_SERIES["TARGET"] not in p.columns:
        return None
    # The high-frequency shock series ends in February 2020, so this exercise
    # cannot follow the rest of the paper to FY2024.  It is restricted here
    # rather than silently estimated on years with no shock.
    p = p[p[gr.YEAR] <= POLICY_LAST_YEAR].copy()
    years = sorted(p[gr.YEAR].unique())
    panels = [("ik", r"Panel A. Investment rate $i/k$", 3),
              ("dlogk", r"Panel B. Capital growth $100\Delta\log k$", 2)]

    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{The gradient's response to monetary policy shocks}",
             r"\label{tab:policy}"]
    lines += tb._open_body(compact=True)
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Series & $h=0$ & $h=1$ & $h=2$ & $h=3$ & "
                 r"Leave-one-year-out, $h=2$ \\")
    for prefix, title, digits in panels:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{6}{l}{\emph{" + title + r"}} \\")
        est = pol.by_series(p, prefix, SHOCK_MRPK, SHOCK_SERIES, years=years)
        for label in SHOCK_SERIES:
            block = est[est["series"] == label].set_index("h")
            cells = []
            for h in range(4):
                if h in block.index:
                    cells.append(f"{block.loc[h, 'coef']:.{digits}f}")
                else:
                    cells.append("")
            row = block.loc[2] if 2 in block.index else None
            rng = ("" if row is None else
                   f"[{row['loyo_min']:.{digits}f}, {row['loyo_max']:.{digits}f}], "
                   f"{int(row['loyo_sig'])}/{int(row['loyo_n'])}")
            lines.append(f"{label} & " + " & ".join(cells) + f" & {rng} \\\\")
            se = [f"({block.loc[h, 'se']:.{digits}f})" if h in block.index else ""
                  for h in range(4)]
            lines.append(" & " + " & ".join(se) + " & \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(tb.BOX_CLOSE)
    lines.append(tb.NOTE_OPEN)
    lines.append(
        "Each cell is a separate regression of the dependent variable $h$ periods "
        "ahead on the interaction of the shock with MRPK, with firm and year "
        "fixed effects and log assets as a control.  The year effects absorb the "
        "shock's own level, so the average response is not estimated.  The shock "
        "is signed so that a positive value is an easing.  Standard errors "
        "clustered on the firm in parentheses.  No significance markers are "
        "shown: the coefficient is identified from the year-to-year movement of "
        "a single annual series, and an asterisk based on a firm-clustered "
        "standard error would overstate what twenty-one years of variation can "
        "support.  The last column reports, for $h=2$, the smallest and largest "
        "estimate obtained when each fiscal year is dropped in turn, and how "
        "many of those re-estimates keep a $t$ ratio above two.")
    lines.append(tb.NOTE_CLOSE)
    lines.append(r"\end{table}")
    return "\n".join(lines)


def renormalise(d: pd.DataFrame) -> pd.DataFrame:
    """Recompute the industry-year moments inside the sample passed in.

    The regressors are deviations from the industry-year mean, scaled by the
    industry-year standard deviation, so a firm's own value moves when its peer
    group changes even if the firm itself does not.  Firm fixed effects do not
    absorb that: they remove what is constant about the firm, not a normalisation
    whose reference group is turning over.  Restricting to firms present
    throughout and then recomputing the moments within that restricted sample
    removes the channel, at the cost of conditioning on survival.
    """
    d = d.copy()
    for label, source in [("mrpk_renorm", "mrpk_os_raw"),
                          ("lev_renorm", "leverage"),
                          ("logk_renorm", "log_ppe")]:
        d[label] = gr.standardize(d, source, "industry_year")
        d[label + "_lag"] = d.groupby(gr.FIRM)[label].shift(1)
    return bp.winsorize(d, [c + "_lag" for c in AXES_RENORMALISED.values()
                            if c.endswith("_lag")]
                        + [v for v in AXES_RENORMALISED.values()])


def table_main(p: pd.DataFrame) -> str:
    """The headline regressions, each dependent variable under two MRPK measures.

    Columns come in pairs.  Within a pair the dependent variable is the same and
    only the measurement of MRPK changes: the single preceding year, or the mean
    of the three preceding years.  MRPK is a flow over a stock and moves far more
    from year to year than the other two regressors -- within a firm its
    autocorrelation is 0.51 against 0.79 for leverage and 0.84 for log capital --
    so the single-year measure asks the three axes an unequal question.
    Averaging over three years brings its persistence to 0.82 and makes the
    comparison across axes a fair one.  Both are reported because they answer
    different questions: whether firms respond to a persistent difference in
    returns, and whether they respond to returns at all.
    """
    specs = [("ik_h1", r"$i/k$, $h=1$", AXES),
             ("ik_h1", r"$i/k$, $h=1$", AXES_AVG),
             ("dlogk_h1", r"$\Delta\log k$, $h=1$", AXES),
             ("dlogk_h1", r"$\Delta\log k$, $h=1$", AXES_AVG),
             ("dlogk_h2", r"$\Delta\log k$, $h=2$", AXES),
             ("dlogk_h2", r"$\Delta\log k$, $h=2$", AXES_AVG)]
    fits = [gr.fit_trend(p, dep, ax) for dep, _, ax in specs]
    rows = [("level_mrpk", "MRPK"), ("drift_mrpk", r"MRPK $\times$ trend"),
            ("level_lev", "Leverage"), ("drift_lev", r"Leverage $\times$ trend"),
            ("level_logk", r"log $k$"), ("drift_logk", r"log $k$ $\times$ trend")]
    n = len(specs)
    footer = [("MRPK measured over", ["1 year", "3 years"] * (n // 2)),
              ("Firm FE", ["Yes"] * n), ("Year FE", ["Yes"] * n),
              ("Controls", ["log assets"] * n)]
    return tb.coef_table(
        fits, [lab for _, lab, _ in specs], rows, footer=footer,
        caption=f"Investment gradients and their drift, {FIRST_YEAR}--{LAST_YEAR}",
        label="tab:main",
        notes=("Each column is a separate regression.  Columns come in pairs: "
               "within a pair only the measurement of MRPK changes, from the "
               "single preceding year to the mean of the three preceding years. "
               " Where fewer than three lags exist the mean is taken over those "
               "that do, which is why the three-year columns have slightly more "
               "observations rather than fewer.  Firm characteristics are "
               "standardised within industry and year, so a coefficient is the "
               "response to a one standard deviation difference measured against "
               "contemporaneous peers.  The trend is the fiscal year centred on "
               "the sample mean, so the level coefficient is the gradient at the "
               "middle of the sample and the trend coefficient is its change per "
               "year."))


def table_robustness(p: pd.DataFrame) -> str:
    rows, breaks = [], {}

    def add(label, fit):
        rows.append((label, fit))

    breaks[0] = "Measurement of MRPK"
    for tag, name in [("os", "Operating surplus"), ("va", "Value added"),
                      ("sa", "Sales")]:
        for suffix, how in [("s", "industry-year units"),
                            ("a", "pooled units, AGK")]:
            col = f"mrpk_{tag}_{suffix}_lag"
            if col not in p.columns:
                continue
            add(f"{name} / {how}",
                gr.fit_trend(p, "dlogk_h1", {**AXES, "mrpk": col}))

    breaks[len(rows)] = "Denominator"
    add(r"MRPK over total assets, $\Delta\log k$",
        gr.fit_trend(p, "dlogk_h1", {**AXES, "mrpk": "mrpk_ta_s_lag"}))
    add(r"MRPK over capital, $\Delta\log$ assets",
        gr.fit_trend(p, "dlogta_h1", AXES))
    add(r"Both over total assets, $\Delta\log$ assets",
        gr.fit_trend(p, "dlogta_h1", {**AXES, "mrpk": "mrpk_ta_s_lag"}))
    add(r"Both over total assets, capex over assets",
        gr.fit_trend(p, "capexta_h1", {**AXES, "mrpk": "mrpk_ta_s_lag"}))

    breaks[len(rows)] = "Multi-period averaging of MRPK"
    # Two things are being checked here.  First, whether the drift is an artefact
    # of measurement error in MRPK that weakened over the sample: averaging over
    # J years cuts the variance of an independent error by roughly 1/J, so a
    # drift produced that way would shrink with J.  It does not.  Second, how
    # much the second column of Table~\\ref{tab:main} owes to taking the mean
    # over whichever lags exist rather than requiring all three.  Requiring all
    # three costs each firm its first three years, and with them FY1999--2001;
    # the MRPK drift is unaffected, but the leverage drift is, because the
    # deepening of the leverage gradient is measured over a sample four years
    # shorter at the start.
    d = p.sort_values([gr.FIRM, gr.YEAR]).copy()
    for J in (2, 4, 5):
        lags = pd.concat([d.groupby(gr.FIRM)["mrpk_os_raw"].shift(k)
                          for k in range(1, J + 1)], axis=1)
        d[f"mavg{J}_raw"] = lags.mean(axis=1).where(lags.notna().sum(axis=1) >= 1)
        d[f"mavg{J}"] = gr.standardize(d, f"mavg{J}_raw", "industry_year")
    d = bp.winsorize(d, [f"mavg{J}" for J in (2, 4, 5)])
    add("MRPK averaged over 2 years",
        gr.fit_trend(d, "dlogk_h1", {**AXES, "mrpk": "mavg2"}))
    add("MRPK averaged over 4 years",
        gr.fit_trend(d, "dlogk_h1", {**AXES, "mrpk": "mavg4"}))
    add("MRPK averaged over 5 years",
        gr.fit_trend(d, "dlogk_h1", {**AXES, "mrpk": "mavg5"}))
    add("MRPK averaged over 3 years, all three lags required",
        gr.fit_trend(p, "dlogk_h1", AXES_AVG_STRICT))

    breaks[len(rows)] = "Dependent variable"
    for dep, label in [("ik_h0", r"$i/k$, $h=0$"), ("ik_h1", r"$i/k$, $h=1$"),
                       ("ik_h3", r"$i/k$, $h=3$"),
                       ("dlogk_h3", r"$\Delta\log k$, $h=3$"),
                       ("ic_h1", r"Completion-based $i/k$, $h=1$")]:
        add(label, gr.fit_trend(p, dep, AXES))

    breaks[len(rows)] = "Controls and fixed effects"
    for ctrl, label in [(("log_assets_lag", "log_age_lag"), "Add firm age"),
                        (("log_emp_lag", "cash_ratio_lag"), "AGK controls"),
                        (("log_assets_lag", "cf_ta_lag"), "Add cash flow")]:
        add(label, gr.fit_trend(p, "dlogk_h1", AXES, controls=ctrl))
    add("Firm-specific linear trends",
        gr.fit_trend(p, "dlogk_h1", AXES, firm_trends=True))
    add("Industry-year FE, no firm FE (between)",
        gr.fit_trend(p, "dlogk_h1", AXES, absorb=("industry_year",)))
    add(r"Industry-year FE, no firm FE, $i/k$",
        gr.fit_trend(p, "ik_h1", AXES, absorb=("industry_year",)))
    add("Industry-year and firm FE",
        gr.fit_trend(p, "dlogk_h1", AXES,
                     absorb=(gr.FIRM, "industry_year")))

    breaks[len(rows)] = "Sample"
    subsets = [
        ("Excluding FY2016", p[p[gr.YEAR] != 2016]),
        ("Excluding FY1999--2001", p[~p[gr.YEAR].isin([1999, 2000, 2001])]),
        ("FY2002--2015 only", p[p[gr.YEAR].between(2002, 2015)]),
        ("Excluding regulated industries", p[~p[gr.INDUSTRY].isin(REGULATED)]),
        ("Manufacturing only", p[p[gr.INDUSTRY].isin(MANUFACTURING)]),
        ("Non-manufacturing only", p[~p[gr.INDUSTRY].isin(MANUFACTURING)]),
        ("Excluding restructured holding companies", p[p["hold_flag"] == 0]),
    ]
    for label, sub in subsets:
        add(label, gr.fit_trend(sub, "dlogk_h1", AXES))

    both_halves = (set(p[p[gr.YEAR].between(2000, 2007)][gr.FIRM])
                   & set(p[p[gr.YEAR].between(2010, 2018)][gr.FIRM]))
    balanced = bp.balanced_firms(p, FIRST_YEAR, LAST_YEAR,
                                 LAST_YEAR - FIRST_YEAR + 1)
    add("Firms present in both halves",
        gr.fit_trend(p[p[gr.FIRM].isin(both_halves)], "dlogk_h1", AXES))
    add("Both halves, moments recomputed within",
        gr.fit_trend(renormalise(p[p[gr.FIRM].isin(both_halves)]),
                     "dlogk_h1", AXES_RENORMALISED))
    add("Observed throughout, moments recomputed within",
        gr.fit_trend(renormalise(p[p[gr.FIRM].isin(balanced)]),
                     "dlogk_h1", AXES_RENORMALISED))

    if "ever_ifrs" in p.columns:
        breaks[len(rows)] = "Accounting regime"
        add(f"FY2006--{LAST_YEAR} only (after mandatory impairment)",
            gr.fit_trend(p[p[gr.YEAR].between(2006, LAST_YEAR)], "dlogk_h1", AXES))
        add("Excluding FY2004--2006",
            gr.fit_trend(p[~p[gr.YEAR].isin([2004, 2005, 2006])],
                         "dlogk_h1", AXES))
        add("Excluding FY2004--2006 and FY2010--2012",
            gr.fit_trend(p[~p[gr.YEAR].isin([2004, 2005, 2006,
                                             2010, 2011, 2012])],
                         "dlogk_h1", AXES))
        add("Excluding firms that ever adopted IFRS",
            gr.fit_trend(p[~p["ever_ifrs"]], "dlogk_h1", AXES))
        add("Japanese GAAP firm-years only",
            gr.fit_trend(p[p["std_flag"] == 1], "dlogk_h1", AXES))

    breaks[len(rows)] = "Capital intensity (fixed at first observation)"
    q = pd.qcut(p["k_intensity_0"].rank(method="first"), 3, labels=[1, 2, 3])
    for tercile, label in [(1, "Low"), (2, "Middle"), (3, "High")]:
        add(label, gr.fit_trend(p[q == tercile], "dlogk_h1", AXES))

    # The table no longer fits on one page.  It is scaled to \\linewidth, so the
    # rendered height is fixed by the aspect ratio and shrinking the font does
    # not help; the only lever is the number of rows.  Split at the point where
    # the exercise changes from varying the measurement to varying the sample.
    cut = min(i for i, name in breaks.items() if name == "Sample")
    common = ("Each row is a separate regression and reports only the "
              "characteristic-by-trend interactions; levels, controls and fixed "
              "effects are as in column (3) of Table~\\ref{tab:main} unless "
              "stated.  The dependent variable is $100\\times\\Delta\\log k$ at "
              "$h=1$ unless the row says otherwise.")
    parts = [
        (rows[:cut], {i: v for i, v in breaks.items() if i < cut},
         "Drift in the investment gradients: measurement and specification",
         "tab:robustness",
         common + "  Under firm-specific linear trends the log capital "
         "coefficient is not interpretable, since log capital is itself close "
         "to a firm-specific trend.  The row without firm fixed effects "
         "estimates the gradient between firms rather than within them; it is "
         "the cross-sectional quantity, and it carries any permanent "
         "heterogeneity correlated with the characteristic.  In the denominator "
         "panel the last row is the one specification in which the drift does "
         "not appear; total assets grew faster than capital over the sample as "
         "firms accumulated cash, which imparts a downward trend to capital "
         "expenditure scaled by assets.  Averaging MRPK over several years cuts "
         "the variance of an independent measurement error by roughly one over "
         "the number of years, so a drift produced by weakening attenuation "
         "would shrink down that panel.  It does not."),
        (rows[cut:], {i - cut: v for i, v in breaks.items() if i >= cut},
         "Drift in the investment gradients: sample",
         "tab:robustness-sample",
         common + "  Where the moments are recomputed within a sample, the "
         "industry-year mean and standard deviation used to normalise a firm's "
         "characteristic are calculated from that sample alone, so that entry "
         "and exit of peers cannot move a continuing firm's regressor.  "
         "Mandatory impairment accounting from FY2006 and voluntary IFRS "
         "adoption from FY2010 both change how the book value of tangible "
         "assets is written down, and that book value is the denominator of "
         "MRPK."),
    ]
    return "\n\n".join(
        tb.summary_table(r, DRIFTS, group_breaks=b, caption=c, label=l, notes=n)
        for r, b, c, l, n in parts)


def table_units(p: pd.DataFrame) -> str:
    """How the leverage drift depends on the units the characteristic is measured in."""
    variants = [
        ("Raw, no industry-year demeaning", "leverage_lag"),
        ("Industry-year demeaned, fixed scale", "lev_f_lag"),
        ("Industry-year demeaned and rescaled", "lev_s_lag"),
    ]
    rows = []
    for label, col in variants:
        rows.append((label + r", $\Delta\log k$",
                     gr.fit_trend(p, "dlogk_h1", {**AXES, "lev": col})))
    for label, col in variants:
        rows.append((label + r", $i/k$",
                     gr.fit_trend(p, "ik_h1", {**AXES, "lev": col})))
    return tb.summary_table(
        rows, DRIFTS,
        caption="Two normalisations, two questions: how the reported drift "
                "depends on the unit of comparison",
        label="tab:units",
        notes=("The rows differ only in how leverage is normalised.  The first "
               "is in units of leverage itself, the second in units of the "
               "industry's full-sample standard deviation, and the third in "
               "units of the contemporaneous industry-year standard deviation. "
               "These are three estimands, not three estimates of one: a slope "
               "measured per unit and a slope measured per contemporaneous "
               "standard deviation are related by that standard deviation, so "
               "they must differ whenever the dispersion of the characteristic "
               "moves.  Over this sample leverage compresses by about a fifth "
               "and MRPK widens by about a fifth, so one and the same change "
               "is reported as a larger or a smaller number depending on the "
               "convention.  For both characteristics the sign and the "
               "significance survive all three; the magnitude does not."))


def table_descriptives(p: pd.DataFrame) -> str:
    """Distribution of the variables the estimation uses.

    Percentiles rather than minima and maxima: the raw series have long tails
    that the estimation trims, and reporting the untrimmed extremes would
    describe observations no coefficient is estimated from.
    """
    p = p.copy()
    p["log_emp_level"] = p["employees"]
    rows = [
        ("ik_h0", r"Investment rate $i/k$"),
        ("dlogk_h1", r"Capital growth $100\Delta\log k$, $h=1$"),
        ("mrpk_os_raw", r"MRPK, logs"),
        ("leverage", "Leverage"),
        ("net_debt", "Net debt / assets"),
        ("cash_ratio", "Cash / assets"),
        ("cf_ta", "Operating cash flow / assets"),
        ("log_ppe", r"log capital"),
        ("log_assets", r"log assets"),
        ("employees", "Employees"),
        ("age", "Firm age, years"),
    ]
    stats = []
    index = []
    for col, label in rows:
        if col not in p.columns:
            continue
        x = p[col].dropna()
        stats.append([len(x), x.mean(), x.std(),
                      x.quantile(0.10), x.quantile(0.50), x.quantile(0.90)])
        index.append(label)
    d = pd.DataFrame(stats, index=index,
                     columns=["Obs.", "Mean", "S.D.", "p10", "Median", "p90"])
    d["Obs."] = d["Obs."].map(lambda v: f"{int(v):,}")
    body = tb.frame_table(
        d.drop(columns=["Obs."]), digits=3,
        caption=f"Descriptive statistics, FY{FIRST_YEAR}--{LAST_YEAR}", label="tab:descriptives",
        index_label="",
        notes=("Firm-years in the estimation window.  Ratios are winsorised at "
               "the first and ninety-ninth percentiles, as in the estimation; "
               "levels are not.  MRPK is the log of operating surplus over "
               "lagged capital, before the industry-year normalisation of "
               "equation~(2)."))
    return body


def table_margins(p: pd.DataFrame) -> str:
    """Where in the distribution, and on which margin, the gradient strengthened.

    Two decompositions of the same fact.  Panel A splits the characteristic at
    the industry-year mean, so that the slope above and below it can move
    separately: a gradient that rose because high-MRPK firms expanded faster is
    a different phenomenon from one that rose because low-MRPK firms contracted
    faster.  Panel B splits the outcome instead, into whether the firm adjusted
    its capital stock materially and by how much given that it did.
    """
    p = p.copy()
    p["mrpk_above"] = p[AXES["mrpk"]].clip(lower=0)
    p["mrpk_below"] = p[AXES["mrpk"]].clip(upper=0)
    growth = p["dlogk_h1"]
    p["expand"] = (growth > 5).astype(float).where(growth.notna())
    p["contract"] = (growth < -5).astype(float).where(growth.notna())
    p["size_expand"] = growth.where(growth > 0)
    p["size_contract"] = growth.where(growth < 0)

    rows, index = [], []
    piece = gr.fit_trend(p, "dlogk_h1",
                         {"hi": "mrpk_above", "lo": "mrpk_below",
                          "lev": AXES["lev"], "logk": AXES["logk"]})
    for key, label in [("hi", "MRPK above the industry-year mean"),
                       ("lo", "MRPK below the industry-year mean")]:
        level, level_se, _ = piece.get("level_" + key)
        drift, drift_se, pval = piece.get("drift_" + key)
        rows.append([level, level_se, drift, drift_se, tb.stars(pval)])
        index.append(label)

    for dep, label in [("expand", r"Pr(expansion above five percent)"),
                       ("contract", r"Pr(contraction below five percent)"),
                       ("size_expand", "Size of expansion, given one"),
                       ("size_contract", "Size of contraction, given one")]:
        f = gr.fit_trend(p, dep, AXES)
        if f is None:
            continue
        level, level_se, _ = f.get("level_mrpk")
        drift, drift_se, pval = f.get("drift_mrpk")
        rows.append([level, level_se, drift, drift_se, tb.stars(pval)])
        index.append(label)

    frame = pd.DataFrame(rows, index=index,
                         columns=["Level", "(s.e.)", "Drift", "(s.e.)", ""])
    return tb.frame_table(
        frame, digits=4, caption="Where the gradient strengthened",
        label="tab:margins", index_label="",
        notes=("The first two rows come from one regression in which the "
               "characteristic enters twice, once truncated below at the "
               "industry-year mean and once above, so the slope on each side "
               "moves separately.  The remaining four are separate regressions "
               "with the dependent variable redefined; the size rows condition "
               "on the sign of the change, so they are estimated on part of the "
               "sample.  Levels, controls and fixed effects are as in "
               "column~(3) of Table~\\ref{tab:main}."))


def table_inference(p: pd.DataFrame, dep: str = "dlogk_h1") -> str:
    """The same estimate under alternative clustering schemes.

    Firm clustering is the baseline, but the interaction of interest pairs a
    firm characteristic with a linear trend, so residuals may be correlated
    within an industry-year as well.  Nothing here changes the estimate; the
    point is to show what happens to its standard error.

    Year clustering is included for completeness and read with care: with
    twenty-one clusters the asymptotic approximation behind the cluster-robust
    variance is not reliable, which is why it is not the baseline.
    """
    p = p.copy()
    p["industry_year"] = (p[gr.INDUSTRY].astype(str) + "_"
                          + p[gr.YEAR].astype(str))
    schemes = [("Firm (baseline)", (gr.FIRM,)),
               ("Industry-year", ("industry_year",)),
               ("Firm and industry-year", (gr.FIRM, "industry_year")),
               ("Industry", (gr.INDUSTRY,)),
               ("Firm and year", (gr.FIRM, gr.YEAR)),
               ("Year", (gr.YEAR,))]
    rows, index = [], []
    for label, cl in schemes:
        f = gr.fit_trend(p, dep, AXES, cluster=cl)
        if f is None:
            continue
        b, se, _ = f.get("drift_mrpk")
        level, level_se, _ = f.get("level_mrpk")
        n_cl = (p[list(cl)].astype(str).agg("|".join, axis=1).nunique()
                if len(cl) > 1 else p[cl[0]].nunique())
        rows.append([level, level_se, b, se, b / se, n_cl])
        index.append(label)
    frame = pd.DataFrame(rows, index=index,
                         columns=["Level", "(s.e.)", "Drift", "(s.e.)",
                                  "$t$", "Clusters"])
    frame["Clusters"] = frame["Clusters"].map(lambda v: f"{int(v):,}")
    return tb.frame_table(
        frame, digits=3, caption="The MRPK gradient under alternative "
                                 "clustering schemes",
        label="tab:inference", index_label="Clustered on",
        notes=("One specification, column (3) of Table~\\ref{tab:main}, with the "
               "variance computed six ways.  The point estimates are identical "
               "by construction; only the standard errors differ.  Two-way "
               "clustering follows Cameron, Gelbach and Miller.  The row "
               "clustered on the year is reported for completeness but is not "
               "the baseline: twenty-one clusters are too few for the "
               "asymptotic approximation behind the cluster-robust variance."))


def table_periods(p: pd.DataFrame, dep: str = "dlogk_h1") -> str:
    """The MRPK gradient by sub-period, as an alternative to the linear trend.

    A linear trend is a parsimonious summary, not a claim that the gradient rose
    at a constant rate.  Replacing it with period dummies interacted with MRPK
    shows where the increase actually sits, and lets the reader see whether the
    linear specification is hiding a step.

    The windows are of roughly equal length.  An earlier version of this table
    used three windows of unequal length, which put the 2008 crisis in the
    middle of one of them and made the rise look concentrated at the end.  With
    equal windows the increments are nearly constant, so the linear trend is not
    hiding a step.  Unequal windows are a choice, and this one was making the
    result.

    The windows match those of Table~\\ref{tab:decomposition} so that the two
    exercises are read on the same calendar.
    """
    cuts = [(FIRST_YEAR, 2004), (2005, 2009), (2010, 2014), (2015, 2019),
            (2020, LAST_YEAR)]
    d = p.copy()
    rhs = []
    for lo, hi in cuts[1:]:
        d[f"mrpk_{lo}"] = d[AXES["mrpk"]] * ((d[gr.YEAR] >= lo)
                                             & (d[gr.YEAR] <= hi))
        d[f"post_{lo}"] = ((d[gr.YEAR] >= lo) & (d[gr.YEAR] <= hi)).astype(float)
        rhs += [f"mrpk_{lo}", f"post_{lo}"]
    rhs += [AXES["mrpk"], AXES["lev"], AXES["logk"], "log_assets_lag"]
    d = d.dropna(subset=[dep] + rhs + [gr.FIRM, gr.YEAR])
    x = gr._absorb(d, [dep] + rhs, [gr.FIRM, gr.YEAR])
    coef, se, pval = gr._ols(x[dep].to_numpy(float), x[rhs].to_numpy(float),
                             rhs, d[gr.FIRM].to_numpy())

    base = coef[AXES["mrpk"]]
    rows, index = [], []
    rows.append([base, se[AXES["mrpk"]], base])
    index.append(f"FY{cuts[0][0]}--{cuts[0][1]}")
    for lo, hi in cuts[1:]:
        rows.append([coef[f"mrpk_{lo}"], se[f"mrpk_{lo}"],
                     base + coef[f"mrpk_{lo}"]])
        index.append(f"FY{lo}--{hi}")
    frame = pd.DataFrame(rows, index=index,
                         columns=["Coefficient", "(s.e.)", "Implied level"])
    return tb.frame_table(
        frame, digits=3, caption="The MRPK gradient by sub-period",
        label="tab:periods", index_label="Period",
        notes=("One regression.  The first row is the gradient in the omitted "
               "period; the others are differences from it, with the implied "
               "level in the last column.  Levels, controls and fixed effects "
               "are as in column (3) of Table~\\ref{tab:main}, with the linear "
               "trend replaced by period interactions.  Dependent variable "
               "$100\\times\\Delta\\log k$ at $h=1$."))


def table_persistence(p: pd.DataFrame, window: int = 5) -> str:
    """Within-firm persistence of MRPK, by sub-period.

    A rising gradient can be generated without any change in technology or in
    frictions.  If innovations to MRPK became more persistent, the optimal
    investment response to a given gap widens, and the estimated gradient rises
    with it.  The test is direct: estimate the within-firm autoregression of
    MRPK over successive windows of equal length and see whether it trends.

    Windows are of equal length because the within transformation biases the
    autoregressive coefficient downward by an amount that depends on the number
    of periods; comparing windows of different lengths would confound that bias
    with the object of interest.
    """
    d = p.sort_values([gr.FIRM, gr.YEAR]).copy()
    d["mrpk_now"] = d.groupby(gr.FIRM)["mrpk_os_s_lag"].shift(-1)
    d["mrpk_lag"] = d["mrpk_os_s_lag"]

    starts = list(range(LAST_YEAR - window + 1, FIRST_YEAR, -window))[::-1]
    rows, index = [], []
    for lo in starts:
        hi = lo + window - 1
        s = d[d[gr.YEAR].between(lo, hi)].dropna(subset=["mrpk_now", "mrpk_lag"])
        if len(s) < 500:
            continue
        y = s["mrpk_now"] - s.groupby(gr.FIRM)["mrpk_now"].transform("mean")
        x = s["mrpk_lag"] - s.groupby(gr.FIRM)["mrpk_lag"].transform("mean")
        rows.append([float((x * y).sum() / (x * x).sum()),
                     float(s["mrpk_now"].corr(s["mrpk_lag"])), len(s)])
        index.append(f"FY{lo}--{hi}")

    # FY2020 disturbs MRPK for a single year and then it returns, which reads as
    # a collapse in persistence when that year sits inside the window.  The last
    # window is therefore also shown without it.  Doing so shortens the window
    # by one year, and the within transformation biases the autoregression down
    # by more when there are fewer periods, so the excluded figure is if
    # anything a lower bound on persistence over those years.
    if LAST_YEAR >= 2024:
        s = d[d[gr.YEAR].between(2021, LAST_YEAR)].dropna(
            subset=["mrpk_now", "mrpk_lag"])
        y = s["mrpk_now"] - s.groupby(gr.FIRM)["mrpk_now"].transform("mean")
        x = s["mrpk_lag"] - s.groupby(gr.FIRM)["mrpk_lag"].transform("mean")
        rows.append([float((x * y).sum() / (x * x).sum()),
                     float(s["mrpk_now"].corr(s["mrpk_lag"])), len(s)])
        index.append(f"FY2021--{LAST_YEAR} (excl. FY2020)")

    frame = pd.DataFrame(rows, index=index,
                         columns=["Within-firm AR(1)", "Raw correlation",
                                  "Obs."])
    frame["Obs."] = frame["Obs."].map(lambda v: f"{int(v):,}")
    return tb.frame_table(
        frame.drop(columns=["Obs."]), digits=3,
        caption="Persistence of MRPK, by sub-period", label="tab:persistence",
        index_label="Window",
        notes=("MRPK is normalised within industry and year, so the level is "
               "comparable across windows.  The first column removes firm means "
               "within the window; the second does not.  Windows are of equal "
               "length so that the downward bias of the within estimator is "
               "common to them.  A gradient that strengthened because shocks to "
               "MRPK became more persistent would show a rising first column."))


def table_counterfactual(p: pd.DataFrame) -> str:
    """What the rise in the gradient did to the dispersion of MRPK."""
    base = (FIRST_YEAR, 2004)   # the first of the four windows of Table~\\ref{tab:periods}
    table, summary = cf.run(p, AXES, base=base)
    rows = [y for y in (2004, 2009, 2014, 2019, LAST_YEAR) if y in table.index]
    frame = table.loc[rows, ["actual", "counterfactual", "difference",
                             "output_equivalent"]]
    frame.index = [f"FY{y}" for y in rows]
    lo, hi = summary["final_window"]
    tail = table.loc[lo:hi, ["actual", "counterfactual", "difference",
                             "output_equivalent"]].mean()
    tail.name = f"FY{lo}--{hi} average"
    frame = pd.concat([frame, tail.to_frame().T])
    frame.columns = ["Actual", "Counterfactual", "Difference",
                     "Output equivalent, per cent"]
    return tb.frame_table(
        frame, digits=4, caption="Dispersion of MRPK under the counterfactual "
                                 "of an unchanged gradient",
        label="tab:counterfactual", index_label="Year",
        notes=("Capital-weighted variance of log MRPK within industry and year. "
               "The counterfactual removes from each firm's capital growth the "
               f"part attributable to the gradient exceeding its FY{base[0]}--"
               f"{base[1]} level of {summary['base_level']:.4f}, which is the "
               "first of the windows of Table~\\ref{tab:periods}.  The excess is "
               "demeaned within industry "
               "and year so that the total capital stock is left almost "
               f"unchanged ({100 * summary['capital_drift']:+.2f} per cent at "
               "the end of the sample).  MRPK is recomputed from counterfactual "
               "capital.  The output equivalent is $(\\theta/2)$ times the "
               "difference with $\\theta=0.30$, a first-order approximation "
               "that fixes an order of magnitude rather than a number.  Two "
               "choices move the answer and are stated rather than buried.  The "
               "base window: taking FY2000--2006 instead, as an earlier version "
               "of this table did, roughly halves the difference, because the "
               "gradient was already rising inside any window wide enough to "
               "estimate and the level held fixed depends on where the window "
               "sits.  And the year read off at the end: capital-weighted "
               "dispersion is noisy from year to year, since the largest tenth "
               "of firms in an industry-year hold about sixty per cent of its "
               "capital, so the last line averages over the final five years "
               "rather than reporting FY{last} alone, which happens to be a low "
               "year.".replace("{last}", str(LAST_YEAR))))


def table_decomposition(p: pd.DataFrame, window: int = 5) -> str:
    """Cross-sectional dispersion of MRPK, split into between- and within-firm.

    The gradient measures how investment responds to MRPK; this measures what
    that response leaves behind.  Faster adjustment closes transitory gaps, so
    it should show up as a falling within-firm component.  A permanent
    difference between firms is not something adjustment can close, so the
    between-firm component answers to something else.

    Both are computed on MRPK demeaned within industry and year but not
    rescaled, so that the numbers are comparable across windows.  Firms are
    required to appear in at least four years of a window, since a firm mean
    estimated from one or two years would attribute noise to the permanent
    component.
    """
    d = p.copy()
    g = d.groupby([gr.INDUSTRY, gr.YEAR])["mrpk_os_raw"]
    d["centred"] = d["mrpk_os_raw"] - g.transform("mean")

    rows, index = [], []
    starts = list(range(LAST_YEAR - window + 1, FIRST_YEAR, -window))[::-1]
    for lo in starts:
        hi = lo + window - 1
        s = d[d[gr.YEAR].between(lo, hi)].dropna(subset=["centred"])
        s = s[s.groupby(gr.FIRM)["centred"].transform("size") >= window - 1]
        if len(s) < 500:
            continue
        firm_mean = s.groupby(gr.FIRM)["centred"].transform("mean")
        total = float(s["centred"].var())
        within = float((s["centred"] - firm_mean).var())
        rows.append([total, total - within, within, (total - within) / total])
        index.append(f"FY{lo}--{hi}")

    # FY2020 alone undoes the contraction of the within-firm component.  The
    # within-industry-year spread of MRPK is 0.79 in FY2019, jumps to 0.87 in
    # FY2020 and falls back the year after.  The last window is therefore also
    # reported without it, so a reader can see how much of the reversal is one
    # pandemic year and how much is a change in how firms adjust.
    if LAST_YEAR >= 2024:
        s = d[d[gr.YEAR].between(2021, LAST_YEAR)].dropna(subset=["centred"])
        s = s[s.groupby(gr.FIRM)["centred"].transform("size") >= window - 1]
        firm_mean = s.groupby(gr.FIRM)["centred"].transform("mean")
        total = float(s["centred"].var())
        within = float((s["centred"] - firm_mean).var())
        rows.append([total, total - within, within, (total - within) / total])
        index.append(f"FY2021--{LAST_YEAR} (excl. FY2020)")

    frame = pd.DataFrame(rows, index=index,
                         columns=["Total", "Between firms", "Within firms",
                                  "Between share"])
    return tb.frame_table(
        frame, digits=3, caption="Dispersion of MRPK, between and within firms",
        label="tab:decomposition", index_label="Window",
        notes=("Variance of MRPK after removing industry-year means, decomposed "
               "into the variance of firm means within the window and the "
               "variance around them.  Firms observed in fewer than "
               f"{window - 1} years of a window are excluded, since a firm mean "
               "estimated from one or two observations would load noise onto "
               "the between-firm component."))


def table_dispersion(p: pd.DataFrame) -> str:
    cols = {"leverage_lag": "Leverage", "mrpk_os_raw": "MRPK (logs)",
            "log_k_s_lag": "log $k$ (standardised)"}
    d = gr.dispersion(p, [c for c in cols if c in p.columns])
    d = d.rename(columns=cols)
    d = d.loc[[y for y in range(2000, LAST_YEAR, 2) if y in d.index]]
    d = 100 * d / d.iloc[0]
    d.index.name = "Fiscal year"
    return tb.frame_table(
        d, digits=1, caption="Cross-sectional dispersion of firm characteristics",
        label="tab:dispersion", index_label="Fiscal year",
        notes=("Mean within industry-year standard deviation, indexed to 100 in "
               "2000.  Leverage compresses over the sample while the dispersion "
               "of the other two characteristics does not."))


def table_productivity(p: pd.DataFrame, dep: str = "dlogk_h1") -> str:
    """Responsiveness to a productivity residual, as the comparison papers measure it.

    Decker et al. and Hambur and Andrews regress input growth on productivity,
    not on MRPK, so a like-for-like comparison needs the same left- and
    right-hand side.  The obstacle is that a productivity residual requires
    output elasticities, and the panel does not identify them; they are imposed.
    The table therefore reports the whole grid rather than one number.  The sign
    survives everywhere and the magnitude does not, which is why this is a check
    and not a headline: MRPK needs no such parameter.
    """
    ok = (p["sales"] > 0) & (p["ppe"] > 0) & (p["employees"] > 0)
    rows, index = [], []
    for tk in (0.20, 0.30, 0.40):
        for gl in (0.50, 0.60, 0.70):
            d = p.copy()
            d["z"] = np.where(ok, np.log(d["sales"]) - tk * np.log(d["ppe"])
                              - gl * np.log(d["employees"]), np.nan)
            d["z_s"] = gr.standardize(d, "z", "industry_year")
            d = bp.winsorize(d, ["z_s"])
            d["z_s_lag"] = d.groupby(gr.FIRM)["z_s"].shift(1)
            f = gr.fit_trend(d, dep, {"z": "z_s_lag", "lev": AXES["lev"],
                                      "logk": AXES["logk"]})
            rows.append([f"{f.coef['level_z']:.2f}",
                         tb.stars_cell(f.coef["drift_z"] * 10,
                                       f.se["drift_z"] * 10, f.pval["drift_z"])
                         if hasattr(tb, "stars_cell") else
                         f"{f.coef['drift_z']*10:.2f}{tb.stars(f.pval['drift_z'])}"
                         f" ({f.se['drift_z']*10:.2f})",
                         f"{f.nobs:,}"])
            index.append(f"$\\theta_K={tk:.2f}$, $\\gamma_L={gl:.2f}$")
    frame = pd.DataFrame(rows, index=index,
                         columns=["Level", "Drift per decade", "$n$"])
    return tb.frame_table(
        frame, caption="Responsiveness to a productivity residual",
        label="tab:productivity", index_label="Output elasticities",
        notes=("The residual is $\\log(\\mathrm{sales}) - \\theta_K\\log k - "
               "\\gamma_L\\log n$, normalised within industry and year like the "
               "other characteristics, and entered in place of MRPK with the same "
               "controls and fixed effects as column (3) of "
               "Table~\\ref{tab:main}.  The elasticities are imposed, not "
               "estimated, so the grid rather than a single row is the result. "
               " The drift is positive and significant at every point of the "
               "grid and its size varies by a factor of two across the grid.  "
               "This is why the paper measures MRPK, which needs no such "
               "parameter, and reports this only as a check that the comparison "
               "with studies that measure productivity is not driven by the "
               "change of object."))


def table_uniformity(p: pd.DataFrame, dep: str = "dlogk_h1") -> str:
    """Does the drift of the MRPK gradient differ across groups of firms?

    The drift, not the level, is the object of interest, so the group is
    interacted with the characteristic *and* the trend, and the test is on the
    two triple interactions.  Interacting with period dummies instead answers a
    different question -- it also picks up departures from a straight line, so
    it can reject when the drift is common and only the path differs.

    Groups are fixed at the firm's average over FY1999--2003.  A time-varying
    split would let a firm change group in response to its own investment,
    which is the dependent variable.
    """
    d = p.sort_values([gr.FIRM, gr.YEAR]).copy()
    init = d[d[gr.YEAR].between(FIRST_YEAR, 2003)].groupby(gr.FIRM).agg(
        ta=("log_assets", "mean"), lk=("log_k", "mean"),
        lev=("leverage", "mean"), nd=("net_debt", "mean"),
        cash=("cash_ratio", "mean"))
    init["ki"] = d.groupby(gr.FIRM)["k_intensity_0"].first()
    axes = [("ta", "Size (total assets)"), ("lk", "Capital stock"),
            ("ki", "Capital intensity"), ("lev", "Leverage"),
            ("nd", "Net debt"), ("cash", "Cash holdings")]

    rows, index = [], []
    for col, label in axes:
        g = pd.qcut(init[col].dropna(), 3, labels=[0, 1, 2]).astype(float)
        s = d.copy()
        s["g"] = s[gr.FIRM].map(g)
        s = s.dropna(subset=["g"])
        m = AXES["mrpk"]
        s["lvl"], s["dft"] = s[m], s[m] * s["tc"]
        rhs, tri = ["lvl", "dft"], []
        for gg in (1, 2):
            gd = (s["g"] == gg).astype(float)
            s[f"G{gg}l"], s[f"G{gg}d"] = s[m] * gd, s[m] * gd * s["tc"]
            s[f"G{gg}"], s[f"G{gg}t"] = gd, gd * s["tc"]
            rhs += [f"G{gg}l", f"G{gg}d", f"G{gg}", f"G{gg}t"]
            tri.append(f"G{gg}d")
        rhs += [AXES["lev"], AXES["logk"], "log_assets_lag"]
        s = s.dropna(subset=[dep] + rhs + [gr.FIRM, gr.YEAR])
        x = gr._absorb(s, [dep] + rhs, [gr.FIRM, gr.YEAR])
        z, y = x[rhs].to_numpy(float), x[dep].to_numpy(float)
        bread = np.linalg.pinv(z.T @ z)
        beta = bread @ (z.T @ y)
        vcov = gr._multiway_vcov(z, y - z @ beta, bread, [s[gr.FIRM].to_numpy()])
        nm = {n: i for i, n in enumerate(rhs)}
        idx = [nm[t] for t in tri]
        b = beta[idx]
        wald = float(b @ np.linalg.pinv(vcov[np.ix_(idx, idx)]) @ b)
        cell = lambda k: (f"{beta[nm[k]] * 10:.2f} "
                          f"({np.sqrt(vcov[nm[k], nm[k]]) * 10:.2f})")
        rows.append([cell("dft"), cell("G1d"), cell("G2d"),
                     f"{1 - stats.chi2.cdf(wald, 2):.3f}", f"{len(s):,}"])
        index.append(label)

    frame = pd.DataFrame(rows, index=index,
                         columns=["Bottom third", "Middle $-$ bottom",
                                  "Top $-$ bottom", "$p$", "$n$"])
    return tb.frame_table(
        frame, caption="Is the drift of the MRPK gradient common across firms?",
        label="tab:uniformity", index_label="Split by",
        notes=("Drift per decade.  One regression per row: the characteristic "
               "and the trend are both interacted with two group dummies, and "
               "the first column is the drift in the omitted (bottom) third.  "
               "The last column but one is the $p$ value of a joint test that "
               "the two triple interactions are zero, that is, that the drift "
               "is common.  Groups are terciles of the firm's average over "
               "FY1999--2003, so a firm cannot change group in response to its "
               "own investment.  Levels, controls and fixed effects are as in "
               "column (3) of Table~\\ref{tab:main}."))


# --------------------------------------------------------------------------
def main(panel_path: str = PANEL) -> None:
    p = attach_shocks(prepare(panel_path))
    print(f"estimation sample: {len(p):,} firm-years, {p[gr.FIRM].nunique():,} firms")
    for builder, filename in [(table_descriptives, "table0_descriptives.tex"),
                              (table_main, "table1_main.tex"),
                              (table_robustness, "table2_robustness.tex"),
                              (table_units, "table3_units.tex"),
                              (table_decomposition, "table4_decomposition.tex"),
                              (table_counterfactual, "table9_counterfactual.tex"),
                              (table_periods, "table5_periods.tex"),
                              (table_margins, "table8_margins.tex"),
                              (table_inference, "table7_inference.tex"),
                              (table_persistence, "table6_persistence.tex"),
                              (table_policy, "table5_policy.tex"),
                              (table_uniformity, "table10_uniformity.tex"),
                              (table_productivity, "table11_productivity.tex")]:
        body = builder(p)
        if body is None:
            continue
        path = tb.write(body, filename)
        print(f"  wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=PANEL,
                    help="path to the constructed panel; pass the synthetic "
                         "panel to test the code without a NEEDS licence")
    main(ap.parse_args().panel)
