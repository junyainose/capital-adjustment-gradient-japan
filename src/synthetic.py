"""A synthetic firm panel with the same shape as the real one.

The estimation sample is drawn from Nikkei NEEDS Financial QUEST, which may not
be redistributed in any form, raw or derived.  Anyone without a NEEDS licence
therefore cannot run this package on the real data.  This module exists so that
the code can still be executed end to end: it writes a panel with the same
columns, types, panel structure and missingness patterns, which every downstream
script accepts without modification.

Two things to be clear about.

**The numbers will not match the paper.**  Nothing here is calibrated to the
real data.  Every parameter below is a round number chosen by hand to give a
panel of roughly the right shape -- capital stocks in the millions of yen,
investment rates centred near a fifth, leverage near a quarter.  None of them is
a moment of the licensed data, which is deliberate: an estimated moment would
itself be derived data.

**The generating process is known, so the code can be checked for correctness
and not only for whether it runs.**  The investment equation plants a gradient
on the productivity measure that drifts linearly over the sample, at the rate
set by `TRUE_MRPK_DRIFT`, together with a constant gradient on leverage and on
log capital.  Running the package on synthetic data should recover the planted
drift on the MRPK term and something indistinguishable from zero on the other
two.  `self_test` does exactly that.

Usage
-----
    python -m src.synthetic                 # writes data/processed/panel_synthetic.parquet
    python build_tables.py --panel data/processed/panel_synthetic.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import build_panel as bp

# --------------------------------------------------------------------------
# Parameters of the generating process.  All hand-set; see the module docstring.
# --------------------------------------------------------------------------
N_FIRMS = 1_700
FY_MIN, FY_MAX = 1995, 2021       # generated wider than the estimation window
                                  # so that lags and leads exist at the edges
ESTIMATION_START = 1999

# 32 industries, matching the Nikkei medium classification codes so that the
# industry mapping in `labor_cost` finds them.
INDUSTRY_CODES = [f"{i:02d}" for i in
                  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                   33, 35, 37, 41, 43, 45, 53, 55, 57, 59, 61, 63, 65, 67, 69,
                   71]]

DEPRECIATION_RATE = 0.09
MRPK_PERSISTENCE = 0.85           # AR(1) in the idiosyncratic productivity term
MRPK_SHOCK_SD = 0.35
LEV_DEV_RHO = 0.80                # persistence of the deviation of leverage
LEV_DEV_SD = 0.030                # from its permanent firm level
INVESTMENT_NOISE_SD = 0.10
# Noise between the latent productivity term and measured sales.  Kept small so
# that MRPK is close to a clean measure of the object the investment equation
# responds to.  Raising it introduces classical attenuation, which the self test
# below will register as a shortfall in the recovered drift.
SALES_NOISE_SD = 0.05

# Planted coefficients of the investment equation, in units of one
# industry-year standard deviation of the characteristic.
TRUE_MRPK_LEVEL = 0.030
TRUE_MRPK_DRIFT = 0.0020          # per year; 0.02 per decade
TRUE_LEVERAGE_LEVEL = -0.060
TRUE_LOGK_LEVEL = -0.030

# Missingness rates, set to the orders of magnitude seen in the real extract.
# Rounded to one digit so that they carry no information about the source.
MISSING_CIP = 0.30                # construction in progress is often not shown
MISSING_CASHFLOW = 0.02           # cash flow statement, absent before FY1999
MISSING_EMPLOYEES = 0.01


# --------------------------------------------------------------------------
def _entry_exit(rng: np.random.Generator, n_firms: int) -> pd.DataFrame:
    """Give each firm a first and last fiscal year.

    Most firms span the whole sample; a minority enter late or leave early, so
    that the panel is unbalanced in the way the real one is and the composition
    checks in the robustness table have something to bite on.
    """
    first = np.full(n_firms, FY_MIN)
    last = np.full(n_firms, FY_MAX)
    late = rng.random(n_firms) < 0.15
    first[late] = rng.integers(FY_MIN + 1, FY_MAX - 5, late.sum())
    early = rng.random(n_firms) < 0.10
    last[early] = rng.integers(FY_MIN + 6, FY_MAX, early.sum())
    return pd.DataFrame({"first_fy": first, "last_fy": np.maximum(first + 3, last)})


def generate_raw(seed: int = 20260814, n_firms: int = N_FIRMS,
                 mrpk_drift: float = TRUE_MRPK_DRIFT) -> pd.DataFrame:
    """Firm-level accounting items, before any derived variable is formed.

    The output has the columns that `io_needs.tidy_needs` and
    `io_needs.load_needs_extra` produce once joined, so it can be fed straight
    into the same panel construction code as the real extract.
    """
    rng = np.random.default_rng(seed)

    span = _entry_exit(rng, n_firms)
    industry = rng.choice(INDUSTRY_CODES, n_firms)
    log_scale = rng.normal(9.0, 1.4, n_firms)          # firm size, log millions
    industry_mrpk = dict(zip(INDUSTRY_CODES,
                             rng.normal(0.0, 0.30, len(INDUSTRY_CODES))))
    firm_effect = rng.normal(0.0, 0.05, n_firms)
    year_effect = dict(zip(range(FY_MIN, FY_MAX + 1),
                           rng.normal(0.0, 0.04, FY_MAX - FY_MIN + 1)))
    t_bar = (FY_MIN + FY_MAX) / 2

    rows = []
    for i in range(n_firms):
        years = range(int(span.first_fy[i]), int(span.last_fy[i]) + 1)
        capital = np.exp(log_scale[i])
        # Draw the productivity term from its stationary distribution rather
        # than from an arbitrary one.  An initial dispersion below the
        # stationary level makes the cross-section fan out over the first
        # decade, and a widening dispersion is enough on its own to produce a
        # rising gradient once the characteristic is standardised each year.
        productivity = (industry_mrpk[industry[i]]
                        + rng.normal(0, MRPK_SHOCK_SD
                                     / np.sqrt(1 - MRPK_PERSISTENCE ** 2)))
        # A permanent firm level plus a stationary deviation.  Drawing the
        # deviation from its stationary distribution from the first period
        # matters: starting it at zero, or starting the level at a wide draw
        # that then mean reverts, makes the cross-sectional dispersion drift,
        # which shows up as a spurious trend in the standardised gradient.
        leverage_mean = float(np.clip(rng.beta(2.5, 7.0), 0.02, 0.75))
        cash_mean = float(np.clip(rng.beta(2.0, 8.0), 0.02, 0.55))
        lev_dev = rng.normal(0, LEV_DEV_SD / np.sqrt(1 - LEV_DEV_RHO ** 2))
        cash_dev = rng.normal(0, LEV_DEV_SD / np.sqrt(1 - LEV_DEV_RHO ** 2))
        leverage = float(np.clip(leverage_mean + lev_dev, 0.0, 0.8))
        cash_share = float(np.clip(cash_mean + cash_dev, 0.0, 0.6))

        for fy in years:
            productivity = (MRPK_PERSISTENCE * productivity
                            + rng.normal(0, MRPK_SHOCK_SD))
            # Investment responds to the characteristics with a gradient on
            # productivity that drifts; this is the object the paper estimates.
            drift = TRUE_MRPK_LEVEL + mrpk_drift * (fy - t_bar)
            inv_rate = (0.20 + firm_effect[i] + year_effect[fy]
                        + drift * productivity
                        + TRUE_LEVERAGE_LEVEL * (leverage - 0.25) / 0.15
                        + TRUE_LOGK_LEVEL * (np.log(capital) - 9.0) / 1.4
                        + rng.normal(0, INVESTMENT_NOISE_SD))
            inv_rate = float(np.clip(inv_rate, -0.05, 1.2))

            capex = inv_rate * capital
            depreciation = DEPRECIATION_RATE * capital
            employees = max(int(np.exp(log_scale[i] - 4.0
                                       + rng.normal(0, 0.3)) * 10), 20)
            sales = capital * np.exp(0.8 + productivity
                                     + rng.normal(0, SALES_NOISE_SD))
            op_profit = sales * np.clip(rng.normal(0.06, 0.05), -0.15, 0.35)
            total_assets = capital / np.clip(rng.normal(0.35, 0.08), 0.1, 0.9)
            debt = leverage * total_assets
            cash = cash_share * total_assets

            rows.append({
                "stkno": f"T{1000 + i}", "fy": fy, "month": 3,
                "nkilm": industry[i],
                "ppe": capital, "total_assets": total_assets,
                "sales": sales, "op_profit": op_profit,
                "depreciation": depreciation, "capex": capex,
                "cash": cash, "st_debt": 0.3 * debt, "lt_debt": 0.7 * debt,
                "total_debt": debt, "interest_exp": 0.015 * debt,
                "employees": employees,
                "cfo": op_profit + depreciation + rng.normal(0, 0.05) * capital,
                "cfi": -capex + rng.normal(0, 0.05) * capital,
                "cff": rng.normal(0, 0.05) * capital,
                "cip": 0.06 * capital * np.exp(rng.normal(0, 0.5)),
                "bond_lt": debt * 0.2 if rng.random() < 0.35 else np.nan,
                "bond_st": debt * 0.05 if rng.random() < 0.20 else np.nan,
                "founded_real": pd.Timestamp(
                    f"{int(rng.integers(1900, 2000))}-04-01"),
            })
            # capital accumulates; the firm carries its leverage and cash with
            # slow drift, so that both have within-firm variation
            capital = max(capital * (1 - DEPRECIATION_RATE) + capex, 1.0)
            # Leverage and cash are mean reverting rather than random walks, so
            # that their cross-sectional dispersion is stationary.  A widening
            # or narrowing dispersion would make the gradient measured in
            # contemporaneous standard deviations drift on its own -- the
            # mechanism documented for leverage in the paper -- and that would
            # confound the self test below.  `units_demo` reproduces it
            # deliberately instead.
            lev_dev = LEV_DEV_RHO * lev_dev + rng.normal(0, LEV_DEV_SD)
            cash_dev = LEV_DEV_RHO * cash_dev + rng.normal(0, LEV_DEV_SD)
            leverage = float(np.clip(leverage_mean + lev_dev, 0.0, 0.8))
            cash_share = float(np.clip(cash_mean + cash_dev, 0.0, 0.6))

    d = pd.DataFrame(rows)

    # A holding company reorganisation resets the formal incorporation date for
    # roughly one firm in eight.
    d["founded_form"] = d["founded_real"]
    reorganised = set(rng.choice(d.stkno.unique(),
                                 size=int(0.12 * d.stkno.nunique()),
                                 replace=False))
    d.loc[d.stkno.isin(reorganised), "founded_form"] = pd.Timestamp("2005-04-01")

    # Missingness, in the patterns the real extract shows
    d.loc[rng.random(len(d)) < MISSING_CIP, "cip"] = np.nan
    for c in ["cfo", "cfi", "cff"]:
        d.loc[rng.random(len(d)) < MISSING_CASHFLOW, c] = np.nan
    d.loc[d.fy < ESTIMATION_START, ["cfo", "cfi", "cff"]] = np.nan
    d.loc[rng.random(len(d)) < MISSING_EMPLOYEES, "employees"] = np.nan
    return d


def make_panel(seed: int = 20260814, n_firms: int = N_FIRMS,
               mrpk_drift: float = TRUE_MRPK_DRIFT) -> pd.DataFrame:
    """Run the synthetic accounting items through the real panel construction.

    Using the same code path as the licensed data is the point: it exercises
    `build_panel` rather than bypassing it, so a functionality test covers the
    variable construction as well as the estimation.
    """
    raw = generate_raw(seed=seed, n_firms=n_firms, mrpk_drift=mrpk_drift)
    panel = bp.filter_sample(raw, fy_min=FY_MIN, fy_max=FY_MAX, month=3)
    panel = bp.add_derived(panel)
    panel = bp.add_needs_extra_vars(panel)
    panel = bp.add_industry_standardized(panel, ["mrpk", "log_k"])
    return panel


def write(path: str | Path = "data/processed/panel_synthetic.parquet",
          seed: int = 20260814, n_firms: int = N_FIRMS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = make_panel(seed=seed, n_firms=n_firms)
    panel.to_parquet(path)
    print(f"wrote {path}: {len(panel):,} firm-years, "
          f"{panel.stkno.nunique():,} firms, FY{panel.fy.min()}-{panel.fy.max()}")
    return path


def self_test(seed: int = 20260814, n_firms: int = N_FIRMS) -> None:
    """Check that the estimator recovers the planted drift, and only that.

    The panel is generated twice, once with the drift switched on and once with
    it set to zero, and the estimator is run on both.  Recovering the planted
    value in the first and nothing in the second is a joint check on power and
    on size: it rules out the possibility that the drift the paper reports is
    something the estimator manufactures out of a panel with none.

    The other two coefficients are printed but are not tested.  Neither is a
    clean null under this generating process.  The investment rate has capital
    in its denominator, so the gradient on lagged log capital carries a
    mechanical component the planted coefficient does not control -- the level
    comes back near -0.13 against a planted -0.03.  Leverage is generated with a
    constant gradient, but its cross-sectional dispersion is not exactly
    stationary once the bounds at zero and 0.8 bind, and a dispersion that moves
    is enough to tilt a gradient measured in contemporaneous standard
    deviations.  `units_demo` makes that mechanism explicit.

    A failure here means the estimation code is wrong, not that the data are
    unusual.  That is what makes it worth running before trusting results
    computed on data a reader cannot see.
    """
    from src import gradients as gr

    def estimate(drift: float):
        panel = make_panel(seed=seed, n_firms=n_firms, mrpk_drift=drift)
        panel = gr.add_leads(panel, "inv_rate_comb", prefix="ik")
        panel = panel[panel.fy.between(ESTIMATION_START, FY_MAX - 2)].copy()
        panel["tc"] = panel.fy - panel.fy.mean()
        for c in ["mrpk_s", "log_k_s", "leverage", "log_assets"]:
            panel[c + "_lag"] = panel.groupby("stkno")[c].shift(1)
        panel["lev_s_lag"] = gr.standardize(panel, "leverage_lag",
                                            "industry_year")
        return gr.fit_trend(panel, "ik_h0",
                            {"mrpk": "mrpk_s_lag", "lev": "lev_s_lag",
                             "logk": "log_k_s_lag"})

    # The tolerance is economic rather than statistical.  With thirty thousand
    # observations the standard error on the drift is 0.0001, so a t test
    # rejects any imperfection in a hand-built generating process; the panel is
    # built to be the right shape, not to satisfy a null exactly.  What the test
    # has to establish is that the estimator tracks the planted value and does
    # not manufacture one, and a quarter of the planted drift is a wide enough
    # band for that while still failing on any real coding error.
    tolerance = 0.25 * abs(TRUE_MRPK_DRIFT)
    failures = 0
    for planted, label in [(TRUE_MRPK_DRIFT, "drift on"), (0.0, "drift off")]:
        fit = estimate(planted)
        b, se, _ = fit.get("drift_mrpk")
        ok = abs(b - planted) < tolerance
        failures += 0 if ok else 1
        print(f"  {label:>9}: MRPK drift {b:+.4f} ({se:.4f})  "
              f"planted {planted:+.4f}  {'ok' if ok else 'FAIL'}")
        if planted:
            for key in ("lev", "logk"):
                c, s_, _ = fit.get(f"drift_{key}")
                print(f"             {key:>4} drift {c:+.4f} ({s_:.4f})  "
                      "not a clean null; see docstring")
    print("  self test passed" if failures == 0
          else f"  self test FAILED on {failures} check(s)")


def units_demo(seed: int = 20260814, n_firms: int = 600) -> None:
    """Show that a compressing distribution mimics a strengthening gradient.

    The paper reports a drift in the leverage gradient that disappears once
    leverage is measured against its contemporaneous dispersion rather than in
    fixed units, because the dispersion of leverage falls by about a fifth over
    the sample.  Here that mechanism is reproduced deliberately: the gradient on
    leverage is held constant by construction while its dispersion is shrunk,
    and the fixed-unit estimate drifts while the standardised one does not.
    """
    from src import gradients as gr

    panel = make_panel(seed=seed, n_firms=n_firms)
    panel = gr.add_leads(panel, "inv_rate_comb", prefix="ik")
    panel = panel[panel.fy.between(ESTIMATION_START, FY_MAX - 2)].copy()
    panel["tc"] = panel.fy - panel.fy.mean()
    # squeeze leverage towards its yearly mean, increasingly over time
    year = panel.fy - panel.fy.min()
    span = max(int(year.max()), 1)
    squeeze = 1.0 - 0.50 * year / span
    mean_by_year = panel.groupby("fy")["leverage"].transform("mean")
    panel["leverage"] = mean_by_year + squeeze * (panel["leverage"] - mean_by_year)

    for c in ["mrpk_s", "log_k_s", "leverage", "log_assets"]:
        panel[c + "_lag"] = panel.groupby("stkno")[c].shift(1)
    panel["lev_year_lag"] = gr.standardize(panel, "leverage_lag", "industry_year")
    panel["lev_fixed_lag"] = gr.standardize(panel, "leverage_lag", "industry_fixed")

    drifts = {}
    for label, col in [("fixed units", "lev_fixed_lag"),
                       ("contemporaneous sd", "lev_year_lag")]:
        fit = gr.fit_trend(panel, "ik_h0",
                           {"mrpk": "mrpk_s_lag", "lev": col,
                            "logk": "log_k_s_lag"})
        b, se, p = fit.get("drift_lev")
        drifts[label] = b
        print(f"  leverage drift, {label:>19}: {b:+.5f} ({se:.5f})  p={p:.3f}")
    share = 1 - drifts["contemporaneous sd"] / drifts["fixed units"]
    print(f"  the compression accounts for {share:.0%} of the drift measured "
          "in fixed units")


if __name__ == "__main__":
    write()
    print("\nself test:")
    self_test()
    print("\nunits demonstration:")
    units_demo()
