"""Build the analysis panel from the raw NEEDS extracts.

    python build_data.py --raw data/raw

Requires the licensed extracts described in README section 2.  The script prints
a sample selection table as it goes and writes it to
output/tables/sample_selection.tex.

Those counts matter as much as the panel does.  A reader with a NEEDS licence
cannot check their extract against the numbers in the paper, because those
numbers depend on the sample they are trying to reproduce.  They check it
against this table, row by row.  Every step that drops an observation is
therefore recorded here, including the ones that look innocuous.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src import build_panel as bp, io_needs as io, shocks as sh, tables as tb  # noqa: E402

RAW_DIR = "data/raw"
EXTRA_DIR = None          # None means: look in the same directory as --raw
SHOCK_FILE = "data/raw/KSdata_VAR_IV.csv"
OUT_PATH = "data/processed/panel_annual.parquet"

# The panel is built wider than the estimation window so that lags at the start
# and leads at the end of the window exist.  Restricting first would silently
# drop the boundary years.
PANEL_FY_MIN, PANEL_FY_MAX = 1995, 2025
ESTIMATION_FY_MIN, ESTIMATION_FY_MAX = 1999, 2019
CLOSING_MONTH = 3

# The extract was assembled in batches; files below this index were trial runs
# with a different item set and are skipped.
MIN_FILE_INDEX = 6

# Industries excluded at the point of extraction rather than here.  Finance is
# excluded because the capital stock of a bank or an insurer is not the object
# measured in this paper; the residual services category is excluded because
# tangible fixed assets there are small enough that MRPK would be dominated by
# measurement error.  Both were left out of the NEEDS download, so the code
# below verifies their absence rather than performing the exclusion: reporting a
# filter that removes nothing would misdescribe how the sample was formed.
EXCLUDED_INDUSTRIES = ["73", "75", "77", "79", "81"]

CORE_VARIABLES = ["ppe", "total_assets", "sales", "op_profit", "depreciation"]

LAGGED = ["leverage", "net_debt", "roa", "cash_ratio", "log_assets",
          "nodiv", "icr", "mrpk", "log_age", "cf_ta"]


class SampleFlow:
    """Record how many firms and observations survive each selection step."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, label: str, df: pd.DataFrame, note: str = "") -> pd.DataFrame:
        firms = df["stkno"].nunique() if "stkno" in df.columns else 0
        row = {"step": label, "firms": firms, "obs": len(df), "note": note}
        prev = self.rows[-1] if self.rows else None
        row["d_firms"] = firms - prev["firms"] if prev else 0
        row["d_obs"] = len(df) - prev["obs"] if prev else 0
        self.rows.append(row)
        print(f"  {label:<44s} {firms:>6,} firms {len(df):>8,} obs"
              f"   ({row['d_firms']:+,} / {row['d_obs']:+,})")
        return df

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)[["step", "firms", "obs",
                                        "d_firms", "d_obs", "note"]]

    def to_latex(self, caption: str = "Sample selection",
                 label: str = "tab:sample") -> str:
        d = self.frame().set_index("step")[["firms", "obs"]]
        d.columns = ["Firms", "Observations"]
        return tb.frame_table(
            d, digits=0, caption=caption, label=label, index_label="Step",
            notes=("Each row reports the sample remaining after the step named. "
                   "The extract covers every fiscal year NEEDS holds, which "
                   "reaches back to the 1970s, so the step that imposes the "
                   "panel span drops the early years rather than any category "
                   "of firm.  The underlying data may not be redistributed, so "
                   "a replicator working from their own extract verifies it by "
                   "matching these counts rather than by comparing estimates."))


def build(raw_dir: str = RAW_DIR, extra_dir: str | None = EXTRA_DIR,
          shock_file: str = SHOCK_FILE) -> tuple[pd.DataFrame, SampleFlow]:
    flow = SampleFlow()

    print("[1/5] reading the financial statement extract")
    raw = io.load_all(raw_dir, min_index=MIN_FILE_INDEX)
    tidy = io.tidy_needs(raw)
    flow.log("Raw extract, all closing months and years", tidy)

    tidy = flow.log(f"Books closing in month {CLOSING_MONTH}",
                    tidy[tidy["month"] == CLOSING_MONTH],
                    "so that the accounting period maps unambiguously to a "
                    "fiscal year")

    print("\n[2/5] reading the later extract and the industry classification")
    extra = io.load_needs_extra(extra_dir if extra_dir else raw_dir)
    extra = io.industry_from_nkil_code(extra)
    drop = [c for c in ["firm_name"] if c in extra.columns]
    merged = tidy.merge(extra.drop(columns=drop), on=["stkno", "fy"], how="left")

    unmatched = int(merged["nkilm"].isna().sum())
    if unmatched:
        print(f"  warning: {unmatched:,} observations carry no industry code. "
              "The later extract is probably incomplete; every firm in the "
              "financial extract should appear in it.")
    merged = flow.log("Merged with the later extract",
                      merged.dropna(subset=["nkilm"]),
                      "cash flow, bonds, construction in progress, "
                      "incorporation dates, industry code")

    still_present = sorted(set(merged["nkilm"]) & set(EXCLUDED_INDUSTRIES))
    if still_present:
        merged = flow.log("Excluding finance and residual services",
                          merged[~merged["nkilm"].isin(EXCLUDED_INDUSTRIES)],
                          "capital not comparable in finance; tangible assets "
                          "too small in residual services to measure MRPK")
    else:
        print("  finance and residual services already absent; they were "
              "excluded at the point of extraction, not here")

    print("\n[3/5] constructing variables")
    panel = bp.filter_sample(merged, fy_min=PANEL_FY_MIN, fy_max=PANEL_FY_MAX,
                             month=CLOSING_MONTH)
    panel = bp.add_derived(panel)
    panel = bp.add_needs_extra_vars(panel)
    panel = bp.add_industry_standardized(panel, ["mrpk", "log_z", "log_k"])
    panel = bp.add_industry_demeaned(panel, ["mrpk", "log_z", "log_k"])
    panel["sector_year"] = (panel["nkilm"].astype(str) + "_"
                            + panel["fy"].astype(str))

    panel = panel.sort_values(["stkno", "fy"])
    for c in LAGGED:
        if c in panel.columns:
            panel[c + "_lag"] = panel.groupby("stkno")[c].shift(1)

    flow.log(f"Panel span FY{PANEL_FY_MIN}-{PANEL_FY_MAX}", panel,
             "the extract reaches back to the 1970s; this step drops the years "
             "before 1995, and lags are formed here, before the estimation "
             "window is imposed")

    estimation = panel[panel["fy"].between(ESTIMATION_FY_MIN,
                                           ESTIMATION_FY_MAX)]
    flow.log(f"Estimation window FY{ESTIMATION_FY_MIN}-{ESTIMATION_FY_MAX}",
             estimation, "cash flow statement available from FY1999")
    estimation = flow.log("Core accounting variables non-missing",
                          estimation.dropna(subset=CORE_VARIABLES),
                          "capital, assets, sales, operating profit, "
                          "depreciation")
    flow.log("Lagged capital available and positive",
             estimation[estimation["ppe_lag"] > 0],
             "required to form the investment rate and MRPK")

    print("\n[4/5] merging the monetary policy shock")
    if Path(shock_file).exists():
        ks = sh.load_ks(shock_file)
        panel = sh.merge_shock(panel, sh.annualize(ks, col="TARGET"))
        share = panel.loc[panel.fy.between(ESTIMATION_FY_MIN,
                                           ESTIMATION_FY_MAX), "shock"].notna()
        print(f"  merged for {share.mean():.1%} of the estimation window")
        variance = sh.identifying_variance(ks, years=(ESTIMATION_FY_MIN,
                                                      ESTIMATION_FY_MAX))
        top = ", ".join(f"FY{int(y)} {v:.0%}" for y, v in variance.head(3).items())
        print(f"  identifying variance concentrated in {top} "
              f"(top four {variance.head(4).sum():.0%})")
    else:
        print(f"  {shock_file} not found. The shock column is omitted; the main "
              "results do not use it, Section 6 does.")

    print("\n[5/5] coverage")
    print(bp.coverage_report(panel, fy_min=ESTIMATION_FY_MIN - 1,
                             fy_max=ESTIMATION_FY_MAX).to_string())
    balanced = bp.balanced_firms(panel, ESTIMATION_FY_MIN, ESTIMATION_FY_MAX)
    print(f"\n  firms observed throughout the window: {len(balanced):,}")
    if len(balanced) < 500:
        print("  warning: the balanced panel is thin; check the extract")

    return panel.sort_values(["stkno", "fy"]).reset_index(drop=True), flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_DIR)
    ap.add_argument("--extra", default=EXTRA_DIR)
    ap.add_argument("--shocks", default=SHOCK_FILE)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    panel, flow = build(args.raw, args.extra, args.shocks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"\nwrote {out}: {len(panel):,} firm-years, "
          f"{panel['stkno'].nunique():,} firms, {panel.shape[1]} columns")

    Path("output/tables").mkdir(parents=True, exist_ok=True)
    tb.write(flow.to_latex(), "sample_selection.tex")
    flow.frame().to_csv("output/tables/sample_selection.csv", index=False)
    print("wrote output/tables/sample_selection.tex and .csv")


if __name__ == "__main__":
    main()
