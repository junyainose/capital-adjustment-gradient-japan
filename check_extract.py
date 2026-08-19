"""Two diagnostics on the raw extracts, run once after build_data.py.

    python check_extract.py

Neither affects the panel.  The first asks whether the industry exclusion in
build_data.py does anything, since the extract may already exclude those
industries at the point of download.  The second asks whether the duplicate
(stkno, fy) rows in the later extract are harmless repetitions or conflicting
values, which matters because the loader keeps the last one it sees.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src import io_needs as io          # noqa: E402
from src import labor_cost as lc        # noqa: E402
import build_data as bd                 # noqa: E402


def industries(panel_path: str = "data/processed/panel_annual.parquet") -> None:
    print("=== industries present in the panel ===")
    p = pd.read_parquet(panel_path)
    counts = (p.groupby("nkilm")["stkno"].nunique().sort_index()
              .rename("firms").to_frame())
    counts["name"] = [lc.NKILM_NAME_EN.get(i, "?") for i in counts.index]
    print(counts.to_string())
    configured = set(bd.EXCLUDED_INDUSTRIES)
    present = set(counts.index)
    print(f"\n  codes configured for exclusion: {sorted(configured)}")
    print(f"  of those, actually present:     {sorted(configured & present)}")
    if not configured & present:
        print("  -> the exclusion step is a no-op; those industries were "
              "already excluded when the data were downloaded.")


FIRST_YEAR, LAST_YEAR = 1999, 2024      # the estimation window; keep in step
                                        # with build_tables.py


def duplicates(extra_dir: str = "data/raw", kind: str = "financial",
               first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> None:
    """Check whether duplicated (stkno, fy) rows carry conflicting values.

    Run this on the financial files, not the corporate ones.  Everything in the
    corporate extract is a firm-level constant repeated across years -- founding
    dates, industry codes -- so duplicated rows there agree almost by
    construction and the check is uninformative.  The financial extract is where
    a conflict would matter, since the loader keeps whichever row it sees last.
    """
    print(f"\n=== duplicate (stkno, fy) rows in the {kind} extract ===")
    folder = Path(extra_dir)
    frames = []
    for f in sorted(folder.glob("*.xlsx")):
        if io._extra_half(f) == kind:
            d = io.read_needs_extra(f)
            d["_file"] = f.name
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    dup_keys = d[d.duplicated(["stkno", "fy"], keep=False)]
    print(f"  rows involved: {len(dup_keys):,}, "
          f"firms: {dup_keys.stkno.nunique():,}, keys: "
          f"{len(dup_keys.drop_duplicates(['stkno', 'fy'])):,}")
    if dup_keys.empty:
        return

    print("  fiscal years involved:")
    span = dup_keys.drop_duplicates(["stkno", "fy"])["fy"]
    print(f"    {int(span.min())}-{int(span.max())}; "
          f"inside FY{first_year}-{last_year}: "
          f"{int(span.between(first_year, last_year).sum()):,} keys")

    value_cols = [c for c in d.columns
                  if c not in ("stkno", "fy", "firm_name", "_file")]

    # Two very different situations hide behind "the rows differ".  One row may
    # carry a value where the other is missing, in which case the rows complete
    # each other and keeping either one alone throws data away.  Or both may
    # carry a value and disagree, in which case a choice has to be made.  Only
    # the second is a problem; counting distinct values with NaN included would
    # merge the two cases and overstate it.
    g = dup_keys.groupby(["stkno", "fy"])[value_cols]
    distinct = g.nunique(dropna=True)       # non-missing values only
    present = g.count()
    total = g.size()

    ambiguous = (distinct > 1)
    complementary = (distinct == 1) & present.lt(total, axis=0)
    bad = ambiguous.any(axis=1)
    # Only a disagreement inside the estimation window can move a result, so
    # that count is reported next to the total rather than left to be inferred.
    years = bad[bad].index.get_level_values("fy").to_series()
    in_window = years.between(first_year, last_year)
    print(f"  keys with a genuine disagreement: {int(bad.sum()):,} "
          f"({int(in_window.sum()):,} inside FY{first_year}-{last_year})")
    print(f"  keys where the rows complete each other: "
          f"{int((complementary.any(axis=1) & ~bad).sum()):,}")

    if ambiguous.any().any():
        keep = (ambiguous.index.get_level_values("fy")
                .to_series().between(first_year, last_year).values)
        allc, inc = ambiguous.sum(), ambiguous.loc[keep].sum()
        print(f"\n  columns that genuinely disagree "
              f"(all years / inside FY{first_year}-{last_year}):")
        for c in allc[allc > 0].sort_values(ascending=False).index:
            print(f"    {c:14s} {int(allc[c]):5d} {int(inc.get(c, 0)):6d}")
        key = ambiguous[ambiguous.any(axis=1)].index[0]
        cols = [c for c in value_cols if ambiguous.loc[key, c]][:5]
        print(f"\n  example, firm {key[0]} in FY{key[1]}:")
        print(dup_keys.set_index(["stkno", "fy"]).loc[[key]]
              [["_file"] + cols].to_string())
    else:
        print("  -> no genuine disagreement; the rows only complete each other.")


if __name__ == "__main__":
    industries()
    duplicates(kind="financial")
    duplicates(kind="corporate")
