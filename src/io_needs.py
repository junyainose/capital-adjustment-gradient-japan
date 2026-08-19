"""Reading raw extracts from Nikkei NEEDS Financial QUEST.

The underlying data are licensed and cannot be redistributed, so this module is
the operative documentation of how the panel is built: which items were pulled,
under which codes, and what has to be repaired before the files can be read at
all.  A reader with a NEEDS licence should be able to reproduce the extract from
the tables below.

Two vintages of extract are handled.

`read_needs_xlsx` / `load_all` read the original financial statement extract,
whose files are named FqReport*.xlsx.  Items there are identified by their
Japanese labels on row 4.

`read_needs_extra` / `load_needs_extra` read the later extract of cash flow
items, bonds, construction in progress and incorporation dates, whose files are
named corporate*.xlsx and financial*.xlsx.  Items there are identified by their
NEEDS item codes on row 5, which is the more robust of the two conventions and
is what a new extract should use.

A quirk worth knowing before anything else: the workbooks NEEDS writes are not
valid xlsx.  The entry names inside the zip container are lower-cased, so the
archive holds "[content_types].xml" and "xl/sharedstrings.xml" where the
standard requires "[Content_Types].xml" and "xl/sharedStrings.xml".  Excel opens
them; openpyxl raises KeyError.  `_repair_workbook` rewrites the archive with
the canonical names before any read.

File layout of Sheet1 in both vintages:

    rows 0-3   header block (company name, ticker and similar labels)
    row  4     item names, in Japanese
    row  5     item codes, e.g. B01001
    row  6     units
    row  7+    data
"""
from __future__ import annotations

import os
import re
import tempfile
import zipfile
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Item dictionary for the original financial statement extract.
# Keys are the Japanese item labels as they appear on row 4 of the workbook.
# Add an entry here whenever a new item is pulled.
# --------------------------------------------------------------------------
COLUMN_MAP: dict[str, str] = {
    # --- balance sheet ---
    "流動資産": "current_assets",                          # current assets
    "固定資産／非流動資産": "fixed_assets",                  # fixed assets
    "有形固定資産": "ppe",                                  # tangible fixed assets
    "資産合計": "total_assets",                             # total assets
    "現金・預金／現金及び現金同等物": "cash",                 # cash and equivalents
    "短期借入金・社債合計": "st_debt",                       # short-term debt
    "長期借入金・社債・転換社債": "lt_debt",                  # long-term debt
    "流動負債": "current_liab",                             # current liabilities
    "利益剰余金": "retained_earnings",                      # retained earnings
    "有利子負債額": "total_debt",                           # interest-bearing debt
    # --- income statement (cumulative within the fiscal year) ---
    "売上高・営業収益［累計］": "sales",                      # sales
    "営業利益［累計］": "op_profit",                         # operating profit
    "支払利息・割引料［累計］": "interest_exp",               # interest expense
    "当期純利益（連結）［累計］": "net_income",               # consolidated net income
    # --- cash flow statement; present from FY1999 only ---
    "減価償却費": "depreciation",                           # depreciation
    "固定資産の取得による支出（▲）": "capex_total",          # purchases of fixed assets
    "固定資産の取得による支出（うち有形固定資産）（▲）": "capex_ppe",
    # --- payout ---
    "配当性向": "payout_ratio",
    "配当性向［累計］": "payout_cum",
    "１株当たり配当金（累計）": "dps",
    "自己株式数": "treasury_shares",
    # --- other ---
    "期末従業員数": "employees",                            # employees, year end
    "連結・単独フラグ": "cons_flag",                         # consolidated flag
    "連結基準フラグ": "std_flag",                            # accounting standard flag
}

NUMERIC_COLS = [
    "current_assets", "fixed_assets", "ppe", "total_assets", "cash",
    "st_debt", "lt_debt", "current_liab", "retained_earnings", "total_debt",
    "sales", "op_profit", "interest_exp", "net_income",
    "depreciation", "capex_total", "capex_ppe",
    "payout_ratio", "payout_cum", "dps", "treasury_shares", "employees",
]

# Identifier columns carry no label on row 4 and must be taken by position.
# Column 5 is the accounting period, "YYYY/MM".  Column 6 is reserved for the
# Nikkei industry aggregate code but returns nothing for individual firms; it
# repeats the accounting period instead, which is why the industry classification
# has to come from somewhere else (see `industry_from_nkil_code`).
POSITIONAL = {0: "basis", 1: "basesub", 2: "firm_name", 3: "stkno",
              4: "frequency", 5: "acc_period", 6: "_col6"}

# Item codes for the later extract, mapped to analysis variable names.
NEEDS_EXTRA_COLS = {
    # corporate attributes
    "CORPORATE'PRMTD1": "founded_real",   # substantive incorporation date; not
                                          # reset when a holding company is formed
    "CORPORATE'PRMTD2": "founded_form",   # formal incorporation date
    "CORPORATE'NKIL":   "nkil_code",      # Nikkei industry code, six digits
    "CORPORATE'JSIC1":  "jsic1",          # Japan Standard Industrial Classification
    # financial statement items
    "FINFSTA'F01065": "cfo",              # cash flow from operations
    "FINFSTA'F01087": "cfi",              # cash flow from investing
    "FINFSTA'F01106": "cff",              # cash flow from financing
    "FINFSTA'F01096": "bond_issue",       # proceeds from bond issuance
    "FINFSTA'F01097": "bond_redeem",      # payments on bond redemption
    "FINFSTA'C01032": "bond_st",          # bonds due within one year
    "FINFSTA'C01059": "bond_lt",          # bonds and convertible bonds
    "FINFSTA'B01074": "cip",              # construction in progress
}

_ZIP_RENAME = {
    "[content_types].xml": "[Content_Types].xml",
    "xl/sharedstrings.xml": "xl/sharedStrings.xml",
}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _repair_workbook(path: str | Path) -> str:
    """Rewrite a NEEDS workbook with canonical zip entry names.

    Returns the path of a temporary file that the caller must delete.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    with zipfile.ZipFile(path) as zin, \
         zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            zout.writestr(_ZIP_RENAME.get(item.filename, item.filename),
                          zin.read(item.filename))
    return tmp.name


def read_needs_xlsx(path: str | Path, sheet: str = "Sheet1") -> pd.DataFrame:
    """Read one file of the original financial statement extract.

    Column names are left as the raw Japanese labels; `tidy_needs` renames and
    types them.  Splitting the two steps keeps the mapping auditable: a label
    that NEEDS changes between vintages shows up as an untranslated column
    rather than as a silently missing variable.
    """
    fixed = _repair_workbook(path)
    try:
        raw = pd.read_excel(fixed, sheet_name=sheet, header=None)
    finally:
        os.unlink(fixed)

    labels = raw.iloc[4].tolist()
    body = raw.iloc[7:].copy()
    columns = []
    for i, label in enumerate(labels):
        if i in POSITIONAL:
            columns.append(POSITIONAL[i])
        elif isinstance(label, str) and label.strip():
            columns.append(label.strip())
        else:
            columns.append(f"_unnamed{i}")
    body.columns = columns
    body["_src"] = os.path.basename(str(path))
    return body.reset_index(drop=True)


def _batch_index(path: str | Path) -> int:
    """Trailing number of an extract file name, or 0 when it carries none.

    NEEDS names its exports FqReport1.xlsx, FqReport2.xlsx and so on, and the
    extract here was built in batches, so the number orders the files and lets
    `min_index` drop the early trial runs.  A file saved under another name has
    no number; it sorts first and is never dropped by `min_index`.
    """
    m = re.search(r"(\d+)\.xlsx$", str(path))
    return int(m.group(1)) if m else 0


def _is_statement_extract(path: str | Path) -> bool:
    """True when a workbook has the layout `read_needs_xlsx` expects.

    Row 5 of the statement extract carries Japanese item labels; the later
    extract carries NEEDS item codes on row 6 instead.  Checking the header
    means a reader may rename the files.
    """
    if _extra_half(path) is not None:
        return False
    try:
        fixed = _repair_workbook(path)
        try:
            head = pd.read_excel(fixed, sheet_name=0, header=None, nrows=6)
        finally:
            os.unlink(fixed)
    except Exception:
        return False
    if head.shape[0] < 5:
        return False
    labels = [str(x) for x in head.iloc[4].tolist()]
    return sum(1 for x in labels if x.strip() and x != "nan") >= 4


def load_all(raw_dir: str | Path, pattern: str = "FqReport*.xlsx",
             min_index: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Read and stack every financial statement extract in `raw_dir`.

    Parameters
    ----------
    min_index : read only files whose trailing number is at least this value.
                The extract was built in batches and the earliest files were
                trial runs with a different item set.
    """
    files = sorted(glob(str(Path(raw_dir) / pattern)), key=_batch_index)
    if min_index is not None:
        files = [f for f in files if _batch_index(f) >= min_index]
    if not files:
        # The name NEEDS gives the export is FqReport<n>.xlsx, but a reader may
        # have saved it under another name, so fall back to reading the header.
        files = sorted((str(f) for f in Path(raw_dir).glob("*.xlsx")
                        if _is_statement_extract(f)), key=_batch_index)
        if verbose and files:
            print(f"  no {pattern} in {raw_dir}; identified "
                  f"{len(files)} statement extract(s) by their header instead")
    if not files:
        raise FileNotFoundError(
            f"no financial statement extract found in {raw_dir}.\n"
            f"Expected files named {pattern}, or workbooks whose row 5 carries "
            "the Japanese item labels of the statement extract.")

    frames = []
    for p in files:
        df = read_needs_xlsx(p)
        frames.append(df)
        if verbose:
            print(f"  {os.path.basename(p):18s} rows={len(df):6d} cols={df.shape[1]}")
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"stacked: {out.shape[0]:,} rows x {out.shape[1]} columns")
    return out


def tidy_needs(df: pd.DataFrame, fiscal_start_month: int = 4) -> pd.DataFrame:
    """Rename, type and date the raw financial statement extract.

    Notes
    -----
    The accounting period arrives as "YYYY/MM".  With `fiscal_start_month=4`,
    books closing in January to March are assigned to the previous fiscal year,
    so a March 2000 balance sheet is a FY1999 observation.  This has to agree
    with `shocks.load_ks`.

    Capital expenditure is recorded as a negative number on the cash flow
    statement.  Absolute values are stored in new columns rather than
    overwriting, so that the sign convention of the source stays visible.
    """
    df = df.rename(columns=COLUMN_MAP).copy()

    ym = df["acc_period"].astype(str).str.extract(r"^(\d{4})/(\d{1,2})")
    df["year"] = pd.to_numeric(ym[0], errors="coerce")
    df["month"] = pd.to_numeric(ym[1], errors="coerce")
    n_bad = int(df["year"].isna().sum())
    if n_bad:
        print(f"  warning: {n_bad} rows have an unparseable accounting period")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["fy"] = np.where(df["month"] < fiscal_start_month,
                        df["year"] - 1, df["year"])

    for c in NUMERIC_COLS + ["cons_flag", "std_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "capex_total" in df.columns:
        df["capex"] = df["capex_total"].abs()
    if "capex_ppe" in df.columns:
        df["capex_tangible"] = df["capex_ppe"].abs()

    # The same firm-period can appear in more than one batch file where the
    # extract windows overlap.  Later files are the later download and win.
    before = len(df)
    df = df.sort_values(["stkno", "fy", "_src"])
    df = df.drop_duplicates(subset=["stkno", "year", "month"], keep="last")
    if before != len(df):
        print(f"  duplicates removed: {before:,} -> {len(df):,} rows")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# later extract: cash flow, bonds, construction in progress, incorporation date
# --------------------------------------------------------------------------
def read_needs_extra(path: str | Path, sheet: str = "Sheet1") -> pd.DataFrame:
    """Read one file of the later extract, keyed on NEEDS item codes.

    Layout: column 2 is the company name, 3 the ticker, 4 the frequency, 5 the
    accounting period, and 6 onwards the requested items, whose codes sit on
    row 5.  Fiscal years follow the same rule as `tidy_needs`.

    Missing values arrive as "-" or "*" and are read as NaN.  For bond balances
    that NaN means zero rather than undisclosed: NEEDS drops the line item for a
    firm with no bonds outstanding.  The substitution is made in
    `build_panel.add_needs_extra_vars`, not here, so that this function returns
    the file as it stands.
    """
    fixed = _repair_workbook(path)
    try:
        d = pd.read_excel(fixed, sheet_name=sheet, header=None)
    finally:
        os.unlink(fixed)

    codes = [str(d.iloc[5, j]) for j in range(6, d.shape[1])]
    names = [NEEDS_EXTRA_COLS.get(c, c) for c in codes]
    body = d.iloc[7:, :].copy()
    body.columns = ["_a", "_b", "firm_name", "stkno", "_freq", "period"] + names
    body = body.dropna(subset=["stkno"])

    year = pd.to_numeric(
        body["period"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    month = pd.to_numeric(
        body["period"].astype(str).str.extract(r"/(\d{2})")[0], errors="coerce")
    body["fy"] = np.where(month <= 3, year - 1, year)

    for c in names:
        if c.startswith("founded"):
            body[c] = pd.to_datetime(body[c], errors="coerce")
        else:
            body[c] = pd.to_numeric(
                body[c].replace({"-": np.nan, "*": np.nan, "": np.nan}),
                errors="coerce")

    keep = ["stkno", "firm_name", "fy"] + names
    return body[keep].dropna(subset=["fy"]).astype({"fy": int})


def _collapse(frame: pd.DataFrame | None, label: str,
              verbose: bool = True) -> pd.DataFrame | None:
    """Reduce duplicated (stkno, fy) rows to one, keeping the non-missing values.

    NEEDS returns a firm twice in the same file when its accounting period was
    restated, and the two rows are usually complementary rather than
    contradictory: one carries construction in progress or a bond balance where
    the other is blank.  Dropping either row wholesale would therefore discard
    real observations, so the rows are combined column by column, taking the
    first non-missing value.

    Where two rows carry different non-missing values the first is kept, and the
    count is reported: it is small in this extract, but it is a choice and
    should not be silent.
    """
    if frame is None:
        return None
    keys = ["stkno", "fy"]
    dup = int(frame.duplicated(keys).sum())
    if not dup:
        return frame
    value_cols = [c for c in frame.columns if c not in keys + ["_file"]]
    distinct = frame.groupby(keys)[value_cols].nunique(dropna=True)
    conflicts = int((distinct > 1).any(axis=1).sum())
    out = frame.groupby(keys, as_index=False)[value_cols].first()
    # first() skips missing values, so the surviving row carries a value
    # wherever either of the originals did.
    if verbose:
        print(f"  {label}: {dup:,} duplicate (stkno, fy) rows combined into "
              f"{len(out):,} unique rows, taking the first non-missing value"
              + (f"; {conflicts:,} keys carried conflicting values"
                 if conflicts else "; no conflicting values"))
    return out


def _extra_half(path: str | Path) -> str | None:
    """Say whether a workbook is the corporate half, the financial half, or neither.

    The file name settles it when NEEDS' own naming survives.  When it does not,
    the item codes on row 6 do: the corporate half requests CORPORATE' items and
    the financial half FINFSTA' items, and no extract mixes the two.  Deciding on
    content means a reader may keep every extract in one directory under
    whatever names they were saved with.
    """
    stem = Path(path).stem.lower()
    if stem.startswith("corporate"):
        return "corporate"
    if stem.startswith("financial"):
        return "financial"
    if stem.startswith("fqreport"):
        return None                      # the financial statement extract
    try:
        fixed = _repair_workbook(path)
        try:
            head = pd.read_excel(fixed, sheet_name=0, header=None, nrows=7)
        finally:
            os.unlink(fixed)
    except Exception:
        return None
    if head.shape[0] < 6 or head.shape[1] < 7:
        return None
    codes = [str(head.iloc[5, j]) for j in range(6, head.shape[1])]
    if any(c.startswith("CORPORATE'") for c in codes):
        return "corporate"
    if any(c.startswith("FINFSTA'") for c in codes):
        return "financial"
    return None


def load_needs_extra(folder: str | Path, pattern: str = "*.xlsx",
                     verbose: bool = True) -> pd.DataFrame:
    """Read and join the corporate and financial halves of the later extract.

    The two halves cover the same firms with different items and are joined on
    (stkno, fy) with an outer join, so that a firm missing from one half is kept
    rather than silently dropped.

    The folder may also hold the financial statement extract; those files are
    recognised and left to `load_all`, so both vintages can live in one
    directory.
    """
    folder = Path(folder)
    corporate, financial = [], []
    for f in sorted(folder.glob(pattern)):
        half = _extra_half(f)
        if half is None:
            continue
        frame = read_needs_extra(f)
        (corporate if half == "corporate" else financial).append(frame)
        if verbose:
            print(f"  {f.name}: {len(frame):,} rows ({half} half)")

    c = _collapse(pd.concat(corporate, ignore_index=True) if corporate else None,
                  "corporate", verbose)
    f = _collapse(pd.concat(financial, ignore_index=True) if financial else None,
                  "financial", verbose)

    if c is None:
        return f
    if f is None:
        return c
    out = f.merge(c.drop(columns=[x for x in ["firm_name"] if x in c.columns]),
                  on=["stkno", "fy"], how="outer")
    if verbose:
        print(f"\njoined: {len(out):,} rows, {out.stkno.nunique():,} firms, "
              f"FY{out.fy.min()}-{out.fy.max()}")
    return out


# --------------------------------------------------------------------------
# industry classification
# --------------------------------------------------------------------------
def industry_from_nkil_code(df: pd.DataFrame, col: str = "nkil_code",
                            out_medium: str = "nkilm",
                            out_small: str = "nkils") -> pd.DataFrame:
    """Derive the Nikkei industry classifications from the six-digit code.

    Digits two and three of `CORPORATE'NKIL` are the medium classification, the
    32-industry level used throughout the paper, and digits two to four are the
    small classification.  Both are kept as zero-padded strings; converting them
    to integers drops the leading zero and silently merges industries.

    This supersedes the earlier route, in which the classification was taken
    from a separate file built from stock price extracts.  Deriving it from the
    same extract as the financial data removes an undocumented dependency; the
    two agree for all 2,549 firms present in both.
    """
    out = df.copy()
    code = out[col].astype(str).str.strip().str.zfill(6)
    out[out_medium] = code.str[1:3]
    out[out_small] = code.str[1:4]
    return out


def attach_sector(df: pd.DataFrame, sector_src: str | Path,
                  stkno_col: str = "stkno",
                  sector_cols: list[str] | None = None,
                  rename: dict[str, str] | None = None) -> pd.DataFrame:
    """Merge an industry classification from a separate file.

    Retained for backward compatibility with panels built before the industry
    code was available in the main extract; new work should use
    `industry_from_nkil_code`.

    The file is expected to hold one row per firm with a ticker column and one
    or more classification columns.  Note that it carries a single, time
    invariant classification per firm, whereas industry can in principle change
    with a change of business or a holding company reorganisation.  Changes are
    rare at the 32-industry level, but the paper states that industry is treated
    as fixed.
    """
    sector_src = str(sector_src)
    smap = (pd.read_parquet(sector_src) if sector_src.endswith(".parquet")
            else pd.read_csv(sector_src, dtype=str))

    sector_cols = sector_cols or [c for c in ["nkilm", "nkils", "sector_est"]
                                  if c in smap.columns]
    if not sector_cols:
        raise KeyError(f"no industry column in {sector_src}: {list(smap.columns)}")

    smap = (smap[[stkno_col] + sector_cols]
            .dropna(subset=[stkno_col])
            .drop_duplicates(subset=stkno_col))
    if rename:
        smap = smap.rename(columns=rename)
        sector_cols = [rename.get(c, c) for c in sector_cols]
    for c in sector_cols:
        if c in smap.columns:
            smap[c] = smap[c].astype(str).str.strip()

    out = df.merge(smap, on=stkno_col, how="left")
    main = sector_cols[0]
    n_all = out[stkno_col].nunique()
    n_missing = out.loc[out[main].isna(), stkno_col].nunique()
    print(f"  industry merge ({main}): {n_all - n_missing}/{n_all} firms matched"
          f" ({n_missing} missing)")
    for c in sector_cols:
        if c in out.columns:
            print(f"    {c}: {out[c].nunique()} categories")
    return out
