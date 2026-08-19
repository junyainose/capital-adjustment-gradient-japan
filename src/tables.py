"""LaTeX tables for the paper.

Two layouts cover everything in the paper.

`coef_table`    specifications in columns, coefficients in rows.  Used for the
                main table, where a handful of specifications are shown in full.
`summary_table` specifications in rows, one coefficient per column.  Used for
                the robustness tables, where many specifications are compared on
                the same small set of coefficients.

Both emit `booktabs` markup and expect the preamble to load `booktabs`.  Nothing
here writes to the paper directly: each function returns a string, and
`write` saves it under `output/tables/`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLE_DIR = Path("output/tables")

# Significance is marked at the conventional levels.  The thresholds are kept
# in one place so that the note under every table stays consistent with them.
LEVELS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))

# Notes sit in a minipage the width of the surrounding text.  \linewidth rather
# than \textwidth so that the block still fits when the table is placed in a
# narrower context, and a newline after \footnotesize so that the size command
# does not run into the first word of the note.
NOTE_OPEN = r"\begin{minipage}{\linewidth}\footnotesize" + "\n"
NOTE_CLOSE = "\n" + r"\end{minipage}"

# Wide tables are shrunk to the text block rather than overflowing into the
# margin.  The \ifdim guard leaves a table that already fits at its natural
# size, so only the tables that need it are scaled.
BOX_OPEN = (r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}"
            r"{!}{%") + "\n"
BOX_CLOSE = "}"


def _open_body(compact: bool = False) -> list[str]:
    """Font and spacing declarations that precede a tabular."""
    lines = [r"\small"] if not compact else [r"\footnotesize"]
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(BOX_OPEN.rstrip("\n"))
    return lines


def stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    for cut, mark in LEVELS:
        if p < cut:
            return mark
    return ""


def _fmt(value: float, pval: float | None = None, digits: int = 3,
         with_stars: bool = True) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    body = f"{value:.{digits}f}"
    if with_stars and pval is not None:
        mark = stars(pval)
        if mark:
            body += r"\sym{" + mark + "}"
    return body


def _escape(text: str) -> str:
    """Escape the characters LaTeX treats specially, outside of math mode.

    Row labels sometimes carry mathematics -- a subscripted parameter name, say
    -- and inside $...$ an underscore is a subscript rather than a character to
    escape.  Splitting on the dollar signs keeps the text segments escaped and
    leaves the mathematics alone.
    """
    out, in_math = [], False
    for part in text.split("$"):
        if in_math:
            out.append(part)
        else:
            for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
                part = part.replace(a, b)
            out.append(part)
        in_math = not in_math
    return "$".join(out)


def coef_table(fits: list, column_labels: list[str], rows: list[tuple[str, str]],
               digits: int = 3, se_digits: int | None = None,
               footer: list[tuple[str, list[str]]] | None = None,
               caption: str = "", label: str = "", notes: str = "") -> str:
    """Specifications in columns; each row is one coefficient with its SE below.

    Parameters
    ----------
    fits : list of `gradients.Fit`, one per column.
    rows : list of (regressor name, printed label).  A regressor missing from a
           given fit prints as blank, which is how specifications with different
           regressor sets are shown side by side.
    footer : extra rows appended below the coefficients, as
             (label, list of cell strings).  Observation and firm counts are
             added automatically after these.
    """
    se_digits = digits if se_digits is None else se_digits
    ncol = len(fits)
    out = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        out.append(r"\caption{" + caption + "}")
    if label:
        out.append(r"\label{" + label + "}")
    out += _open_body()
    out.append(r"\begin{tabular}{l" + "c" * ncol + "}")
    out.append(r"\toprule")
    out.append(" & " + " & ".join(f"({i + 1})" for i in range(ncol)) + r" \\")
    out.append(" & " + " & ".join(_escape(c) for c in column_labels) + r" \\")
    out.append(r"\midrule")

    for name, printed in rows:
        cells, se_cells = [], []
        for f in fits:
            if f is None or name not in f.coef:
                cells.append("")
                se_cells.append("")
                continue
            b, s, p = f.get(name)
            cells.append(_fmt(b, p, digits))
            se_cells.append(f"({s:.{se_digits}f})")
        out.append(_escape(printed) + " & " + " & ".join(cells) + r" \\")
        out.append(" & " + " & ".join(se_cells) + r" \\")

    out.append(r"\midrule")
    for lab, cells in (footer or []):
        out.append(_escape(lab) + " & " + " & ".join(cells) + r" \\")
    out.append("Observations & " + " & ".join(
        f"{f.nobs:,}" if f else "" for f in fits) + r" \\")
    out.append("Firms & " + " & ".join(
        f"{f.nfirms:,}" if f else "" for f in fits) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(BOX_CLOSE)
    if notes:
        out.append(NOTE_OPEN)
        out.append(notes + " Standard errors clustered on the firm in "
                            "parentheses. "
                   + r"\sym{*}, \sym{**} and \sym{***} denote significance at "
                     "the 10, 5 and 1 percent level.")
        out.append(NOTE_CLOSE)
    out.append(r"\end{table}")
    return "\n".join(out)


def summary_table(rows: list[tuple[str, object]], columns: list[tuple[str, str]],
                  digits: int = 3, show_n: bool = True,
                  group_breaks: dict[int, str] | None = None,
                  caption: str = "", label: str = "", notes: str = "") -> str:
    """Specifications in rows; one column per reported coefficient.

    Parameters
    ----------
    rows : list of (printed label, `gradients.Fit`).
    columns : list of (regressor name, printed header).
    group_breaks : row index -> panel heading inserted above that row.  Used to
                   separate the robustness blocks (measurement, dependent
                   variable, controls, sample) inside a single table.
    """
    ncol = len(columns) + (1 if show_n else 0)
    out = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        out.append(r"\caption{" + caption + "}")
    if label:
        out.append(r"\label{" + label + "}")
    out += _open_body(compact=True)
    out.append(r"\begin{tabular}{l" + "c" * ncol + "}")
    out.append(r"\toprule")
    head = [_escape(h) for _, h in columns] + (["Obs."] if show_n else [])
    out.append(" & " + " & ".join(head) + r" \\")
    out.append(r"\midrule")

    for i, (lab, fit) in enumerate(rows):
        if group_breaks and i in group_breaks:
            out.append(r"\addlinespace")
            out.append(r"\multicolumn{" + str(ncol + 1) + r"}{l}{\emph{"
                       + _escape(group_breaks[i]) + r"}} \\")
        if fit is None:
            out.append(_escape(lab) + " & " + " & ".join([""] * ncol) + r" \\")
            continue
        cells = []
        for name, _ in columns:
            b, s, p = fit.get(name)
            cells.append(_fmt(b, p, digits) + f" ({s:.{digits}f})")
        if show_n:
            cells.append(f"{fit.nobs:,}")
        out.append(_escape(lab) + " & " + " & ".join(cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(BOX_CLOSE)
    if notes:
        out.append(NOTE_OPEN)
        out.append(notes + " Standard errors clustered on the firm in "
                            "parentheses. "
                   + r"\sym{*}, \sym{**} and \sym{***} denote significance at "
                     "the 10, 5 and 1 percent level.")
        out.append(NOTE_CLOSE)
    out.append(r"\end{table}")
    return "\n".join(out)


def frame_table(df: pd.DataFrame, digits: int = 3, caption: str = "",
                label: str = "", notes: str = "",
                index_label: str = "") -> str:
    """A plain numeric table, for descriptive exhibits such as the dispersion series."""
    out = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        out.append(r"\caption{" + caption + "}")
    if label:
        out.append(r"\label{" + label + "}")
    out += _open_body()
    out.append(r"\begin{tabular}{l" + "c" * df.shape[1] + "}")
    out.append(r"\toprule")
    out.append(_escape(index_label) + " & "
               + " & ".join(_escape(str(c)) for c in df.columns) + r" \\")
    out.append(r"\midrule")
    for ix, row in df.iterrows():
        # Columns holding counts or labels are already strings; only numbers are
        # formatted, so a frame may mix the two.
        cells = ["" if pd.isna(v) else (v if isinstance(v, str)
                                        else f"{v:.{digits}f}") for v in row]
        out.append(_escape(str(ix)) + " & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(BOX_CLOSE)
    if notes:
        out.append(NOTE_OPEN + notes + NOTE_CLOSE)
    out.append(r"\end{table}")
    return "\n".join(out)


def write(text: str, filename: str, directory: Path | str = TABLE_DIR) -> Path:
    """Save a table and return its path.  Creates the directory if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    return path
