# Replication package

**Have Firms Become More Responsive to Returns?
A Changing Capital-Adjustment Gradient in Japan**
Junya Inose, Faculty of Economics, Toyo University

This package reproduces every table and figure in the paper.

---

## 1. Overview

The paper estimates how the cross-sectional gradient of investment with respect
to firm characteristics — the marginal revenue product of capital, leverage and
size — has moved over FY1999–2024 for firms listed on the Tokyo Stock
Exchange.

The estimation sample is drawn from Nikkei NEEDS Financial QUEST. **Those data
may not be redistributed in any form, raw or derived**, so they are not included
here. Everything else is: all code, the public statistics, the paper's exhibits
as produced, and a synthetic panel that lets the code be executed and checked
without a NEEDS licence.

Approximate run time on a recent laptop: building the panel from the raw
extracts takes about three minutes, tables about two minutes, figures under a
minute.

---

## 2. Data availability and provenance

### 2.1 Nikkei NEEDS Financial QUEST — **not redistributable**

| | |
|---|---|
| Provider | Nikkei Media Marketing, Inc. |
| Access | Institutional subscription. Widely held by Japanese universities; individual access is also sold. |
| Terms | The licence prohibits redistribution of the data, whether raw or processed, for commercial and non-commercial purposes alike. |
| Coverage used | Consolidated annual accounts, firms listed on the Tokyo Stock Exchange, fiscal years 1995–2025. |
| Retrieved | Financial statement extract 2026-05; cash flow, bond, construction in progress and incorporation date extract 2026-08. |

The extract cannot be shipped, so it is documented instead, in enough detail
that a licensed user can rebuild it.

**Items in the first extract** are identified by their Japanese labels; see
`COLUMN_MAP` in `src/io_needs.py` for the full list with English glosses.

**Items in the second extract** are identified by NEEDS item codes, which is the
more robust convention and the one a fresh extract should use. The full list is
`NEEDS_EXTRA_COLS` in `src/io_needs.py`:

| Code | Variable |
|---|---|
| `CORPORATE'PRMTD1` | substantive incorporation date |
| `CORPORATE'PRMTD2` | formal incorporation date |
| `CORPORATE'NKIL` | Nikkei industry code, six digits |
| `CORPORATE'JSIC1` | Japan Standard Industrial Classification |
| `FINFSTA'F01065` | cash flow from operations |
| `FINFSTA'F01087` | cash flow from investing |
| `FINFSTA'F01106` | cash flow from financing |
| `FINFSTA'F01096` | proceeds from bond issuance |
| `FINFSTA'F01097` | payments on bond redemption |
| `FINFSTA'C01032` | bonds due within one year |
| `FINFSTA'C01059` | bonds and convertible bonds |
| `FINFSTA'B01074` | construction in progress |

Two practical notes. The workbooks NEEDS writes are not valid xlsx: entry names
inside the zip container are lower-cased, so Excel opens them but `openpyxl`
does not. `src/io_needs.py` repairs them before reading, so no manual step is
needed. And the industry classification is derived from digits two and three of
`CORPORATE'NKIL`; no separate classification file is required.

**Industries.** Finance and the residual services category are excluded at the
point of extraction, not in the code. Finance is excluded because the capital
stock of a bank or an insurer is not the object measured here; residual services
because tangible fixed assets there are small enough that MRPK would be
dominated by measurement error. What remains is the 32 industries of the Nikkei
medium classification. `check_extract.py` verifies that none of the excluded
codes is present.

**Verifying the extract.** Since the data cannot be compared directly, a
replicator checks their extract against the sample selection table that
`build_data.py` writes to `output/tables/sample_selection.tex`. Matching those
counts row by row establishes that the extract is the same one. For reference,
the counts obtained from the extract used in the paper are:

| Step | Firms | Observations |
|---|---:|---:|
| Raw extract, all closing months and years | 2,558 | 90,963 |
| Books closing in March | 1,754 | 62,082 |
| Merged with the later extract | 1,754 | 62,082 |
| Panel span FY1995--2025 | 1,749 | 46,546 |
| Estimation window FY1999--2024 | 1,738 | 39,558 |
| Core accounting variables non-missing | 1,736 | 39,426 |
| Lagged capital available and positive | 1,729 | 39,037 |

Restricting to firms closing their books in March is by far the largest
selection, removing about a third of firms and just under half of the
observations. It is imposed so that an accounting period maps unambiguously to
a fiscal year, and it means the sample under-represents retailers and
subsidiaries of foreign parents, which more often close in February or December.

The later extract returns 705 duplicated (firm, fiscal year) keys, almost all of
them firms that changed their accounting period at some point since 1974. The
loader keeps the last row of each. `check_extract.py` reports what that choice
costs, and the answer is nothing for any result in the paper.

Of the 705 keys, 87 fall inside FY1999--2024 and 70 of those carry a genuine
disagreement, meaning both rows hold a value and the values differ. The columns
involved, counted inside the estimation window, are operating cash flow (58),
cash flow from investing (58), cash flow from financing (57), construction in
progress (35), long-term bonds (12), short-term bonds (3), bond redemptions (3)
and bond issuance (1).

None of these reaches the main specification. Leverage is built from short- and
long-term borrowing in the financial statement extract, not from the bond
balances here; the bond items feed only `bond`, `has_bond` and `ever_bond`,
which no table uses, and cash flow from investing and from financing are not
used at all. Operating cash flow enters one robustness row and the descriptive
table, and construction in progress enters the completion-based investment rate
of Appendix A. MRPK, capital and leverage come from the financial statement
extract throughout, where `read_needs_xlsx` reads one row per firm-year.

The corporate half carries no genuine disagreement at all, as it should: founding
dates and industry codes are firm-level constants repeated across years.

#### What the export must look like

NEEDS lets the user choose how the header block is laid out at download time, so
an extract saved under different settings will not read. Both loaders expect the
layout below, which is what the export produces when the header option is left
at its default.

| | financial statement extract | later extract |
|---|---|---|
| worksheet | the first sheet, named `Sheet1` | the same |
| header row | Excel row 5 carries the Japanese item labels | Excel row 6 carries the NEEDS item codes, e.g. `FINFSTA'F01065` |
| first data row | Excel row 8 | Excel row 8 |
| leading columns | company name, ticker, accounting period | two spare columns, then company name, ticker, frequency, accounting period |
| accounting period | `YYYY/MM` | the same |
| missing values | blank, `-` or `*` | the same |

`read_needs_xlsx` reads labels from row 5 and takes company name, ticker and
period by position, so those three must keep their places; the requested items
are matched by label and may be reordered. `read_needs_extra` reads item codes
from row 6, so those items may be reordered freely, but the six leading columns
must keep their places.

A label that NEEDS renames between vintages comes through as an untranslated
column rather than as a silently missing variable, which is deliberate: the
mismatch is visible in `check_extract.py` instead of propagating into the
estimates as a column of NaN.

#### Where to put the files

Everything goes in `data/raw`, in one directory. The loaders tell the two
vintages apart by the header, so the files may be saved under any name:

- the financial statement extract is read by `load_all`. It first takes anything
  matching `FqReport*.xlsx`, the name NEEDS gives the export. If nothing
  matches, it reads the header of every workbook in the directory and takes
  those whose row 5 carries item labels.
- the later extract is read by `load_needs_extra`, which calls a workbook the
  corporate half if its row 6 carries `CORPORATE'` codes and the financial half
  if it carries `FINFSTA'` codes.

One caveat about names. `load_all` orders the statement extracts by the number
at the end of the file name, because this extract was assembled in batches and
`MIN_FILE_INDEX` in `build_data.py` drops the early trial runs. A file with no
number sorts first and is never dropped, which is the right behaviour for a
fresh single-batch extract but means the trial-run guard does nothing once the
files have been renamed. Keep the NEEDS names if the extract came in batches.

Older copies of this package expected the later extract in `data/raw/extra`, and
`build_data.py --extra <dir>` still accepts that. With no `--extra`, both are
read from `--raw`, and `data/raw/extra` no longer exists.

### 2.2 High-frequency monetary policy shocks — obtain from the authors

Kubota, H., and M. Shintani (2022), "High-frequency identification of monetary
policy shocks in Japan", *The Japanese Economic Review* 73(3), 483–513. The
series is used in Section 6 only; the main results do not depend on it. Place
the file at `data/raw/KSdata_VAR_IV.csv`.

### 2.3 Public statistics — **included**

| File | Source | Used for |
|---|---|---|
| `data/raw/macro_annual.csv` | Statistics Bureau, Bank of Japan, Ministry of Finance, Cabinet Office | aggregate controls, Section 6 |
| `data/raw/hojin_kigyo.csv` | Financial Statements Statistics of Corporations by Industry, Ministry of Finance, via e-Stat | imputed labour cost for the value-added robustness row |

Neither is needed for the main results.

### 2.4 Statement

The author certifies that he has legitimate access to the data used here, that
the NEEDS data are available to any researcher who purchases a licence on the
same terms, and that he will preserve the code and the constructed panel for at
least five years and assist with reasonable requests for clarification.

---

## 3. Computational requirements

Python 3.11 or later. `pip install -r requirements.txt`. No proprietary software
and no cluster; the largest object in memory is a panel of about 50,000 rows.

---

## 4. Running the package

### With a NEEDS licence

```bash
python build_data.py            # reads every extract in data/raw
python build_tables.py
python build_figures.py
```

### Without one

```bash
python -m src.synthetic          # writes the synthetic panel; runs the self test
python build_tables.py  --panel data/processed/panel_synthetic.parquet
python build_figures.py --panel data/processed/panel_synthetic.parquet
```

The second route runs the whole pipeline and produces tables and figures of the
same shape. **The numbers will not match the paper**: nothing in the synthetic
generator is calibrated to the licensed data, and no parameter in it is a moment
of that data.

What the synthetic route does establish is that the estimation code is correct.
The generator plants a known drift in the MRPK gradient, and `src/synthetic.py`
generates the panel twice, once with the drift and once without, checking that
the estimator recovers the planted value in the first case and nothing in the
second:

```
 drift on : MRPK drift +0.0017 (0.0001)  planted +0.0020  ok
 drift off: MRPK drift +0.0003 (0.0001)  planted +0.0000  ok
```

`src.synthetic.units_demo()` reproduces, in data with a known generating
process, the mechanism behind Table 3: a characteristic whose dispersion
compresses shows a strengthening gradient in fixed units even when the relation
per standard deviation is unchanged.

---

## 5. Exhibits and the code that produces them

| Exhibit | Script | Output |
|---|---|---|
| Table 1, main results | `build_tables.py` → `table_main` | `output/tables/table1_main.tex` |
| Table 2, robustness | `build_tables.py` → `table_robustness` | `output/tables/table2_robustness.tex` |
| Table 3, units | `build_tables.py` → `table_units` | `output/tables/table3_units.tex` |
| Table 4, dispersion | `build_tables.py` → `table_dispersion` | `output/tables/table4_dispersion.tex` |
| Appendix, sample selection | `build_data.py` → `SampleFlow` | `output/tables/sample_selection.tex` |
| Figure 1, gradients | `build_figures.py` → `figure_gradients` | `output/figures/fig1_gradients.pdf` |

Tables use `booktabs` and a `\sym` command for significance markers:

```latex
\usepackage{booktabs}
\newcommand{\sym}[1]{\ensuremath{^{#1}}}
```

Section 6, on the response to monetary policy shocks, is not yet in this
package; its estimation code will be added when that section is final.

---

## 6. Contents

```
build_data.py        raw extracts -> analysis panel, with sample selection counts
build_tables.py      panel -> the paper's tables
build_figures.py     panel -> the paper's figures
src/
  io_needs.py        reading the NEEDS extracts; the item dictionary lives here
  build_panel.py     derived variables; the ordering rules are in the docstring
  shocks.py          annualising the Kubota-Shintani shocks
  labor_cost.py      imputed labour cost and value added (robustness only)
  gradients.py       the paper's estimators
  tables.py          LaTeX output
  synthetic.py       synthetic panel, self test, units demonstration
data/raw/            public statistics; licensed extracts go here too
output/tables/       tables as produced
output/figures/      figures as produced
```

---

## 7. Three things that are easy to get wrong

These cost time during the project and are recorded so that they need not cost
it again. Each is also documented at the point in the code where it applies.

**Order of operations.** Lags and leads are formed on the full panel before the
estimation window is imposed; restricting first drops the boundary years, and
for a five-year sub-period a three-period lead leaves two usable years.
Winsorising comes after leads are formed and must cover the dependent variable:
the raw investment rate reaches 1,985 in this panel, so a lead of an untrimmed
series carries the tail onto the left-hand side.

**Units.** A gradient measured in fixed units drifts whenever the dispersion of
the characteristic moves, even if the relation per standard deviation is
constant. Leverage compresses by about a fifth over this sample and shows
exactly that. `gradients.standardize` offers the three conventions and the paper
reports all of them.

**Duplicated firm-years in the later extract.** NEEDS returns a firm twice for
the same fiscal year when its accounting period was restated, and both rows sit
in the same file. The two are usually complementary — one carries construction
in progress or a bond balance where the other is blank — so dropping either
wholesale discards real observations. `io_needs` combines them column by column,
taking the first non-missing value, and reports how many keys carry genuinely
conflicting values. In this extract 705 keys are duplicated, of which 81 fall
inside the estimation window; conflicts are concentrated in construction in
progress, which enters only the completion-based investment rate reported as a
robustness check.

**Missing values that are not missing.** A missing bond balance in NEEDS means a
zero balance: the line item disappears for a firm with no bonds outstanding.
Treating those as missing would drop precisely the firms without bond market
access. Construction in progress is genuinely missing for about a third of
observations, and the completion-based investment rate must not have those
filled with zero — doing so produced a series correlated 1.000 with the
uncorrected rate.

---

## 8. Licence

Code is released under the MIT licence. The data are not the author's to
license; see section 2.
