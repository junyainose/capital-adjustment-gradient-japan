"""Locate why the yearly gradients come back unidentified.

    python diagnose_figure.py

Prints, at each stage, whether the year-by-year interaction columns actually
carry variation: after construction, after the fixed effects are absorbed, and
in the solved coefficients.  The stage at which the numbers collapse to zero
identifies the cause.
"""
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from build_tables import AXES, prepare   # noqa: E402
from src import gradients as gr          # noqa: E402

DEP = "dlogk_h1"
CONTROLS = ("log_assets_lag",)

p = prepare()
print(f"pandas {pd.__version__}, numpy {np.__version__}")
print(f"rows {len(p):,}  fy dtype {p[gr.YEAR].dtype}  "
      f"stkno dtype {p[gr.FIRM].dtype}")

years = sorted(p[gr.YEAR].unique())
print(f"years: {len(years)}  first {years[0]!r} type {type(years[0]).__name__}")

d = p.copy()
names = []
for label, col in AXES.items():
    for t in years:
        nm = f"{label}@{t}"
        d[nm] = (d[col] * (d[gr.YEAR] == t)).astype(float)
        names.append(nm)
names += list(CONTROLS)
d = d.dropna(subset=[DEP] + list(AXES.values()) + list(CONTROLS)
             + [gr.FIRM, gr.YEAR])
print(f"\nafter dropna: {len(d):,} rows")

# 1. do the interaction columns carry anything at all?
built = d[[c for c in names if "@" in c]]
nonzero = (built != 0).sum()
print(f"1. constructed columns with any non-zero entry: "
      f"{int((nonzero > 0).sum())} of {built.shape[1]}")
print(f"   example mrpk@{years[5]}: {int(nonzero[f'mrpk@{years[5]}']):,} "
      "non-zero rows")
if int((nonzero > 0).sum()) == 0:
    print("   -> the year mask never matched; comparing "
          f"{p[gr.YEAR].dtype} against {type(years[0]).__name__} failed")

# 2. does the variation survive the within transformation?
x = gr._absorb(d, [DEP] + names, [gr.FIRM, gr.YEAR])
norms = np.linalg.norm(x[[c for c in names if "@" in c]].to_numpy(float), axis=0)
print(f"\n2. after absorbing firm and year effects: "
      f"{int((norms > 1e-8).sum())} of {len(norms)} columns retain variation")
print(f"   norms: min {norms.min():.3e}  median {np.median(norms):.3e}  "
      f"max {norms.max():.3e}")

# 3. does the solve return anything?
z = x[names].to_numpy(float)
y = x[DEP].to_numpy(float)
print(f"\n3. design matrix {z.shape}, rank {np.linalg.matrix_rank(z)}, "
      f"any NaN: {bool(np.isnan(z).any())}, y NaN: {bool(np.isnan(y).any())}")
bread = np.linalg.pinv(z.T @ z)
beta = bread @ (z.T @ y)
print(f"   non-zero coefficients: {int((np.abs(beta) > 1e-12).sum())} "
      f"of {len(beta)}")
resid = y - z @ beta
print(f"   residual sd {resid.std():.4f}")
groups = d[gr.FIRM].to_numpy()
print(f"   cluster array dtype {groups.dtype}, "
      f"{len(np.unique(groups)):,} clusters")
vcov = gr._cluster_vcov(z, resid, groups, bread)
se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
print(f"   positive standard errors: {int((se > 0).sum())} of {len(se)}")
