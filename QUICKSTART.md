# Quick start

Every script takes the panel through `--panel`. The default is
`data/processed/panel_annual.parquet`, which is built from Nikkei NEEDS
Financial QUEST and is not in this package.

## Without a NEEDS licence

```bash
pip install -r requirements.txt
python -m src.synthetic
python build_tables.py  --panel data/processed/panel_synthetic.parquet
python build_figures.py --panel data/processed/panel_synthetic.parquet
```

This writes 13 tables to `output/tables/` and the figure to `output/figures/`.

**The numbers will not match the paper, and are not meant to.** Nothing in the
generator is calibrated to the real data; every parameter is a round number,
because a moment estimated from licensed data would itself be derived data. What
the run checks is that the estimator recovers a gradient that was planted in the
data and manufactures none where none was planted:

```
drift on:  MRPK drift +0.0017 (0.0001)  planted +0.0020  ok
drift off: MRPK drift +0.0003 (0.0001)  planted +0.0000  ok
```

Appendix D of the paper describes the exercise and the two coefficients that are
reported but not tested.

## With a NEEDS licence

```bash
python build_data.py        # writes data/processed/panel_annual.parquet
python build_tables.py
python build_figures.py
```

Section 2 of `README.md` lists the NEEDS item codes for every variable and the
exclusions applied at extraction. Because the data cannot be compared directly,
check the extract against the sample selection counts of Table C.1, which
`build_tables.py` regenerates.

## Working with the panel directly

```python
import sys; sys.path.insert(0, '.')
from build_tables import AXES, prepare
from src import gradients as gr

p = prepare('data/processed/panel_synthetic.parquet')
f = gr.fit_trend(p, 'dlogk_h1', AXES)
print(f.get('level_mrpk'), f.get('drift_mrpk'))
```

`prepare()` builds the columns the tables need. The main ones:

| column | contents |
|---|---|
| `mrpk_os_s_lag` | MRPK from operating surplus, industry-year normalized, lagged |
| `mrpk_os_raw` | the same in raw logs, before normalization |
| `lev_s_lag` | leverage, industry-year normalized, lagged |
| `log_k_s_lag` | log capital, industry-year normalized, lagged |
| `dlogk_h{0..3}` | 100 x the cumulative log change in capital at horizon h |

## Three things that will bite

- **Five-year rolling windows do not identify the gradient.** MRPK is persistent
  enough that firm fixed effects absorb almost all of the within-window
  variation. Estimate on the full sample with period interactions instead.
- **Order of operations.** Build lags and leads before restricting the period,
  and winsorize after that, with the dependent variable included.
- **Table files share their names across panels.** Running on the synthetic
  panel overwrites `output/tables/`, so keep the two runs in separate trees.
