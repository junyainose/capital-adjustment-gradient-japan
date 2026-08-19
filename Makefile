# Full pipeline.  `make synthetic` needs no licensed data.
PANEL = data/processed/panel_annual.parquet
SYNTH = data/processed/panel_synthetic.parquet

.PHONY: all data tables figures synthetic test clean

all: data tables figures

data:
	python build_data.py

tables:
	python build_tables.py

figures:
	python build_figures.py

synthetic:
	python -m src.synthetic
	python build_tables.py  --panel $(SYNTH)
	python build_figures.py --panel $(SYNTH)

test:
	python -c "from src import synthetic; synthetic.self_test()"

clean:
	rm -f output/tables/*.tex output/tables/*.csv
	rm -f output/figures/*.pdf output/figures/*.png
