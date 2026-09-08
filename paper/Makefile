PYTHON := .venv/bin/python3

.PHONY: paper numbers figures clean

paper: numbers figures
	tectonic main.tex

# Regenerate paper/numbers.tex from the report JSONs under reports/ and
# outputs/ (plus the derived-stats JSONs under paper/data_derived/).
numbers:
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/collect_numbers.py

# Figure-generation scripts. compute_exotic_spans.py and
# compute_windowing_stats.py are NOT run by default here -- they read raw
# ELAsTiCC2 FITS data and take several minutes; run them by hand once
# (`PYTHONPATH=. ../.venv/bin/python3 scripts/compute_exotic_spans.py` etc.)
# and their output under data_derived/ is reused on subsequent builds.
figures:
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/make_lightcurve_schematics.py
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/make_windowing_figure.py
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/make_phase_curve.py
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/make_confusion_grid.py
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/make_prediction_panels.py
	cd .. && PYTHONPATH=. $(PYTHON) paper/scripts/make_anomaly_slope.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.xdv
