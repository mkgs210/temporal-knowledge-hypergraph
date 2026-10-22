PYTHON ?= .venv-cpu/bin/python
BOOTSTRAP_PYTHON ?= python3.12
OUT ?= out/cpu

.PHONY: all offline build test validate metrics science components labels label-audit illustrate tables pdf package
all:
	$(BOOTSTRAP_PYTHON) scripts/run_pipeline.py --python "$(PYTHON)" --out "$(OUT)"

# The same local study without paid/remote model calls; T5 remains unmeasured.
offline:
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_pipeline --out $(OUT) --build-only
	PYTHONPATH=src $(PYTHON) validate.py $(OUT)/hierarchy
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_pipeline --out $(OUT) --assess-only
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_science $(OUT)
	PYTHONPATH=src $(PYTHON) make_tables.py $(OUT) --report report.md --pdf report.pdf

build:
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_pipeline --out $(OUT) --build-only

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

validate:
	PYTHONPATH=src $(PYTHON) validate.py $(OUT)/hierarchy

metrics:
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_pipeline --out $(OUT) --assess-only

science:
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_science $(OUT)

components:
	OPENBLAS_NUM_THREADS=4 PYTHONPATH=src $(PYTHON) -m tkh.cpu_components $(OUT)

illustrate:
	PYTHONPATH=src $(PYTHON) -m tkh.cpu_figures $(OUT)

labels:
	PYTHONPATH=src $(PYTHON) -m tkh.cpu_annotations --run $(OUT)

label-audit:
	PYTHONPATH=src $(PYTHON) -m tkh.cpu_label_audit --run $(OUT) --check

tables:
	PYTHONPATH=src $(PYTHON) make_tables.py $(OUT) --report report.md

pdf:
	PYTHONPATH=src $(PYTHON) make_tables.py $(OUT) --report report.md --pdf report.pdf

package:
	PYTHONPATH=src $(PYTHON) -m tkh.cpu_package --run $(OUT)

