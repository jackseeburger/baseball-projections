.PHONY: setup data test clean

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) run_pipeline.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/raw/*.parquet data/parquet/*.parquet data/processed/*.parquet
	rm -f *.log
	@echo "Data files cleaned. Run 'make data' to re-fetch."
