.PHONY: help install test lint data features train evaluate causal report all clean

PYTHON ?= python

help:
	@echo "crypto-foundation-forecast CLI"
	@echo "Available targets:"
	@echo "  make install    - Install dependencies and local package in editable mode"
	@echo "  make test       - Run all test suites (anti-leakage, targets, availability, risk)"
	@echo "  make lint       - Run ruff linter and code formatting checks"
	@echo "  make data       - Download public Binance market data and clean to parquet"
	@echo "  make features   - Generate feature store with point-in-time availability"
	@echo "  make train      - Train classical baselines and foundation model heads"
	@echo "  make evaluate   - Run out-of-sample purged validation, VaR/ES scoring, DSR/PBO"
	@echo "  make causal     - Run event study, double ML, RDD, and transfer entropy"
	@echo "  make report     - Build summary tables, calibration plots, and performance charts"
	@echo "  make all        - Execute pipeline end-to-end and reproduce main figures"
	@echo "  make clean      - Clean temporary files and build caches"

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

data:
	$(PYTHON) -m cff.data.download --config configs/data.yaml
	$(PYTHON) -m cff.data.clean --config configs/data.yaml

features:
	$(PYTHON) -m cff.features.build --config configs/features.yaml

train:
	$(PYTHON) -m cff.train --config configs/model.yaml --grid configs/grid.yaml

evaluate:
	$(PYTHON) -m cff.evaluate --config configs/eval.yaml

causal:
	$(PYTHON) -m cff.causal.run --config configs/eval.yaml

report:
	$(PYTHON) -m cff.plots
	$(PYTHON) -m cff.report --output results/
	$(PYTHON) scripts/build_notebook.py

all: data features train evaluate causal report

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov
