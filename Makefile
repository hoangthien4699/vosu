.PHONY: help install install-cuda install-macos test lint run gate clean

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "make install-cuda   - cài dependency cho build NVIDIA/CUDA"
	@echo "make install-macos  - cài dependency cho build macOS (Apple Silicon)"
	@echo "make test           - chạy unit test"
	@echo "make lint           - ruff check"
	@echo "make run            - khởi động backend"
	@echo "make gate           - chạy Benchmark Gate (B1-B10) và in PASS/FAIL"

.venv:
	python3 -m venv .venv || uv venv --python 3.11 .venv

install-cuda: .venv
	$(PIP) install -r requirements/cuda.txt -r requirements/dev.txt

install-macos: .venv
	$(PIP) install -r requirements/macos.txt -r requirements/dev.txt

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check backend benchmarks tests

run:
	$(PY) -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

gate:
	$(PY) -m benchmarks.run_gate

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
