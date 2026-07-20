.PHONY: build sample marts test lint format clean

# Full default-scale synthetic dataset + marts.
build:
	python -m src.data.build --scale default

# Tiny dataset for CI / quick local checks.
sample:
	python -m src.data.build --scale sample

# Rebuild marts from an existing data/raw (no regeneration).
marts:
	python -m src.data.build --skip-generate

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

# Remove generated data (keeps the gitignored dirs).
clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/processed/*.duckdb data/processed/*.wal
