.PHONY: build sample marts test lint format eval clean

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

# Retrain all six V2 models from the built marts and check the regenerated
# metrics against the committed outputs/ within tolerance. Needs `make build`
# (or `make marts` over an existing data/raw) first. Overwrites outputs/.
eval:
	python -m analysis.eval_reproduction_check

# Remove generated data (keeps the gitignored dirs).
clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/processed/*.duckdb data/processed/*.wal
