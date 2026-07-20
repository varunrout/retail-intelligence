"""One-command build: generate synthetic raw data, then materialise the marts.

    python -m src.data.build                 # default scale
    python -m src.data.build --scale sample  # tiny, for CI/tests
    python -m src.data.build --skip-generate  # rebuild marts from existing data/raw

Writes raw tables to data/raw/, then runs sql/*.sql via DuckDB into
data/processed/ (both gitignored). After this, notebooks, model runners and the
full pytest suite run against real, reproducible data.
"""

from __future__ import annotations

import argparse

from src.data import run_phase3_marts
from src.data.generate import SCALES, Config, generate


def main() -> None:
    p = argparse.ArgumentParser(description="Generate raw data and build DuckDB marts")
    p.add_argument("--scale", choices=list(SCALES), default="default")
    p.add_argument("--n-customers", type=int, default=None)
    p.add_argument("--n-products", type=int, default=None)
    p.add_argument("--n-stores", type=int, default=None)
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-generate", action="store_true", help="reuse existing data/raw")
    args = p.parse_args()

    if not args.skip_generate:
        preset = SCALES[args.scale]
        cfg = Config(
            n_customers=args.n_customers or preset["n_customers"],
            n_products=args.n_products or preset["n_products"],
            n_stores=args.n_stores or preset["n_stores"],
            months=args.months or preset["months"],
            seed=args.seed,
        )
        generate(cfg)

    print("\nBuilding marts from data/raw …")
    run_phase3_marts.main()
    print("Build complete. Marts are in data/processed/.")


if __name__ == "__main__":
    main()
