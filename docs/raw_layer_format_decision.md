# Decision: keep `data/raw/` as CSV, not partitioned parquet

An earlier finish-off plan (`FIXES.md`, RETA-05) proposed committing the raw
generator output as partitioned parquet under `data/raw/` instead of CSV.
That plan was written before the generator existed. The generator that was
actually built and merged (`src/data/generate.py`, PR #27) writes 16 flat CSVs
plus `returns_hidden_labels.csv`, and `sql/*.sql` reads them via DuckDB's
`read_csv`. This note records the decision to keep that, rather than convert.

## Why not switch to parquet

- **No working problem to fix.** The stated motivation was "no committed,
  defined on-disk format" — that's resolved: the generator's schema and
  output format are both committed and covered by mart validators
  (`src/data/mart_validators.py`). Partitioning by month only pays off at
  data volumes or read patterns this project doesn't have; at `full` scale
  (50k customers, ~530k orders) DuckDB reads the CSVs directly into marts in
  well under a minute.
- **Real cost.** Converting means rewriting the generator's I/O, every
  `sql/*.sql` source reference, `mart_loaders.py`, and the raw-file existence
  checks in `raw_data_exists()` — touching code that just shipped and is
  covered by CI. That's a meaningful diff for a format change with no
  measured benefit here.
- **CSV is more inspectable.** Being able to `head`/grep a raw table directly
  is worth something for a portfolio/demo project, and outweighs parquet's
  columnar-read advantage at this scale.

## When to revisit

If the generated dataset grows enough that DuckDB's CSV scan becomes the
bottleneck in `make build`, or if a partitioned/incremental generation mode
is needed (e.g. appending new months without rewriting the whole table),
parquet under `data/raw/<table>/year=YYYY/month=MM/part-*.parquet` is still
the right target layout, as originally scoped in RETA-05.
