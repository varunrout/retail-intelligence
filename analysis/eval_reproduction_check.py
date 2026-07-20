"""Offline eval harness — regenerates V2 metrics from the built marts and
checks them against the committed ``outputs/`` files.

This is a local reproducibility check, not a CI gate: it retrains all six V2
models (LightGBM x4, an X-learner, and an SVD), which takes real time, and it
needs the `default` or `full` scale build to compare against numbers of the
right order of magnitude (CI only builds `sample` scale for speed, which
would report every metric as a large "mismatch" for reasons that have nothing
to do with a regression).

**This overwrites `outputs/*_metrics.json` / `outputs/*_model_comparison.csv`
and the other files each runner writes.** That is the point — it proves the
committed numbers are actually reproducible from the pipeline, not just typed
into a CSV once. Review the diff after running it.

Tolerance: the generator is seeded and models are seeded, but the currently
committed `data/raw` is not guaranteed to be byte-identical to whatever
produced the committed `outputs/` (see README "Limitations"). So this checks
metrics land within a relative tolerance of the committed value, not an exact
match — a metric outside tolerance is a WARN worth looking at, not
automatically a bug.

Run:
    make eval
    python -m analysis.eval_reproduction_check
    python -m analysis.eval_reproduction_check --only churn anomaly
    python -m analysis.eval_reproduction_check --tolerance 0.15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd

from src.config import OUTPUTS_DIR

DEFAULT_TOLERANCE = 0.25  # relative; see module docstring for why this is loose


@dataclass
class Workstream:
    name: str
    runner_module: str
    metric_file: str  # relative to OUTPUTS_DIR
    file_kind: str  # "csv" or "json"
    metrics: list[str]  # keys/columns to compare
    row_filter: tuple[str, str] | None = None  # (column, value) for csv row selection


WORKSTREAMS: list[Workstream] = [
    Workstream(
        name="churn",
        runner_module="src.data.run_phase_churn_v2",
        metric_file="phase_churn_v2_model_comparison.csv",
        file_kind="csv",
        row_filter=("model", "lightgbm_v2"),
        metrics=["roc_auc", "pr_auc", "brier_score"],
    ),
    Workstream(
        name="uplift",
        runner_module="src.data.run_phase_uplift_v2",
        metric_file="phase_uplift_v2_model_comparison.csv",
        file_kind="csv",
        row_filter=("model", "xlearner_v2"),
        metrics=["overall_ate_test", "qini_like_area"],
    ),
    Workstream(
        name="forecast",
        runner_module="src.data.run_phase_forecast_v2",
        metric_file="phase_forecast_v2_model_comparison.csv",
        file_kind="csv",
        row_filter=("model", "lgbm_tweedie_v2"),
        metrics=["mae", "rmse", "smape"],
    ),
    Workstream(
        name="segmentation",
        runner_module="src.data.run_phase_segmentation_v2",
        metric_file="phase_segmentation_v2_metrics.json",
        file_kind="json",
        metrics=["silhouette_buyers", "churn_spread_pp"],
    ),
    Workstream(
        name="recsys",
        runner_module="src.data.run_phase_recsys_v2",
        metric_file="phase_recsys_v2_metrics.json",
        file_kind="json",
        metrics=["hit_rate_at_10", "mrr_at_10"],
    ),
    Workstream(
        name="anomaly",
        runner_module="src.data.run_phase_anomaly_v2",
        metric_file="phase_anomaly_v2_metrics.json",
        file_kind="json",
        metrics=["ap", "precision", "recall"],
    ),
]


def _read_metrics(ws: Workstream) -> dict[str, float]:
    path = OUTPUTS_DIR / ws.metric_file
    if not path.exists():
        return {}
    if ws.file_kind == "json":
        with open(path) as f:
            data = json.load(f)
        return {k: float(data[k]) for k in ws.metrics if k in data}
    df = pd.read_csv(path)
    if ws.row_filter is not None:
        col, val = ws.row_filter
        df = df[df[col] == val]
    if df.empty:
        return {}
    row = df.iloc[0]
    return {k: float(row[k]) for k in ws.metrics if k in row}


def _run_workstream(ws: Workstream) -> None:
    print(f"  running python -m {ws.runner_module} ...")
    result = subprocess.run(
        [sys.executable, "-m", ws.runner_module],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"{ws.runner_module} exited with code {result.returncode}")


def main(only: list[str] | None = None, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    workstreams = [w for w in WORKSTREAMS if only is None or w.name in only]
    if not workstreams:
        raise ValueError(f"No workstreams matched --only {only}")

    rows = []
    all_within_tolerance = True

    for ws in workstreams:
        print(f"[{ws.name}]")
        before = _read_metrics(ws)
        _run_workstream(ws)
        after = _read_metrics(ws)

        for metric in ws.metrics:
            b = before.get(metric)
            a = after.get(metric)
            if b is None or a is None:
                rows.append(
                    {
                        "workstream": ws.name,
                        "metric": metric,
                        "committed": b,
                        "regenerated": a,
                        "rel_delta": None,
                        "status": "MISSING",
                    }
                )
                all_within_tolerance = False
                continue
            rel_delta = abs(a - b) / abs(b) if b != 0 else abs(a - b)
            within = rel_delta <= tolerance
            all_within_tolerance = all_within_tolerance and within
            rows.append(
                {
                    "workstream": ws.name,
                    "metric": metric,
                    "committed": round(b, 4),
                    "regenerated": round(a, 4),
                    "rel_delta": round(rel_delta, 4),
                    "status": "OK" if within else "WARN",
                }
            )

    report = pd.DataFrame(rows)
    print()
    print(report.to_string(index=False))
    print()
    n_warn = (report["status"] != "OK").sum()
    if n_warn:
        print(
            f"{n_warn} metric(s) outside +/-{tolerance:.0%} tolerance — worth a look, but not "
            "necessarily a bug (see module docstring: regenerated data is not guaranteed "
            "byte-identical to what produced the committed outputs)."
        )
    else:
        print(f"All metrics within +/-{tolerance:.0%} tolerance.")
    return all_within_tolerance


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        nargs="+",
        choices=[w.name for w in WORKSTREAMS],
        default=None,
        help="run only these workstreams (default: all six)",
    )
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = p.parse_args()
    ok = main(only=args.only, tolerance=args.tolerance)
    raise SystemExit(0 if ok else 1)
