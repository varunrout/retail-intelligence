# Phase 1 Operating Setup

This document defines how the sprint will operate before data mart and modeling implementation starts.

## Objective

Create a stable operating baseline so all downstream modules are consistent, explainable, and interview-ready.

## Entry Criteria

- Raw synthetic retail data is available under data/raw.
- Processed inventory artifact exists under data/processed/row_counts.csv.
- Team has agreed to a two-day sprint objective focused on breadth with coherent baselines.

## Exit Criteria

Phase 1 is complete only if all items are true:

1. Stack is locked for SQL and Python.
2. Folder contract is documented and accepted.
3. Naming conventions are documented and accepted.
4. Quality gate checklist is documented.
5. Module-to-business-question mapping exists.

## Stack Lock

- SQL engine: DuckDB
- Python version: 3.11+
- Python libraries: defined in requirements.txt
- Explainability standard: permutation importance by default, SHAP only if environment-compatible
- Notebook workflow: JupyterLab

## Module To Question Mapping

1. Churn prediction and retention prioritization
- Questions addressed: 1 and 2
- Core mart dependency: customer feature mart, campaign response mart

2. Recommendation engine
- Questions addressed: 3
- Core mart dependency: recommendation interaction mart

3. Demand forecasting
- Questions addressed: 4
- Core mart dependency: product demand mart, store-week performance mart

4. Anomaly detection for returns and transactions
- Questions addressed: 5
- Core mart dependency: returns risk mart, store-week performance mart

5. Segmentation and action design
- Questions addressed: 6
- Core mart dependency: customer feature mart

## Naming Standards

- SQL marts: mart_<domain>.sql
- Python module files: <verb>_<domain>.py
- Notebooks: NN_<module_name>.ipynb
- Model artifacts: <task>_<model>_v<version>.pkl
- Chart artifacts: <module>_<metric>_<date>.png
- Evaluation tables: <module>_<evaluation_type>.csv

## Quality Gates

Use these checks after each major module:

1. Data Integrity
- Required keys are present.
- Join cardinality assumptions are validated.
- Null rates on critical columns are reported.

2. Statistical Integrity
- Baseline summary and distribution checks are present.
- Assumptions for tests or models are clearly stated.
- Uncertainty is reported where relevant.

3. Modeling Integrity
- Leakage checks are documented.
- Baseline models are compared with at least one stronger model.
- Validation split strategy is justified.

4. Business Integrity
- Decision threshold is explicitly stated.
- Tradeoff discussion is included.
- Recommended action and owner are specified.

## AI Usage Policy For This Sprint

- AI can generate boilerplate SQL and Python scaffolding.
- All business logic, assumptions, and leakage checks must be manually reviewed.
- Every model output used in conclusions must be reproducible from saved scripts or notebooks.

## Immediate Next Step

Proceed to Phase 2: raw data audit and schema mapping.
