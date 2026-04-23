# Retail Growth Intelligence System

An end-to-end retail analytics and machine learning capstone for an omnichannel retailer. The system supports marketing, CRM, merchandising, inventory planning, and risk monitoring decisions through one connected data and modeling workflow.

## Business Questions

1. Which customers are likely to churn soon?
2. Which customers should receive a retention offer for incremental impact?
3. What products should be recommended to each customer?
4. What will future demand look like at product, store, and week level?
5. Which customers, orders, stores, or SKUs show unusual behavior?
6. What customer segments exist, and what actions should the business take?

## Two-Day Sprint Scope

### In Scope

- One shared SQL mart layer powering all workstreams
- Reusable Python data loading and validation utilities
- Core EDA and statistical framing
- Baseline implementations for churn, uplift, forecasting, segmentation, recommendations, and anomaly detection
- Deployment and monitoring design document

### Out of Scope For Sprint

- Full production orchestration
- Extensive hyperparameter search across all models
- Full dashboard application implementation
- Deep causal identification beyond interview-ready framing

## Execution Stack (Phase 1 Lock)

- SQL engine: DuckDB (local analytics over CSV and Parquet)
- Python: 3.11+
- Core libraries: pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn
- Explainability: permutation importance baseline, SHAP if environment supports installation
- Notebook runtime: jupyterlab, ipykernel

## Repository Contract

- sql: raw-to-mart SQL scripts, one script per mart plus shared staging logic
- src: reusable Python modules for IO, validation, feature engineering, modeling utilities
- notebooks: stepwise analysis notebooks aligned to project modules
- docs: design docs, schema references, operating checklist, and case-study content
- tests: lightweight checks for data assumptions and utility functions
- outputs: generated charts, tables, model artifacts, and evaluation summaries

## Naming Conventions

- Mart scripts: mart_<domain>.sql
- Notebooks: NN_<module_name>.ipynb
- Python scripts: <verb>_<domain>.py
- Models: <task>_<model>_v<version>.pkl
- Charts: <module>_<metric>_<date>.png

## Quality Gate (Reusable For All Phases)

Each module is complete only if:

1. Inputs and grain are documented.
2. Join keys and null behavior are validated.
3. Baseline metric results are produced.
4. Business interpretation is written.
5. Limitations and next improvements are listed.

## Planned Build Order

1. SQL marts
2. Python cleaning and EDA
3. Statistical analysis layer
4. Churn model
5. Uplift model
6. Demand forecasting
7. Segmentation and PCA
8. Recommendation system
9. Anomaly detection
10. Deployment and monitoring design

## First Implementation Milestone

Phase 1 completion criteria:

- Stack and tools frozen
- Naming and folder contract documented
- Module-level done definition documented
- Ready to begin Phase 2 schema audit and key mapping

