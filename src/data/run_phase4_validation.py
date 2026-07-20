from __future__ import annotations

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.data.feature_dictionary import feature_dictionary_df
from src.data.mart_loaders import load_all_marts
from src.data.mart_validators import summarize_validation_results, validate_all_marts


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    marts = load_all_marts(processed_dir=PROCESSED_DIR)
    validation_details = validate_all_marts(marts)
    validation_summary = summarize_validation_results(validation_details)
    feature_dict = feature_dictionary_df()

    detail_path = OUTPUTS_DIR / "phase4_validation_details.csv"
    summary_path = OUTPUTS_DIR / "phase4_validation_summary.csv"
    feature_dict_path = OUTPUTS_DIR / "phase4_feature_dictionary.csv"

    validation_details.to_csv(detail_path, index=False)
    validation_summary.to_csv(summary_path, index=False)
    feature_dict.to_csv(feature_dict_path, index=False)

    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {feature_dict_path}")


if __name__ == "__main__":
    main()
