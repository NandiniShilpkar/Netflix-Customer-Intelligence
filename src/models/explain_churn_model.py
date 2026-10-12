
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "churn_feature_importance.csv"


def main():
    model = joblib.load(MODEL_PATH)

    if "classifier" not in model.named_steps:
        raise ValueError("Expected a pipeline step named 'classifier'.")

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    importances = classifier.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    if len(importances) != len(feature_names):
        raise ValueError("Feature names and importance values do not match.")

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nTop 15 model feature importances:")
    print(result.head(15).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
