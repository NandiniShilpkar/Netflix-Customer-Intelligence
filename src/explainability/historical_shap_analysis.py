from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/features/churn_historical_snapshots.csv"
MODEL_PATH = ROOT / "models/historical_random_forest.joblib"

OUTPUT_DIR = ROOT / "reports/explainability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_next_30_days"

EXCLUDED_COLUMNS = [
    "customer_id",
    TARGET,
    "snapshot_date",
    "outcome_window_end",
]

RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 1000
TOP_CUSTOMERS = 50


def main():

    print("=" * 65)
    print("HISTORICAL CHURN — SHAP EXPLAINABILITY")
    print("=" * 65)

    # 1. Load data and model
    df = pd.read_csv(DATA_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    test = df[
        df["snapshot_date"] == pd.Timestamp("2026-05-31")
    ].copy().reset_index(drop=True)

    model = joblib.load(MODEL_PATH)

    X_test = test.drop(columns=EXCLUDED_COLUMNS)
    customer_ids = test["customer_id"]

    print("May test customers:", len(test))

    # 2. Transform using the model's fitted preprocessor
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_test)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=test.index
    )

    # 3. Select a reproducible sample for global explanations
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_transformed))

    sample = X_transformed.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    print("SHAP sample size:", len(sample))

    # 4. Calculate SHAP values
    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(classifier)

    raw_values = explainer.shap_values(sample)

    if isinstance(raw_values, list):
        shap_values = raw_values[1]
    else:
        shap_values = np.asarray(raw_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    expected_value = np.asarray(explainer.expected_value)

    if expected_value.ndim > 0:
        expected_value = expected_value.reshape(-1)[-1]

    expected_value = float(expected_value)

    print("SHAP calculation completed.")

    # 5. Global feature importance
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "global_shap_feature_importance.csv",
        index=False
    )

    print("\nTop 15 global SHAP features:")
    print(importance.head(15).to_string(index=False))

    # 6. SHAP summary plot
    shap.summary_plot(
        shap_values,
        sample,
        show=False,
        max_display=20
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()

    # 7. SHAP bar plot
    shap.summary_plot(
        shap_values,
        sample,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_feature_importance.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()

    # 8. Explain highest-risk customers
    print("\nGenerating individual customer explanations...")

    risk_scores = model.predict_proba(X_test)[:, 1]

    top_indices = np.argsort(risk_scores)[::-1][
        :min(TOP_CUSTOMERS, len(test))
    ]

    top_customers = X_transformed.iloc[top_indices]

    individual_raw = explainer.shap_values(top_customers)

    if isinstance(individual_raw, list):
        individual_values = individual_raw[1]
    else:
        individual_values = np.asarray(individual_raw)

        if individual_values.ndim == 3:
            individual_values = individual_values[:, :, 1]

    rows = []

    for position, original_index in enumerate(top_indices):

        customer_shap = individual_values[position]

        top_features = np.argsort(
            np.abs(customer_shap)
        )[::-1][:5]

        for feature_index in top_features:

            value = float(customer_shap[feature_index])

            rows.append({
                "customer_id": customer_ids.iloc[original_index],
                "risk_score": float(risk_scores[original_index]),
                "actual_churn": int(
                    test[TARGET].iloc[original_index]
                ),
                "feature": feature_names[feature_index],
                "transformed_feature_value": float(
                    top_customers.iloc[position, feature_index]
                ),
                "shap_value": value,
                "contribution": (
                    "Increases model churn score"
                    if value > 0
                    else "Decreases model churn score"
                ),
            })

    individual_report = pd.DataFrame(rows)

    individual_report.to_csv(
        OUTPUT_DIR / "top_customer_shap_explanations.csv",
        index=False
    )

    # 9. Save overall run information
    summary = {
        "model": "historical_random_forest",
        "snapshot_date": "2026-05-31",
        "test_customers": len(test),
        "shap_sample_size": sample_size,
        "customers_individually_explained": len(top_indices),
        "expected_model_output": expected_value,
        "shap_version": shap.__version__,
        "interpretation": (
            "SHAP values explain model output contributions; "
            "they do not establish causation."
        ),
    }

    pd.Series(summary).to_json(
        OUTPUT_DIR / "shap_run_summary.json",
        indent=4
    )

    print("\nSHAP analysis completed successfully.")

    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.glob("*")):
        print(path.name, "-", path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()