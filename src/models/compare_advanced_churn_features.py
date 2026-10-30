import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ORIGINAL_FILE = os.path.join(
    BASE_DIR, "data/features/churn_historical_snapshots.csv"
)
ADVANCED_FILE = os.path.join(
    BASE_DIR, "data/features/churn_advanced_snapshots.csv"
)
REPORT_DIR = os.path.join(BASE_DIR, "reports/advanced_features")
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET = "churn_next_30_days"
ID_COLS = ["customer_id", "snapshot_date", "outcome_window_end"]

TRAIN_DATES = ["2026-02-28", "2026-03-31"]
TEST_DATE = "2026-05-31"


def make_model(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    return model


def evaluate_dataset(df, dataset_name):
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    train = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TRAIN_DATES)
    ].copy()

    test = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d") == TEST_DATE
    ].copy()

    feature_cols = [
        col for col in df.columns
        if col not in ID_COLS + [TARGET]
    ]

    X_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    X_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(f"{dataset_name}: target must contain both classes")

    model = make_model(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Top 10% risk ranking
    top_n = max(1, int(np.ceil(len(test) * 0.10)))
    top_indices = np.argsort(probabilities)[-top_n:]
    top_churners = int(y_test.iloc[top_indices].sum())

    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "dataset": dataset_name,
        "feature_count": len(feature_cols),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "top_10_percent_customers": top_n,
        "top_10_percent_churners": top_churners,
        "top_10_percent_precision": float(top_churners / top_n),
        "top_10_percent_recall": float(
            top_churners / max(1, int(y_test.sum()))
        ),
        "confusion_matrix": matrix.tolist()
    }

    print(f"\n--- {dataset_name} ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    advanced = pd.read_csv(ADVANCED_FILE)

    # Ensure both datasets represent the same observations.
    original_keys = set(zip(
        original["customer_id"], original["snapshot_date"]
    ))
    advanced_keys = set(zip(
        advanced["customer_id"], advanced["snapshot_date"]
    ))

    if original_keys != advanced_keys:
        raise ValueError("Original and advanced datasets have different records")

    if not original[TARGET].reset_index(drop=True).equals(
        advanced[TARGET].reset_index(drop=True)
    ):
        raise ValueError("Target labels differ between datasets")

    print("Original rows:", len(original))
    print("Advanced rows:", len(advanced))
    print("Training snapshots:", TRAIN_DATES)
    print("Testing snapshot:", TEST_DATE)

    results = [
        evaluate_dataset(original, "Original features"),
        evaluate_dataset(advanced, "Advanced features")
    ]

    comparison = pd.DataFrame(results)

    comparison_file = os.path.join(
        REPORT_DIR, "feature_model_comparison.csv"
    )
    json_file = os.path.join(
        REPORT_DIR, "feature_model_comparison.json"
    )

    comparison.to_csv(comparison_file, index=False)

    with open(json_file, "w") as file:
        json.dump(results, file, indent=4)

    print("\n--- FINAL COMPARISON ---")
    columns = [
        "dataset", "feature_count", "roc_auc", "pr_auc",
        "precision", "recall", "f1",
        "top_10_percent_precision", "top_10_percent_recall"
    ]
    print(comparison[columns].to_string(index=False))

    print("\nReports saved:")
    print(comparison_file)
    print(json_file)


if __name__ == "__main__":
    main()
