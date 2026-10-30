import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ORIGINAL_FILE = os.path.join(
    BASE, "data/features/churn_historical_snapshots.csv"
)
ADVANCED_FILE = os.path.join(
    BASE, "data/features/churn_advanced_snapshots.csv"
)

OUTPUT_DIR = os.path.join(BASE, "reports/advanced_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "churn_next_30_days"
EXCLUDED = [
    "customer_id", "snapshot_date",
    "outcome_window_end", TARGET
]

TRAIN_DATES = ["2026-02-28", "2026-03-31"]
TEST_DATE = "2026-05-31"


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    advanced = pd.read_csv(ADVANCED_FILE)

    new_features = [
        c for c in advanced.columns if c not in original.columns
    ]

    engagement = [
        c for c in new_features
        if any(x in c for x in [
            "viewing", "watch_", "active_days",
            "completion", "session_duration",
            "device", "declining_engagement"
        ])
    ]

    selected_features = [
        c for c in advanced.columns
        if c not in EXCLUDED + engagement
    ]

    df = advanced.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    train = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TRAIN_DATES)
    ]
    test = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d") == TEST_DATE
    ]

    X_train = train[selected_features]
    y_train = train[TARGET].astype(int)
    X_test = test[selected_features]
    y_test = test[TARGET].astype(int)

    numeric = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
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

    print("Training selected feature-group model...")
    model.fit(X_train, y_train)

    baseline_probabilities = model.predict_proba(X_test)[:, 1]
    baseline_ap = average_precision_score(
        y_test, baseline_probabilities
    )

    print("Baseline test PR-AUC:", round(baseline_ap, 6))
    print("Calculating permutation importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "feature": selected_features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(
        "importance_mean", ascending=False
    )

    output = os.path.join(
        OUTPUT_DIR, "advanced_permutation_importance.csv"
    )

    importance.to_csv(output, index=False)

    print("\nTOP 20 FEATURES:")
    print(importance.head(20).round(6).to_string(index=False))

    print("\nFeatures with negative or zero importance:")
    print(int((importance["importance_mean"] <= 0).sum()))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
