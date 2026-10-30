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
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ORIGINAL_FILE = os.path.join(BASE, "data/features/churn_historical_snapshots.csv")
ADVANCED_FILE = os.path.join(BASE, "data/features/churn_advanced_snapshots.csv")
REPORT_DIR = os.path.join(BASE, "reports/advanced_features")
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET = "churn_next_30_days"
EXCLUDED = ["customer_id", "snapshot_date", "outcome_window_end", TARGET]

TRAIN_DATES = ["2026-02-28", "2026-03-31"]
TEST_DATE = "2026-05-31"


def build_model(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])


def evaluate(df, name, features):
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    train = df[df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TRAIN_DATES)]
    test = df[df["snapshot_date"].dt.strftime("%Y-%m-%d") == TEST_DATE]

    X_train = train[features]
    y_train = train[TARGET].astype(int)
    X_test = test[features]
    y_test = test[TARGET].astype(int)

    model = build_model(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    top_n = int(np.ceil(len(test) * 0.10))
    top_idx = np.argsort(probabilities)[-top_n:]
    top_churners = int(y_test.iloc[top_idx].sum())

    result = {
        "experiment": name,
        "feature_count": len(features),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "top10_precision": top_churners / top_n,
        "top10_recall": top_churners / max(1, int(y_test.sum())),
        "top10_churners_captured": top_churners
    }

    print(f"\n{name}")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    advanced = pd.read_csv(ADVANCED_FILE)

    original_features = [c for c in original.columns if c not in EXCLUDED]
    new_features = [c for c in advanced.columns if c not in original.columns]

    engagement = [
        c for c in new_features
        if any(x in c for x in [
            "viewing", "watch_", "active_days", "completion",
            "session_duration", "device", "declining_engagement"
        ])
    ]

    payment_support = [
        c for c in new_features
        if c not in engagement
    ]

    experiments = [
        ("A - Original", original, original_features),
        ("B - Original + Engagement", advanced,
         original_features + engagement),
        ("C - Original + Payment + Support", advanced,
         original_features + payment_support),
        ("D - Original + All Advanced", advanced,
         original_features + new_features)
    ]

    results = []

    for name, dataset, features in experiments:
        results.append(evaluate(dataset, name, features))

    result_df = pd.DataFrame(results)

    csv_path = os.path.join(REPORT_DIR, "feature_group_experiments.csv")
    json_path = os.path.join(REPORT_DIR, "feature_group_experiments.json")

    result_df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n========== FEATURE GROUP COMPARISON ==========")
    columns = [
        "experiment", "feature_count", "roc_auc", "pr_auc",
        "precision", "recall", "f1",
        "top10_precision", "top10_recall",
        "top10_churners_captured"
    ]
    print(result_df[columns].to_string(index=False))

    print("\nReports saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
