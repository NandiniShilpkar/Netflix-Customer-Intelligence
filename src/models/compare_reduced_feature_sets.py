import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ORIGINAL_FILE = os.path.join(
    BASE, "data/features/churn_historical_snapshots.csv"
)
ADVANCED_FILE = os.path.join(
    BASE, "data/features/churn_advanced_snapshots.csv"
)

OUT = os.path.join(BASE, "reports/advanced_features")
os.makedirs(OUT, exist_ok=True)

TARGET = "churn_next_30_days"
EXCLUDED = [
    "customer_id", "snapshot_date", "outcome_window_end", TARGET
]

TRAIN_DATES = ["2026-01-31", "2026-02-28"]
TEST_DATE = "2026-04-30"

WEAK_VIEWING = [
    "viewing_sessions_90d",
    "total_watch_minutes_90d",
    "active_viewing_days_90d",
    "days_since_last_viewing",
    "avg_completion_pct_90d",
    "avg_watch_minutes_90d"
]

PAYMENT_FEATURES = [
    "successful_payment_count",
    "failed_payments_30d",
    "failed_payment_rate_30d",
    "payment_risk_flag",
    "payment_failures_change"
]

SUPPORT_FEATURES = [
    "support_tickets_previous_30d"
]


def build_model(X):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

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

    train = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TRAIN_DATES)
    ]

    test = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d") == TEST_DATE
    ]

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

    return {
        "experiment": name,
        "feature_count": len(features),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "top10_churners": top_churners,
        "top10_recall": top_churners / max(1, int(y_test.sum()))
    }


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    advanced = pd.read_csv(ADVANCED_FILE)

    original_features = [
        c for c in original.columns if c not in EXCLUDED
    ]

    # Group A: original features minus weak viewing features.
    group_a = [
        c for c in original_features if c not in WEAK_VIEWING
    ]

    # Group B: add selected payment features not already present.
    group_b = list(dict.fromkeys(
        group_a + PAYMENT_FEATURES
    ))

    # Group C: add selected support features not already present.
    group_c = list(dict.fromkeys(
        group_b + SUPPORT_FEATURES
    ))

    experiments = [
        ("A - Original minus weak viewing", group_a),
        ("B - Add selected payment", group_b),
        ("C - Add selected support", group_c)
    ]

    results = []

    for name, features in experiments:
        missing = [c for c in features if c not in advanced.columns]

        if missing:
            raise ValueError(f"Missing features: {missing}")

        print(f"\nRunning {name}")
        print("Feature count:", len(features))
        print("Added features:", [
            c for c in features if c not in group_a
        ])

        results.append(evaluate(advanced, name, features))

    result_df = pd.DataFrame(results)

    print("\n========== CORRECTED FEATURE COMPARISON ==========")
    print(result_df.round(6).to_string(index=False))

    output = os.path.join(OUT, "temporal_validation_april.csv")
    result_df.to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
