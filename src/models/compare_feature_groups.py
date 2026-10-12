
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "features" / "churn_features.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_next_30_days"

df = pd.read_csv(DATA_PATH)
y = df[TARGET].astype(int)

# Customer ID is excluded from all model inputs.
X = df.drop(columns=[TARGET, "customer_id"])

# Compare meaningful feature groups.
feature_groups = {
    "Demographics and plan": [
        "age", "gender", "country", "city",
        "preferred_language", "plan_id",
        "subscription_tenure_days",
    ],
    "Viewing behavior": [
        "viewing_sessions_90d", "active_viewing_days_90d",
        "total_watch_minutes_90d", "avg_watch_minutes_90d",
        "avg_completion_pct_90d", "days_since_last_viewing",
    ],
    "Payment behavior": [
        "payment_count", "failed_payment_count",
        "successful_payment_count", "avg_payment_amount",
        "failed_payment_rate",
    ],
    "Support history": [
        "support_ticket_count", "high_priority_ticket_count",
        "unresolved_ticket_count", "avg_support_satisfaction",
        "avg_resolution_hours", "has_support_tickets",
    ],
    "All features": list(X.columns),
}

# Keep one consistent stratified holdout for every comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

results = []

for group_name, columns in feature_groups.items():
    missing_columns = sorted(set(columns) - set(X.columns))
    if missing_columns:
        raise ValueError(
            f"Missing columns for {group_name}: {missing_columns}"
        )

    train_group = X_train[columns].copy()
    test_group = X_test[columns].copy()

    categorical = train_group.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()
    numeric = [
        col for col in columns if col not in categorical
    ]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        )
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(train_group, y_train)
    probabilities = model.predict_proba(test_group)[:, 1]

    ap = average_precision_score(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    results.append(
        {
            "feature_group": group_name,
            "feature_count": len(columns),
            "average_precision": ap,
            "roc_auc": auc,
        }
    )

    print(
        f"{group_name}: "
        f"features={len(columns)}, "
        f"average_precision={ap:.4f}, "
        f"roc_auc={auc:.4f}"
    )

results_df = pd.DataFrame(results).sort_values(
    "average_precision",
    ascending=False,
)

output_path = REPORT_DIR / "feature_group_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nResults ranked by average precision:")
print(results_df.to_string(index=False))
print(f"\nSaved to: {output_path}")

print(
    "\nCaution: these are exploratory feature comparisons. "
    "The observation window for negative labels is not yet verified."
)
