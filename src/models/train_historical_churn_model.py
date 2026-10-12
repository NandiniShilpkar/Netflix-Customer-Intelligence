
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "features" / "churn_historical_snapshots.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_next_30_days"
RANDOM_STATE = 42

# Train on February and March.
TRAIN_DATES = ["2026-02-28", "2026-03-31"]

# Test on later, chronologically held-out months.
TEST_DATES = ["2026-05-31"]

# Metadata must not be used as model features.
EXCLUDED_COLUMNS = [
    "customer_id",
    TARGET,
    "snapshot_date",
    "outcome_window_end",
]


# --------------------------------------------------
# 2. Load and validate historical snapshots
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required = set(EXCLUDED_COLUMNS + ["customer_id"])
missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"], errors="raise"
)

df["outcome_window_end"] = pd.to_datetime(
    df["outcome_window_end"], errors="raise"
)

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

if df.duplicated(["customer_id", "snapshot_date"]).any():
    raise ValueError("Duplicate customer-snapshot pairs found.")

train_df = df[
    df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TRAIN_DATES)
].copy()

test_df = df[
    df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TEST_DATES)
].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing dataset is empty.")

if train_df["snapshot_date"].max() >= test_df["snapshot_date"].min():
    raise ValueError("Training dates must precede testing dates.")

# The last training outcome window must finish before the
# first test snapshot's outcome period begins.
last_train_window_end = train_df["outcome_window_end"].max()
first_test_snapshot = test_df["snapshot_date"].min()

if last_train_window_end > first_test_snapshot:
    raise ValueError(
        "Training outcome windows overlap the test period. "
        "Revisit the split before training."
    )

X_train = train_df.drop(columns=EXCLUDED_COLUMNS)
y_train = train_df[TARGET].astype(int)

X_test = test_df.drop(columns=EXCLUDED_COLUMNS)
y_test = test_df[TARGET].astype(int)

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both training and testing data must contain both target classes."
    )

print("=" * 65)
print("HISTORICAL CHURN MODEL — CHRONOLOGICAL EVALUATION")
print("=" * 65)

print("\nTraining snapshots:")
print(train_df.groupby("snapshot_date")[TARGET].agg(["count", "sum"]))

print("\nTesting snapshots:")
print(test_df.groupby("snapshot_date")[TARGET].agg(["count", "sum"]))

print(f"\nTraining rows: {len(train_df):,}")
print(f"Training positives: {int(y_train.sum()):,}")
print(f"Testing rows: {len(test_df):,}")
print(f"Testing positives: {int(y_test.sum()):,}")

print(
    "\nCustomers appearing in both train and test:",
    len(
        set(train_df["customer_id"])
        & set(test_df["customer_id"])
    ),
)


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

def make_preprocessor(scale_numeric):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

models = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

results = {}

for name, model in models.items():
    print("\n" + "=" * 65)
    print(f"MODEL: {name}")
    print("=" * 65)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    results[name] = metrics

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Use separate filenames to preserve existing baseline models.
    model_path = MODEL_DIR / f"historical_{name}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")


# --------------------------------------------------
# 7. Save a separate report
# --------------------------------------------------

report = {
    "experiment": "historical_churn_chronological_evaluation",
    "training_snapshots": TRAIN_DATES,
    "testing_snapshots": TEST_DATES,
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "training_positive_cases": int(y_train.sum()),
    "testing_positive_cases": int(y_test.sum()),
    "training_max_outcome_window_end": str(
        last_train_window_end.date()
    ),
    "first_test_snapshot": str(first_test_snapshot.date()),
    "random_state": RANDOM_STATE,
    "metrics": results,
    "caveats": [
        "Labels are provisional until complete follow-up is verified.",
        "Repeated customers across periods are allowed for an existing-customer forecasting use case.",
        "Evaluation is not a test of generalization to entirely new customers.",
        "Probability thresholds have not been tuned.",
        "The dataset is observational and does not establish causal drivers of churn.",
    ],
}

report_path = REPORT_DIR / "historical_churn_metrics.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print(f"\nSaved historical evaluation report: {report_path}")
print("\nHistorical evaluation finished.")
