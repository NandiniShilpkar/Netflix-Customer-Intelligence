
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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "features" / "churn_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_next_30_days"
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs detected.")

X = df.drop(columns=[TARGET, "customer_id"])
y = df[TARGET].astype(int)

if y.nunique() != 2:
    raise ValueError("Expected a binary target with two classes.")

print("=" * 60)
print("NETFLIX CUSTOMER CHURN — BASELINE MODELING")
print("=" * 60)
print(f"Dataset rows: {len(df):,}")
print(f"Feature columns: {X.shape[1]}")
print(f"Churn cases: {int(y.sum()):,}")
print(f"Non-churn cases: {int((y == 0).sum()):,}")
print(f"Churn rate: {y.mean():.2%}")


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Note: plan_id and demographics are included as features.
# Final subscription_status, cancellation_date, churn_date,
# and churn_reason are intentionally excluded.


# --------------------------------------------------
# 4. Preprocessing pipelines
# --------------------------------------------------

numeric_logistic = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_forest = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


def build_preprocessor(numeric_pipeline):
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Training churn cases: {int(y_train.sum()):,}")
print(f"Testing churn cases: {int(y_test.sum()):,}")


# --------------------------------------------------
# 6. Define baseline models
# --------------------------------------------------

models = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_logistic)),
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
            ("preprocessor", build_preprocessor(numeric_forest)),
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
# 7. Train and evaluate
# --------------------------------------------------

results = {}

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)

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

    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")


# --------------------------------------------------
# 8. Save evaluation results
# --------------------------------------------------

results_path = REPORT_DIR / "baseline_model_metrics.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "experiment": "historical_april_2026_churn_baseline",
            "cutoff_date": "2026-03-31",
            "outcome_window_end": "2026-04-30",
            "dataset_rows": int(len(df)),
            "positive_cases": int(y.sum()),
            "negative_cases": int((y == 0).sum()),
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "metrics": results,
            "caveat": (
                "Exploratory evaluation only. The observation window "
                "for negative labels has not been independently verified."
            ),
        },
        file,
        indent=2,
    )

print("\n" + "=" * 60)
print(f"Metrics saved to: {results_path}")
print("Baseline experiment completed.")
print("=" * 60)
