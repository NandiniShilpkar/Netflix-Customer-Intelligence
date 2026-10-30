import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ORIGINAL_FILE = os.path.join(
    BASE, "data/features/churn_historical_snapshots.csv"
)
DATA_FILE = os.path.join(
    BASE, "data/features/churn_advanced_snapshots.csv"
)

OUT_DIR = os.path.join(BASE, "reports/model_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "churn_next_30_days"
EXCLUDED = [
    "customer_id",
    "snapshot_date",
    "outcome_window_end",
    TARGET
]

TRAIN_DATES = ["2026-01-31", "2026-02-28"]
VALIDATION_DATES = ["2026-04-30", "2026-05-31"]

WEAK_VIEWING = [
    "viewing_sessions_90d",
    "total_watch_minutes_90d",
    "active_viewing_days_90d",
    "days_since_last_viewing",
    "avg_completion_pct_90d",
    "avg_watch_minutes_90d"
]

PAYMENT_FEATURES = [
    "failed_payments_30d",
    "failed_payment_rate_30d",
    "payment_risk_flag",
    "payment_failures_change"
]


def make_preprocessor(X, scale_numeric=False):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = [
        ("numeric", Pipeline(numeric_steps), numeric)
    ]

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    return ColumnTransformer(transformers)


def make_models(X_train, y_train):
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)

    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(X_train, True)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", XGBClassifier(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=5,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=4
            ))
        ]),

        "LightGBM": Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", LGBMClassifier(
                n_estimators=400,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5,
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,
                random_state=42,
                n_jobs=4
            ))
        ]),

        "CatBoost": Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", CatBoostClassifier(
                iterations=400,
                depth=4,
                learning_rate=0.03,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=42,
                thread_count=4
            ))
        ])
    }


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    top_n = max(1, int(np.ceil(len(y_test) * 0.10)))
    top_indices = np.argsort(probabilities)[-top_n:]
    top_churners = int(y_test.iloc[top_indices].sum())

    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision_at_0.5": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_at_0.5": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_at_0.5": f1_score(
            y_test, predictions, zero_division=0
        ),
        "top10_churners": top_churners,
        "top10_recall": top_churners / max(1, int(y_test.sum()))
    }


def main():
    print("Loading advanced churn dataset...")
    original = pd.read_csv(ORIGINAL_FILE)
    df = pd.read_csv(DATA_FILE)

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["outcome_window_end"] = pd.to_datetime(
        df["outcome_window_end"]
    )

    original_features = [
        c for c in original.columns if c not in EXCLUDED
    ]

    group_a = [
        c for c in original_features if c not in WEAK_VIEWING
    ]

    group_b = list(dict.fromkeys(
        group_a + PAYMENT_FEATURES
    ))

    groups = {
        "A - Reduced original": group_a,
        "B - Reduced original + payment": group_b
    }

    train = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(TRAIN_DATES)
    ].copy()

    if train.empty:
        raise ValueError("Training data is empty.")

    latest_train_outcome = train["outcome_window_end"].max()

    print("\nTraining rows:", len(train))
    print("Training positives:", int(train[TARGET].sum()))
    print("Latest training outcome window:", latest_train_outcome.date())

    all_results = []

    for validation_date in VALIDATION_DATES:
        test = df[
            df["snapshot_date"].dt.strftime("%Y-%m-%d")
            == validation_date
        ].copy()

        if test.empty:
            raise ValueError(
                f"No validation rows found for {validation_date}"
            )

        if latest_train_outcome >= pd.Timestamp(validation_date):
            raise ValueError(
                f"Training outcome window overlaps {validation_date}"
            )

        print("\n" + "=" * 65)
        print("VALIDATION SNAPSHOT:", validation_date)
        print("Rows:", len(test))
        print("Actual churners:", int(test[TARGET].sum()))
        print(
            "Churn rate:",
            round(test[TARGET].mean() * 100, 2), "%"
        )
        print("=" * 65)

        y_train = train[TARGET].astype(int)
        y_test = test[TARGET].astype(int)

        for group_name, features in groups.items():
            missing = [
                feature for feature in features
                if feature not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing features in {group_name}: {missing}"
                )

            X_train = train[features].copy()
            X_test = test[features].copy()

            print(
                f"\nFeature group: {group_name} "
                f"({len(features)} features)"
            )

            models = make_models(X_train, y_train)

            for model_name, model in models.items():
                print("Training:", model_name)

                try:
                    metrics = evaluate_model(
                        model, X_train, y_train, X_test, y_test
                    )

                    row = {
                        "validation_date": validation_date,
                        "feature_group": group_name,
                        "model": model_name,
                        "feature_count": len(features),
                        "train_rows": len(train),
                        "validation_rows": len(test),
                        "validation_churners": int(y_test.sum()),
                        **metrics
                    }

                    all_results.append(row)

                    print(
                        f"  PR-AUC={metrics['pr_auc']:.4f} | "
                        f"ROC-AUC={metrics['roc_auc']:.4f} | "
                        f"Recall={metrics['recall_at_0.5']:.4f} | "
                        f"Top10={metrics['top10_churners']}"
                    )

                except Exception as error:
                    print("FAILED:", model_name, str(error))
                    all_results.append({
                        "validation_date": validation_date,
                        "feature_group": group_name,
                        "model": model_name,
                        "feature_count": len(features),
                        "error": str(error)
                    })

    results = pd.DataFrame(all_results)

    csv_path = os.path.join(
        OUT_DIR, "algorithm_temporal_comparison.csv"
    )

    results.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    display_columns = [
        "validation_date",
        "feature_group",
        "model",
        "roc_auc",
        "pr_auc",
        "precision_at_0.5",
        "recall_at_0.5",
        "f1_at_0.5",
        "top10_churners",
        "top10_recall"
    ]

    successful = results[
        results["pr_auc"].notna()
    ].copy()

    if not successful.empty:
        print(
            successful[display_columns]
            .round(4)
            .to_string(index=False)
        )

    summary = {
        "training_dates": TRAIN_DATES,
        "validation_dates": VALIDATION_DATES,
        "training_rows": len(train),
        "training_positive_labels": int(train[TARGET].sum()),
        "latest_training_outcome_window": str(
            latest_train_outcome.date()
        ),
        "results_file": csv_path,
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
        "note": (
            "May 31 was previously used during feature selection. "
            "Treat it as a secondary development evaluation, "
            "not an independent final test."
        )
    }

    with open(
        os.path.join(OUT_DIR, "algorithm_comparison_summary.json"),
        "w"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nSaved results:", csv_path)
    print("Saved summary: algorithm_comparison_summary.json")


if __name__ == "__main__":
    main()
