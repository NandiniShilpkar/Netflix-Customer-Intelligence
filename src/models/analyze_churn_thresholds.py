import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve
)

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ORIGINAL_FILE = os.path.join(
    BASE, "data/features/churn_historical_snapshots.csv"
)
ADVANCED_FILE = os.path.join(
    BASE, "data/features/churn_advanced_snapshots.csv"
)

OUT = os.path.join(BASE, "reports/threshold_analysis")
os.makedirs(OUT, exist_ok=True)

TARGET = "churn_next_30_days"
TRAIN_DATES = ["2026-01-31", "2026-02-28"]
VALIDATION_DATE = "2026-04-30"

EXCLUDED = [
    "customer_id", "snapshot_date",
    "outcome_window_end", TARGET
]

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


def make_preprocessor(X):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    transformers = [
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric
        )
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


def build_candidates():
    return {
        "LightGBM_A": (
            "lightgbm",
            LGBMClassifier(
                n_estimators=400,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5,
                scale_pos_weight=(
                (len(y_train) - int(y_train.sum()))
                / max(int(y_train.sum()), 1)
            ),
                verbosity=-1,
                random_state=42,
                n_jobs=4
            )
        ),
        "CatBoost_A": (
            "catboost",
            CatBoostClassifier(
                iterations=400,
                depth=4,
                learning_rate=0.03,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=42,
                thread_count=4
            )
        ),
        "CatBoost_B": (
            "catboost",
            CatBoostClassifier(
                iterations=400,
                depth=4,
                learning_rate=0.03,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=42,
                thread_count=4
            )
        )
    }


def calculate_threshold_metrics(y_true, probabilities, model_name):
    rows = []

    for threshold in np.arange(0.05, 0.951, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "model": model_name,
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "customers_flagged": int(predictions.sum())
        })

    return rows


def main():
    print("Loading datasets...")

    original = pd.read_csv(ORIGINAL_FILE)
    advanced = pd.read_csv(ADVANCED_FILE)

    original["snapshot_date"] = pd.to_datetime(
        original["snapshot_date"]
    )
    advanced["snapshot_date"] = pd.to_datetime(
        advanced["snapshot_date"]
    )
    advanced["outcome_window_end"] = pd.to_datetime(
        advanced["outcome_window_end"]
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

    train = advanced[
        advanced["snapshot_date"].isin(
            pd.to_datetime(TRAIN_DATES)
        )
    ].copy()

    valid = advanced[
        advanced["snapshot_date"] == pd.Timestamp(VALIDATION_DATE)
    ].copy()

    if train.empty or valid.empty:
        raise ValueError("Training or validation data is empty.")

    latest_outcome = train["outcome_window_end"].max()

    if latest_outcome >= pd.Timestamp(VALIDATION_DATE):
        raise ValueError("Training outcome windows overlap validation.")

    y_train = train[TARGET].astype(int)
    y_valid = valid[TARGET].astype(int)

    print("\nTraining rows:", len(train))
    print("Validation rows:", len(valid))
    print("Training positives:", int(y_train.sum()))
    print("Validation positives:", int(y_valid.sum()))
    print("Latest training outcome:", latest_outcome.date())
    print("Validation date:", VALIDATION_DATE)
    print("Group A features:", len(group_a))
    print("Group B features:", len(group_b))

    experiments = [
        ("LightGBM_A", group_a),
        ("CatBoost_A", group_a),
        ("CatBoost_B", group_b)
    ]

    all_thresholds = []
    all_predictions = []
    all_calibration = []
    summaries = []

    for model_name, features in experiments:
        print("\nTraining:", model_name)

        X_train = train[features].copy()
        X_valid = valid[features].copy()

        preprocessor = make_preprocessor(X_train)

        if model_name == "LightGBM_A":
            model = LGBMClassifier(
                n_estimators=400,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5,
                scale_pos_weight=(
                    (len(y_train) - int(y_train.sum()))
                    / max(int(y_train.sum()), 1)
                ),
                verbosity=-1,
                random_state=42,
                n_jobs=4
            )
        else:
            model = CatBoostClassifier(
                iterations=400,
                depth=4,
                learning_rate=0.03,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=42,
                thread_count=4
            )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_valid)[:, 1]

        ap = average_precision_score(y_valid, probabilities)
        auc = roc_auc_score(y_valid, probabilities)
        brier = brier_score_loss(y_valid, probabilities)

        print(
            f"PR-AUC: {ap:.4f} | ROC-AUC: {auc:.4f} "
            f"| Brier: {brier:.4f}"
        )

        threshold_rows = calculate_threshold_metrics(
            y_valid, probabilities, model_name
        )
        all_thresholds.extend(threshold_rows)

        # Save customer-level validation predictions.
        prediction_df = pd.DataFrame({
            "customer_id": valid["customer_id"].values,
            "snapshot_date": VALIDATION_DATE,
            "actual_churn": y_valid.values,
            "predicted_churn_probability": probabilities,
            "model": model_name
        })

        prediction_df["risk_percentile"] = (
            prediction_df["predicted_churn_probability"]
            .rank(pct=True, ascending=True)
        )

        prediction_df["risk_group"] = pd.cut(
            prediction_df["risk_percentile"],
            bins=[0, 0.50, 0.80, 0.90, 0.95, 1.0],
            labels=[
                "Bottom 50%",
                "50-80%",
                "80-90%",
                "90-95%",
                "Top 5%"
            ],
            include_lowest=True
        )

        all_predictions.append(prediction_df)

        # Calibration by equal-width probability bins.
        calibration_df = pd.DataFrame({
            "probability": probabilities,
            "actual": y_valid.values
        })

        calibration_df["bin"] = pd.cut(
            calibration_df["probability"],
            bins=np.linspace(0, 1, 11),
            include_lowest=True
        )

        calibration = calibration_df.groupby(
            "bin", observed=True
        ).agg(
            customers=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            actual_churn_rate=("actual", "mean")
        ).reset_index()

        calibration["model"] = model_name
        all_calibration.append(calibration)

        summaries.append({
            "model": model_name,
            "feature_count": len(features),
            "pr_auc": ap,
            "roc_auc": auc,
            "brier_score": brier,
            "actual_churn_rate": float(y_valid.mean()),
            "mean_predicted_probability": float(probabilities.mean())
        })

        # Precision-recall curve.
        precision, recall, _ = precision_recall_curve(
            y_valid, probabilities
        )

        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision)
        plt.axhline(
            y_valid.mean(),
            linestyle="--",
            label="Baseline churn rate"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT, f"{model_name}_pr_curve.png"),
            dpi=150
        )
        plt.close()

    threshold_df = pd.DataFrame(all_thresholds)
    prediction_df = pd.concat(all_predictions, ignore_index=True)
    calibration_df = pd.concat(all_calibration, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    threshold_df.to_csv(
        os.path.join(OUT, "threshold_metrics.csv"), index=False
    )
    prediction_df.to_csv(
        os.path.join(OUT, "april_customer_predictions.csv"), index=False
    )
    calibration_df.to_csv(
        os.path.join(OUT, "probability_calibration.csv"), index=False
    )
    summary_df.to_csv(
        os.path.join(OUT, "model_probability_summary.csv"), index=False
    )

    print("\n========== MODEL SUMMARY ==========")
    print(summary_df.round(4).to_string(index=False))

    print("\n========== THRESHOLD RESULTS ==========")

    for model_name in summary_df["model"]:
        subset = threshold_df[
            threshold_df["model"] == model_name
        ]

        print(f"\n{model_name}")
        print(
            subset[[
                "threshold", "precision", "recall", "f1",
                "customers_flagged", "true_positives",
                "false_positives"
            ]].round(3).to_string(index=False)
        )

    print("\n========== RISK GROUP RESULTS ==========")

    for model_name in summary_df["model"]:
        subset = prediction_df[
            prediction_df["model"] == model_name
        ]

        risk_summary = subset.groupby(
            "risk_group", observed=True
        ).agg(
            customers=("actual_churn", "size"),
            actual_churners=("actual_churn", "sum"),
            actual_churn_rate=("actual_churn", "mean")
        )

        print(f"\n{model_name}")
        print(risk_summary.round(4).to_string())

    print("\nFiles saved in:", OUT)


if __name__ == "__main__":
    main()
