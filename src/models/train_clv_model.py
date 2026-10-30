from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(".")
DATA = ROOT / "data/processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports/clv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2026-05-31")
RANDOM_STATE = 42


def load_data():
    names = [
        "customers",
        "payments",
        "subscriptions",
        "viewing_activity",
        "support_tickets",
        "customer_feedback",
    ]
    return {
        name: pd.read_csv(DATA / f"{name}.csv")
        for name in names
    }


def build_dataset(data):
    customers = data["customers"].copy()
    payments = data["payments"].copy()
    subscriptions = data["subscriptions"].copy()
    viewing = data["viewing_activity"].copy()
    tickets = data["support_tickets"].copy()
    feedback = data["customer_feedback"].copy()

    for frame, columns in [
        (payments, ["payment_date"]),
        (subscriptions, ["subscription_start_date", "subscription_end_date"]),
        (viewing, ["viewing_date"]),
        (tickets, ["ticket_date"]),
        (feedback, ["feedback_date"]),
    ]:
        for col in columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")

    # Target: successful payments up to the historical cutoff.
    successful = payments[
        (payments["payment_status"].astype(str).str.strip().str.lower() == "success")
        & (payments["payment_date"] <= CUTOFF)
    ].copy()

    successful["amount"] = pd.to_numeric(
        successful["amount"], errors="coerce"
    )

    target = (
        successful.groupby("customer_id")["amount"]
        .sum()
        .rename("customer_lifetime_value")
    )

    # Customer demographics and tenure.
    customers["registration_date"] = pd.to_datetime(
        customers["registration_date"], errors="coerce"
    )
    customers["customer_tenure_days"] = (
        CUTOFF - customers["registration_date"]
    ).dt.days.clip(lower=0)

    base = customers[
        [
            "customer_id", "age", "gender", "country",
            "acquisition_channel", "customer_segment",
            "preferred_language", "customer_tenure_days",
        ]
    ].drop_duplicates("customer_id").set_index("customer_id")

    # Latest subscription record available at the cutoff.
    eligible_subs = subscriptions[
        (subscriptions["subscription_start_date"] <= CUTOFF)
    ].copy()

    eligible_subs = eligible_subs.sort_values(
        ["subscription_start_date", "subscription_id"]
    )

    latest_sub = eligible_subs.groupby("customer_id").tail(1).copy()
    latest_sub = latest_sub.set_index("customer_id")

    latest_sub["subscription_tenure_days"] = (
        CUTOFF - latest_sub["subscription_start_date"]
    ).dt.days.clip(lower=0)

    sub_features = latest_sub[
        [
            "plan_id", "monthly_price", "subscription_tenure_days",
        ]
    ].copy()

    base = base.join(sub_features, how="left")

    # Viewing engagement during the 90 days before cutoff.
    start_90d = CUTOFF - pd.Timedelta(days=90)
    recent_viewing = viewing[
        (viewing["viewing_date"] > start_90d)
        & (viewing["viewing_date"] <= CUTOFF)
    ].copy()

    recent_viewing["watch_duration_minutes"] = pd.to_numeric(
        recent_viewing["watch_duration_minutes"], errors="coerce"
    )
    recent_viewing["completion_percentage"] = pd.to_numeric(
        recent_viewing["completion_percentage"], errors="coerce"
    )

    view_features = recent_viewing.groupby("customer_id").agg(
        viewing_sessions_90d=("viewing_id", "count"),
        active_viewing_days_90d=("viewing_date", "nunique"),
        watch_minutes_90d=("watch_duration_minutes", "sum"),
        avg_completion_90d=("completion_percentage", "mean"),
    )

    base = base.join(view_features, how="left")

    # Support experience up to cutoff.
    recent_tickets = tickets[
        tickets["ticket_date"] <= CUTOFF
    ].copy()

    recent_tickets["customer_satisfaction_score"] = pd.to_numeric(
        recent_tickets["customer_satisfaction_score"], errors="coerce"
    )

    ticket_features = recent_tickets.groupby("customer_id").agg(
        support_ticket_count=("ticket_id", "count"),
        avg_support_satisfaction=("customer_satisfaction_score", "mean"),
    )

    base = base.join(ticket_features, how="left")

    # Feedback signals up to cutoff.
    recent_feedback = feedback[
        feedback["feedback_date"] <= CUTOFF
    ].copy()

    recent_feedback["rating"] = pd.to_numeric(
        recent_feedback["rating"], errors="coerce"
    )

    feedback_features = recent_feedback.groupby("customer_id").agg(
        feedback_count=("feedback_id", "count"),
        avg_feedback_rating=("rating", "mean"),
    )

    base = base.join(feedback_features, how="left")

    # Activity counts can be zero when no records exist.
    count_columns = [
        "viewing_sessions_90d",
        "active_viewing_days_90d",
        "watch_minutes_90d",
        "support_ticket_count",
        "feedback_count",
    ]

    for col in count_columns:
        base[col] = base[col].fillna(0)

    dataset = base.join(target, how="inner").reset_index()

    dataset = dataset[
        dataset["customer_lifetime_value"].notna()
        & (dataset["customer_lifetime_value"] >= 0)
    ].copy()

    return dataset


def train():
    print("Loading processed datasets...")
    data = load_data()

    print("Building CLV dataset...")
    df = build_dataset(data)

    if len(df) < 100:
        raise ValueError("Insufficient customer records for CLV training.")

    target_col = "customer_lifetime_value"

    categorical = [
        "gender", "country", "acquisition_channel",
        "preferred_language", "plan_id",
    ]

    numeric = [
        "age", "customer_tenure_days", "monthly_price",
        "subscription_tenure_days", "viewing_sessions_90d",
        "active_viewing_days_90d", "watch_minutes_90d",
        "avg_completion_90d", "support_ticket_count",
        "avg_support_satisfaction", "feedback_count",
        "avg_feedback_rating",
    ]

    X = df[categorical + numeric].copy()
    y = df[target_col].astype(float)

    # Customer-level holdout: each customer appears only once.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    candidates = {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not installed; comparing available models only.")

    results = []
    fitted_models = {}

    print(f"Customers: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Target mean: {y.mean():.2f}")
    print(f"Target median: {y.median():.2f}")

    for name, estimator in candidates.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        fitted_models[name] = pipeline

        print(
            f"{name}: MAE={mae:.3f} | "
            f"RMSE={rmse:.3f} | R2={r2:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        ["RMSE", "MAE"], ascending=True
    )

    results_df.to_csv(
        REPORT_DIR / "clv_model_comparison.csv", index=False
    )

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "clv_regression.joblib")

    test_predictions = pd.DataFrame({
        "actual_clv": y_test.values,
        "predicted_clv": best_model.predict(X_test),
    })

    test_predictions["absolute_error"] = (
        test_predictions["actual_clv"]
        - test_predictions["predicted_clv"]
    ).abs()

    test_predictions.to_csv(
        REPORT_DIR / "clv_test_predictions.csv", index=False
    )

    metrics = {
        "target": "Successful payments accumulated through 2026-05-31",
        "target_definition": "Sum of successful payment amounts per customer",
        "cutoff_date": str(CUTOFF.date()),
        "best_model": best_name,
        "customer_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "metrics": results_df.to_dict(orient="records"),
        "limitations": [
            "This estimates historical accumulated customer value, not future lifetime revenue.",
            "Random customer holdout evaluates unseen customers from the same historical population.",
            "The target is affected by customer tenure and the available observation period.",
            "No payment-derived predictors are used, avoiding direct target leakage.",
        ],
    }

    with open(REPORT_DIR / "clv_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("CLV MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'clv_regression.joblib'}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    train()
