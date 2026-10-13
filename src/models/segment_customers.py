from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data" / "features"

for directory in (MODEL_DIR, REPORT_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = "2026-06-30"
RANDOM_STATE = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 7
SILHOUETTE_SAMPLE_SIZE = 5000


def load_customer_features(engine):
    """Build one row per customer using data available by the cutoff."""
    query = text("""
        WITH latest_subscription AS (
            SELECT DISTINCT ON (customer_id)
                customer_id,
                subscription_start_date,
                monthly_price,
                auto_renew,
                plan_id
            FROM subscriptions
            WHERE subscription_start_date <= CAST(:cutoff AS DATE)
            ORDER BY customer_id,
                     subscription_start_date DESC,
                     subscription_id DESC
        ),
        viewing AS (
            SELECT
                customer_id,
                COUNT(*) AS viewing_sessions_90d,
                COALESCE(SUM(watch_duration_minutes), 0)
                    AS total_watch_minutes_90d,
                AVG(completion_percentage) AS avg_completion_pct_90d
            FROM viewing_activity
            WHERE viewing_date > CAST(:cutoff AS DATE) - INTERVAL '90 days'
              AND viewing_date <= CAST(:cutoff AS DATE)
            GROUP BY customer_id
        ),
        payment AS (
            SELECT
                customer_id,
                COUNT(*) AS payment_count,
                COUNT(*) FILTER (
                    WHERE LOWER(payment_status) = 'failed'
                ) AS failed_payment_count,
                AVG(amount) AS avg_payment_amount
            FROM payments
            WHERE payment_date <= CAST(:cutoff AS DATE)
            GROUP BY customer_id
        ),
        support AS (
            SELECT
                customer_id,
                COUNT(*) AS support_ticket_count,
                COUNT(*) FILTER (
                    WHERE LOWER(ticket_status) IN ('pending', 'escalated')
                ) AS unresolved_ticket_count,
                AVG(customer_satisfaction_score) AS avg_support_satisfaction
            FROM support_tickets
            WHERE ticket_date <= CAST(:cutoff AS DATE)
            GROUP BY customer_id
        ),
        feedback AS (
            SELECT
                customer_id,
                AVG(rating) AS avg_feedback_rating
            FROM customer_feedback
            WHERE feedback_date <= CAST(:cutoff AS DATE)
            GROUP BY customer_id
        )
        SELECT
            c.customer_id,
            c.age,
            c.country,
            c.gender,
            c.acquisition_channel,
            s.plan_id,
            s.monthly_price,
            CASE WHEN s.auto_renew IS TRUE THEN 1
                 WHEN s.auto_renew IS FALSE THEN 0
                 ELSE NULL END AS auto_renew,
            GREATEST(
                0,
                CAST(:cutoff AS DATE) - s.subscription_start_date
            ) AS subscription_tenure_days,
            COALESCE(v.viewing_sessions_90d, 0) AS viewing_sessions_90d,
            COALESCE(v.total_watch_minutes_90d, 0)
                AS total_watch_minutes_90d,
            v.avg_completion_pct_90d,
            COALESCE(p.payment_count, 0) AS payment_count,
            COALESCE(p.failed_payment_count, 0)
                AS failed_payment_count,
            p.avg_payment_amount,
            COALESCE(
                p.failed_payment_count::FLOAT
                / NULLIF(p.payment_count, 0),
                0
            ) AS failed_payment_rate,
            COALESCE(su.support_ticket_count, 0)
                AS support_ticket_count,
            COALESCE(su.unresolved_ticket_count, 0)
                AS unresolved_ticket_count,
            su.avg_support_satisfaction,
            f.avg_feedback_rating
        FROM customers c
        LEFT JOIN latest_subscription s USING (customer_id)
        LEFT JOIN viewing v USING (customer_id)
        LEFT JOIN payment p USING (customer_id)
        LEFT JOIN support su USING (customer_id)
        LEFT JOIN feedback f USING (customer_id)
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"cutoff": CUTOFF_DATE})

    if df.empty:
        raise ValueError("No customer records were returned.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Customer features contain duplicate customer IDs.")

    return df


def main():
    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env.")

    engine = create_engine(database_url, pool_pre_ping=True)

    try:
        customers = load_customer_features(engine)
    finally:
        engine.dispose()

    ids = customers["customer_id"].copy()
    features = customers.drop(columns=["customer_id"]).copy()

    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable segmentation features were found.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    transformed = preprocessor.fit_transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float64)

    if not np.isfinite(transformed).all():
        raise ValueError("Transformed features contain non-finite values.")

    # Evaluate a small, reproducible range of cluster counts.
    evaluation = []
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(transformed))

    for k in range(MIN_CLUSTERS, min(MAX_CLUSTERS, len(transformed) - 1) + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(transformed)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            transformed,
            labels,
            sample_size=sample_size if sample_size < len(transformed) else None,
            random_state=RANDOM_STATE,
        )

        evaluation.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_score": float(score),
            "smallest_cluster_size": int(pd.Series(labels).value_counts().min()),
        })

    if not evaluation:
        raise RuntimeError("No valid cluster evaluations were produced.")

    evaluation_df = pd.DataFrame(evaluation)
    evaluation_df.to_csv(REPORT_DIR / "segmentation_evaluation.csv", index=False)

    # Select the highest silhouette score among the tested candidates.
    # Review cluster size and interpretability before treating this as final.
    best_k = int(
        evaluation_df.sort_values(
            ["silhouette_score", "k"],
            ascending=[False, True],
        ).iloc[0]["k"]
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = final_model.fit_predict(transformed)

    assignments = pd.DataFrame({
        "customer_id": ids,
        "segment_id": labels,
    })
    assignments.to_csv(DATA_DIR / "customer_segments.csv", index=False)

    profile = features.copy()
    profile["segment_id"] = labels
    profile["customer_id"] = ids

    numeric_profile_columns = profile.select_dtypes(
        include=["number"]
    ).columns.tolist()
    numeric_profile_columns = [
        column for column in numeric_profile_columns
        if column not in {"segment_id", "customer_id"}
    ]

    profile_summary = (
        profile.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "nunique"),
            **{
                column: (column, "mean")
                for column in numeric_profile_columns
            },
        )
        .reset_index()
    )
    profile_summary.to_csv(
        REPORT_DIR / "customer_segment_profiles.csv", index=False
    )

    artifact = {
        "cutoff_date": CUTOFF_DATE,
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": preprocessor,
        "model": final_model,
        "selected_k": best_k,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, MODEL_DIR / "customer_segmentation.joblib")

    report = {
        "cutoff_date": CUTOFF_DATE,
        "customers_segmented": int(len(customers)),
        "selected_k": best_k,
        "selection_rule": "Highest silhouette score among tested k values",
        "random_state": RANDOM_STATE,
        "feature_columns": features.columns.tolist(),
        "evaluation": evaluation,
        "caveats": [
            "Cluster IDs are arbitrary labels, not ordered value tiers.",
            "K-Means finds statistical groupings, not causal customer types.",
            "Review cluster sizes and profiles before assigning business names.",
            "This model describes the selected cutoff snapshot.",
        ],
    }
    (REPORT_DIR / "customer_segmentation_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Customer segmentation completed.")
    print("Customers:", len(customers))
    print("Selected clusters:", best_k)
    print("\nEvaluation:")
    print(evaluation_df.to_string(index=False))
    print("\nSegment sizes:")
    print(assignments["segment_id"].value_counts().sort_index().to_string())
    print("\nSaved:")
    print("  models/customer_segmentation.joblib")
    print("  data/features/customer_segments.csv")
    print("  reports/segmentation_evaluation.csv")
    print("  reports/customer_segment_profiles.csv")
    print("  reports/customer_segmentation_metrics.json")


if __name__ == "__main__":
    main()
