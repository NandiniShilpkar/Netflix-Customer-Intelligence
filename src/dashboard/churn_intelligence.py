
from pathlib import Path
import shap


import plotly.express as px

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "historical_random_forest.joblib"

CUTOFF_DATE = "2026-05-31"
VIEWING_WINDOW_START = "2026-03-02"


@st.cache_resource
def load_churn_model():
    """Load the saved preprocessing and Random Forest pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


def build_current_customer_features(engine):
    """
    Build model input features using the same column definitions
    and historical cutoff as the existing training pipeline.
    """
    cutoff = CUTOFF_DATE
    viewing_start = VIEWING_WINDOW_START

    cohort_query = text("""
        WITH latest_subscription AS (
            SELECT DISTINCT ON (customer_id)
                customer_id,
                subscription_start_date,
                plan_id,
                cancellation_date
            FROM subscriptions
            WHERE subscription_start_date <= CAST(:cutoff AS DATE)
            ORDER BY customer_id, subscription_start_date DESC
        )
        SELECT
            c.customer_id,
            c.age,
            c.gender,
            c.country,
            c.city,
            c.preferred_language,
            ls.plan_id,
            ls.subscription_start_date
        FROM customers c
        JOIN latest_subscription ls
            ON c.customer_id = ls.customer_id
        WHERE ls.cancellation_date IS NULL
           OR ls.cancellation_date > CAST(:cutoff AS DATE)
    """)

    with engine.connect() as conn:
        features = pd.read_sql(
            cohort_query,
            conn,
            params={"cutoff": cutoff},
        )

    viewing_query = text("""
        SELECT
            customer_id,
            COUNT(*) AS viewing_sessions_90d,
            COUNT(DISTINCT viewing_date) AS active_viewing_days_90d,
            COALESCE(SUM(watch_duration_minutes), 0)
                AS total_watch_minutes_90d,
            AVG(watch_duration_minutes)
                AS avg_watch_minutes_90d,
            AVG(completion_percentage)
                AS avg_completion_pct_90d,
            MAX(viewing_date) AS last_viewing_date
        FROM viewing_activity
        WHERE viewing_date > CAST(:window_start AS DATE)
          AND viewing_date <= CAST(:cutoff AS DATE)
        GROUP BY customer_id
    """)

    with engine.connect() as conn:
        viewing = pd.read_sql(
            viewing_query,
            conn,
            params={
                "window_start": viewing_start,
                "cutoff": cutoff,
            },
        )

    features = features.merge(
        viewing, on="customer_id", how="left"
    )

    viewing_count_numeric = [
        "viewing_sessions_90d",
        "active_viewing_days_90d",
        "total_watch_minutes_90d",
    ]
    features[viewing_count_numeric] = (
        features[viewing_count_numeric].fillna(0)
    )

    features["days_since_last_viewing"] = (
        pd.Timestamp(cutoff)
        - pd.to_datetime(features["last_viewing_date"])
    ).dt.days.fillna(91)

    payments_query = text("""
        SELECT
            customer_id,
            COUNT(*) AS payment_count,
            COUNT(*) FILTER (
                WHERE payment_status = 'Failed'
            ) AS failed_payment_count,
            COUNT(*) FILTER (
                WHERE payment_status = 'Success'
            ) AS successful_payment_count,
            AVG(amount) AS avg_payment_amount
        FROM payments
        WHERE payment_date <= CAST(:cutoff AS DATE)
        GROUP BY customer_id
    """)

    with engine.connect() as conn:
        payments = pd.read_sql(
            payments_query,
            conn,
            params={"cutoff": cutoff},
        )

    features = features.merge(
        payments, on="customer_id", how="left"
    )

    payment_count_numeric = [
        "payment_count",
        "failed_payment_count",
        "successful_payment_count",
    ]
    features[payment_count_numeric] = (
        features[payment_count_numeric].fillna(0)
    )

    features["failed_payment_rate"] = (
        features["failed_payment_count"]
        / features["payment_count"].replace(0, float("nan"))
    ).fillna(0)

    support_query = text("""
        SELECT
            customer_id,
            COUNT(*) AS support_ticket_count,
            COUNT(*) FILTER (
                WHERE priority IN ('High', 'Critical')
            ) AS high_priority_ticket_count,
            COUNT(*) FILTER (
                WHERE ticket_status IN ('Pending', 'Escalated')
            ) AS unresolved_ticket_count,
            AVG(customer_satisfaction_score)
                AS avg_support_satisfaction,
            AVG(resolution_time_hours)
                AS avg_resolution_hours
        FROM support_tickets
        WHERE ticket_date <= CAST(:cutoff AS DATE)
        GROUP BY customer_id
    """)

    with engine.connect() as conn:
        support = pd.read_sql(
            support_query,
            conn,
            params={"cutoff": cutoff},
        )

    features = features.merge(
        support, on="customer_id", how="left"
    )

    features[
        [
            "support_ticket_count",
            "high_priority_ticket_count",
            "unresolved_ticket_count",
        ]
    ] = features[
        [
            "support_ticket_count",
            "high_priority_ticket_count",
            "unresolved_ticket_count",
        ]
    ].fillna(0)

    features["has_support_tickets"] = (
        features["support_ticket_count"] > 0
    ).astype(int)

    features["subscription_tenure_days"] = (
        pd.Timestamp(cutoff)
        - pd.to_datetime(features["subscription_start_date"])
    ).dt.days

    features = features.drop(
        columns=[
            "subscription_start_date",
            "last_viewing_date",
        ],
        errors="ignore",
    )

    return features.drop_duplicates(
        subset=["customer_id"]
    )


def score_customers(engine):
    """Return customers with experimental model-based risk scores."""
    model = load_churn_model()
    features = build_current_customer_features(engine)

    if features.empty:
        return features

    customer_ids = features["customer_id"].copy()
    model_features = features.drop(
        columns=["customer_id"]
    )

    probabilities = model.predict_proba(
        model_features
    )[:, 1]

    results = pd.DataFrame({
        "customer_id": customer_ids,
        "experimental_churn_risk_pct": probabilities * 100,
    })

    results["risk_band"] = pd.cut(
        results["experimental_churn_risk_pct"],
        bins=[-0.001, 30, 60, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return results.sort_values(
        "experimental_churn_risk_pct",
        ascending=False,
    )


def render_churn_intelligence(engine):
    """Display experimental churn scoring in Streamlit."""

    st.header("AI Churn Intelligence")
    render_feature_importance()
    st.caption(
        "Experimental churn scores from a historical Random Forest model "
        "trained on earlier customer snapshots and evaluated on May 2026 data. "
        "Predictive performance is limited, and the scores are not calibrated "
        "or validated individual churn probabilities. Use them for exploration, "
        "not as definitive predictions about individual customers."
    )

    explain_customer(engine)



    if not st.button(
        "Calculate experimental churn risk",
        key="calculate_experimental_churn_risk",
        type="secondary",
        use_container_width=False,
    ):
        st.info(
            "Click the button to score the historical customer cohort."
        )
        return


    try:
        with st.spinner("Scoring customers..."):
            scores = score_customers(engine)

        if scores.empty:
            st.warning("No eligible customers were found.")
            return

        high_risk = int((scores["risk_band"] == "High").sum())
        medium_risk = int((scores["risk_band"] == "Medium").sum())
        low_risk = int((scores["risk_band"] == "Low").sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("High risk", f"{high_risk:,}")
        col2.metric("Medium risk", f"{medium_risk:,}")
        col3.metric("Low risk", f"{low_risk:,}")

        selected_band = st.selectbox(
            "Filter by experimental risk band",
            ["All", "High", "Medium", "Low"],
        )

        visible = scores
        if selected_band != "All":
            visible = scores[
                scores["risk_band"] == selected_band
            ]

        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download risk scores (CSV)",
            data=scores.to_csv(index=False).encode("utf-8"),
            file_name="experimental_churn_risk.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error("Unable to calculate churn risk.")
        st.caption(
            "Check the model file, database connection, and feature schema."
        )
        # explain_customer(engine)
        st.exception(exc)



def render_feature_importance():
    """Display global Random Forest feature importance."""
    st.subheader("What influences the model's decisions?")

    try:
        model = load_churn_model()
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        importance_df = pd.DataFrame({
            "Feature": preprocessor.get_feature_names_out(),
            "Importance": classifier.feature_importances_,
        })

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        importance_df = importance_df.nlargest(
            15, "Importance"
        ).sort_values("Importance")


        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Global Feature Importances",
            labels={
                "Importance": "Relative importance",
                "Feature": "",
            },
            template="plotly_dark",
            color_discrete_sequence=["#E50914"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,8,9,0.95)",
            font=dict(color="#F5F5F1"),
            title_font=dict(color="#F5F5F1"),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Global feature importance describes the model as a whole. "
            "It does not explain an individual customer's score or prove causation."
        )

    except Exception as exc:
        st.error("Unable to display feature importance.")
        st.exception(exc)


def explain_customer(engine):
    """Explain one customer's experimental Random Forest score."""
    st.subheader("Individual Customer Explanation")

    try:
        model = load_churn_model()
        features = build_current_customer_features(engine)

        if features.empty:
            st.warning("No eligible customers found.")
            return

        customer_ids = features["customer_id"].astype(str)
        selected_id = st.selectbox(
            "Select a customer",
            customer_ids.tolist(),
            key="shap_customer_id",
        )

        selected_index = customer_ids[
            customer_ids == selected_id
        ].index[0]

        customer_features = features.loc[[selected_index]].drop(
            columns=["customer_id"]
        )

        model_probability = float(
            model.predict_proba(customer_features)[0, 1]
        )

        st.metric(
            "Experimental churn-risk score",
            f"{model_probability:.1%}",
        )

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed = preprocessor.transform(customer_features)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed, dtype=np.float64)
        feature_names = preprocessor.get_feature_names_out()

        # TreeExplainer explains the fitted forest on the
        # preprocessed feature representation.
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            customer_shap = shap_values[1][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            customer_shap = shap_values[0, :, 1]
        else:
            customer_shap = shap_values[0]

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "SHAP value": customer_shap,
        })

        explanation["Feature"] = (
            explanation["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        explanation["Absolute impact"] = (
            explanation["SHAP value"].abs()
        )

        explanation = explanation.nlargest(
            12, "Absolute impact"
        ).sort_values("SHAP value")


        fig = px.bar(
            explanation,
            x="SHAP value",
            y="Feature",
            orientation="h",
            color="SHAP value",
            color_continuous_scale=[
                [0, "#777777"],
                [0.5, "#241012"],
                [1, "#E50914"],
            ],
            title="Features pushing the model score up or down",
            labels={"SHAP value": "Contribution to model output"},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,8,9,0.95)",
            font=dict(color="#F5F5F1"),
            title_font=dict(color="#F5F5F1"),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


        st.caption(
            "SHAP values explain contributions on the model's output scale. "
            "For a classifier, they may represent class-output contributions "
            "rather than percentage-point changes in churn probability. "
            "They are not proof of causation."
        )

    except Exception as exc:
        st.error("Unable to explain this customer's model score.")
        st.caption(
            "The explanation method may need adjustment for the installed "
            "SHAP version or model output format."
        )
        st.exception(exc)
