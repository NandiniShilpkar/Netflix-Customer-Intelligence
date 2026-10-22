import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

from src.dashboard.churn_intelligence import (
    build_current_customer_features,
    load_churn_model,
)


def _minmax(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(50.0, index=series.index)

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    ).clip(0, 100)


def _get_monthly_plan_prices(engine):
    query = text(
        """
        SELECT
            plan_id,
            monthly_price
        FROM subscription_plans
        """
    )

    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _action_for_customer(row):
    if row["failed_payment_rate"] >= 50:
        return "Payment recovery and billing support"

    if row["unresolved_ticket_count"] > 0:
        return "Escalate unresolved support issues"

    if row["days_since_last_viewing"] >= 30:
        return "Launch personalized re-engagement"

    if row["viewing_sessions_90d"] <= 3:
        return "Send personalized content recommendations"

    if (
        row["subscription_tenure_days"] <= 60
        and row["churn_risk"] >= 60
    ):
        return "Early-tenure retention intervention"

    return "Proactive retention outreach"


def _decision_for_customer(row):
    risk = row["churn_risk"]
    value = row["customer_value_score"]

    if risk >= 70 or (risk >= 60 and value >= 60):
        return "Save Now"

    if risk >= 45:
        return "Engage"

    if risk >= 30:
        return "Monitor"

    return "Low Priority"


@st.cache_data(ttl=300)
def build_customer_intelligence(_engine):
    """
    Build a deterministic business decision layer on top of
    the validated historical churn model.

    The churn model remains the predictive source of truth.
    Health, value, priority, and actions are business rules.
    """

    model = load_churn_model()
    features = build_current_customer_features(_engine).copy()

    if features.empty:
        return features

    # --------------------------------------------------------
    # Predict churn risk
    # --------------------------------------------------------

    model_features = features.drop(
        columns=["customer_id"],
        errors="ignore",
    )

    churn_probability = model.predict_proba(
        model_features
    )[:, 1]

    result = features.copy()

    result["churn_risk"] = (
        churn_probability * 100
    )

    # --------------------------------------------------------
    # Customer Health Score
    # --------------------------------------------------------

    inactivity_score = (
        result["days_since_last_viewing"]
        .clip(lower=0, upper=90)
        / 90
        * 100
    )

    payment_score = (
        result["failed_payment_rate"]
        .clip(lower=0, upper=100)
    )

    support_score = (
        result["unresolved_ticket_count"]
        .clip(lower=0, upper=3)
        / 3
        * 100
    )

    engagement_score = (
        100
        - (
            result["viewing_sessions_90d"]
            .clip(lower=0, upper=20)
            / 20
            * 100
        )
    )

    result["customer_health_score"] = (
        100
        - (
            result["churn_risk"] * 0.55
            + inactivity_score * 0.15
            + payment_score * 0.15
            + support_score * 0.10
            + engagement_score * 0.05
        )
    ).clip(0, 100)

    # --------------------------------------------------------
    # Customer Value Score
    # --------------------------------------------------------

    result["engagement_value"] = _minmax(
        result["total_watch_minutes_90d"]
    )

    result["tenure_value"] = _minmax(
        result["subscription_tenure_days"]
    )

    # Use payment amount only if available.
    # Otherwise use plan revenue as the financial value signal.
    if "avg_payment_amount" in result.columns:
        result["payment_value"] = _minmax(
            result["avg_payment_amount"]
        )
    else:
        result["payment_value"] = 50.0

    plans = _get_monthly_plan_prices(_engine)

    result = result.merge(
        plans,
        on="plan_id",
        how="left",
    )

    result["monthly_price"] = pd.to_numeric(
        result["monthly_price"],
        errors="coerce",
    ).fillna(0)

    result["revenue_value"] = _minmax(
        result["monthly_price"]
    )

    result["customer_value_score"] = (
        result["revenue_value"] * 0.45
        + result["tenure_value"] * 0.20
        + result["engagement_value"] * 0.20
        + result["payment_value"] * 0.15
    ).clip(0, 100)

    # --------------------------------------------------------
    # Retention decision
    # --------------------------------------------------------

    result["decision"] = result.apply(
        _decision_for_customer,
        axis=1,
    )

    result["recommended_action"] = result.apply(
        _action_for_customer,
        axis=1,
    )

    # --------------------------------------------------------
    # Retention Priority Score
    # --------------------------------------------------------

    result["retention_priority_score"] = (
        result["churn_risk"] * 0.60
        + result["customer_value_score"] * 0.25
        + result["failed_payment_rate"].clip(0, 100) * 0.10
        + (
            result["unresolved_ticket_count"]
            .clip(0, 3)
            / 3
            * 100
            * 0.05
        )
    ).clip(0, 100)

    # --------------------------------------------------------
    # Revenue at risk
    # --------------------------------------------------------

    result["monthly_revenue_at_risk"] = np.where(
        result["churn_risk"] >= 60,
        result["monthly_price"],
        0,
    )

    result["annualized_revenue_at_risk"] = (
        result["monthly_revenue_at_risk"] * 12
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output_columns = [
        "customer_id",
        "plan_id",
        "churn_risk",
        "customer_health_score",
        "customer_value_score",
        "retention_priority_score",
        "decision",
        "recommended_action",
        "monthly_price",
        "monthly_revenue_at_risk",
        "annualized_revenue_at_risk",
        "failed_payment_rate",
        "days_since_last_viewing",
        "viewing_sessions_90d",
        "unresolved_ticket_count",
        "subscription_tenure_days",
    ]

    result = result[output_columns].copy()

    return result.sort_values(
        [
            "retention_priority_score",
            "churn_risk",
        ],
        ascending=False,
    ).reset_index(drop=True)


def calculate_retention_impact(
    intelligence,
    retention_rate,
):
    """
    Scenario estimate for revenue protection if a percentage
    of high-risk customers are successfully retained.
    """

    high_risk = intelligence[
        intelligence["churn_risk"] >= 60
    ]

    monthly_risk = high_risk[
        "monthly_revenue_at_risk"
    ].sum()

    protected_monthly = (
        monthly_risk * retention_rate
    )

    return {
        "high_risk_customers": int(
            len(high_risk)
        ),
        "monthly_revenue_at_risk": float(
            monthly_risk
        ),
        "protected_monthly_revenue": float(
            protected_monthly
        ),
        "protected_annual_revenue": float(
            protected_monthly * 12
        ),
    }


def render_customer_decision_engine(engine):
    st.header(
        "Customer Intelligence Decision Engine"
    )

    st.caption(
        "A deterministic business decision layer built on top "
        "of the validated historical churn model. Scores support "
        "prioritization and scenario analysis; they are not "
        "guaranteed outcomes."
    )

    with st.spinner(
        "Building customer intelligence..."
    ):
        intelligence = build_customer_intelligence(
            engine
        )

    if intelligence.empty:
        st.warning(
            "No eligible customers were found."
        )
        return

    # --------------------------------------------------------
    # Executive metrics
    # --------------------------------------------------------

    save_now = int(
        (
            intelligence["decision"]
            == "Save Now"
        ).sum()
    )

    engage = int(
        (
            intelligence["decision"]
            == "Engage"
        ).sum()
    )

    high_risk = intelligence[
        intelligence["churn_risk"] >= 60
    ]

    monthly_risk = high_risk[
        "monthly_revenue_at_risk"
    ].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers analyzed",
        f"{len(intelligence):,}",
    )

    col2.metric(
        "Save Now",
        f"{save_now:,}",
    )

    col3.metric(
        "Engage",
        f"{engage:,}",
    )

    col4.metric(
        "Monthly revenue at risk",
        f"${monthly_risk:,.2f}",
    )

    # --------------------------------------------------------
    # Decision distribution
    # --------------------------------------------------------

    st.subheader(
        "Retention decision distribution"
    )

    decision_counts = (
        intelligence["decision"]
        .value_counts()
        .rename_axis("Decision")
        .reset_index(
            name="Customers"
        )
    )

    fig = px.bar(
        decision_counts,
        x="Decision",
        y="Customers",
        title="Customers by retention decision",
        template="plotly_dark",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,8,9,0.95)",
        font=dict(color="#F5F5F1"),
        title_font=dict(color="#F5F5F1"),
        hoverlabel=dict(
            bgcolor="#1A1112",
            font_color="#F5F5F1",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Priority actions
    # --------------------------------------------------------

    st.subheader(
        "Priority customer actions"
    )

    display = intelligence.head(50).copy()

    display = display.rename(
        columns={
            "customer_id": "Customer ID",
            "plan_id": "Plan",
            "churn_risk": "Churn Risk %",
            "customer_health_score": "Health Score",
            "customer_value_score": "Value Score",
            "retention_priority_score": "Priority Score",
            "decision": "Decision",
            "recommended_action": "Recommended Action",
            "monthly_price": "Monthly Price",
            "monthly_revenue_at_risk": "Revenue at Risk",
            "failed_payment_rate": "Failed Payment %",
            "days_since_last_viewing": "Days Since Viewing",
            "viewing_sessions_90d": "Viewing Sessions",
            "unresolved_ticket_count": "Unresolved Tickets",
            "subscription_tenure_days": "Tenure Days",
        }
    )

    for column in [
        "Churn Risk %",
        "Health Score",
        "Value Score",
        "Priority Score",
        "Monthly Price",
        "Revenue at Risk",
        "Failed Payment %",
    ]:
        if column in display.columns:
            display[column] = display[
                column
            ].round(2)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Decision explorer
    # --------------------------------------------------------

    st.subheader(
        "Decision explorer"
    )

    selected_decision = st.selectbox(
        "Filter by retention decision",
        [
            "All",
            "Save Now",
            "Engage",
            "Monitor",
            "Low Priority",
        ],
        key="decision_engine_filter",
    )

    filtered = intelligence.copy()

    if selected_decision != "All":
        filtered = filtered[
            filtered["decision"]
            == selected_decision
        ]

    filtered_display = filtered[
        [
            "customer_id",
            "churn_risk",
            "customer_health_score",
            "customer_value_score",
            "retention_priority_score",
            "decision",
            "recommended_action",
        ]
    ].rename(
        columns={
            "customer_id": "Customer ID",
            "churn_risk": "Churn Risk %",
            "customer_health_score": "Health Score",
            "customer_value_score": "Value Score",
            "retention_priority_score": "Priority Score",
            "decision": "Decision",
            "recommended_action": "Recommended Action",
        }
    )

    st.caption(
        f"Matching customers: "
        f"{len(filtered_display):,}"
    )

    st.dataframe(
        filtered_display.head(500),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Retention impact simulator
    # --------------------------------------------------------

    st.subheader(
        "Retention impact simulator"
    )

    st.caption(
        "Scenario analysis: assumes the selected percentage "
        "of high-risk customers are successfully retained."
    )

    scenario_rows = []

    for rate in [
        0.10,
        0.20,
        0.30,
    ]:
        impact = calculate_retention_impact(
            intelligence,
            rate,
        )

        scenario_rows.append(
            {
                "Retention Success": (
                    f"{int(rate * 100)}%"
                ),
                "High-Risk Customers": impact[
                    "high_risk_customers"
                ],
                "Monthly Revenue at Risk": round(
                    impact[
                        "monthly_revenue_at_risk"
                    ],
                    2,
                ),
                "Protected Monthly Revenue": round(
                    impact[
                        "protected_monthly_revenue"
                    ],
                    2,
                ),
                "Protected Annual Revenue": round(
                    impact[
                        "protected_annual_revenue"
                    ],
                    2,
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(scenario_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Revenue protection figures are scenario estimates "
        "based on current subscription revenue associated "
        "with high-risk customers. They are not guaranteed "
        "revenue forecasts."
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv_data = intelligence.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download customer intelligence decisions",
        data=csv_data,
        file_name=(
            "customer_intelligence_decisions.csv"
        ),
        mime="text/csv",
        key="download_customer_decision_engine",
    )


# Compatibility alias for Copilot usage.
build_customer_intelligence_context = (
    build_customer_intelligence
)
