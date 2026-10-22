from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.customer_decision_engine import build_customer_intelligence
from src.dashboard.retention_optimization import (
    build_retention_optimization,
    calculate_recovery_scenarios,
)


def build_executive_analytics(engine):
    customer_df = build_customer_intelligence(engine)

    if customer_df.empty:
        return {}

    retention_df = build_retention_optimization(engine)

    total_customers = int(len(customer_df))

    save_now = int(
        (customer_df["decision"] == "Save Now").sum()
    )

    engage = int(
        (customer_df["decision"] == "Engage").sum()
    )

    high_risk = int(
        (customer_df["churn_risk"] >= 55).sum()
    )

    high_value = int(
        (customer_df["customer_value_score"] >= 60).sum()
    )

    high_value_high_risk = int(
        (
            (customer_df["customer_value_score"] >= 60)
            & (customer_df["churn_risk"] >= 55)
        ).sum()
    )

    monthly_revenue_at_risk = float(
        customer_df["monthly_revenue_at_risk"].sum()
    )

    annual_revenue_at_risk = monthly_revenue_at_risk * 12

    avg_health = float(
        customer_df["customer_health_score"].mean()
    )

    avg_value = float(
        customer_df["customer_value_score"].mean()
    )

    avg_churn = float(
        customer_df["churn_risk"].mean()
    )

    target_customers = int(
        retention_df[
            retention_df["decision"].isin(["Save Now", "Engage"])
        ].shape[0]
    )

    scenarios = calculate_recovery_scenarios(retention_df)

    recovery_30 = 0.0

    if not scenarios.empty:
        row = scenarios[
            scenarios["Retention Success Rate"] == "30%"
        ]

        if not row.empty:
            recovery_30 = float(
                row.iloc[0]["Annual Revenue Recovered"]
            )

    return {
        "customer_df": customer_df,
        "retention_df": retention_df,
        "total_customers": total_customers,
        "save_now": save_now,
        "engage": engage,
        "high_risk": high_risk,
        "high_value": high_value,
        "high_value_high_risk": high_value_high_risk,
        "monthly_revenue_at_risk": monthly_revenue_at_risk,
        "annual_revenue_at_risk": annual_revenue_at_risk,
        "avg_health": avg_health,
        "avg_value": avg_value,
        "avg_churn": avg_churn,
        "target_customers": target_customers,
        "recovery_30": recovery_30,
    }


def _recommendations(metrics):
    recommendations = []

    if metrics["high_value_high_risk"] > 0:
        recommendations.append(
            f"Prioritize {metrics['high_value_high_risk']:,} high-value "
            "customers with elevated churn risk."
        )

    if metrics["save_now"] > 0:
        recommendations.append(
            f"Launch immediate retention actions for "
            f"{metrics['save_now']:,} Save Now customers."
        )

    if metrics["engage"] > 0:
        recommendations.append(
            f"Use proactive engagement campaigns for "
            f"{metrics['engage']:,} Engage customers."
        )

    if metrics["monthly_revenue_at_risk"] > 0:
        recommendations.append(
            f"Track ${metrics['monthly_revenue_at_risk']:,.2f} "
            "of monthly revenue currently exposed to retention risk."
        )

    if metrics["recovery_30"] > 0:
        recommendations.append(
            f"A 30% retention-success scenario represents approximately "
            f"${metrics['recovery_30']:,.2f} in annual revenue recovery."
        )

    return recommendations


def render_executive_analytics(engine):
    st.title("Executive Analytics")
    st.caption(
        "Executive view of customer health, churn exposure, "
        "retention opportunity, and revenue risk."
    )

    data = build_executive_analytics(engine)

    if not data:
        st.warning("No executive analytics data is currently available.")
        return

    customer_df = data["customer_df"]

    st.subheader("Executive Scorecard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Customers",
            f"{data['total_customers']:,}",
        )

    with c2:
        st.metric(
            "High-Risk Customers",
            f"{data['high_risk']:,}",
        )

    with c3:
        st.metric(
            "Monthly Revenue at Risk",
            f"${data['monthly_revenue_at_risk']:,.2f}",
        )

    with c4:
        st.metric(
            "Annual Revenue at Risk",
            f"${data['annual_revenue_at_risk']:,.2f}",
        )

    st.divider()

    st.subheader("Customer Health & Value")

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric(
            "Average Health Score",
            f"{data['avg_health']:.1f}",
        )

    with h2:
        st.metric(
            "Average Customer Value",
            f"{data['avg_value']:.1f}",
        )

    with h3:
        st.metric(
            "Average Churn Risk",
            f"{data['avg_churn']:.1f}%",
        )

    with h4:
        st.metric(
            "High-Value / High-Risk",
            f"{data['high_value_high_risk']:,}",
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        decision_df = (
            customer_df["decision"]
            .value_counts()
            .rename_axis("Decision")
            .reset_index(name="Customers")
        )

        fig = px.bar(
            decision_df,
            x="Decision",
            y="Customers",
            text="Customers",
            title="Customer Decision Distribution",
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Customers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        fig = px.scatter(
            customer_df.sample(
                min(len(customer_df), 2500),
                random_state=42,
            ),
            x="customer_value_score",
            y="churn_risk",
            size="monthly_revenue_at_risk",
            hover_data=[
                "customer_id",
                "decision",
                "customer_health_score",
            ],
            title="Customer Value × Churn Risk",
        )

        fig.update_layout(
            xaxis_title="Customer Value Score",
            yaxis_title="Churn Risk %",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Retention Opportunity")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Customers to Target",
            f"{data['target_customers']:,}",
        )

    with r2:
        st.metric(
            "Save Now",
            f"{data['save_now']:,}",
        )

    with r3:
        st.metric(
            "Engage",
            f"{data['engage']:,}",
        )

    with r4:
        st.metric(
            "30% Recovery Scenario",
            f"${data['recovery_30']:,.2f}",
        )

    st.divider()

    st.subheader("What Needs Attention?")

    recommendations = _recommendations(data)

    if recommendations:
        for recommendation in recommendations:
            st.info(recommendation)
    else:
        st.success("No major retention actions are currently required.")

    st.divider()

    st.subheader("Priority Executive Customers")

    priority = (
        customer_df[
            customer_df["decision"].isin(
                ["Save Now", "Engage"]
            )
        ]
        .sort_values(
            "retention_priority_score",
            ascending=False,
        )
        .head(50)
        .copy()
    )

    columns = [
        "customer_id",
        "churn_risk",
        "customer_health_score",
        "customer_value_score",
        "retention_priority_score",
        "decision",
        "recommended_action",
        "monthly_revenue_at_risk",
    ]

    columns = [
        column
        for column in columns
        if column in priority.columns
    ]

    st.dataframe(
        priority[columns].style.format(
            {
                "churn_risk": "{:.1f}",
                "customer_health_score": "{:.1f}",
                "customer_value_score": "{:.1f}",
                "retention_priority_score": "{:.1f}",
                "monthly_revenue_at_risk": "${:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv = priority[columns].to_csv(index=False)

    st.download_button(
        "Download Executive Priority List",
        data=csv,
        file_name="executive_priority_customers.csv",
        mime="text/csv",
    )


build_executive_analytics_context = build_executive_analytics
