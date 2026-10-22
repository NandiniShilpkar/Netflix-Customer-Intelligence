from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from src.dashboard.customer_decision_engine import build_customer_intelligence


# ============================================================
# RETENTION ACTION & REVENUE OPTIMIZATION ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _add_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add retention campaign and business-action classifications."""

    result = df.copy()

    def campaign(row):
        decision = str(row.get("decision", ""))
        health = float(row.get("customer_health_score", 50))
        value = float(row.get("customer_value_score", 50))
        churn = float(row.get("churn_risk", 0))

        if decision == "Save Now" and value >= 60:
            return "High-Value Rescue"

        if decision == "Save Now":
            return "Immediate Retention"

        if decision == "Engage" and health < 50:
            return "Customer Experience Recovery"

        if decision == "Engage":
            return "Proactive Engagement"

        if churn >= 55:
            return "Churn Watch"

        return "Always-On Monitoring"

    def matrix(row):
        churn = float(row.get("churn_risk", 0))
        value = float(row.get("customer_value_score", 0))

        if value >= 60 and churn >= 55:
            return "High Value / High Risk"

        if value >= 60 and churn < 55:
            return "High Value / Lower Risk"

        if value < 60 and churn >= 55:
            return "Lower Value / High Risk"

        return "Lower Value / Lower Risk"

    result["campaign_segment"] = result.apply(campaign, axis=1)
    result["risk_value_segment"] = result.apply(matrix, axis=1)

    return result


def build_retention_optimization(engine) -> pd.DataFrame:
    """
    Build the retention optimization dataset from the existing
    Customer Intelligence Decision Engine.
    """

    df = build_customer_intelligence(engine)

    if df.empty:
        return df

    return _add_action_columns(df)


def calculate_recovery_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate monthly and annual revenue recovery at different success rates."""

    if df.empty:
        return pd.DataFrame()

    target = df[df["decision"].isin(["Save Now", "Engage"])].copy()

    monthly_risk = float(target["monthly_revenue_at_risk"].sum())
    customers = int(len(target))

    rows = []

    for success_rate in [10, 20, 30, 40, 50]:
        recovered_monthly = monthly_risk * success_rate / 100

        rows.append(
            {
                "Retention Success Rate": f"{success_rate}%",
                "Customers Targeted": customers,
                "Monthly Revenue Recovered": recovered_monthly,
                "Annual Revenue Recovered": recovered_monthly * 12,
            }
        )

    return pd.DataFrame(rows)


def build_campaign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize actionable retention campaign segments."""

    if df.empty:
        return pd.DataFrame()

    result = (
        df.groupby("campaign_segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Churn_Risk=("churn_risk", "mean"),
            Avg_Customer_Value=("customer_value_score", "mean"),
            Monthly_Revenue_at_Risk=("monthly_revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values("Monthly_Revenue_at_Risk", ascending=False)
    )

    return result


def render_retention_optimization(engine):
    """Render Phase 3 dashboard."""

    st.title("Retention Action & Revenue Optimization")

    st.caption(
        "Turn churn predictions and customer intelligence into prioritized "
        "retention campaigns and measurable revenue opportunities."
    )

    df = build_retention_optimization(engine)

    if df.empty:
        st.warning("No retention optimization data is currently available.")
        return

    target_df = df[df["decision"].isin(["Save Now", "Engage"])].copy()

    monthly_risk = float(target_df["monthly_revenue_at_risk"].sum())
    annual_risk = monthly_risk * 12
    customers_targeted = len(target_df)
    high_value_risk = len(
        df[df["risk_value_segment"] == "High Value / High Risk"]
    )

    # --------------------------------------------------------
    # Executive metrics
    # --------------------------------------------------------

    st.subheader("Retention opportunity")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Customers to Target", f"{customers_targeted:,}")

    with c2:
        st.metric("Monthly Revenue at Risk", f"${monthly_risk:,.2f}")

    with c3:
        st.metric("Annual Revenue at Risk", f"${annual_risk:,.2f}")

    with c4:
        st.metric("High-Value / High-Risk", f"{high_value_risk:,}")

    st.divider()

    # --------------------------------------------------------
    # Risk-value matrix
    # --------------------------------------------------------

    st.subheader("Customer Risk × Value Matrix")

    matrix_df = (
        df.groupby("risk_value_segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Churn_Risk=("churn_risk", "mean"),
            Avg_Customer_Value=("customer_value_score", "mean"),
            Revenue_at_Risk=("monthly_revenue_at_risk", "sum"),
        )
        .reset_index()
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            matrix_df,
            x="risk_value_segment",
            y="Customers",
            title="Customers by Risk / Value Segment",
            text="Customers",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            matrix_df,
            x="risk_value_segment",
            y="Revenue_at_Risk",
            title="Revenue at Risk by Segment",
            text_auto=".2f",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Monthly Revenue at Risk")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        matrix_df.style.format(
            {
                "Avg_Churn_Risk": "{:.1f}",
                "Avg_Customer_Value": "{:.1f}",
                "Revenue_at_Risk": "${:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Campaign segments
    # --------------------------------------------------------

    st.subheader("Retention Campaign Builder")

    campaign_df = build_campaign_segments(df)

    st.dataframe(
        campaign_df.style.format(
            {
                "Avg_Churn_Risk": "{:.1f}",
                "Avg_Customer_Value": "{:.1f}",
                "Monthly_Revenue_at_Risk": "${:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Recommended actions
    # --------------------------------------------------------

    st.subheader("Recommended Retention Actions")

    action_summary = (
        df[df["decision"].isin(["Save Now", "Engage"])]
        .groupby("recommended_action")
        .agg(
            Customers=("customer_id", "count"),
            Revenue_at_Risk=("monthly_revenue_at_risk", "sum"),
            Avg_Churn_Risk=("churn_risk", "mean"),
        )
        .reset_index()
        .sort_values("Revenue_at_Risk", ascending=False)
    )

    st.dataframe(
        action_summary.style.format(
            {
                "Revenue_at_Risk": "${:,.2f}",
                "Avg_Churn_Risk": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Revenue recovery simulator
    # --------------------------------------------------------

    st.subheader("Revenue Recovery Simulator")

    scenarios = calculate_recovery_scenarios(df)

    if not scenarios.empty:

        success_rate = st.slider(
            "Expected retention success rate",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
        )

        selected = scenarios.iloc[
            (success_rate // 10) - 1
        ]

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric(
                "Customers Targeted",
                f"{int(selected['Customers Targeted']):,}",
            )

        with s2:
            st.metric(
                "Monthly Revenue Recovered",
                f"${selected['Monthly Revenue Recovered']:,.2f}",
            )

        with s3:
            st.metric(
                "Annual Revenue Recovered",
                f"${selected['Annual Revenue Recovered']:,.2f}",
            )

        st.dataframe(
            scenarios.style.format(
                {
                    "Monthly Revenue Recovered": "${:,.2f}",
                    "Annual Revenue Recovered": "${:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --------------------------------------------------------
    # Priority campaign customer list
    # --------------------------------------------------------

    st.subheader("Priority Campaign Customers")

    priority = (
        df[df["decision"].isin(["Save Now", "Engage"])]
        .sort_values("retention_priority_score", ascending=False)
        .head(100)
        .copy()
    )

    display_columns = [
        "customer_id",
        "churn_risk",
        "customer_health_score",
        "customer_value_score",
        "retention_priority_score",
        "decision",
        "campaign_segment",
        "recommended_action",
        "monthly_revenue_at_risk",
    ]

    display_columns = [
        c for c in display_columns if c in priority.columns
    ]

    st.dataframe(
        priority[display_columns].style.format(
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

    csv = priority[display_columns].to_csv(index=False)

    st.download_button(
        "Download Retention Campaign List",
        data=csv,
        file_name="retention_campaign_customers.csv",
        mime="text/csv",
    )


# Compatibility alias for future modules.
build_retention_optimization_context = build_retention_optimization
