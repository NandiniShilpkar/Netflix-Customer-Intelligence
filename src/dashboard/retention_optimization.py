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


    render_historical_campaign_roi()

# Compatibility alias for future modules.
build_retention_optimization_context = build_retention_optimization


# ============================================================
# HISTORICAL CATBOOST RETENTION CAMPAIGN ROI
# ============================================================

def render_historical_campaign_roi():
    st.divider()
    st.header("Historical Retention Campaign ROI")
    st.caption(
        "Retrospective campaign simulation using the April 2026 "
        "CatBoost B validation cohort."
    )

    prediction_path = (
        PROJECT_ROOT
        / "reports/retention_roi/catboost_b_ranked_customers_with_prices.csv"
    )
    sensitivity_path = (
        PROJECT_ROOT
        / "reports/retention_roi/retention_sensitivity_results.csv"
    )

    if not prediction_path.exists():
        st.warning("Historical customer prediction report was not found.")
        return

    predictions = pd.read_csv(prediction_path)

    required = {
        "customer_id",
        "actual_churn",
        "predicted_churn_probability",
        "monthly_price",
    }

    if not required.issubset(predictions.columns):
        st.error("The historical prediction report is missing required columns.")
        return

    predictions["actual_churn"] = pd.to_numeric(
        predictions["actual_churn"], errors="coerce"
    ).fillna(0).astype(int)

    predictions["monthly_price"] = pd.to_numeric(
        predictions["monthly_price"], errors="coerce"
    )

    predictions = predictions.sort_values(
        "predicted_churn_probability", ascending=False
    ).reset_index(drop=True)

    total_customers = len(predictions)
    total_churners = int(predictions["actual_churn"].sum())

    if total_customers == 0 or total_churners == 0:
        st.warning("The historical cohort has no usable customer records.")
        return

    st.subheader("Campaign assumptions")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        capacity = st.slider(
            "Contact capacity (%)",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            key="historical_roi_capacity",
        )

    with c2:
        retention_rate = st.slider(
            "Assumed retention success (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="historical_roi_success",
        )

    with c3:
        campaign_cost = st.number_input(
            "Cost per contacted customer",
            min_value=0.0,
            value=1.0,
            step=0.25,
            key="historical_roi_cost",
        )

    with c4:
        months_saved = st.selectbox(
            "Revenue-saving period (months)",
            options=[1, 3, 6],
            index=1,
            key="historical_roi_months",
        )

    contact_count = max(
        1, round(total_customers * capacity / 100)
    )

    selected = predictions.head(contact_count).copy()

    selected_churners = selected[
        (selected["actual_churn"] == 1)
        & selected["monthly_price"].notna()
    ]

    actual_churners_selected = len(selected_churners)

    expected_retained = (
        actual_churners_selected * retention_rate / 100
    )

    monthly_revenue_at_risk = selected_churners[
        "monthly_price"
    ].sum()

    gross_revenue_saved = (
        monthly_revenue_at_risk
        * retention_rate / 100
        * months_saved
    )

    total_campaign_cost = contact_count * campaign_cost
    net_benefit = gross_revenue_saved - total_campaign_cost

    roi = (
        net_benefit / total_campaign_cost * 100
        if total_campaign_cost > 0
        else 0
    )

    precision = actual_churners_selected / contact_count
    recall = actual_churners_selected / total_churners

    st.divider()
    st.subheader("Campaign performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Customers Contacted", f"{contact_count:,}")
    m2.metric("Historical Churners Selected", f"{actual_churners_selected:,}")
    m3.metric("Expected Retained", f"{expected_retained:.1f}")
    m4.metric("Campaign ROI", f"{roi:.2f}%")

    m5, m6, m7, m8 = st.columns(4)

    m5.metric("Campaign Cost", f"{total_campaign_cost:,.2f}")
    m6.metric("Estimated Revenue Saved", f"{gross_revenue_saved:,.2f}")
    m7.metric("Estimated Net Benefit", f"{net_benefit:,.2f}")
    m8.metric("Churner Recall", f"{recall:.2%}")

    st.caption(
        f"Selected-group precision: {precision:.2%} | "
        f"Overall historical churn rate: "
        f"{predictions['actual_churn'].mean():.2%}"
    )

    if roi >= 0:
        st.success("This scenario has positive estimated net benefit.")
    else:
        st.warning("This scenario has negative estimated net benefit.")

    st.info(
        "This is a retrospective what-if simulation. Actual historical "
        "churn outcomes are used to estimate the opportunity, while the "
        "retention success rate is an assumption. It does not measure "
        "the causal effect of a real retention campaign."
    )

    st.subheader("Sensitivity Analysis")

    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)

        available_costs = sorted(
            sensitivity["campaign_cost_per_customer"].unique()
        )
        available_months = sorted(
            sensitivity["revenue_saving_months"].unique()
        )

        h1, h2 = st.columns(2)

        with h1:
            heatmap_cost = st.selectbox(
                "Sensitivity campaign cost",
                available_costs,
                index=available_costs.index(1.0)
                if 1.0 in available_costs else 0,
                key="historical_heatmap_cost",
            )

        with h2:
            heatmap_months = st.selectbox(
                "Sensitivity revenue period",
                available_months,
                index=available_months.index(3)
                if 3 in available_months else 0,
                key="historical_heatmap_months",
            )

        filtered = sensitivity[
            (sensitivity["campaign_cost_per_customer"] == heatmap_cost)
            & (sensitivity["revenue_saving_months"] == heatmap_months)
        ]

        pivot = filtered.pivot(
            index="retention_success_rate",
            columns="contact_capacity_pct",
            values="estimated_roi_pct",
        ) * 1.0

        pivot.index = [f"{int(x * 100)}%" for x in pivot.index]

        fig = px.imshow(
            pivot,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={
                "x": "Contact capacity",
                "y": "Retention success rate",
                "color": "Estimated ROI (%)",
            },
            title="Campaign ROI Sensitivity Heatmap",
        )

        st.plotly_chart(fig, use_container_width=True)

        average_roi = (
            sensitivity.groupby(
                ["contact_capacity_pct", "retention_success_rate"],
                as_index=False,
            )["estimated_roi_pct"].mean()
        )

        average_roi["retention_success_rate"] *= 100

        fig2 = px.line(
            average_roi,
            x="retention_success_rate",
            y="estimated_roi_pct",
            color="contact_capacity_pct",
            markers=True,
            title="Average ROI by Retention Success Rate",
            labels={
                "retention_success_rate": "Retention success (%)",
                "estimated_roi_pct": "Average ROI (%)",
                "contact_capacity_pct": "Contact capacity (%)",
            },
        )

        fig2.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)

        profitable_count = int(sensitivity["profitable"].sum())
        total_scenarios = len(sensitivity)

        p1, p2 = st.columns(2)
        p1.metric("Profitable Scenarios", f"{profitable_count:,}")
        p2.metric(
            "Unprofitable Scenarios",
            f"{total_scenarios - profitable_count:,}",
        )

        st.download_button(
            "Download Sensitivity Analysis CSV",
            data=sensitivity.to_csv(index=False),
            file_name="retention_sensitivity_analysis.csv",
            mime="text/csv",
            key="download_retention_sensitivity",
        )
    else:
        st.warning("Sensitivity analysis report was not found.")

    st.subheader("Priority Historical Campaign Customers")

    display = selected[
        [
            "customer_id",
            "predicted_churn_probability",
            "actual_churn",
            "monthly_price",
            "risk_group",
        ]
    ].copy()

    display["predicted_churn_probability"] *= 100

    st.dataframe(
        display.head(100).style.format(
            {
                "predicted_churn_probability": "{:.2f}%",
                "monthly_price": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Historical Campaign Customer List",
        data=selected.to_csv(index=False),
        file_name="historical_catboost_retention_campaign.csv",
        mime="text/csv",
        key="download_historical_campaign",
    )
