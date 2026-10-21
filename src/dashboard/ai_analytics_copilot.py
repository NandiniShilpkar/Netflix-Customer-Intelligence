import streamlit as st
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.dashboard.churn_intelligence import (
    build_current_customer_features,
    load_churn_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


@st.cache_resource
def get_copilot_engine():
    load_dotenv(ENV_PATH)

    import os
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


@st.cache_data(ttl=300)
def get_top_churn_customers(limit=20):
    engine = get_copilot_engine()

    features = build_current_customer_features(engine)
    model = load_churn_model()

    X = features.drop(columns=["customer_id"])

    churn_probability = model.predict_proba(X)[:, 1]

    result = features[
        [
            "customer_id",
            "age",
            "gender",
            "country",
            "plan_id",
            "viewing_sessions_90d",
            "active_viewing_days_90d",
            "total_watch_minutes_90d",
            "days_since_last_viewing",
            "failed_payment_rate",
            "support_ticket_count",
            "unresolved_ticket_count",
            "subscription_tenure_days",
        ]
    ].copy()

    result["churn_risk"] = churn_probability * 100

    result = (
        result.sort_values("churn_risk", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )

    result["churn_risk"] = result["churn_risk"].round(2)
    result["failed_payment_rate"] = (
        result["failed_payment_rate"] * 100
    ).round(2)

    return result


@st.cache_data(ttl=300)
def get_churn_by_plan():
    engine = get_copilot_engine()

    query = """
    SELECT
        sp.plan_id,
        sp.plan_name,
        sp.monthly_price,
        COUNT(DISTINCT s.customer_id) AS customers,
        COUNT(
            DISTINCT CASE
                WHEN cl.churned = TRUE THEN s.customer_id
            END
        ) AS churned_customers,
        ROUND(
            100.0
            * COUNT(
                DISTINCT CASE
                    WHEN cl.churned = TRUE THEN s.customer_id
                END
            )
            / NULLIF(COUNT(DISTINCT s.customer_id), 0),
            2
        ) AS churn_rate
    FROM subscriptions s
    JOIN subscription_plans sp
        ON s.plan_id = sp.plan_id
    LEFT JOIN churn_labels cl
        ON s.customer_id = cl.customer_id
    GROUP BY
        sp.plan_id,
        sp.plan_name,
        sp.monthly_price
    ORDER BY churn_rate DESC
    """

    return pd.read_sql(query, engine)


@st.cache_data(ttl=300)
def get_churn_risk_segments():
    engine = get_copilot_engine()

    features = build_current_customer_features(engine)
    model = load_churn_model()

    X = features.drop(columns=["customer_id"])
    features["churn_risk"] = model.predict_proba(X)[:, 1] * 100

    features["customer_segment"] = pd.cut(
        features["churn_risk"],
        bins=[-1, 30, 60, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    result = (
        features.groupby("customer_segment", observed=False)
        .agg(
            customers=("customer_id", "count"),
            average_churn_risk=("churn_risk", "mean"),
        )
        .reset_index()
    )

    result["average_churn_risk"] = result["average_churn_risk"].round(2)

    return result.sort_values(
        "average_churn_risk",
        ascending=False,
    ).reset_index(drop=True)


@st.cache_data(ttl=300)
def get_churn_trend_analysis():
    engine = get_copilot_engine()

    monthly_query = """
    SELECT
        DATE_TRUNC('month', churn_date) AS churn_month,
        COUNT(*) AS churned_customers
    FROM churn_labels
    WHERE churned = TRUE
      AND churn_date IS NOT NULL
    GROUP BY DATE_TRUNC('month', churn_date)
    ORDER BY churn_month
    """

    reasons_query = """
    SELECT
        churn_reason,
        COUNT(*) AS churned_customers
    FROM churn_labels
    WHERE churned = TRUE
      AND churn_reason IS NOT NULL
    GROUP BY churn_reason
    ORDER BY churned_customers DESC
    """

    monthly = pd.read_sql(monthly_query, engine)
    reasons = pd.read_sql(reasons_query, engine)

    monthly["churn_month"] = pd.to_datetime(monthly["churn_month"])

    return monthly, reasons


@st.cache_data(ttl=300)
def get_high_risk_recommendations(limit=20):
    customers = get_top_churn_customers(limit)

    recommendations = []

    for _, customer in customers.iterrows():
        actions = []

        if customer["failed_payment_rate"] >= 50:
            actions.append("Prioritize payment recovery and billing support")

        if customer["days_since_last_viewing"] >= 30:
            actions.append("Launch a re-engagement campaign")

        if customer["viewing_sessions_90d"] <= 3:
            actions.append("Send personalized content recommendations")

        if customer["unresolved_ticket_count"] > 0:
            actions.append("Escalate unresolved support issues")

        if (
            customer["subscription_tenure_days"] <= 60
            and customer["churn_risk"] >= 60
        ):
            actions.append("Offer an early-tenure retention intervention")

        if not actions:
            actions.append(
                "Prioritize for proactive retention outreach based on high predicted churn risk"
            )

        recommendations.append(
            {
                "customer_id": customer["customer_id"],
                "churn_risk": customer["churn_risk"],
                "recommended_actions": "; ".join(actions),
            }
        )

    return pd.DataFrame(recommendations)


@st.cache_data(ttl=300)
def get_retention_targets(limit=20):
    customers = get_top_churn_customers(100)

    result = customers.copy()

    result["retention_priority_score"] = (
        result["churn_risk"] * 0.60
        + (result["failed_payment_rate"] / 100) * 100 * 0.15
        + (
            result["days_since_last_viewing"].clip(lower=0, upper=90) / 90
        ) * 100 * 0.15
        + (
            result["unresolved_ticket_count"].clip(lower=0, upper=3) / 3
        ) * 100 * 0.10
    )

    result["retention_priority_score"] = (
        result["retention_priority_score"].round(2)
    )

    result["priority"] = pd.cut(
        result["retention_priority_score"],
        bins=[-1, 50, 70, 100],
        labels=[
            "Priority 3 - Standard",
            "Priority 2 - High",
            "Priority 1 - Immediate",
        ],
    )

    result = (
        result.sort_values(
            ["retention_priority_score", "churn_risk"],
            ascending=False,
        )
        .head(limit)
        .reset_index(drop=True)
    )

    return result[
        [
            "customer_id",
            "churn_risk",
            "retention_priority_score",
            "priority",
            "failed_payment_rate",
            "days_since_last_viewing",
            "unresolved_ticket_count",
            "plan_id",
        ]
    ]


@st.cache_data(ttl=300)
def get_revenue_at_risk():
    engine = get_copilot_engine()

    features = build_current_customer_features(engine)
    model = load_churn_model()

    X = features.drop(columns=["customer_id"])
    features["churn_risk"] = model.predict_proba(X)[:, 1] * 100

    subscription_query = """
    SELECT
        s.customer_id,
        s.plan_id,
        sp.plan_name,
        sp.monthly_price
    FROM subscriptions s
    JOIN subscription_plans sp
        ON s.plan_id = sp.plan_id
    """

    subscriptions = pd.read_sql(subscription_query, engine)

    result = features[
        ["customer_id", "churn_risk"]
    ].merge(
        subscriptions,
        on="customer_id",
        how="left",
    )

    high_risk = result[result["churn_risk"] >= 60].copy()

    high_risk["monthly_revenue_at_risk"] = high_risk["monthly_price"]

    summary = {
        "high_risk_customers": len(high_risk),
        "monthly_revenue_at_risk": high_risk[
            "monthly_revenue_at_risk"
        ].sum(),
        "annualized_revenue_at_risk": high_risk[
            "monthly_revenue_at_risk"
        ].sum() * 12,
    }

    by_plan = (
        high_risk.groupby(
            ["plan_id", "plan_name"],
            as_index=False,
        )
        .agg(
            high_risk_customers=("customer_id", "count"),
            monthly_revenue_at_risk=(
                "monthly_revenue_at_risk",
                "sum",
            ),
        )
        .sort_values(
            "monthly_revenue_at_risk",
            ascending=False,
        )
    )

    by_plan["monthly_revenue_at_risk"] = (
        by_plan["monthly_revenue_at_risk"].round(2)
    )

    return summary, by_plan


def render_ai_analytics_copilot():
    st.title("AI Analytics Copilot")

    st.caption(
        "Ask questions about customers, churn, subscriptions, "
        "viewing behavior, and business performance."
    )

    question = st.text_input(
        "Ask a business question",
        placeholder="Example: Which customers have the highest churn risk?",
    )

    if not question:
        st.info(
            "Try asking: Which customers have the highest churn risk?"
        )
        return

    normalized_question = question.lower().strip()

    if (
        "subscription plan" not in normalized_question
        and "which plan" not in normalized_question
        and "plan has the highest churn" not in normalized_question
        and "plan with the highest churn" not in normalized_question
        and "churn rate by plan" not in normalized_question
        and "customer segment" not in normalized_question
        and "customer segments" not in normalized_question
        and "segment has the highest churn" not in normalized_question
        and "segments have the highest churn" not in normalized_question
        and "highest churn risk segment" not in normalized_question
        and "highest churn risk segments" not in normalized_question
        and (
            "highest churn" in normalized_question
            or "top churn" in normalized_question
            or "high churn risk" in normalized_question
            or "churning" in normalized_question
        )
    ):
        st.subheader("Highest Churn-Risk Customers")

        with st.spinner("Analyzing customer churn risk..."):
            result = get_top_churn_customers(20)

        top_customer = result.iloc[0]

        st.write(
            f"**{len(result)} customers** have been ranked by predicted churn risk. "
            f"The highest-risk customer is **{top_customer['customer_id']}** "
            f"with a predicted churn risk of **{top_customer['churn_risk']:.2f}%**."
        )

        risk_factors = []

        if top_customer["days_since_last_viewing"] >= 30:
            risk_factors.append(
                f"{int(top_customer['days_since_last_viewing'])} days since last viewing"
            )

        if top_customer["failed_payment_rate"] > 0:
            risk_factors.append(
                f"{top_customer['failed_payment_rate']:.0f}% failed payment rate"
            )

        if top_customer["viewing_sessions_90d"] <= 2:
            risk_factors.append(
                f"only {int(top_customer['viewing_sessions_90d'])} viewing sessions in the last 90 days"
            )

        if top_customer["support_ticket_count"] > 0:
            risk_factors.append(
                f"{int(top_customer['support_ticket_count'])} support tickets"
            )

        if risk_factors:
            st.write(
                "**Observed risk signals:** "
                + "; ".join(risk_factors)
                + "."
            )
        else:
            st.write(
                "**Observed risk signals:** The model identifies this "
                "customer as high risk based on the combined customer "
                "behavior and account features."
            )

        st.dataframe(
            result,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Churn risk is based on the validated Random Forest model "
            "used by the Churn Intelligence dashboard."
        )

    elif (
        "why is churn increasing" in normalized_question
        or "why churn is increasing" in normalized_question
        or "why is churn rising" in normalized_question
        or "why churn is rising" in normalized_question
        or "what is causing churn" in normalized_question
        or "what causes churn" in normalized_question
    ):
        st.subheader("Why Is Churn Increasing?")

        with st.spinner("Analyzing churn trends and customer churn reasons..."):
            monthly, reasons = get_churn_trend_analysis()

        first_month = monthly.iloc[0]
        latest_month = monthly.iloc[-1]

        change = (
            (latest_month["churned_customers"] - first_month["churned_customers"])
            / first_month["churned_customers"]
        ) * 100

        highest_month = monthly.loc[
            monthly["churned_customers"].idxmax()
        ]

        top_reason = reasons.iloc[0]
        second_reason = reasons.iloc[1]
        third_reason = reasons.iloc[2]

        if change > 0:
            trend_text = (
                f"Monthly churn increased from **{int(first_month['churned_customers']):,}** "
                f"in {first_month['churn_month'].strftime('%b %Y')} to "
                f"**{int(latest_month['churned_customers']):,}** in "
                f"{latest_month['churn_month'].strftime('%b %Y')}, "
                f"an overall increase of **{change:.1f}%**."
            )
        elif change < 0:
            trend_text = (
                f"Monthly churn decreased by **{abs(change):.1f}%** between "
                f"{first_month['churn_month'].strftime('%b %Y')} and "
                f"{latest_month['churn_month'].strftime('%b %Y')}."
            )
        else:
            trend_text = "Monthly churn was broadly unchanged over the observed period."

        st.write(trend_text)

        st.write(
            f"The highest monthly churn occurred in "
            f"**{highest_month['churn_month'].strftime('%b %Y')}**, "
            f"with **{int(highest_month['churned_customers']):,} churned customers**."
        )

        st.write(
            "**Main churn drivers:** "
            f"**{top_reason['churn_reason']}** ({int(top_reason['churned_customers']):,}), "
            f"**{second_reason['churn_reason']}** ({int(second_reason['churned_customers']):,}), "
            f"and **{third_reason['churn_reason']}** "
            f"({int(third_reason['churned_customers']):,})."
        )

        st.dataframe(
            reasons,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "This analysis uses historical churn dates and recorded churn "
            "reasons from the churn_labels table. It describes observed "
            "patterns and does not establish causal relationships."
        )

    elif (
        "customer segment" in normalized_question
        or "customer segments" in normalized_question
        or "segments have the highest churn" in normalized_question
        or "segment has the highest churn" in normalized_question
        or "highest churn risk segment" in normalized_question
        or "highest churn risk segments" in normalized_question
    ):
        st.subheader("Churn Risk by Customer Segment")
        with st.spinner("Analyzing customer churn-risk segments..."):
            result = get_churn_risk_segments()

        top_segment = result.iloc[0]

        st.write(
            f"The **{top_segment['customer_segment']}** segment has the "
            f"highest average predicted churn risk at "
            f"**{top_segment['average_churn_risk']:.2f}%**. "
            f"It contains **{int(top_segment['customers']):,} customers**."
        )

        st.dataframe(result, width="stretch", hide_index=True)

        st.caption(
            "Segments are created from the validated Random Forest churn "
            "probabilities: Low Risk (<30%), Medium Risk (30–60%), "
            "and High Risk (>60%)."
        )

    elif (
        "what should the business do" in normalized_question
        or "what should we do" in normalized_question
        or "what action should" in normalized_question
        or "how should the business respond" in normalized_question
        or "how should we respond" in normalized_question
        or "retention action" in normalized_question
        or "retention actions" in normalized_question
        or "high-risk customers" in normalized_question
        or "high risk customers" in normalized_question
    ):
        st.subheader("Recommended Actions for High-Risk Customers")

        with st.spinner("Analyzing high-risk customers and recommended retention actions..."):
            result = get_high_risk_recommendations(20)

        st.write(
            f"The Copilot identified the **top {len(result)} high-risk customers** "
            "and mapped their observed risk signals to recommended retention actions."
        )

        st.write(
            "The recommendations prioritize payment recovery, customer "
            "re-engagement, personalized content, support intervention, "
            "and early-tenure retention where the available customer signals "
            "support those actions."
        )

        st.dataframe(
            result,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Recommendations are rule-based actions derived from predicted "
            "churn risk and observed customer behavior. They are decision-support "
            "recommendations and do not establish causal relationships."
        )

    elif (
        "how much revenue is at risk" in normalized_question
        or "what is the revenue at risk" in normalized_question
        or "revenue at risk" in normalized_question
        or "revenue risk" in normalized_question
    ):
        st.subheader("Revenue at Risk")

        with st.spinner("Calculating revenue exposure from high-risk customers..."):
            summary, by_plan = get_revenue_at_risk()

        monthly_risk = float(summary["monthly_revenue_at_risk"])
        annual_risk = float(summary["annualized_revenue_at_risk"])
        high_risk_customers = int(summary["high_risk_customers"])

        st.write(
            f"There are **{high_risk_customers:,} high-risk customers** "
            f"with **${monthly_risk:,.2f} in monthly revenue at risk**."
        )

        st.write(
            f"On an annualized basis, this represents "
            f"**${annual_risk:,.2f} of subscription revenue associated "
            "with high-risk customers**."
        )

        st.write("**Revenue exposure by subscription plan:**")

        st.dataframe(
            by_plan,
            width="stretch",
            hide_index=True,
        )

        top_plan = by_plan.iloc[0]

        st.write(
            f"The **{top_plan['plan_name']}** plan has the largest revenue "
            f"exposure at **${float(top_plan['monthly_revenue_at_risk']):,.2f} "
            "per month**."
        )

        st.caption(
            "Revenue at risk represents subscription revenue associated with "
            "customers whose predicted churn probability is at least 60%. "
            "It is an exposure metric, not a forecast of guaranteed revenue loss."
        )

    elif (
        "top 20 customers to target" in normalized_question
        or "top customers to target" in normalized_question
        or "customers to target for retention" in normalized_question
        or "target for retention" in normalized_question
        or "retention targets" in normalized_question
        or "who should we target" in normalized_question
    ):
        st.subheader("Top Customers to Target for Retention")

        with st.spinner("Prioritizing customers for retention..."):
            result = get_retention_targets(20)

        immediate_count = int(
            (result["priority"] == "Priority 1 - Immediate").sum()
        )
        high_count = int(
            (result["priority"] == "Priority 2 - High").sum()
        )

        st.write(
            f"The Copilot ranked the **top {len(result)} customers** "
            "for retention based on predicted churn risk and observed "
            "customer risk signals."
        )

        st.write(
            f"Among these customers, **{immediate_count}** are classified as "
            f"**Priority 1 - Immediate** and **{high_count}** as "
            "**Priority 2 - High**."
        )

        st.dataframe(
            result,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Retention priority combines predicted churn risk with observed "
            "payment failure, inactivity, and unresolved support signals. "
            "It is a decision-support score, not a causal measure."
        )

    elif (
        "subscription plan" in normalized_question
        or "which plan" in normalized_question
        or "plan has the highest churn" in normalized_question
        or "plan with the highest churn" in normalized_question
        or "churn rate by plan" in normalized_question
    ):
        st.subheader("Churn Rate by Subscription Plan")

        with st.spinner("Analyzing churn by subscription plan..."):
            result = get_churn_by_plan()

        top_plan = result.iloc[0]

        st.write(
            f"The **{top_plan['plan_name']}** plan has the highest churn rate "
            f"at **{top_plan['churn_rate']:.2f}%**. "
            f"It has **{int(top_plan['churned_customers']):,} churned customers "
            f"out of {int(top_plan['customers']):,} customers**."
        )

        st.dataframe(
            result,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Churn rate is calculated from historical churn labels grouped "
            "by subscription plan."
        )

    else:
        st.info(
            "I can currently answer questions about the highest churn-risk "
            "customers and churn by subscription plan."
        )
