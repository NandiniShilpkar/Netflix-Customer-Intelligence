import streamlit as st
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.services.llm_service import generate_business_answer

from src.dashboard.churn_intelligence import (
    build_current_customer_features,
    load_churn_model,
)

from src.dashboard.customer_decision_engine import (
    build_customer_intelligence,
)

def classify_question(question):
    q = question.lower().strip()

    if any(
        phrase in q
        for phrase in [
            "data quality",
            "data quality score",
            "missing values",
            "missing features",
            "missing feature values",
            "feature completeness",
            "data completeness",
            "mlops",
            "ml ops",
            "explainability",
            "model explanation",
            "why is this customer high risk",
            "why is this customer risky",
            "churn drivers",
            "churn driver",
            "feature importance",
            "risk explanation",
        ]
    ):
        return "data_quality_mlops"

    if any(
        phrase in q
        for phrase in [
            "production monitoring",
            "production readiness",
            "production health",
            "model monitoring",
            "model health",
            "model status",
            "ai health",
            "ai service health",
            "ollama health",
            "data quality",
            "feature quality",
            "prediction coverage",
            "production ai",
        ]
    ):
        return "production_ai_monitoring"

    if any(
        phrase in q
        for phrase in [
            "executive analytics",
            "executive dashboard",
            "executive summary",
            "business overview",
            "business health",
            "overall business health",
            "what needs attention",
            "what should management focus on",
            "what should leadership focus on",
            "management priorities",
            "leadership priorities",
            "executive priorities",
            "customer health overview",
        ]
    ):
        return "executive_analytics"

    if any(
        phrase in q
        for phrase in [
            "retention campaign",
            "retention campaigns",
            "retention strategy",
            "retention strategies",
            "retention success",
            "retention rate",
            "retention optimization",
            "retention opportunity",
            "retention plan",
            "retention action",
            "retention actions",
            "revenue recovered",
            "revenue recovery",
            "recover revenue",
            "revenue can we recover",
            "how much revenue can we recover",
            "customers to target",
            "customer to target",
            "who should we target",
            "which customers should we target",
            "target for a retention campaign",
            "target for retention",
            "campaign segments",
            "campaign segment",
            "save customers",
            "save customer",
            "customers should we save",
            "who should we save",
            "prevent churn",
            "prevent customer churn",
        ]
    ):
        return "retention_optimization"


    """
    Convert a natural-language business question into a supported
    analytics intent.
    """

    q = question.lower().strip()

    if any(
        phrase in q
        for phrase in [
            "revenue at risk",
            "how much revenue is at risk",
            "how much revenue could we lose",
            "revenue risk",
            "revenue could we lose",
            "revenue might we lose",
            "financial risk",
        ]
    ):
        return "revenue_at_risk"

    if any(
        phrase in q
        for phrase in [
            "customer health score",
            "customer health",
            "which customers should we save first",
            "who should we save first",
            "highest value high-risk customers",
            "highest value customers at risk",
            "highest business value",
            "highest customer value",
            "high-risk customers with the highest business value",
            "high-risk customers with the highest customer value",
            "customer decision engine",
            "retention decision",
        ]
    ):
        return "customer_intelligence"

    if any(
        phrase in q
        for phrase in [
            "who should we target",
            "customers should we target",
            "target for retention",
            "retention target",
            "retention targets",
            "who needs retention",
        ]
    ):
        return "retention_targets"

    if any(
        phrase in q
        for phrase in [
            "what should we do",
            "what should the business do",
            "what action should",
            "how should the business respond",
            "recommended action",
            "recommendations for high-risk",
        ]
    ):
        return "retention_actions"

    if any(
        phrase in q
        for phrase in [
            "why are customers leaving",
            "why are customers churning",
            "why is churn increasing",
            "why churn is increasing",
            "why is churn rising",
            "what is causing churn",
            "what causes churn",
            "churn reasons",
        ]
    ):
        return "churn_reasons"

    if any(
        phrase in q
        for phrase in [
            "customer segment",
            "customer segments",
            "highest churn risk segment",
            "highest churn risk segments",
            "which segment has the highest churn",
            "which segments have the highest churn",
        ]
    ):
        return "churn_segments"

    if any(
        phrase in q
        for phrase in [
            "subscription plan",
            "which plan",
            "which subscription",
            "plan has the highest churn",
            "plan with the highest churn",
            "churn rate by plan",
        ]
    ):
        return "churn_by_plan"

    if any(
        phrase in q
        for phrase in [
            "highest churn",
            "top churn",
            "highest churn risk",
            "high churn risk",
            "most likely to churn",
            "most likely customers to churn",
            "who is most likely to churn",
            "which customers are likely to churn",
            "churning customers",
        ]
    ):
        return "highest_churn"

    return "unknown"

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

def build_analytics_context(intent):
    """
    Build a compact, factual analytics context for the local LLM.

    The deterministic analytics remain the source of truth.
    The LLM only explains the supplied results.
    """

    if intent == "data_quality_mlops":
        from src.dashboard.data_quality_mlops import (
            build_data_quality_mlops,
        )

        engine = get_copilot_engine()
        data = build_data_quality_mlops(engine)

        if not data:
            return "No data-quality or MLOps results are available."

        checks = data["checks"]
        passed = sum(checks.values())
        total = len(checks)

        top_features = data["explainability_df"].head(10)

        if top_features.empty:
            driver_text = "Feature importance is unavailable."
        else:
            driver_text = ", ".join(
                top_features["Feature"].astype(str).tolist()
            )

        return f"""
Data Quality & MLOps:

Quality score: {data['quality_score']:.0f}%
Missing feature values: {data['missing_cells']:,}
Missing feature rate: {data['missing_rate']:.2f}%
Feature completeness: {data['feature_completeness']:.1f}%
Duplicate feature rows: {data['duplicate_rows']:,}
Customer coverage: {data['customer_coverage']:.1f}%

MLOps checks passed: {passed}/{total}

Top global model features:
{driver_text}

Important:
Customer-level explanation values are risk-contribution proxies,
not causal explanations.

Answer using only these analytics.
Do not invent missing-value counts, features, customer IDs,
causal explanations, or model results.
"""

    if intent == "production_ai_monitoring":
        from src.dashboard.production_ai_monitoring import (
            build_production_monitoring,
        )

        engine = get_copilot_engine()
        data = build_production_monitoring(engine)

        if not data:
            return "No production monitoring results are available."

        checks = data["checks"]

        passed = sum(checks.values())
        total = len(checks)

        return f"""
Production AI Monitoring:

Production readiness score: {data['readiness_score']:.0f}%
Customers scored: {data['total_customers']:,}
Prediction coverage: {data['prediction_coverage']:.1f}%

Low-risk customers: {data['low_risk']:,}
Medium-risk customers: {data['medium_risk']:,}
High-risk customers: {data['high_risk']:,}

Missing feature values: {data['missing_values']:,}
Duplicate feature rows: {data['duplicate_rows']:,}

Churn model status: {data['model_health']['status']}
Model type: {data['model_health']['model_type']}
Model features: {data['model_health']['features']}
Model trees: {data['model_health']['trees']}

Local AI service: {data['ollama_health']['status']}
Configured AI model: {data['ollama_health']['model']}

Validation checks passed: {passed}/{total}

Answer using only these monitoring results.
Do not invent metrics, model details, or production outcomes.
"""

    if intent == "executive_analytics":
        from src.dashboard.executive_analytics import build_executive_analytics

        engine = get_copilot_engine()
        data = build_executive_analytics(engine)

        if not data:
            return "No executive analytics results are available."

        return f"""
Executive Business Overview:

Customers analyzed: {data['total_customers']:,}
High-risk customers: {data['high_risk']:,}
Average customer health score: {data['avg_health']:.1f}
Average customer value score: {data['avg_value']:.1f}
Average churn risk: {data['avg_churn']:.1f}%

Save Now customers: {data['save_now']:,}
Engage customers: {data['engage']:,}
Customers to target: {data['target_customers']:,}
High-value / high-risk customers: {data['high_value_high_risk']:,}

Monthly revenue at risk: ${data['monthly_revenue_at_risk']:,.2f}
Annualized revenue at risk: ${data['annual_revenue_at_risk']:,.2f}

30% retention-success annual recovery scenario:
${data['recovery_30']:,.2f}

Answer using only these analytics.
Do not invent numbers, customer IDs, financial results, or business outcomes.
Revenue recovery is a scenario estimate, not a guarantee.
"""

    if intent == "retention_optimization":
        from src.dashboard.retention_optimization import (
            build_retention_optimization,
            calculate_recovery_scenarios,
            build_campaign_segments,
        )

        engine = get_copilot_engine()
        df = build_retention_optimization(engine)

        if df.empty:
            return "No retention optimization results are available."

        target = df[df["decision"].isin(["Save Now", "Engage"])].copy()

        monthly_risk = float(
            target["monthly_revenue_at_risk"].sum()
        )
        annual_risk = monthly_risk * 12

        campaigns = build_campaign_segments(df)
        scenarios = calculate_recovery_scenarios(df)

        top_customers = target.head(10)[
            [
                "customer_id",
                "churn_risk",
                "customer_value_score",
                "retention_priority_score",
                "decision",
                "campaign_segment",
                "recommended_action",
                "monthly_revenue_at_risk",
            ]
        ]

        return f"""
Retention Action & Revenue Optimization

Customers to target: {len(target)}
Monthly revenue at risk: ${monthly_risk:,.2f}
Annualized revenue at risk: ${annual_risk:,.2f}

Campaign segments:
{campaigns.to_string(index=False)}

Recovery scenarios:
{scenarios.to_string(index=False)}

Priority customers:
{top_customers.to_string(index=False)}

Use only the analytics above.
Do not invent customer IDs, numbers, campaign segments, or financial results.
Revenue recovery figures are scenario estimates, not guarantees.
"""

    if intent == "customer_intelligence":
        engine = get_copilot_engine()
        df = build_customer_intelligence(engine)

        if df.empty:
            return "No customer intelligence results are available."

        top = df.head(10)[
            [
                "customer_id",
                "churn_risk",
                "customer_health_score",
                "customer_value_score",
                "retention_priority_score",
                "decision",
                "recommended_action",
            ]
        ].copy()

        return f"""
Customer Intelligence Decision Engine:

{top.to_string(index=False)}

Decision definitions:
- Save Now: highest-priority retention intervention.
- Engage: proactive retention engagement.
- Monitor: continue observation.
- Low Priority: no immediate retention intervention.

Important:
- churn_risk is a model-based risk signal, not a guaranteed outcome.
- health and value scores are deterministic business scores.
- Preserve customer IDs and decision labels exactly as supplied.
- Do not invent customers or numbers.
"""

    if intent == "highest_churn":
        df = get_top_churn_customers(limit=10)

        columns = [
            "customer_id",
            "churn_risk",
            "plan_id",
            "days_since_last_viewing",
        ]
        compact = df[columns]

        return f"""
Top 10 customers by predicted churn risk:

{compact.to_string(index=False)}

Important:
- The list above is the complete supplied list.
- Preserve every customer ID in the list.
- churn_risk is a model prediction, not a guaranteed churn outcome.
"""

    if intent == "retention_targets":
        df = get_retention_targets(limit=5)

        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"{row['customer_id']} - "
                f"churn risk {row['churn_risk']:.2f} - "
                f"{row['priority']}"
            )

        return f"""
The analytics system selected these 5 retention targets:

{chr(10).join(lines)}

Important:
- Preserve every customer ID exactly as supplied.
- Preserve the priority exactly as supplied.
- churn risk is a model prediction, not a guaranteed churn outcome.
- Do not add or remove customers from this list.
"""

    if intent == "churn_by_plan":
        df = get_churn_by_plan()

        return f"""
Churn by subscription plan:

{df.to_string(index=False)}

Important:
- Use the plan values exactly as provided.
- Do not claim that subscription plan causes churn.
"""

    if intent == "churn_segments":
        df = get_churn_risk_segments()

        return f"""
Current churn-risk segments:

{df.to_string(index=False)}

Important:
- These are model-based risk segments.
- They describe predicted risk, not guaranteed outcomes.
"""

    if intent == "churn_reasons":
        monthly, reasons = get_churn_trend_analysis()

        return f"""
Observed churn trend:

{monthly.to_string(index=False)}

Observed churn reasons:

{reasons.to_string(index=False)}

Important:
- These are observed patterns.
- Do not claim that these patterns prove causation.
"""

    if intent == "revenue_at_risk":
        summary, by_plan = get_revenue_at_risk()

        return f"""
Revenue at risk:

High-risk customers: {summary["high_risk_customers"]}
Monthly revenue at risk: {summary["monthly_revenue_at_risk"]:.2f}
Annualized revenue at risk: {summary["annualized_revenue_at_risk"]:.2f}

Revenue at risk by plan:

{by_plan.to_string(index=False)}

Important:
- Revenue at risk represents subscription revenue associated with high-risk customers.
- It is not guaranteed revenue loss.
- Use the plan values exactly as provided.
"""

    if intent == "retention_actions":
        df = get_high_risk_recommendations(limit=10)

        columns = [
            "customer_id",
            "churn_risk",
            "recommendation",
        ]

        compact = df[columns]

        return f"""
High-risk customer retention recommendations:

{compact.to_string(index=False)}

Important:
- Preserve every customer ID and recommendation exactly as supplied.
- These are suggested actions based on observed customer signals.
- They are not guaranteed outcomes.
"""

    return "No supported analytics context is available for this question."

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
    intent = classify_question(question)
    if intent != "unknown":
        try:
           analytics_context = build_analytics_context(intent)

           with st.spinner("Analyzing your question..."):
            ai_answer = generate_business_answer(
                question,
                analytics_context,
            )

            st.markdown("### AI Business Analysis")
            st.markdown(ai_answer)

        except Exception as exc:
           st.warning(
                 "The local AI explanation is temporarily unavailable. "
                 "Showing the underlying analytics instead."
           )
           st.caption(f"AI service: {str(exc)}")

    if intent == "highest_churn":
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

    elif intent == "churn_reasons":
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

    elif intent == "churn_segments":
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

    if intent == "retention_actions":
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

    if intent == "revenue_at_risk":
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

    if intent == "retention_targets":
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

    elif intent == "churn_by_plan":
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
        "I can answer business questions about churn, retention, "
        "customers, subscription plans, customer segments, "
        "and revenue at risk."
    )

    st.write(
        "Try questions such as:"
    )

    st.markdown(
        """
        - Which customers are most likely to churn?
        - Which customers should we target for retention?
        - Which subscription plan has the highest churn?
        - Why are customers leaving?
        - How much revenue is at risk?
        - What should the business do about high-risk customers?
        - Which customer segment has the highest churn?
- Which customers should we save first?
- Which high-risk customers have the highest business value?
- What is the customer health score?
        """
    )
