from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def _safe(value, default="N/A"):
    if pd.isna(value):
        return default
    return value


def _get_customer_history(engine, customer_id):
    query = text(
        """
        SELECT
            c.customer_id,
            c.age,
            c.gender,
            c.city,
            c.country,
            c.preferred_language,

            s.subscription_id,
            s.plan_id,
            s.start_date,
            s.end_date,
            s.status,

            sp.plan_name,
            sp.monthly_price,

            COALESCE(
                va.watch_minutes,
                0
            ) AS watch_minutes,

            COALESCE(
                va.session_count,
                0
            ) AS session_count

        FROM customers c

        LEFT JOIN subscriptions s
            ON c.customer_id = s.customer_id

        LEFT JOIN subscription_plans sp
            ON s.plan_id = sp.plan_id

        LEFT JOIN (
            SELECT
                customer_id,
                SUM(
                    COALESCE(
                        watch_minutes,
                        0
                    )
                ) AS watch_minutes,
                COUNT(*) AS session_count
            FROM viewing_activity
            GROUP BY customer_id
        ) va
            ON c.customer_id = va.customer_id

        WHERE c.customer_id = :customer_id
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={
                "customer_id": customer_id,
            },
        )


def _get_support_summary(engine, customer_id):
    query = text(
        """
        SELECT
            COUNT(*) AS ticket_count,
            SUM(
                CASE
                    WHEN status NOT IN (
                        'resolved',
                        'closed'
                    )
                    THEN 1
                    ELSE 0
                END
            ) AS unresolved_tickets
        FROM support_tickets
        WHERE customer_id = :customer_id
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "customer_id": customer_id,
            },
        ).mappings().first()

    if not result:
        return {
            "ticket_count": 0,
            "unresolved_tickets": 0,
        }

    return {
        "ticket_count": int(
            result["ticket_count"] or 0
        ),
        "unresolved_tickets": int(
            result["unresolved_tickets"] or 0
        ),
    }


def _get_feedback_summary(engine, customer_id):
    query = text(
        """
        SELECT
            COUNT(*) AS feedback_count,
            AVG(rating) AS average_rating
        FROM customer_feedback
        WHERE customer_id = :customer_id
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "customer_id": customer_id,
            },
        ).mappings().first()

    if not result:
        return {
            "feedback_count": 0,
            "average_rating": None,
        }

    return {
        "feedback_count": int(
            result["feedback_count"] or 0
        ),
        "average_rating": result[
            "average_rating"
        ],
    }


def _customer_action(row):
    decision = str(
        row.get(
            "decision",
            "Monitor",
        )
    )

    if decision == "Save Now":
        return (
            "Immediate retention intervention: "
            "contact the customer, investigate the "
            "primary friction point, and offer a "
            "targeted retention intervention."
        )

    if decision == "Engage":
        return (
            "Proactive engagement: increase product "
            "engagement and monitor behavior closely "
            "before risk increases."
        )

    if decision == "Low Priority":
        return (
            "Low-priority monitoring: maintain normal "
            "engagement and monitor for risk changes."
        )

    return (
        "Continue monitoring customer health and "
        "engagement."
    )


def _build_customer_briefing(data):
    prompt = f"""
You are a customer intelligence analyst.

Create a concise business briefing for this customer.

Customer ID: {data['customer_id']}
Plan: {data['plan']}
Monthly Price: {data['monthly_price']}
Churn Risk: {data['churn_risk']:.1f}%
Health Score: {data['health_score']:.1f}
Customer Value Score: {data['value_score']:.1f}
Decision: {data['decision']}
Monthly Revenue at Risk: ${data['monthly_revenue_at_risk']:.2f}
Annual Revenue at Risk: ${data['annual_revenue_at_risk']:.2f}
Watch Minutes: {data['watch_minutes']:,.0f}
Viewing Sessions: {data['viewing_sessions']:,}
Support Tickets: {data['support_tickets']:,}
Unresolved Tickets: {data['unresolved_tickets']:,}
Feedback Count: {data['feedback_count']:,}
Average Feedback Rating: {data['average_rating']}
Recommended Action: {data['recommended_action']}

Explain:
1. Current customer status
2. Main business risk
3. Recommended action

Use only the supplied analytics.
Do not invent customer behavior, financial results,
or causal explanations.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={
                "Content-Type": "application/json",
            },
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 180,
                },
            },
            timeout=120,
        )

        if response.status_code != 200:
            return (
                "Local AI briefing is unavailable. "
                "The structured customer analytics "
                "are still available."
            )

        result = response.json()

        answer = result.get(
            "response",
            "",
        ).strip()

        if not answer:
            return (
                "Local AI returned no briefing. "
                "Use the structured customer analytics."
            )

        return answer

    except Exception:
        return (
            "Local AI briefing is temporarily "
            "unavailable. The structured customer "
            "analytics remain available."
        )


def build_customer_360(engine):
    from src.dashboard.customer_decision_engine import (
        build_customer_intelligence,
    )

    customer_df = build_customer_intelligence(
        engine
    )

    if customer_df.empty:
        return {}

    return {
        "customer_df": customer_df,
    }


def get_customer_360_profile(
    engine,
    customer_id,
):
    data = build_customer_360(engine)

    if not data:
        return {}

    customer_df = data["customer_df"]

    matches = customer_df[
        customer_df["customer_id"].astype(str)
        == str(customer_id)
    ]

    if matches.empty:
        return {}

    row = matches.iloc[0]

    history = _get_customer_history(
        engine,
        customer_id,
    )

    support = _get_support_summary(
        engine,
        customer_id,
    )

    feedback = _get_feedback_summary(
        engine,
        customer_id,
    )

    watch_minutes = 0.0
    viewing_sessions = 0
    plan = "Unknown"
    monthly_price = 0.0

    if not history.empty:
        watch_minutes = float(
            history["watch_minutes"].fillna(0).sum()
        )

        viewing_sessions = int(
            history["session_count"].fillna(0).sum()
        )

        valid_plans = history[
            history["plan_name"].notna()
        ]

        if not valid_plans.empty:
            latest = valid_plans.iloc[-1]

            plan = str(
                _safe(
                    latest["plan_name"],
                    "Unknown",
                )
            )

            monthly_price = float(
                _safe(
                    latest["monthly_price"],
                    0,
                )
            )

    profile = {
        "customer_id": str(
            row["customer_id"]
        ),
        "age": _safe(
            row.get("age")
        ),
        "city": _safe(
            row.get("city")
        ),
        "country": _safe(
            row.get("country")
        ),
        "preferred_language": _safe(
            row.get("preferred_language")
        ),
        "plan": plan,
        "monthly_price": monthly_price,
        "churn_risk": float(
            row["churn_risk"]
        ),
        "health_score": float(
            row["customer_health_score"]
        ),
        "value_score": float(
            row["customer_value_score"]
        ),
        "priority_score": float(
            row["retention_priority_score"]
        ),
        "decision": str(
            row["decision"]
        ),
        "recommended_action": str(
            row["recommended_action"]
        ),
        "monthly_revenue_at_risk": float(
            row["monthly_revenue_at_risk"]
        ),
        "annual_revenue_at_risk": float(
            row["annual_revenue_at_risk"]
        ),
        "watch_minutes": watch_minutes,
        "viewing_sessions": viewing_sessions,
        "support_tickets": support[
            "ticket_count"
        ],
        "unresolved_tickets": support[
            "unresolved_tickets"
        ],
        "feedback_count": feedback[
            "feedback_count"
        ],
        "average_rating": _safe(
            feedback["average_rating"],
            "N/A",
        ),
    }

    profile["recommended_action"] = (
        _customer_action(profile)
    )

    return profile


def render_customer_360(engine):
    st.title("Customer 360 Intelligence")

    st.caption(
        "Individual customer profile combining "
        "churn risk, health, value, behavior, "
        "support, feedback, and retention action."
    )

    data = build_customer_360(engine)

    if not data:
        st.warning(
            "Customer intelligence data is unavailable."
        )
        return

    customer_df = data["customer_df"]

    customer_ids = (
        customer_df["customer_id"]
        .astype(str)
        .sort_values()
        .tolist()
    )

    selected_customer = st.selectbox(
        "Search Customer",
        customer_ids,
    )

    profile = get_customer_360_profile(
        engine,
        selected_customer,
    )

    if not profile:
        st.warning(
            "Customer profile could not be created."
        )
        return

    st.divider()

    st.subheader(
        f"Customer Profile — "
        f"{profile['customer_id']}"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Churn Risk",
            f"{profile['churn_risk']:.1f}%",
        )

    with c2:
        st.metric(
            "Health Score",
            f"{profile['health_score']:.1f}",
        )

    with c3:
        st.metric(
            "Customer Value",
            f"{profile['value_score']:.1f}",
        )

    with c4:
        st.metric(
            "Decision",
            profile["decision"],
        )

    st.divider()

    st.subheader("Customer Details")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.write(
            f"**Plan:** {profile['plan']}"
        )
        st.write(
            f"**Monthly Price:** "
            f"${profile['monthly_price']:.2f}"
        )

    with d2:
        st.write(
            f"**Age:** {profile['age']}"
        )
        st.write(
            f"**Location:** "
            f"{profile['city']}, "
            f"{profile['country']}"
        )

    with d3:
        st.write(
            f"**Language:** "
            f"{profile['preferred_language']}"
        )
        st.write(
            f"**Viewing Sessions:** "
            f"{profile['viewing_sessions']:,}"
        )

    with d4:
        st.write(
            f"**Watch Minutes:** "
            f"{profile['watch_minutes']:,.0f}"
        )
        st.write(
            f"**Feedback Rating:** "
            f"{profile['average_rating']}"
        )

    st.divider()

    st.subheader("Revenue Exposure")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Monthly Revenue at Risk",
            f"${profile['monthly_revenue_at_risk']:,.2f}",
        )

    with r2:
        st.metric(
            "Annual Revenue at Risk",
            f"${profile['annual_revenue_at_risk']:,.2f}",
        )

    with r3:
        st.metric(
            "Retention Priority",
            f"{profile['priority_score']:.1f}",
        )

    st.divider()

    st.subheader("Customer Behavior & Experience")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.metric(
            "Viewing Sessions",
            f"{profile['viewing_sessions']:,}",
        )

    with b2:
        st.metric(
            "Watch Minutes",
            f"{profile['watch_minutes']:,.0f}",
        )

    with b3:
        st.metric(
            "Support Tickets",
            f"{profile['support_tickets']:,}",
        )

    with b4:
        st.metric(
            "Unresolved Tickets",
            f"{profile['unresolved_tickets']:,}",
        )

    st.divider()

    st.subheader("Retention Recommendation")

    if profile["decision"] == "Save Now":
        st.error(
            profile["recommended_action"]
        )
    elif profile["decision"] == "Engage":
        st.warning(
            profile["recommended_action"]
        )
    else:
        st.info(
            profile["recommended_action"]
        )

    st.divider()

    st.subheader("AI Customer Briefing")

    if st.button(
        "Generate AI Customer Briefing",
        type="primary",
    ):
        with st.spinner(
            "Generating customer briefing..."
        ):
            briefing = _build_customer_briefing(
                profile
            )

        st.write(briefing)

    st.divider()

    st.subheader("Customer Export")

    export_df = pd.DataFrame(
        [profile]
    )

    st.download_button(
        "Download Customer 360 Profile",
        data=export_df.to_csv(
            index=False
        ),
        file_name=(
            f"{profile['customer_id']}"
            "_customer_360.csv"
        ),
        mime="text/csv",
    )


build_customer_360_context = build_customer_360
