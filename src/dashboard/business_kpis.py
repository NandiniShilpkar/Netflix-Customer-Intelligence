
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text


def render_business_kpis(engine):
    st.header("Business Performance Analytics")

    queries = {
        "payments": """
            SELECT payment_status, amount, payment_date
            FROM payments
        """,
        "viewing": """
            SELECT viewing_date, watch_duration_minutes,
                   completion_percentage, device_type
            FROM viewing_activity
        """,
        "support": """
            SELECT issue_category, ticket_status,
                   resolution_time_hours,
                   customer_satisfaction_score
            FROM support_tickets
        """,
        "subscriptions": """
            SELECT s.plan_id, p.plan_name, s.subscription_status,
                   s.monthly_price
            FROM subscriptions s
            LEFT JOIN subscription_plans p ON s.plan_id = p.plan_id
        """,
    }

    try:
        with engine.connect() as connection:
            data = {
                name: pd.read_sql(text(query), connection)
                for name, query in queries.items()
            }
    except Exception as exc:
        st.error("Unable to load business analytics.")
        st.exception(exc)
        return

    payments = data["payments"]
    viewing = data["viewing"]
    support = data["support"]
    subscriptions = data["subscriptions"]

    # Payments
    st.subheader("Payments")
    total_payments = len(payments)
    successful = (
        payments["payment_status"]
        .astype(str)
        .str.lower()
        .eq("success")
        .sum()
    )
    failed = (
        payments["payment_status"]
        .astype(str)
        .str.lower()
        .eq("failed")
        .sum()
    )
    success_rate = successful / total_payments * 100 if total_payments else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Payment records", f"{total_payments:,}")
    c2.metric("Successful", f"{successful:,}")
    c3.metric("Failed", f"{failed:,}")
    c4.metric("Success rate", f"{success_rate:.1f}%")

    st.caption(
        "Revenue below is the sum of amounts on successful payment records, "
        "not necessarily recognized accounting revenue."
    )
    successful_amount = payments.loc[
        payments["payment_status"].astype(str).str.lower().eq("success"),
        "amount",
    ].sum()
    st.metric("Successful payment amount", f"{successful_amount:,.2f}")

    if not payments.empty:
        payment_counts = (
            payments["payment_status"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Records")
        )

        fig = px.bar(
            payment_counts,
            x="Status",
            y="Records",
            title="Payment status distribution",
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


    # Viewing activity
    st.subheader("Viewing Activity")
    total_sessions = len(viewing)
    total_watch_hours = (
        pd.to_numeric(viewing["watch_duration_minutes"], errors="coerce")
        .sum() / 60
    )
    avg_completion = pd.to_numeric(
        viewing["completion_percentage"], errors="coerce"
    ).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Viewing records", f"{total_sessions:,}")
    c2.metric("Recorded watch hours", f"{total_watch_hours:,.0f}")
    c3.metric(
        "Average completion",
        f"{avg_completion:.1f}%" if pd.notna(avg_completion) else "N/A",
    )

    if not viewing.empty:


        devices = (
            viewing["device_type"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Device")
            .reset_index(name="Viewing records")
        )


        fig = px.bar(
            devices,
            x="Device",
            y="Viewing records",
            title="Viewing records by device",
            template="plotly_dark",
            color_discrete_sequence=["#E50914"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,8,9,0.95)",
            font=dict(color="#F5F5F1"),
            margin=dict(l=20, r=20, t=55, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        # devices = (
        #     viewing["device_type"]
        #     .fillna("Unknown")
        #     .value_counts()
        #     .rename_axis("Device")
        #     .reset_index(name="Viewing records")
        # )
        # st.plotly_chart(
        #     px.bar(devices, x="Device", y="Viewing records",
        #            title="Viewing records by device"),
        #     use_container_width=True,
        # )

    # Customer support
    st.subheader("Customer Support")
    total_tickets = len(support)
    avg_resolution = pd.to_numeric(
        support["resolution_time_hours"], errors="coerce"
    ).mean()
    avg_satisfaction = pd.to_numeric(
        support["customer_satisfaction_score"], errors="coerce"
    ).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Support tickets", f"{total_tickets:,}")
    c2.metric(
        "Average resolution (hours)",
        f"{avg_resolution:.1f}" if pd.notna(avg_resolution) else "N/A",
    )
    c3.metric(
        "Average satisfaction",
        f"{avg_satisfaction:.2f}/5"
        if pd.notna(avg_satisfaction) else "N/A",
    )

    if not support.empty:

        categories = (
            support["issue_category"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Issue category")
            .reset_index(name="Tickets")
        )


        fig = px.bar(
            categories,
            x="Issue category",
            y="Tickets",
            title="Tickets by issue category",
            template="plotly_dark",
            color_discrete_sequence=["#E50914"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,8,9,0.95)",
            font=dict(color="#F5F5F1"),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # categories = (
        #     support["issue_category"]
        #     .fillna("Unknown")
        #     .value_counts()
        #     .rename_axis("Issue category")
        #     .reset_index(name="Tickets")
        # )
        # st.plotly_chart(
        #     px.bar(categories, x="Issue category", y="Tickets",
        #            title="Tickets by issue category"),
        #     use_container_width=True,
        # )

    # Subscriptions
    st.subheader("Subscription Plans")
    if subscriptions.empty:
        st.info("No subscription records found.")
    else:
        plan_counts = (
            subscriptions["plan_name"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Plan")
            .reset_index(name="Subscriptions")
        )

        fig = px.bar(
            plan_counts,
            x="Plan",
            y="Subscriptions",
            title="Subscription records by plan",
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


        avg_price = pd.to_numeric(
            subscriptions["monthly_price"], errors="coerce"
        ).mean()
        st.metric(
            "Average recorded subscription price",
            f"{avg_price:,.2f}" if pd.notna(avg_price) else "N/A",
        )
        st.caption(
            "These are subscription records, not necessarily distinct "
            "customers or currently active subscriptions."
        )
