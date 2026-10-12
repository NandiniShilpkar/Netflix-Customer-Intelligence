
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

st.set_page_config(
    page_title="Netflix Customer Intelligence",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Netflix Customer Intelligence")
st.caption("Customer analytics • Subscription insights • Churn intelligence")

DATABASE_URL = os.getenv("DATABASE_URL")


@st.cache_resource
def get_engine(database_url):
    return create_engine(database_url, pool_pre_ping=True)


@st.cache_data(ttl=300)
def load_data(query, database_url):
    engine = get_engine(database_url)
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


if not DATABASE_URL:
    st.error(
        "DATABASE_URL is missing. Add it to your local .env file "
        "using your Neon PostgreSQL connection string."
    )
    st.stop()

try:
    customers = load_data(
        """
        SELECT customer_id, age, country, gender
        FROM customers
        """,
        DATABASE_URL,
    )

    labels = load_data(
        """
        SELECT customer_id, churned
        FROM churn_labels
        """,
        DATABASE_URL,
    )

    plans = load_data(
        """
        SELECT customer_id, plan_id
        FROM subscriptions
        """,
        DATABASE_URL,
    )

    customer_data = customers.merge(
        labels, on="customer_id", how="left"
    )

    total_customers = customer_data["customer_id"].nunique()
    churned_customers = customer_data.loc[
        customer_data["churned"] == True, "customer_id"
    ].nunique()
    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total customers", f"{total_customers:,}")
    col2.metric("Recorded churn", f"{churned_customers:,}")
    col3.metric("Recorded churn rate", f"{churn_rate:.1f}%")

    st.info(
        "Recorded churn reflects the dataset's existing labels. "
        "It is not a live or prospective churn forecast."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Customers by country")
        country_counts = (
            customer_data["country"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .rename_axis("Country")
            .reset_index(name="Customers")
        )
        fig_country = px.bar(
            country_counts,
            x="Country",
            y="Customers",
            title="Top 10 countries",
        )
        st.plotly_chart(fig_country, use_container_width=True)

    with right:
        st.subheader("Recorded churn distribution")
        churn_counts = (
            customer_data["churned"]
            .map({True: "Churned", False: "Not recorded as churned"})
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Label")
            .reset_index(name="Customers")
        )
        fig_churn = px.pie(
            churn_counts,
            names="Label",
            values="Customers",
            hole=0.55,
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    st.subheader("Customer explorer")
    country_options = sorted(
        customer_data["country"].dropna().astype(str).unique()
    )
    selected_country = st.selectbox(
        "Filter by country",
        ["All countries"] + country_options,
    )

    filtered = customer_data.copy()
    if selected_country != "All countries":
        filtered = filtered[
            filtered["country"].astype(str) == selected_country
        ]

    st.dataframe(
        filtered.head(500),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing up to 500 of {len(filtered):,} matching customers."
    )

except Exception as exc:
    st.error("Unable to load dashboard data from PostgreSQL.")
    st.caption(
        "Check your DATABASE_URL, database connectivity, and table names."
    )
    st.exception(exc)
