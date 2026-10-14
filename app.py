
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from src.dashboard.churn_intelligence import render_churn_intelligence
from src.dashboard.business_kpis import render_business_kpis
from src.dashboard.customer_segmentation import render_customer_segmentation
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# Netflix Cherry theme for all Plotly charts
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#E50914",  # Netflix red
    "#B20710",  # Cherry red
    "#F5F5F1",  # Off-white
    "#8B1E2D",  # Deep cherry
    "#777777",  # Grey
]
px.defaults.color_continuous_scale = [
    "#241012",
    "#8B1E2D",
    "#B20710",
    "#E50914",
]

# Force dark styling on charts created with Plotly Express
import plotly.io as pio

pio.templates["netflix_dark"] = pio.templates["plotly_dark"]
pio.templates.default = "netflix_dark"

px.defaults.template = "netflix_dark"



load_dotenv()

st.set_page_config(
    page_title="Netflix Customer Intelligence",
    page_icon="🎬",
    layout="wide",
)

# Netflix-style brand header
st.markdown("""
<style>
.netflix-brand {
    text-align: center;
    padding: 8px 0 22px;
    border-bottom: 1px solid #461216;
    margin-bottom: 24px;
}
.netflix-logo {
    color: #E50914;
    font-family: Impact, "Arial Narrow", sans-serif;
    font-size: clamp(38px, 5vw, 66px);
    font-weight: 900;
    letter-spacing: 3px;
    line-height: 1.1;
}
.netflix-subtitle {
    color: #F5F5F1;
    font-size: clamp(23px, 3vw, 38px);
    font-weight: 750;
    margin-top: 5px;
}
.netflix-tagline {
    color: #999;
    font-size: 12px;
    letter-spacing: 3px;
    margin-top: 10px;
}
</style>

<div class="netflix-brand">
    <div class="netflix-logo">NETFLIX</div>
    <div class="netflix-subtitle">Customer Intelligence</div>
    <div class="netflix-tagline">
        DATA-DRIVEN INSIGHTS FOR A SMARTER STREAMING BUSINESS
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Netflix-inspired dashboard theme
# --------------------------------------------------

st.markdown("""
<style>
    :root {
        --netflix-red: #E50914;
        --netflix-black: #141414;
        --netflix-panel: #222222;
        --netflix-muted: #B3B3B3;
    }

    .stApp {
        background-color: var(--netflix-black);
        color: #FFFFFF;
    }

    [data-testid="stHeader"] {
        background: rgba(20, 20, 20, 0.95);
    }

    [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #100607 0%, #080808 100%);
    border-right: 1px solid #7F1D2D;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #F5F5F1;
}

[data-testid="stSidebar"] hr {
    border-color: #492024;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #2A171A, #101010);
    border: 1px solid #7F1D2D;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(229, 9, 20, 0.12);
}

[data-testid="stMetricValue"] {
    color: #E50914 !important;
    font-weight: 800 !important;
}

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background: var(--netflix-panel);
        border: 1px solid #353535;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--netflix-muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background-color: var(--netflix-red);
        color: #FFFFFF;
        border: 1px solid var(--netflix-red);
        border-radius: 7px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #B20710;
        color: #FFFFFF;
        border-color: #B20710;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #353535;
        border-radius: 8px;
    }

    hr {
        border-color: #353535;
    }

    .stCaption {
        color: var(--netflix-muted);
    }

/* Netflix-style dashboard boards */
[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg, #201719, #111111);
    border: 1px solid #492024;
    border-radius: 14px;
    padding: 10px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.3);
}

/* KPI and metric boards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #2A171A, #151515);
    border: 1px solid #7F1D2D;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(229, 9, 20, 0.08);
}

/* Expandable dashboard panels */
[data-testid="stExpander"] {
    background: #181313;
    border: 1px solid #492024;
    border-radius: 12px;
}

/* Tables and data boards */
[data-testid="stDataFrame"] {
    border: 1px solid #492024;
    border-radius: 12px;
    overflow: hidden;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #B3B3B3;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #E50914;
    border-bottom-color: #E50914;
}

/* Fixed Netflix collage background across the entire app */
.stApp {
    background-color: #080808;
    background-image:
        linear-gradient(
            rgba(0, 0, 0, 0.68),
            rgba(0, 0, 0, 0.84)
        ),
        url("assets/netflix-hero-image.avif");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Keep the page transparent so the background remains visible */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}

/* Dark, translucent analytics boards */
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stExpander"] {
    background-color: rgba(12, 8, 9, 0.90);
    border: 1px solid rgba(229, 9, 20, 0.55);
    border-radius: 14px;
}

/* Keep the sidebar dark and readable */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 8, 0.96);
    border-right: 1px solid #7F1D2D;
}


</style>
""", unsafe_allow_html=True)


# st.title("🎬 Netflix Customer Intelligence")
# st.caption("Customer analytics • Subscription insights • Churn intelligence")

# Netflix-style sidebar navigation
st.sidebar.markdown(
    """
    <div style="text-align:center; padding:10px 0 24px;">
        <div style="
            color:#E50914;
            font-family:Impact, 'Arial Narrow', sans-serif;
            font-size:32px;
            font-weight:900;
            letter-spacing:2px;">
            NETFLIX
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Netflix-style sidebar navigation without radio buttons
# st.sidebar.markdown(
#     """
#     <div style="
#         text-align:center;
#         padding:12px 0 28px;
#         color:#E50914;
#         font-family:Impact, 'Arial Narrow', sans-serif;
#         font-size:32px;
#         font-weight:900;
#         letter-spacing:2px;">
#         NETFLIX
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

if "dashboard_section" not in st.session_state:
    st.session_state.dashboard_section = "Dashboard"

menu_items = [
    ("Dashboard", "🏠"),
    ("Business KPIs", "📊"),
    ("Churn Intelligence", "👥"),
    ("Customer Segmentation", "📈"),
]

for label, icon in menu_items:
    is_active = st.session_state.dashboard_section == label
    background = "#B20710" if is_active else "transparent"
    border = "#E50914" if is_active else "transparent"

    if st.sidebar.button(
        f"{icon}   {label}",
        key=f"nav_{label}",
        use_container_width=True,
    ):
        st.session_state.dashboard_section = label
        st.rerun()

    st.sidebar.markdown(
        f"""
        <style>
        div[data-testid="stSidebar"] button[kind="secondary"] {{
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

dashboard_section = st.session_state.dashboard_section

# dashboard_section = st.sidebar.radio(
#     "NAVIGATION",
#     [
#         "🏠 Dashboard",
#         "📊 Business KPIs",
#         "👥 Churn Intelligence",
#         "📈 Customer Segmentation",
#     ],
#     label_visibility="collapsed",
# )

st.sidebar.markdown("---")
st.sidebar.caption("CUSTOMER INTELLIGENCE")

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
    if dashboard_section == "Dashboard":
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
            template="plotly_dark",
            color_discrete_sequence=["#E50914"],
        )
        fig_country.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,8,9,0.95)",
            font=dict(color="#F5F5F1"),
            title_font=dict(color="#F5F5F1"),
            margin=dict(l=20, r=20, t=55, b=20),
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
            template="plotly_dark",
            color_discrete_sequence=["#E50914", "#777777", "#B20710"],
        )
        fig_churn.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,8,9,0.95)",
            font=dict(color="#F5F5F1"),
            title_font=dict(color="#F5F5F1"),
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(font=dict(color="#F5F5F1")),
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
    if dashboard_section == "Business KPIs":
        render_business_kpis(get_engine(DATABASE_URL))

    elif dashboard_section == "Churn Intelligence":
        render_churn_intelligence(get_engine(DATABASE_URL))

    elif dashboard_section == "Customer Segmentation":
        render_customer_segmentation()
    # render_business_kpis(get_engine(DATABASE_URL))
    # render_churn_intelligence(get_engine(DATABASE_URL))
    # render_customer_segmentation()

except Exception as exc:
    st.error("Unable to load dashboard data from PostgreSQL.")
    st.caption(
        "Check your DATABASE_URL, database connectivity, and table names."
    )
    st.exception(exc)
