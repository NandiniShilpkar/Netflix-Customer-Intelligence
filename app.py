
import os



import pandas as pd

import plotly.express as px

import plotly.io as pio

import streamlit as st

from dotenv import load_dotenv

from sqlalchemy import create_engine, text



from src.dashboard.business_kpis import render_business_kpis

from src.dashboard.churn_intelligence import render_churn_intelligence

from src.dashboard.customer_segmentation import render_customer_segmentation
from src.dashboard.ai_analytics_copilot import render_ai_analytics_copilot
from src.dashboard.customer_decision_engine import render_customer_decision_engine
from src.dashboard.retention_optimization import render_retention_optimization
from src.dashboard.executive_analytics import render_executive_analytics
from src.dashboard.production_ai_monitoring import render_production_ai_monitoring
from src.dashboard.data_quality_mlops import render_data_quality_mlops
from src.dashboard.customer_360 import render_customer_360





# --------------------------------------------------

# Configuration

# --------------------------------------------------



load_dotenv()



st.set_page_config(

    page_title="Netflix Customer Intelligence",

    page_icon="🎬",

    layout="wide",

    initial_sidebar_state="expanded",

)



NETFLIX_RED = "#E50914"

NETFLIX_DARK_RED = "#B20710"

NETFLIX_WHITE = "#F5F5F1"

NETFLIX_MUTED = "#B3B3B3"



px.defaults.template = "plotly_dark"

px.defaults.color_discrete_sequence = [

    NETFLIX_RED,

    NETFLIX_DARK_RED,

    NETFLIX_WHITE,

    "#8B1E2D",

    "#777777",

]

px.defaults.color_continuous_scale = [

    "#241012",

    "#8B1E2D",

    NETFLIX_DARK_RED,

    NETFLIX_RED,

]



pio.templates["netflix_dark"] = pio.templates["plotly_dark"]

pio.templates.default = "netflix_dark"

px.defaults.template = "netflix_dark"





# --------------------------------------------------

# Netflix-inspired styling

# --------------------------------------------------



st.markdown(

    """

    <style>

    :root {

        --netflix-red: #E50914;

        --netflix-dark-red: #B20710;

        --netflix-black: #080808;

        --netflix-panel: rgba(18, 12, 13, 0.94);

        --netflix-border: rgba(229, 9, 20, 0.38);

        --netflix-muted: #B3B3B3;

    }



    .stApp {

        background-color: var(--netflix-black);

        background-image:

            linear-gradient(

                rgba(0, 0, 0, 0.80),

                rgba(0, 0, 0, 0.94)

            ),

            url("assets/netflix-hero-image.avif");

        background-size: cover;

        background-position: center top;

        background-repeat: no-repeat;

        background-attachment: fixed;

        color: #FFFFFF;

    }



    [data-testid="stAppViewContainer"],

    [data-testid="stMain"],

    [data-testid="stMainBlockContainer"] {

        background: transparent;

    }



    [data-testid="stHeader"] {

        background: rgba(8, 8, 8, 0.88);

    }



    [data-testid="stMainBlockContainer"] {

        padding-top: 1.5rem;

        padding-bottom: 3rem;

    }



    /* Brand header */

    .netflix-brand {

        text-align: center;

        padding: 12px 0 26px;

        margin-bottom: 26px;

        border-bottom: 1px solid #492024;

    }



    .netflix-logo {

        color: var(--netflix-red);

        font-family: Impact, "Arial Narrow", sans-serif;

        font-size: clamp(40px, 6vw, 68px);

        font-weight: 900;

        letter-spacing: 4px;

        line-height: 1.1;

        text-shadow: 0 4px 24px rgba(229, 9, 20, 0.20);

    }



    .netflix-subtitle {

        color: #FFFFFF;

        font-size: clamp(23px, 3vw, 36px);

        font-weight: 750;

        margin-top: 8px;

    }



    .netflix-tagline {

        color: var(--netflix-muted);

        font-size: 11px;

        letter-spacing: 2px;

        margin-top: 12px;

    }



    /* Sidebar */

    [data-testid="stSidebar"] {

        background: linear-gradient(

            180deg,

            #16090B 0%,

            #0A0A0A 55%,

            #080808 100%

        );

        border-right: 1px solid #492024;

    }



    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {

        color: var(--netflix-white);

    }



    [data-testid="stSidebar"] hr {

        border-color: #492024;

    }



    [data-testid="stSidebar"] .stButton > button {

        width: 100%;

        min-height: 44px;

        text-align: left;

        border-radius: 9px;

        font-weight: 600;

        transition: all 0.2s ease;

    }



    [data-testid="stSidebar"] .stButton > button[kind="primary"] {

        background: var(--netflix-dark-red);

        color: #FFFFFF;

        border: 1px solid var(--netflix-red);

        box-shadow: 0 4px 14px rgba(229, 9, 20, 0.18);

    }



    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {

        background: rgba(255, 255, 255, 0.025);

        color: #FFFFFF;

        border: 1px solid transparent;

    }



    [data-testid="stSidebar"] .stButton > button:hover {

        background: #3A1115;

        border-color: var(--netflix-red);

        color: #FFFFFF;

    }



    /* Headings and text */

    h1, h2, h3 {

        color: #FFFFFF !important;

        font-weight: 700 !important;

    }



    h2 {

        padding-bottom: 5px;

    }



    [data-testid="stCaptionContainer"],

    .stCaption {

        color: var(--netflix-muted);

    }



    /* KPI cards */

    [data-testid="stMetric"] {

        background: linear-gradient(

            145deg,

            rgba(47, 20, 24, 0.96),

            rgba(15, 12, 12, 0.96)

        );

        border: 1px solid var(--netflix-border);

        border-radius: 14px;

        padding: 18px 20px;

        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.22);

        transition: border-color 0.2s ease;

    }



    [data-testid="stMetric"]:hover {

        border-color: var(--netflix-red);

    }



    [data-testid="stMetricLabel"] {

        color: var(--netflix-muted) !important;

    }



    [data-testid="stMetricValue"] {

        color: var(--netflix-red) !important;

        font-weight: 800 !important;

    }



    [data-testid="stMetricDelta"] {

        color: var(--netflix-white);

    }



    /* Chart panels */

    [data-testid="stPlotlyChart"] {

        background: var(--netflix-panel);

        border: 1px solid var(--netflix-border);

        border-radius: 14px;

        padding: 8px;

        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.22);

    }



    /* Data tables */

    [data-testid="stDataFrame"] {

        border: 1px solid var(--netflix-border);

        border-radius: 12px;

        overflow: hidden;

    }



    /* Expanders */

    [data-testid="stExpander"] {

        background: rgba(18, 12, 13, 0.90);

        border: 1px solid var(--netflix-border);

        border-radius: 12px;

    }



    /* Buttons */

    .stButton > button,

    .stDownloadButton > button {

        border-radius: 8px;

        font-weight: 600;

        transition: all 0.2s ease;

    }



    .stButton > button[kind="primary"],

    .stDownloadButton > button {

        background: var(--netflix-red);

        color: #FFFFFF;

        border: 1px solid var(--netflix-red);

    }



    .stButton > button[kind="primary"]:hover,

    .stDownloadButton > button:hover {

        background: var(--netflix-dark-red);

        border-color: var(--netflix-dark-red);

        color: #FFFFFF;

    }



    /* Inputs */

    [data-baseweb="select"] > div,

    [data-baseweb="input"] > div {

        border-radius: 8px;

    }



    /* Tabs */

    button[data-baseweb="tab"] {

        color: var(--netflix-muted);

    }



    button[data-baseweb="tab"][aria-selected="true"] {

        color: var(--netflix-red);

        border-bottom-color: var(--netflix-red);

    }



    hr {

        border-color: #353535;

    }



    /* Alerts */

    [data-testid="stAlert"] {

        border-radius: 10px;

    }



    /* Responsive spacing */

    @media (max-width: 768px) {

        [data-testid="stMainBlockContainer"] {

            padding: 1rem 1rem 2rem;

        }



        .netflix-brand {

            padding-bottom: 18px;

        }



        .netflix-tagline {

            font-size: 9px;

            letter-spacing: 1px;

        }

    }



# /* White button with hidden text until hover */

# .stApp [data-testid="stBaseButton-secondary"] {

#     background: #FFFFFF !important;

#     color: #FFFFFF !important;

#     border: 1px solid #FFFFFF !important;

#     transition: all 0.2s ease;

# }



# .stApp [data-testid="stBaseButton-secondary"] p,

# .stApp [data-testid="stBaseButton-secondary"] span {

#     color: #FFFFFF !important;

# }



# .stApp [data-testid="stBaseButton-secondary"]:hover {

#     background: #E50914 !important;

#     color: #FFFFFF !important;

#     border-color: #E50914 !important;

# }



# .stApp [data-testid="stBaseButton-secondary"]:hover p,

# .stApp [data-testid="stBaseButton-secondary"]:hover span {

#     color: #FFFFFF !important;

# }





# /* Strong override for Streamlit buttons */

# .stApp button,

# .stApp button p,

# .stApp button span {

#     color: #FFFFFF !important;

# }



# .stApp [data-testid="stBaseButton-secondary"],

# .stApp [data-testid="stBaseButton-primary"] {

#     background: #E50914 !important;

#     color: #FFFFFF !important;

#     border: 1px solid #E50914 !important;

#     opacity: 1 !important;

# }



# .stApp [data-testid="stBaseButton-secondary"]:hover,

# .stApp [data-testid="stBaseButton-primary"]:hover {

#     background: #B20710 !important;

#     color: #FFFFFF !important;

#     border-color: #B20710 !important;

# }



# .stApp [data-testid="stBaseButton-secondary"] p,

# .stApp [data-testid="stBaseButton-primary"] p {

#     color: #FFFFFF !important;

# }





    # /* Fix experimental churn risk button */

    #     .stButton > button {

    #         background-color: #E50914 !important;

    #         color: #FFFFFF !important;

    #         border: 1px solid #E50914 !important;

    #         border-radius: 8px !important;

    #         font-weight: 700 !important;

    #         opacity: 1 !important;

    #     }



    #     .stButton > button:hover {

    #         background-color: #B20710 !important;

    #         color: #FFFFFF !important;

    #         border-color: #B20710 !important;

    #     }



    #     .stButton > button p,

    #     .stButton > button span {

    #         color: #FFFFFF !important;

    #     }



/* Only the experimental churn risk button */

.stApp [class*="st-key-calculate_experimental_churn_risk"]

[data-testid="stBaseButton-secondary"] {

    background: transparent !important;

    color: #FFFFFF !important;

    border: 1px solid transparent !important;

    transition: background-color 0.2s ease, border-color 0.2s ease;

}



.stApp [class*="st-key-calculate_experimental_churn_risk"]

[data-testid="stBaseButton-secondary"]:hover {

    background: #B20710 !important;

    color: #FFFFFF !important;

    border-color: #E50914 !important;

}



.stApp [class*="st-key-calculate_experimental_churn_risk"]

[data-testid="stBaseButton-secondary"] p {

    color: #FFFFFF !important;

}
    </style>

    """,
    unsafe_allow_html=True,

)





# --------------------------------------------------

# Header

# --------------------------------------------------



st.markdown(
    """
    <div class="netflix-brand">
        <div class="netflix-logo">NETFLIX</div>
        <div class="netflix-subtitle">Customer Intelligence</div>
        <div class="netflix-tagline">
            DATA-DRIVEN INSIGHTS FOR A SMARTER STREAMING BUSINESS
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)





# --------------------------------------------------

# Sidebar navigation

# --------------------------------------------------








st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] h1 {
        color: #E50914 !important;
        font-family: Impact, "Arial Narrow", sans-serif;
        letter-spacing: 3px;
        text-align: center !important;
        width: 100%;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        width: 100%;
        text-align: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        width: 100%;
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("NETFLIX")

st.sidebar.caption("ANALYTICS COPILOT")

st.sidebar.divider()






menu_items = [

    ("Dashboard", "🏠"),

    ("Business KPIs", "📊"),

    ("Churn Intelligence", "👥"),

    ("Customer Segmentation", "📈"),
    ("Customer Intelligence", "🎯"),
    ("Retention Optimization", "💰"),
("Executive Analytics", "📊"),
("Production AI & Monitoring", "🧠"),
("Data Quality & MLOps", "🔍"),
    ("AI Analytics Copilot", "🤖"),

]



if "dashboard_section" not in st.session_state:

    st.session_state.dashboard_section = "Dashboard"



for label, icon in menu_items:

    is_active = st.session_state.dashboard_section == label



    if st.sidebar.button(

        f"{icon}   {label}",

        key=f"nav_{label}",

        type="primary" if is_active else "secondary",

        use_container_width=True,

    ):

        st.session_state.dashboard_section = label

        st.rerun()



st.sidebar.markdown("---")

st.sidebar.caption("CUSTOMER INTELLIGENCE")

st.sidebar.caption("PostgreSQL • Streamlit • AI Analytics")





# --------------------------------------------------

# Database helpers

# --------------------------------------------------



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

        "using your PostgreSQL connection string."

    )

    st.stop()





# --------------------------------------------------

# Page rendering

# --------------------------------------------------



try:
    dashboard_section = st.session_state.dashboard_section



    if dashboard_section == "Dashboard":

        # -------------------------------

        # Load and prepare customer data

        # -------------------------------

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



        customer_data = customers.merge(

            labels,

            on="customer_id",

            how="left",

            validate="one_to_one",

        )



        total_customers = customer_data["customer_id"].nunique()



        churned_customers = customer_data.loc[

            customer_data["churned"].eq(True),

            "customer_id",

        ].nunique()



        churn_rate = (

            churned_customers / total_customers * 100

            if total_customers > 0

            else 0

        )



        # -------------------------------

        # Dashboard heading

        # -------------------------------

        st.title("Customer Intelligence Dashboard")

        st.caption(

            "Explore customer demographics, historical churn labels, "

            "and customer-level records."

        )

        st.divider()



        # -------------------------------

        # KPI metrics

        # -------------------------------

        st.subheader("Customer overview")



        metric1, metric2, metric3 = st.columns(3, gap="medium")



        with metric1:

            st.metric(

                label="Total customers",

                value=f"{total_customers:,}",

            )



        with metric2:

            st.metric(

                label="Recorded churn",

                value=f"{churned_customers:,}",

            )



        with metric3:

            st.metric(

                label="Recorded churn rate",

                value=f"{churn_rate:.1f}%",

            )



        st.caption(

            "These figures use existing historical churn labels. "

            "They are not predictions of future customer churn."

        )



        st.divider()



        # -------------------------------

        # Chart data

        # -------------------------------

        customer_data["country_display"] = (

            customer_data["country"]

            .fillna("Unknown")

            .astype(str)

        )



        customer_data["churn_status"] = (

            customer_data["churned"]

            .map({

                True: "Churned",

                False: "Not recorded as churned",

            })

            .fillna("Unknown")

        )



        left, right = st.columns(2, gap="large")



        # -------------------------------

        # Customers by country

        # -------------------------------

        with left:

            st.subheader("Customer geography")



            country_counts = (

                customer_data["country_display"]

                .value_counts()

                .head(10)

                .rename_axis("Country")

                .reset_index(name="Customers")

                .sort_values("Customers", ascending=True)

            )



            fig_country = px.bar(

                country_counts,

                x="Customers",

                y="Country",

                orientation="h",

                title="Top 10 countries by customer count",

                template="netflix_dark",

                color_discrete_sequence=[NETFLIX_RED],

                text="Customers",

            )



            fig_country.update_traces(

                textposition="outside",

                cliponaxis=False,

                hovertemplate=(

                    "<b>%{y}</b><br>"

                    "Customers: %{x:,}"

                    "<extra></extra>"

                ),

            )



            fig_country.update_layout(

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(12,8,9,0.95)",

                font=dict(color=NETFLIX_WHITE, size=12),

                title_font=dict(color=NETFLIX_WHITE, size=16),

                margin=dict(l=10, r=35, t=65, b=25),

                xaxis_title="Customers",

                yaxis_title="",

                showlegend=False,

            )



            fig_country.update_xaxes(

                showgrid=True,

                gridcolor="rgba(255,255,255,0.08)",

                zeroline=False,

            )



            fig_country.update_yaxes(

                showgrid=False,

                zeroline=False,

            )



            st.plotly_chart(

                fig_country,

                use_container_width=True,

                key="dashboard_country_chart",

            )



        # -------------------------------

        # Historical churn distribution

        # -------------------------------

        with right:

            st.subheader("Historical churn distribution")



            churn_counts = (

                customer_data["churn_status"]

                .value_counts()

                .reindex(

                    [

                        "Churned",

                        "Not recorded as churned",

                        "Unknown",

                    ],

                    fill_value=0,

                )

                .rename_axis("Status")

                .reset_index(name="Customers")

            )



            churn_counts = churn_counts[

                churn_counts["Customers"] > 0

            ]



            fig_churn = px.pie(

                churn_counts,

                names="Status",

                values="Customers",

                hole=0.58,

                title="Distribution of existing churn labels",

                template="netflix_dark",

                color="Status",

                color_discrete_map={

                    "Churned": NETFLIX_RED,

                    "Not recorded as churned": "#777777",

                    "Unknown": NETFLIX_DARK_RED,

                },

            )



            fig_churn.update_traces(

                textposition="inside",

                textinfo="percent",

                hovertemplate=(

                    "<b>%{label}</b><br>"

                    "Customers: %{value:,}<br>"

                    "Share: %{percent}"

                    "<extra></extra>"

                ),

            )



            fig_churn.update_layout(

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(12,8,9,0.95)",

                font=dict(color=NETFLIX_WHITE, size=12),

                title_font=dict(color=NETFLIX_WHITE, size=16),

                margin=dict(l=10, r=10, t=65, b=25),

                legend=dict(

                    orientation="h",

                    yanchor="top",

                    y=-0.08,

                    xanchor="center",

                    x=0.5,

                    font=dict(color=NETFLIX_WHITE, size=10),

                ),

            )



            st.plotly_chart(

                fig_churn,

                use_container_width=True,

                key="dashboard_churn_chart",

            )



        st.divider()



        # -------------------------------

        # Customer explorer

        # -------------------------------

        st.subheader("Customer explorer")

        st.caption(

            "Filter customer records and download the matching results."

        )



        filter1, filter2 = st.columns(2, gap="medium")



        country_options = sorted(

            customer_data["country_display"].unique().tolist()

        )



        with filter1:

            selected_country = st.selectbox(

                "Country",

                options=["All countries"] + country_options,

                key="dashboard_country_filter",

            )



        churn_options = [

            "All statuses",

            "Churned",

            "Not recorded as churned",

            "Unknown",

        ]



        with filter2:

            selected_churn_status = st.selectbox(

                "Historical churn status",

                options=churn_options,

                key="dashboard_churn_filter",

            )



        filtered = customer_data.copy()



        if selected_country != "All countries":

            filtered = filtered[

                filtered["country_display"] == selected_country

            ]



        if selected_churn_status != "All statuses":

            filtered = filtered[

                filtered["churn_status"] == selected_churn_status

            ]



        # Keep internal helper columns out of the customer table.

        display_columns = [

            "customer_id",

            "age",

            "country_display",

            "gender",

            "churn_status",

        ]



        display_data = filtered[display_columns].rename(

            columns={

                "customer_id": "Customer ID",

                "age": "Age",

                "country_display": "Country",

                "gender": "Gender",

                "churn_status": "Historical churn status",

            }

        )



        st.caption(

            f"Matching customers: {len(display_data):,}"

        )



        st.dataframe(

            display_data.head(500),

            use_container_width=True,

            hide_index=True,

        )



        st.download_button(

            label="Download filtered customers (CSV)",

            data=display_data.to_csv(index=False).encode("utf-8"),

            file_name="netflix_filtered_customers.csv",

            mime="text/csv",

            key="download_dashboard_customers",

        )



        if len(display_data) > 500:

            st.info(

                "The table displays the first 500 matching records. "

                "The CSV download includes all matching records."

            )



    elif dashboard_section == "Business KPIs":

        render_business_kpis(get_engine(DATABASE_URL))



    elif dashboard_section == "Churn Intelligence":

        render_churn_intelligence(get_engine(DATABASE_URL))



    elif dashboard_section == "Customer Segmentation":

        render_customer_segmentation()


    elif dashboard_section == "Customer Intelligence":

        render_customer_decision_engine(get_engine(DATABASE_URL))

        st.divider()

        render_customer_360(get_engine(DATABASE_URL))


    elif dashboard_section == "Retention Optimization":

        render_retention_optimization(get_engine(DATABASE_URL))



    elif dashboard_section == "Executive Analytics":

        render_executive_analytics(get_engine(DATABASE_URL))


    elif dashboard_section == "Production AI & Monitoring":

        render_production_ai_monitoring(get_engine(DATABASE_URL))


    elif dashboard_section == "Data Quality & MLOps":

        render_data_quality_mlops(get_engine(DATABASE_URL))

    elif dashboard_section == "AI Analytics Copilot":

        render_ai_analytics_copilot()



except Exception as exc:

    st.error("Unable to load this dashboard section.")

    st.caption(

        "Check the terminal for the technical error. "

        "Verify database connectivity and required table names."

    )

    st.exception(exc)
