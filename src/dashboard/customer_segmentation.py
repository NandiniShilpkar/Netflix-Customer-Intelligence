
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEGMENTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "segmentation_experiment"
    / "behavior_focused_assignments.csv"
)

CHURN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "churn_labels.csv"
)

NETFLIX_RED = "#E50914"
NETFLIX_DARK_RED = "#B20710"
NETFLIX_WHITE = "#F5F5F1"


# --------------------------------------------------
# Chart styling
# --------------------------------------------------

def style_chart(fig):
    """Apply consistent Netflix-inspired chart styling."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,8,9,0.95)",
        font=dict(color=NETFLIX_WHITE, size=12),
        title_font=dict(color=NETFLIX_WHITE, size=18),
        title_x=0.02,
        margin=dict(l=20, r=20, t=60, b=35),
        hoverlabel=dict(
            bgcolor="#1A1112",
            font_color=NETFLIX_WHITE,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
    )

    return fig


# --------------------------------------------------
# Customer segmentation page
# --------------------------------------------------

def render_customer_segmentation():
    st.header("Customer Segmentation")

    st.caption(
        "Explore behavior-focused customer groups, compare their sizes, "
        "and examine recorded historical churn across segments."
    )

    st.divider()

    # --------------------------------------------------
    # Load input files
    # --------------------------------------------------

    if not SEGMENTS_PATH.exists():
        st.error(
            "Customer segment assignments were not found."
        )
        st.caption(f"Expected file: {SEGMENTS_PATH}")
        return

    if not CHURN_PATH.exists():
        st.error(
            "The processed churn labels file was not found."
        )
        st.caption(f"Expected file: {CHURN_PATH}")
        return

    try:
        segments = pd.read_csv(SEGMENTS_PATH)

        churn = pd.read_csv(
            CHURN_PATH,
            usecols=["customer_id", "churned"],
        )

    except Exception as exc:
        st.error("Unable to load customer segmentation data.")
        st.exception(exc)
        return

    # --------------------------------------------------
    # Validate records
    # --------------------------------------------------

    required_segment_columns = {
        "customer_id",
        "segment_id",
    }

    if not required_segment_columns.issubset(segments.columns):
        st.error(
            "The segment assignments file is missing required columns: "
            "customer_id and/or segment_id."
        )
        return

    if segments["customer_id"].duplicated().any():
        st.error(
            "Duplicate customer IDs were found in segment assignments."
        )
        return

    if churn["customer_id"].duplicated().any():
        st.error(
            "Duplicate customer IDs were found in churn labels."
        )
        return

    try:
        data = segments.merge(
            churn,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    except Exception as exc:
        st.error("Unable to match segment assignments with churn labels.")
        st.exception(exc)
        return

    data["churned"] = (
        data["churned"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    data = data.dropna(
        subset=["segment_id", "churned"]
    ).copy()

    if data.empty:
        st.warning(
            "No valid matched customer records were found."
        )
        return

    numeric_segment_ids = pd.to_numeric(
        data["segment_id"],
        errors="coerce",
    )

    data = data.loc[
        numeric_segment_ids.notna()
    ].copy()

    data["segment_id"] = numeric_segment_ids.loc[
        data.index
    ].astype(int)

    if data.empty:
        st.warning(
            "No valid segment IDs were found in the matched records."
        )
        return

    # --------------------------------------------------
    # Segment profiles
    # --------------------------------------------------

    profiles = (
        data.groupby("segment_id")
        .agg(
            customers=("customer_id", "nunique"),
            recorded_churned=("churned", "sum"),
            churn_rate=("churned", "mean"),
        )
        .reset_index()
    )

    segment_names = {
        0: "Frequent viewers",
        1: "Infrequent viewers",
    }

    profiles["segment_name"] = (
        profiles["segment_id"]
        .map(segment_names)
        .fillna(
            "Segment " + profiles["segment_id"].astype(str)
        )
    )

    # --------------------------------------------------
    # KPI overview
    # --------------------------------------------------

    st.subheader("Customer Overview")

    total_customers = data["customer_id"].nunique()
    recorded_churned = int(data["churned"].sum())
    overall_churn_rate = data["churned"].mean()

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:
        st.metric(
            "Matched Customers",
            f"{total_customers:,}",
        )

    with col2:
        st.metric(
            "Recorded Churn",
            f"{recorded_churned:,}",
        )

    with col3:
        st.metric(
            "Overall Recorded Churn",
            f"{overall_churn_rate:.1%}",
        )

    st.caption(
        "Metrics are calculated from customers matched between the "
        "segment assignment file and the historical churn labels."
    )

    st.divider()

    # --------------------------------------------------
    # Segment comparison
    # --------------------------------------------------

    st.subheader("Segment Comparison")

    table = profiles[
        [
            "segment_name",
            "customers",
            "recorded_churned",
            "churn_rate",
        ]
    ].rename(
        columns={
            "segment_name": "Segment",
            "customers": "Customers",
            "recorded_churned": "Recorded Churned Customers",
            "churn_rate": "Recorded Churn Rate",
        }
    )

    table["Recorded Churn Rate"] = (
        table["Recorded Churn Rate"] * 100
    ).round(2).map(
        lambda value: f"{value:.2f}%"
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # Segment charts
    # --------------------------------------------------

    st.subheader("Segment Insights")

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        st.markdown("#### Customer Distribution")

        fig = px.bar(
            profiles,
            x="segment_name",
            y="customers",
            labels={
                "segment_name": "Customer Segment",
                "customers": "Customers",
            },
            title="Customers by Segment",
            color_discrete_sequence=[NETFLIX_RED],
            text="customers",
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            cliponaxis=False,
        )

        style_chart(fig)

        fig.update_yaxes(
            rangemode="tozero",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.markdown("#### Recorded Churn by Segment")

        fig = px.bar(
            profiles,
            x="segment_name",
            y="churn_rate",
            labels={
                "segment_name": "Customer Segment",
                "churn_rate": "Recorded Churn Rate",
            },
            title="Historical Churn Rate by Segment",
            color_discrete_sequence=[NETFLIX_RED],
            text="churn_rate",
        )

        fig.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
        )

        style_chart(fig)

        fig.update_yaxes(
            tickformat=".0%",
            rangemode="tozero",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------
    # Interpretation and limitations
    # --------------------------------------------------

    st.divider()

    st.subheader("How to Interpret These Results")

    st.markdown(
        """
        - **Customer distribution:** shows how matched customers are
          distributed across the existing behavioral segments.
        - **Recorded churn rate:** shows the proportion of matched
          customers labeled as churned in the historical dataset.
        - **Segment comparison:** helps identify differences between
          the groups for further analysis.
        """
    )

    st.info(
        "These are historical dataset labels, not predictions of future "
        "churn. Differences between segments do not establish causation. "
        "Segment names describe the existing experiment's assignments."
    )



# from pathlib import Path

# import pandas as pd
# import plotly.express as px
# import streamlit as st


# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SEGMENTS_PATH = (
#     PROJECT_ROOT
#     / "reports"
#     / "segmentation_experiment"
#     / "behavior_focused_assignments.csv"
# )
# CHURN_PATH = PROJECT_ROOT / "data" / "processed" / "churn_labels.csv"


# def render_customer_segmentation():
#     st.header("Customer Segmentation")
#     st.caption(
#         "Experimental behavior-focused segments. "
#         "Churn metrics reflect existing historical labels."
#     )

#     if not SEGMENTS_PATH.exists() or not CHURN_PATH.exists():
#         st.warning(
#             "Required files are missing. Please run the segmentation "
#             "workflow and check the processed data files."
#         )
#         return

#     segments = pd.read_csv(SEGMENTS_PATH)
#     churn = pd.read_csv(CHURN_PATH, usecols=["customer_id", "churned"])

#     if segments["customer_id"].duplicated().any():
#         st.error("Duplicate customer IDs in segment assignments.")
#         return

#     if churn["customer_id"].duplicated().any():
#         st.error("Duplicate customer IDs in churn labels.")
#         return

#     data = segments.merge(
#         churn,
#         on="customer_id",
#         how="inner",
#         validate="one_to_one",
#     )

#     data["churned"] = (
#         data["churned"]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#         .map({"true": True, "false": False, "1": True, "0": False})
#     )

#     data = data.dropna(subset=["segment_id", "churned"])

#     if data.empty:
#         st.warning("No valid matched customer records were found.")
#         return

#     data["segment_id"] = data["segment_id"].astype(int)

#     profiles = (
#         data.groupby("segment_id")
#         .agg(
#             customers=("customer_id", "nunique"),
#             recorded_churned=("churned", "sum"),
#             churn_rate=("churned", "mean"),
#         )
#         .reset_index()
#     )

#     profiles["segment_name"] = profiles["segment_id"].map(
#         {0: "Frequent viewers", 1: "Infrequent viewers"}
#     ).fillna("Segment " + profiles["segment_id"].astype(str))

#     # KPI cards
#     col1, col2, col3 = st.columns(3)

#     col1.metric(
#         "Matched customers",
#         f"{data['customer_id'].nunique():,}",
#     )
#     col2.metric(
#         "Recorded churn",
#         f"{int(data['churned'].sum()):,}",
#     )
#     col3.metric(
#         "Overall recorded churn",
#         f"{data['churned'].mean():.1%}",
#     )

#     # Segment comparison table
#     st.subheader("Segment comparison")

#     table = profiles[
#         ["segment_name", "customers", "recorded_churned", "churn_rate"]
#     ].rename(
#         columns={
#             "segment_name": "Segment",
#             "customers": "Customers",
#             "recorded_churned": "Recorded churned customers",
#             "churn_rate": "Recorded churn rate",
#         }
#     )

#     table["Recorded churn rate"] = (
#         table["Recorded churn rate"] * 100
#     ).round(2).astype(str) + "%"

#     st.dataframe(
#         table,
#         use_container_width=True,
#         hide_index=True,
#     )

#     # Charts in two columns
#     left, right = st.columns(2)

#     with left:
#         st.subheader("Customer distribution")

#         fig = px.bar(
#             profiles,
#             x="segment_name",
#             y="customers",
#             labels={
#                 "segment_name": "Customer segment",
#                 "customers": "Customers",
#             },
#             title="Customers by segment",
#             template="plotly_dark",
#             color_discrete_sequence=["#E50914"],
#         )

#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(12,8,9,0.95)",
#             font=dict(color="#F5F5F1"),
#             title_font=dict(color="#F5F5F1"),
#             margin=dict(l=20, r=20, t=55, b=20),
#         )

#         st.plotly_chart(fig, use_container_width=True)

#     with right:
#         st.subheader("Recorded churn by segment")

#         fig = px.bar(
#             profiles,
#             x="segment_name",
#             y="churn_rate",
#             labels={
#                 "segment_name": "Customer segment",
#                 "churn_rate": "Recorded churn rate",
#             },
#             title="Churn rate by segment",
#             template="plotly_dark",
#             color_discrete_sequence=["#E50914"],
#         )

#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(12,8,9,0.95)",
#             font=dict(color="#F5F5F1"),
#             title_font=dict(color="#F5F5F1"),
#             margin=dict(l=20, r=20, t=55, b=20),
#         )

#         fig.update_yaxes(tickformat=".0%")
#         st.plotly_chart(fig, use_container_width=True)

#     st.info(
#         "These are historical dataset labels, not predictions of future "
#         "churn. The observed relationship does not establish causation."
#     )

#     # Customer explorer
#     with st.expander("Explore customers by segment"):
#         segment_names = dict(
#             zip(profiles["segment_id"], profiles["segment_name"])
#         )

#         selected = st.selectbox(
#             "Choose a segment",
#             sorted(segment_names),
#             format_func=lambda x: segment_names[x],
#         )

#         st.dataframe(
#             data.loc[
#                 data["segment_id"] == selected,
#                 ["customer_id", "segment_id", "churned"],
#             ],
#             use_container_width=True,
#             hide_index=True,
#         )



# from pathlib import Path

# import pandas as pd
# import plotly.express as px
# import streamlit as st


# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SEGMENTS_PATH = (
#     PROJECT_ROOT
#     / "reports"
#     / "segmentation_experiment"
#     / "behavior_focused_assignments.csv"
# )
# CHURN_PATH = PROJECT_ROOT / "data" / "processed" / "churn_labels.csv"


# def render_customer_segmentation():
#     st.header("Customer Segmentation")
#     st.caption(
#         "Experimental behavior-focused segments. "
#         "Churn metrics reflect existing historical labels."
#     )

#     if not SEGMENTS_PATH.exists() or not CHURN_PATH.exists():
#         st.warning(
#             "Required files are missing. Please run the segmentation "
#             "workflow and check the processed data files."
#         )
#         return

#     segments = pd.read_csv(SEGMENTS_PATH)
#     churn = pd.read_csv(CHURN_PATH, usecols=["customer_id", "churned"])

#     if segments["customer_id"].duplicated().any():
#         st.error("Duplicate customer IDs in segment assignments.")
#         return

#     if churn["customer_id"].duplicated().any():
#         st.error("Duplicate customer IDs in churn labels.")
#         return

#     data = segments.merge(
#         churn, on="customer_id", how="inner", validate="one_to_one"
#     )

#     data["churned"] = (
#         data["churned"].astype(str).str.strip().str.lower()
#         .map({"true": True, "false": False, "1": True, "0": False})
#     )
#     data = data.dropna(subset=["segment_id", "churned"])

#     if data.empty:
#         st.warning("No valid matched customer records were found.")
#         return

#     data["segment_id"] = data["segment_id"].astype(int)

#     profiles = (
#         data.groupby("segment_id")
#         .agg(
#             customers=("customer_id", "nunique"),
#             recorded_churned=("churned", "sum"),
#             churn_rate=("churned", "mean"),
#         )
#         .reset_index()
#     )

#     profiles["segment_name"] = profiles["segment_id"].map(
#         {0: "Frequent viewers", 1: "Infrequent viewers"}
#     ).fillna("Segment " + profiles["segment_id"].astype(str))

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Matched customers", f"{data['customer_id'].nunique():,}")
#     col2.metric("Recorded churn", f"{int(data['churned'].sum()):,}")
#     col3.metric("Overall recorded churn", f"{data['churned'].mean():.1%}")

#     st.subheader("Segment comparison")

#     table = profiles[
#         ["segment_name", "customers", "recorded_churned", "churn_rate"]
#     ].rename(
#         columns={
#             "segment_name": "Segment",
#             "customers": "Customers",
#             "recorded_churned": "Recorded churned customers",
#             "churn_rate": "Recorded churn rate",
#         }
#     )
#     table["Recorded churn rate"] = (
#         table["Recorded churn rate"] * 100
#     ).round(2).astype(str) + "%"

#     st.dataframe(table, use_container_width=True, hide_index=True)

#     left, right = st.columns(2)

#     with left:
#         st.subheader("Customer distribution")

#         fig = px.bar(
#             profiles,
#             x="segment_name",
#             y="churn_rate",
#             labels={
#                 "segment_name": "Customer segment",
#                 "churn_rate": "Recorded churn rate",
#             },
#             title="Recorded churn by segment",
#             template="plotly_dark",
#             color_discrete_sequence=["#E50914"],
#         )
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(12,8,9,0.95)",
#             font=dict(color="#F5F5F1"),
#             title_font=dict(color="#F5F5F1"),
#             margin=dict(l=20, r=20, t=55, b=20),
#         )
#         fig.update_yaxes(tickformat=".0%")
#         st.plotly_chart(fig, use_container_width=True)

#         fig = px.bar(
#             profiles,
#             x="segment_name",
#             y="churn_rate",
#             labels={
#                 "segment_name": "Customer segment",
#                 "churn_rate": "Recorded churn rate",
#             },
#         )
#         fig.update_yaxes(tickformat=".0%")
#         st.plotly_chart(fig, use_container_width=True)

#     st.info(
#         "These are historical dataset labels, not predictions of future "
#         "churn. The observed relationship does not establish causation."
#     )

#     with st.expander("Explore customers by segment"):
#         segment_names = dict(
#             zip(profiles["segment_id"], profiles["segment_name"])
#         )
#         selected = st.selectbox(
#             "Choose a segment",
#             sorted(segment_names),
#             format_func=lambda x: segment_names[x],
#         )
#         st.dataframe(
#             data.loc[
#                 data["segment_id"] == selected,
#                 ["customer_id", "segment_id", "churned"],
#             ],
#             use_container_width=True,
#             hide_index=True,
#         )
