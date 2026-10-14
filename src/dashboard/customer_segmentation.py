
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEGMENTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "segmentation_experiment"
    / "behavior_focused_assignments.csv"
)
CHURN_PATH = PROJECT_ROOT / "data" / "processed" / "churn_labels.csv"


def render_customer_segmentation():
    st.header("Customer Segmentation")
    st.caption(
        "Experimental behavior-focused segments. "
        "Churn metrics reflect existing historical labels."
    )

    if not SEGMENTS_PATH.exists() or not CHURN_PATH.exists():
        st.warning(
            "Required files are missing. Please run the segmentation "
            "workflow and check the processed data files."
        )
        return

    segments = pd.read_csv(SEGMENTS_PATH)
    churn = pd.read_csv(CHURN_PATH, usecols=["customer_id", "churned"])

    if segments["customer_id"].duplicated().any():
        st.error("Duplicate customer IDs in segment assignments.")
        return

    if churn["customer_id"].duplicated().any():
        st.error("Duplicate customer IDs in churn labels.")
        return

    data = segments.merge(
        churn,
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    data["churned"] = (
        data["churned"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    data = data.dropna(subset=["segment_id", "churned"])

    if data.empty:
        st.warning("No valid matched customer records were found.")
        return

    data["segment_id"] = data["segment_id"].astype(int)

    profiles = (
        data.groupby("segment_id")
        .agg(
            customers=("customer_id", "nunique"),
            recorded_churned=("churned", "sum"),
            churn_rate=("churned", "mean"),
        )
        .reset_index()
    )

    profiles["segment_name"] = profiles["segment_id"].map(
        {0: "Frequent viewers", 1: "Infrequent viewers"}
    ).fillna("Segment " + profiles["segment_id"].astype(str))

    # KPI cards
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Matched customers",
        f"{data['customer_id'].nunique():,}",
    )
    col2.metric(
        "Recorded churn",
        f"{int(data['churned'].sum()):,}",
    )
    col3.metric(
        "Overall recorded churn",
        f"{data['churned'].mean():.1%}",
    )

    # Segment comparison table
    st.subheader("Segment comparison")

    table = profiles[
        ["segment_name", "customers", "recorded_churned", "churn_rate"]
    ].rename(
        columns={
            "segment_name": "Segment",
            "customers": "Customers",
            "recorded_churned": "Recorded churned customers",
            "churn_rate": "Recorded churn rate",
        }
    )

    table["Recorded churn rate"] = (
        table["Recorded churn rate"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    # Charts in two columns
    left, right = st.columns(2)

    with left:
        st.subheader("Customer distribution")

        fig = px.bar(
            profiles,
            x="segment_name",
            y="customers",
            labels={
                "segment_name": "Customer segment",
                "customers": "Customers",
            },
            title="Customers by segment",
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

    with right:
        st.subheader("Recorded churn by segment")

        fig = px.bar(
            profiles,
            x="segment_name",
            y="churn_rate",
            labels={
                "segment_name": "Customer segment",
                "churn_rate": "Recorded churn rate",
            },
            title="Churn rate by segment",
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

        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "These are historical dataset labels, not predictions of future "
        "churn. The observed relationship does not establish causation."
    )

    # Customer explorer
    with st.expander("Explore customers by segment"):
        segment_names = dict(
            zip(profiles["segment_id"], profiles["segment_name"])
        )

        selected = st.selectbox(
            "Choose a segment",
            sorted(segment_names),
            format_func=lambda x: segment_names[x],
        )

        st.dataframe(
            data.loc[
                data["segment_id"] == selected,
                ["customer_id", "segment_id", "churned"],
            ],
            use_container_width=True,
            hide_index=True,
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
