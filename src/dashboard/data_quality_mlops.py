from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, inspect, text

from src.dashboard.churn_intelligence import (
    build_current_customer_features,
    load_churn_model,
)
from src.dashboard.customer_decision_engine import build_customer_intelligence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _feature_quality(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in feature_df.columns:
        missing = int(feature_df[column].isna().sum())
        total = len(feature_df)

        rows.append(
            {
                "Feature": column,
                "Missing Values": missing,
                "Missing %": (
                    missing / total * 100
                    if total
                    else 0
                ),
                "Unique Values": int(
                    feature_df[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Missing Values",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def _source_data_quality(engine):
    inspector = inspect(engine)
    tables = [
        "customers",
        "subscriptions",
        "subscription_plans",
        "viewing_activity",
        "content",
        "payments",
        "support_tickets",
        "customer_feedback",
        "churn_labels",
    ]

    results = []

    for table in tables:
        if table not in inspector.get_table_names():
            continue

        with engine.connect() as connection:
            total = connection.execute(
                text(f'SELECT COUNT(*) FROM "{table}"')
            ).scalar()

        results.append(
            {
                "Table": table,
                "Rows": int(total or 0),
                "Status": "Available",
            }
        )

    return pd.DataFrame(results)


def _get_feature_names(model, feature_df):
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)

        if len(names) == len(feature_df.columns):
            return names

    return list(feature_df.columns)


def _global_explainability(feature_df):
    model = load_churn_model()

    if model is None:
        return pd.DataFrame()

    importance = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importance is None:
        return pd.DataFrame()

    names = _get_feature_names(
        model,
        feature_df,
    )

    # build_current_customer_features may contain helper
    # columns that are not model inputs. Keep only the
    # model feature count when the artifact exposes it.
    if hasattr(model, "n_features_in_"):
        names = names[: model.n_features_in_]

    length = min(
        len(names),
        len(importance),
    )

    if length == 0:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "Feature": names[:length],
            "Importance": importance[:length],
        }
    )

    return result.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)


def _customer_risk_explanation(
    customer_df,
    feature_df,
    customer_id,
):
    model = load_churn_model()

    if model is None:
        return pd.DataFrame()

    if "customer_id" not in feature_df.columns:
        return pd.DataFrame()

    customer_rows = feature_df[
        feature_df["customer_id"].astype(str)
        == str(customer_id)
    ]

    if customer_rows.empty:
        return pd.DataFrame()

    row = customer_rows.iloc[0]

    importance = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importance is None:
        return pd.DataFrame()

    names = _get_feature_names(
        model,
        feature_df,
    )

    median_values = feature_df[
        names
    ].median(numeric_only=True)

    explanations = []

    for index, feature in enumerate(names):
        if feature not in feature_df.columns:
            continue

        value = row.get(feature)

        if pd.isna(value):
            continue

        baseline = median_values.get(feature)

        if pd.isna(baseline):
            continue

        importance_value = float(
            importance[index]
        )

        if importance_value <= 0:
            continue

        difference = float(value) - float(baseline)

        if difference == 0:
            continue

        explanations.append(
            {
                "Feature": feature,
                "Customer Value": float(value),
                "Cohort Median": float(baseline),
                "Model Importance": importance_value,
                "Risk Direction": (
                    "Higher than cohort baseline"
                    if difference > 0
                    else "Lower than cohort baseline"
                ),
                "Risk Contribution Proxy": abs(
                    difference
                ) * importance_value,
            }
        )

    result = pd.DataFrame(explanations)

    if result.empty:
        return result

    return result.sort_values(
        "Risk Contribution Proxy",
        ascending=False,
    ).head(10)


def build_data_quality_mlops(engine):
    feature_df = build_current_customer_features(engine)
    customer_df = build_customer_intelligence(engine)

    if feature_df.empty:
        return {}

    quality_df = _feature_quality(feature_df)
    source_df = _source_data_quality(engine)
    explainability_df = _global_explainability(
        feature_df
    )

    total_cells = (
        feature_df.shape[0]
        * feature_df.shape[1]
    )

    missing_cells = int(
        feature_df.isna().sum().sum()
    )

    missing_rate = (
        missing_cells / total_cells * 100
        if total_cells
        else 0
    )

    complete_features = int(
        (quality_df["Missing Values"] == 0).sum()
    )

    total_features = len(quality_df)

    feature_completeness = (
        complete_features
        / total_features
        * 100
        if total_features
        else 0
    )

    duplicate_rows = int(
        feature_df.duplicated().sum()
    )

    customer_coverage = (
        len(customer_df)
        / len(feature_df)
        * 100
        if len(feature_df)
        else 0
    )

    checks = {
        "Feature table available": len(feature_df) > 0,
        "Customer coverage >= 95%": customer_coverage >= 95,
        "Duplicate feature rows = 0": duplicate_rows == 0,
        "Missing feature rate < 10%": missing_rate < 10,
        "Active model available": load_churn_model() is not None,
        "Source tables available": len(source_df) >= 8,
    }

    quality_score = (
        sum(checks.values())
        / len(checks)
        * 100
    )

    return {
        "feature_df": feature_df,
        "customer_df": customer_df,
        "quality_df": quality_df,
        "source_df": source_df,
        "explainability_df": explainability_df,
        "missing_cells": missing_cells,
        "missing_rate": missing_rate,
        "feature_completeness": feature_completeness,
        "duplicate_rows": duplicate_rows,
        "customer_coverage": customer_coverage,
        "checks": checks,
        "quality_score": quality_score,
    }


def render_data_quality_mlops(engine):
    st.title("Data Quality & MLOps")

    st.caption(
        "Data-quality monitoring, model explainability, "
        "feature completeness, and ML production checks."
    )

    data = build_data_quality_mlops(engine)

    if not data:
        st.warning(
            "Data-quality monitoring is currently unavailable."
        )
        return

    st.subheader("Data Quality Score")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Quality Score",
            f"{data['quality_score']:.0f}%",
        )

    with c2:
        st.metric(
            "Missing Values",
            f"{data['missing_cells']:,}",
        )

    with c3:
        st.metric(
            "Feature Completeness",
            f"{data['feature_completeness']:.1f}%",
        )

    with c4:
        st.metric(
            "Customer Coverage",
            f"{data['customer_coverage']:.1f}%",
        )

    if data["missing_cells"] > 0:
        st.warning(
            f"{data['missing_cells']:,} missing feature values "
            "were detected. Review the feature-level breakdown below."
        )

    st.divider()

    st.subheader("Feature-Level Data Quality")

    quality_display = data["quality_df"].copy()

    st.dataframe(
        quality_display.style.format(
            {
                "Missing %": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    missing_plot = quality_display[
        quality_display["Missing Values"] > 0
    ].head(20)

    if not missing_plot.empty:
        fig = px.bar(
            missing_plot.sort_values(
                "Missing Values",
                ascending=True,
            ),
            x="Missing Values",
            y="Feature",
            orientation="h",
            title="Features With Missing Values",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Source Data Inventory")

    st.dataframe(
        data["source_df"],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Model Explainability")

    st.caption(
        "Global feature importance is derived from the active "
        "Random Forest model. Customer-level explanations below "
        "are risk-contribution proxies based on feature deviation "
        "from the current cohort median weighted by model importance."
    )

    explanation_df = data["explainability_df"]

    if explanation_df.empty:
        st.info(
            "The active model artifact does not expose "
            "feature_importances_. Customer-level model "
            "explanations are therefore limited to the "
            "available risk and cohort analytics."
        )

    if not explanation_df.empty:
        top_features = explanation_df.head(15)

        fig = px.bar(
            top_features.sort_values(
                "Importance",
                ascending=True,
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Global Churn Drivers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            explanation_df.head(20).style.format(
                {
                    "Importance": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Model feature importance is unavailable."
        )

    st.divider()

    st.subheader("Customer Risk Explanation")

    customer_ids = (
        data["customer_df"]["customer_id"]
        .astype(str)
        .tolist()
    )

    if customer_ids:
        selected_customer = st.selectbox(
            "Select a customer",
            customer_ids,
        )

        selected_customer_row = data["customer_df"][
            data["customer_df"]["customer_id"].astype(str)
            == str(selected_customer)
        ]

        if not selected_customer_row.empty:
            row = selected_customer_row.iloc[0]

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Churn Risk",
                    f"{row['churn_risk']:.1f}%",
                )

            with m2:
                st.metric(
                    "Health Score",
                    f"{row['customer_health_score']:.1f}",
                )

            with m3:
                st.metric(
                    "Decision",
                    str(row["decision"]),
                )

            explanation = _customer_risk_explanation(
                data["customer_df"],
                data["feature_df"],
                selected_customer,
            )

            if not explanation.empty:
                st.dataframe(
                    explanation.style.format(
                        {
                            "Customer Value": "{:.2f}",
                            "Cohort Median": "{:.2f}",
                            "Model Importance": "{:.4f}",
                            "Risk Contribution Proxy": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No explainability signals are available "
                    "for this customer."
                )

    st.divider()

    st.subheader("MLOps Validation Checks")

    checks_df = pd.DataFrame(
        [
            {
                "Check": name,
                "Status": (
                    "PASS"
                    if passed
                    else "REVIEW"
                ),
            }
            for name, passed in data["checks"].items()
        ]
    )

    st.dataframe(
        checks_df,
        use_container_width=True,
        hide_index=True,
    )

    report = pd.DataFrame(
        [
            {
                "Metric": "Quality Score %",
                "Value": round(
                    data["quality_score"],
                    2,
                ),
            },
            {
                "Metric": "Missing Feature Values",
                "Value": data["missing_cells"],
            },
            {
                "Metric": "Missing Feature Rate %",
                "Value": round(
                    data["missing_rate"],
                    2,
                ),
            },
            {
                "Metric": "Feature Completeness %",
                "Value": round(
                    data["feature_completeness"],
                    2,
                ),
            },
            {
                "Metric": "Duplicate Feature Rows",
                "Value": data["duplicate_rows"],
            },
            {
                "Metric": "Customer Coverage %",
                "Value": round(
                    data["customer_coverage"],
                    2,
                ),
            },
        ]
    )

    st.divider()

    st.subheader("MLOps Report")

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Data Quality & MLOps Report",
        data=report.to_csv(index=False),
        file_name="data_quality_mlops_report.csv",
        mime="text/csv",
    )


build_data_quality_mlops_context = build_data_quality_mlops
