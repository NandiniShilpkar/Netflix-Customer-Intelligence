from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.dashboard.churn_intelligence import (
    build_current_customer_features,
    load_churn_model,
)
from src.dashboard.customer_decision_engine import build_customer_intelligence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"


def _check_ollama():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5,
        )

        if response.status_code != 200:
            return {
                "status": "Unavailable",
                "model": OLLAMA_MODEL,
                "details": "Local AI service is not available in this environment.",
            }

        models = response.json().get("models", [])
        model_names = [
            model.get("name", "")
            for model in models
        ]

        model_available = any(
            OLLAMA_MODEL in name
            for name in model_names
        )

        if model_available:
            return {
                "status": "Healthy",
                "model": OLLAMA_MODEL,
                "details": "Local LLM service is responding.",
            }

        return {
            "status": "Unavailable",
            "model": OLLAMA_MODEL,
            "details": "Ollama is running but the configured model is not installed.",
        }

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return {
            "status": "Unavailable",
            "model": OLLAMA_MODEL,
            "details": "Local AI is unavailable in this deployment environment.",
        }

    except (requests.exceptions.RequestException, ValueError):
        return {
            "status": "Unavailable",
            "model": OLLAMA_MODEL,
            "details": "Local AI service could not be reached.",
        }

def _model_health():
    model = load_churn_model()

    if model is None:
        return {
            "status": "Unavailable",
            "model_type": "Unknown",
            "features": 0,
            "trees": 0,
            "importance": pd.DataFrame(),
        }

    model_type = type(model).__name__

    feature_count = int(
        getattr(model, "n_features_in_", 0)
    )

    tree_count = int(
        getattr(model, "n_estimators", 0)
    )

    importance = pd.DataFrame()

    raw_importance = getattr(
        model,
        "feature_importances_",
        None,
    )

    if raw_importance is not None:
        importance = pd.DataFrame(
            {
                "Feature": [
                    f"Feature {index + 1}"
                    for index in range(len(raw_importance))
                ],
                "Importance": raw_importance,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

    return {
        "status": "Healthy",
        "model_type": model_type,
        "features": feature_count,
        "trees": tree_count,
        "importance": importance,
    }


def build_production_monitoring(engine):
    customer_df = build_customer_intelligence(engine)

    if customer_df.empty:
        return {}

    feature_df = build_current_customer_features(engine)

    model_health = _model_health()
    ollama_health = _check_ollama()

    total_customers = len(customer_df)

    churn_risk = customer_df["churn_risk"]

    low_risk = int((churn_risk < 35).sum())
    medium_risk = int(
        ((churn_risk >= 35) & (churn_risk < 55)).sum()
    )
    high_risk = int((churn_risk >= 55).sum())

    missing_values = int(
        feature_df.isna().sum().sum()
    )

    duplicate_rows = int(
        feature_df.duplicated().sum()
    )

    prediction_coverage = (
        len(customer_df) / len(feature_df) * 100
        if len(feature_df)
        else 0
    )

    checks = {
        "Model available": model_health["status"] == "Healthy",
        "Predictions generated": total_customers > 0,
        "Prediction coverage": prediction_coverage >= 95,
        "Feature data complete": missing_values == 0,
        "No duplicate feature rows": duplicate_rows == 0,
        "AI service available": ollama_health["status"] == "Healthy",
    }

    readiness_score = (
        sum(checks.values()) / len(checks) * 100
    )

    risk_distribution = pd.DataFrame(
        {
            "Risk Level": [
                "Low",
                "Medium",
                "High",
            ],
            "Customers": [
                low_risk,
                medium_risk,
                high_risk,
            ],
        }
    )

    return {
        "customer_df": customer_df,
        "feature_df": feature_df,
        "model_health": model_health,
        "ollama_health": ollama_health,
        "total_customers": total_customers,
        "low_risk": low_risk,
        "medium_risk": medium_risk,
        "high_risk": high_risk,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "prediction_coverage": prediction_coverage,
        "checks": checks,
        "readiness_score": readiness_score,
        "risk_distribution": risk_distribution,
    }


def render_production_ai_monitoring(engine):
    st.title("Production AI & Model Monitoring")

    st.caption(
        "Production-readiness view for the churn model, "
        "customer predictions, feature quality, and local AI service."
    )

    data = build_production_monitoring(engine)

    if not data:
        st.warning(
            "Production monitoring data is currently unavailable."
        )
        return

    model_health = data["model_health"]
    ollama_health = data["ollama_health"]

    st.subheader("Production Readiness")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Readiness Score",
            f"{data['readiness_score']:.0f}%",
        )

    with c2:
        st.metric(
            "Customers Scored",
            f"{data['total_customers']:,}",
        )

    with c3:
        st.metric(
            "Prediction Coverage",
            f"{data['prediction_coverage']:.1f}%",
        )

    with c4:
        st.metric(
            "High-Risk Customers",
            f"{data['high_risk']:,}",
        )

    st.divider()

    st.subheader("System Health")

    h1, h2, h3 = st.columns(3)

    with h1:
        if model_health["status"] == "Healthy":
            st.success("Churn Model: Healthy")
        else:
            st.error("Churn Model: Unavailable")

        st.write(
            f"Model type: **{model_health['model_type']}**"
        )
        st.write(
            f"Features: **{model_health['features']}**"
        )
        st.write(
            f"Trees: **{model_health['trees']}**"
        )

    with h2:
        if ollama_health["status"] == "Healthy":
            st.success("Local AI: Healthy")
        else:
            st.warning("Local AI: Unavailable")

        st.write(
            f"Configured model: **{ollama_health['model']}**"
        )
        st.caption(ollama_health["details"])

    with h3:
        if data["missing_values"] == 0:
            st.success("Feature Quality: Healthy")
        else:
            st.warning("Feature Quality: Review Required")

        st.write(
            f"Missing values: **{data['missing_values']:,}**"
        )
        st.write(
            f"Duplicate rows: **{data['duplicate_rows']:,}**"
        )

    st.divider()

    st.subheader("Churn Risk Monitoring")

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            data["risk_distribution"],
            x="Risk Level",
            y="Customers",
            text="Customers",
            title="Current Churn-Risk Distribution",
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Customers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        fig = px.histogram(
            data["customer_df"],
            x="churn_risk",
            nbins=20,
            title="Churn-Risk Score Distribution",
        )

        fig.update_layout(
            xaxis_title="Churn Risk %",
            yaxis_title="Customers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Model Feature Importance")

    importance = model_health["importance"]

    if not importance.empty:
        top_features = importance.head(15)

        fig = px.bar(
            top_features.sort_values(
                "Importance",
                ascending=True,
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Model Features",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    else:
        st.info(
            "Feature importance is not available for the active model."
        )

    st.divider()

    st.subheader("Production Validation Checks")

    check_rows = []

    for check_name, passed in data["checks"].items():
        check_rows.append(
            {
                "Check": check_name,
                "Status": "PASS" if passed else "REVIEW",
            }
        )

    check_df = pd.DataFrame(check_rows)

    st.dataframe(
        check_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Monitoring Summary")

    summary = pd.DataFrame(
        [
            {
                "Metric": "Customers Scored",
                "Value": data["total_customers"],
            },
            {
                "Metric": "Low Risk Customers",
                "Value": data["low_risk"],
            },
            {
                "Metric": "Medium Risk Customers",
                "Value": data["medium_risk"],
            },
            {
                "Metric": "High Risk Customers",
                "Value": data["high_risk"],
            },
            {
                "Metric": "Missing Feature Values",
                "Value": data["missing_values"],
            },
            {
                "Metric": "Duplicate Feature Rows",
                "Value": data["duplicate_rows"],
            },
            {
                "Metric": "Prediction Coverage %",
                "Value": round(
                    data["prediction_coverage"],
                    2,
                ),
            },
            {
                "Metric": "Production Readiness %",
                "Value": round(
                    data["readiness_score"],
                    2,
                ),
            },
        ]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    csv = summary.to_csv(index=False)

    st.download_button(
        "Download Production Monitoring Report",
        data=csv,
        file_name="production_ai_monitoring_report.csv",
        mime="text/csv",
    )


build_production_ai_monitoring_context = build_production_monitoring
