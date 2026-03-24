"""tab_regression.py – Regression: Predict Monthly Spend."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils import preprocess, TARGET_REG


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    ridge    = joblib.load("models/reg_ridge.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return ridge, encoders


def render():
    st.title("📈 Predictive Analysis — Regression")
    st.markdown("*How much will a customer spend? Ridge Regression predicts monthly book spend (₹).*")

    df = load_data()
    ridge, encoders = load_models()

    X, _ = preprocess(df, fit_encoders=encoders, return_encoders=True)
    y    = df[TARGET_REG].fillna(0)
    feat_names = list(X.columns)

    mask   = y < 2000
    X_clean = X[mask]
    y_clean = y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    y_pred = ridge.predict(X_te)
    y_pred = np.clip(y_pred, 0, None)

    # ── Metrics ───────────────────────────────────────────────────────────────
    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)

    st.subheader("Regression Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",  f"{r2:.3f}")
    m2.metric("RMSE",      f"₹{rmse:.1f}")
    m3.metric("MAE",       f"₹{mae:.1f}")

    st.markdown("---")

    # ── Actual vs Predicted scatter ────────────────────────────────────────────
    st.subheader("Actual vs Predicted Monthly Spend")
    scat_df = pd.DataFrame({
        "Actual (₹)": y_te.values,
        "Predicted (₹)": y_pred
    })
    fig_scat = px.scatter(scat_df, x="Actual (₹)", y="Predicted (₹)",
                          opacity=0.5,
                          title="Actual vs Predicted Spend",
                          trendline="ols",
                          color_discrete_sequence=["#1D9E75"])
    max_val = max(scat_df["Actual (₹)"].max(), scat_df["Predicted (₹)"].max())
    fig_scat.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                  mode="lines", name="Perfect Prediction",
                                  line=dict(color="grey", dash="dash")))
    fig_scat.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_scat, use_container_width=True)

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.subheader("Residual Analysis")
    col_r1, col_r2 = st.columns(2)

    residuals = y_te.values - y_pred
    with col_r1:
        fig_res = px.histogram(residuals, nbins=40,
                               title="Residuals Distribution",
                               color_discrete_sequence=["#378ADD"],
                               labels={"value": "Residual (₹)"})
        fig_res.update_layout(showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        fig_res2 = px.scatter(x=y_pred, y=residuals,
                              opacity=0.5,
                              title="Residuals vs Fitted",
                              labels={"x": "Fitted Value (₹)", "y": "Residual"},
                              color_discrete_sequence=["#EF9F27"])
        fig_res2.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res2.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_res2, use_container_width=True)

    # ── Ridge coefficients (feature importance) ────────────────────────────────
    st.subheader("Top Predictors of Monthly Spend (Ridge Coefficients)")
    coefs = ridge.coef_
    coef_df = pd.DataFrame({
        "Feature": feat_names[:len(coefs)],
        "Coefficient": coefs
    }).reindex(range(min(len(feat_names), len(coefs))))
    coef_df["Feature"] = feat_names[:len(coef_df)]
    coef_df["Abs Coef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs Coef", ascending=False).head(20)

    fig_coef = px.bar(coef_df, x="Coefficient", y="Feature",
                      orientation="h",
                      title="Top 20 Ridge Coefficients",
                      color="Coefficient",
                      color_continuous_scale="RdYlGn")
    fig_coef.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                           yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_coef, use_container_width=True)

    # ── Average predicted spend by cluster ────────────────────────────────────
    if "cluster_name" in df.columns:
        st.subheader("Predicted Spend by Customer Segment")
        df_pred = df.copy()
        X_all, _ = preprocess(df_pred, fit_encoders=encoders, return_encoders=True)
        df_pred["predicted_spend"] = np.clip(ridge.predict(X_all), 0, None)
        seg_spend = (df_pred.groupby("cluster_name")["predicted_spend"]
                     .mean().reset_index()
                     .sort_values("predicted_spend", ascending=False))
        seg_spend.columns = ["Segment", "Avg Predicted Spend (₹)"]
        fig_seg = px.bar(seg_spend, x="Segment", y="Avg Predicted Spend (₹)",
                         color="Segment",
                         color_discrete_sequence=["#378ADD","#1D9E75","#EF9F27",
                                                   "#E24B4A","#7F77DD"],
                         title="Average Predicted Monthly Spend per Segment")
        fig_seg.update_layout(showlegend=False,
                              margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    # ── Spend distribution ─────────────────────────────────────────────────────
    st.subheader("Actual Spend Distribution")
    fig_dist = px.histogram(df, x="monthly_book_spend_numeric",
                             nbins=40,
                             title="Monthly Book Spend Distribution (₹)",
                             color_discrete_sequence=["#7F77DD"])
    fig_dist.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_dist, use_container_width=True)
