"""tab_classification.py – Predictive Classification."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

from utils import preprocess, TARGET_CLF


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    rf      = joblib.load("models/clf_rf.pkl")
    lr      = joblib.load("models/clf_lr.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    encoders= joblib.load("models/encoders.pkl")
    return rf, lr, scaler, encoders


def render():
    st.title("🎯 Predictive Analysis — Classification")
    st.markdown("*Who is likely to buy? Binary classification: Interested (1) vs Not Interested (0)*")

    df = load_data()
    rf, lr, scaler, encoders = load_models()

    X, _ = preprocess(df, fit_encoders=encoders, return_encoders=True)
    y = df[TARGET_CLF].astype(int)
    feat_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    X_te_s = scaler.transform(X_te)

    # ── Model selector ────────────────────────────────────────────────────────
    model_choice = st.radio("Select Model", ["Random Forest", "Logistic Regression"],
                            horizontal=True)

    if model_choice == "Random Forest":
        y_pred = rf.predict(X_te)
        y_prob = rf.predict_proba(X_te)[:, 1]
        model  = rf
        is_rf  = True
    else:
        y_pred = lr.predict(X_te_s)
        y_prob = lr.predict_proba(X_te_s)[:, 1]
        model  = lr
        is_rf  = False

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_prob)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall",    f"{rec:.3f}")
    m4.metric("F1-Score",  f"{f1:.3f}")
    m5.metric("ROC-AUC",   f"{auc:.3f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Not Interested", "Interested"],
                    yticklabels=["Not Interested", "Interested"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    with col_right:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{model_choice} (AUC={auc:.3f})",
                                     line=dict(color="#1D9E75", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Random Baseline",
                                     line=dict(color="grey", dash="dash")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title=f"ROC Curve – {model_choice}",
            margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(x=0.5, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("Feature Importance (Top 20)")
    if is_rf:
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        imp_df = pd.DataFrame({
            "Feature": [feat_names[i] for i in idx],
            "Importance": importances[idx]
        })
        color_scale = "Teal"
    else:
        coef = np.abs(lr.coef_[0])
        idx  = np.argsort(coef)[::-1][:20]
        imp_df = pd.DataFrame({
            "Feature": [feat_names[i] for i in idx],
            "Importance": coef[idx]
        })
        color_scale = "Blues"

    fig_imp = px.bar(imp_df, x="Importance", y="Feature",
                     orientation="h",
                     title=f"Top 20 Features – {model_choice}",
                     color="Importance",
                     color_continuous_scale=color_scale)
    fig_imp.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Classification report ─────────────────────────────────────────────────
    st.subheader("Detailed Classification Report")
    report_dict = classification_report(y_te, y_pred,
                                        target_names=["Not Interested", "Interested"],
                                        output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # ── Score distribution ─────────────────────────────────────────────────────
    st.subheader("Predicted Probability Distribution")
    score_df = pd.DataFrame({
        "Probability of Buying": y_prob,
        "Actual Label": y_te.map({1: "Interested", 0: "Not Interested"}).values
    })
    fig_dist = px.histogram(score_df, x="Probability of Buying",
                             color="Actual Label", nbins=30,
                             barmode="overlay", opacity=0.75,
                             title="Model Confidence – Buyer vs Non-Buyer",
                             color_discrete_map={"Interested": "#1D9E75",
                                                 "Not Interested": "#E24B4A"})
    fig_dist.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_dist, use_container_width=True)
