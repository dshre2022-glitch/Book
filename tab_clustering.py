"""tab_clustering.py – K-Means Clustering."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from utils import preprocess


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    kmeans   = joblib.load("models/kmeans.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return kmeans, encoders


CLUSTER_NAMES = {
    0: "Urban Gen Z Escapist",
    1: "Aspirational Tier-2 Learner",
    2: "Premium Gifting Buyer",
    3: "Reluctant Non-Reader",
    4: "Traditional Homemaker",
}

CLUSTER_COLORS = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A", "#7F77DD"]


def render():
    st.title("🔵 Predictive Analysis — Customer Clustering")
    st.markdown("*Who are the distinct customer segments? K-Means (k=5) clustering.*")

    df = load_data()
    kmeans, encoders = load_models()

    X, _ = preprocess(df, fit_encoders=encoders, return_encoders=True)

    if "cluster" not in df.columns:
        df["cluster"] = kmeans.predict(X)

    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

    # ── Elbow + silhouette ────────────────────────────────────────────────────
    st.subheader("Optimal k — Elbow Method & Silhouette Score")

    @st.cache_data
    def compute_elbow(_X):
        X_arr = np.array(_X)
        inertias, sil_scores, ks = [], [], list(range(2, 9))
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_arr)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_arr, labels, sample_size=500))
        return ks, inertias, sil_scores

    ks, inertias, sil_scores = compute_elbow(X)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                                       line=dict(color="#378ADD", width=2),
                                       marker=dict(size=7)))
        fig_elbow.add_vline(x=5, line_dash="dash", line_color="#E24B4A",
                            annotation_text="k=5 selected")
        fig_elbow.update_layout(title="Elbow Method — Inertia vs k",
                                xaxis_title="Number of Clusters (k)",
                                yaxis_title="Inertia",
                                margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_e2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=ks, y=sil_scores, mode="lines+markers",
                                     line=dict(color="#1D9E75", width=2),
                                     marker=dict(size=7)))
        fig_sil.add_vline(x=5, line_dash="dash", line_color="#E24B4A",
                          annotation_text="k=5 selected")
        fig_sil.update_layout(title="Silhouette Score vs k",
                               xaxis_title="Number of Clusters (k)",
                               yaxis_title="Silhouette Score",
                               margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # ── Cluster size distribution ─────────────────────────────────────────────
    st.subheader("Cluster Size Distribution")
    cluster_counts = df["cluster_name"].value_counts().reset_index()
    cluster_counts.columns = ["Segment", "Count"]
    fig_size = px.bar(cluster_counts, x="Segment", y="Count",
                      color="Segment",
                      color_discrete_sequence=CLUSTER_COLORS,
                      title="Respondents per Customer Segment")
    fig_size.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_size, use_container_width=True)

    # ── 2D PCA scatter ────────────────────────────────────────────────────────
    st.subheader("Cluster Visualisation — PCA (2D Projection)")

    @st.cache_data
    def pca_2d(_X):
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(np.array(_X))

    coords = pca_2d(X)
    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                            "Segment": df["cluster_name"].values})
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Segment",
                         title="Customer Segments in 2D PCA Space",
                         color_discrete_sequence=CLUSTER_COLORS,
                         opacity=0.6)
    fig_pca.update_traces(marker=dict(size=4))
    fig_pca.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")

    # ── Cluster profile cards ─────────────────────────────────────────────────
    st.subheader("Segment Profile Cards")
    profile_cols = ["stress_level", "online_comfort", "eco_importance",
                    "aspiration_gap_score", "social_influence_score",
                    "past_purchase_count", "monthly_book_spend_numeric",
                    "purchase_intent_label"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profiles = df.groupby("cluster_name")[profile_cols].mean().round(2)

    for i, (seg_name, row) in enumerate(profiles.iterrows()):
        with st.expander(f"Segment: {seg_name}", expanded=(i == 0)):
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Avg Stress",        f"{row.get('stress_level', 0):.1f}/5")
            pc2.metric("Online Comfort",    f"{row.get('online_comfort', 0):.1f}/5")
            pc3.metric("Eco Importance",    f"{row.get('eco_importance', 0):.1f}/5")
            pc4.metric("Past Purchases",    f"{row.get('past_purchase_count', 0):.1f} items")
            pc5, pc6, pc7, _ = st.columns(4)
            pc5.metric("Aspiration Gap",    f"{row.get('aspiration_gap_score', 0):.1f}")
            pc6.metric("Avg Book Spend",    f"₹{row.get('monthly_book_spend_numeric', 0):.0f}")
            pc7.metric("% Likely to Buy",   f"{row.get('purchase_intent_label', 0)*100:.0f}%")

    # ── Radar chart per cluster ────────────────────────────────────────────────
    st.subheader("Segment Radar Comparison")
    radar_cols = ["stress_level", "online_comfort", "eco_importance",
                  "social_influence_score", "aspiration_gap_score"]
    radar_cols = [c for c in radar_cols if c in df.columns]

    radar_df = df.groupby("cluster_name")[radar_cols].mean()
    # normalise to 0-1
    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-6)

    fig_radar = go.Figure()
    for i, (seg, vals) in enumerate(radar_norm.iterrows()):
        vals_list = vals.tolist()
        vals_list.append(vals_list[0])  # close polygon
        labels = radar_cols + [radar_cols[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_list, theta=labels, fill="toself",
            name=seg, line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Trait Comparison (normalised)",
        margin=dict(t=60, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
