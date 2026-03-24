"""tab_diagnostic.py – Diagnostic Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


def render():
    st.title("🔍 Diagnostic Analysis")
    st.markdown("*Why do some customers buy and others don't?*")

    df = load_data()

    # ── Correlation heatmap (numeric features) ────────────────────────────────
    st.subheader("Correlation Heatmap — Numeric Features")
    num_cols = ["stress_level", "online_comfort", "social_proof_need",
                "eco_importance", "aspiration_gap_score", "social_influence_score",
                "past_purchase_count", "monthly_book_spend_numeric",
                "purchase_intent_label"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Pearson Correlation – Numeric Survey Features", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Cross-tab: personality × intent ──────────────────────────────────────
    st.subheader("Reading Personality × Purchase Intent")
    ct = pd.crosstab(df["reading_personality"], df["purchase_intent_label"],
                     normalize="index") * 100
    ct = ct.rename(columns={0: "Not Interested (%)", 1: "Interested (%)"})
    ct = ct.reset_index()

    fig = px.bar(ct, x="reading_personality",
                 y=["Interested (%)", "Not Interested (%)"],
                 barmode="stack",
                 title="Purchase Intent by Reading Personality Type",
                 color_discrete_map={"Interested (%)": "#1D9E75",
                                     "Not Interested (%)": "#E24B4A"},
                 labels={"reading_personality": "Personality Type",
                         "value": "% of Segment"})
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Stress level vs genre ─────────────────────────────────────────────────
    st.subheader("Stress Level vs Genre Preference")
    genre_stress = (
        df[["stress_level", "genres_enjoyed"]].copy()
        .assign(genre=df["genres_enjoyed"].str.split("|"))
        .explode("genre")
        .groupby("genre")["stress_level"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    genre_stress.columns = ["Genre", "Avg Stress Level"]
    fig = px.bar(genre_stress, x="Avg Stress Level", y="Genre",
                 orientation="h",
                 title="Average Stress Level by Genre Preference",
                 color="Avg Stress Level",
                 color_continuous_scale="RdYlGn_r")
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Aspiration gap vs spend willingness ───────────────────────────────────
    st.subheader("Aspiration Gap Score vs Monthly Spend")
    col_a, col_b = st.columns(2)

    with col_a:
        agg_gap = (df.groupby("aspiration_gap_score")["monthly_book_spend_numeric"]
                   .mean().reset_index())
        agg_gap.columns = ["Aspiration Gap Score", "Avg Monthly Spend (₹)"]
        fig = px.bar(agg_gap, x="Aspiration Gap Score", y="Avg Monthly Spend (₹)",
                     title="Higher Aspiration Gap → Higher Spend?",
                     color="Avg Monthly Spend (₹)",
                     color_continuous_scale="Teal")
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        agg_past = (df.groupby("past_purchase_count")["purchase_intent_label"]
                    .mean().reset_index())
        agg_past.columns = ["Past Purchase Count", "% Interested"]
        agg_past["% Interested"] = (agg_past["% Interested"] * 100).round(1)
        fig = px.line(agg_past, x="Past Purchase Count", y="% Interested",
                      title="Past Purchases → Purchase Intent",
                      markers=True,
                      color_discrete_sequence=["#1D9E75"])
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── City tier vs product interest ─────────────────────────────────────────
    st.subheader("City Tier × Product Interest Overlap")
    city_prod = (
        df[["city_tier", "products_interested"]].copy()
        .assign(product=df["products_interested"].str.split("|"))
        .explode("product")
        .groupby(["city_tier", "product"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(city_prod, x="city_tier", y="count", color="product",
                 barmode="stack",
                 title="Product Interest by City Tier",
                 labels={"city_tier": "City Tier", "count": "Mentions",
                         "product": "Product"},
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Purchase motivation breakdown ─────────────────────────────────────────
    st.subheader("Purchase Motivation vs Intent")
    mot_intent = pd.crosstab(df["purchase_motivation"],
                             df["purchase_intent_label"],
                             normalize="index") * 100
    mot_intent = mot_intent.rename(
        columns={0: "Not Interested (%)", 1: "Interested (%)"}).reset_index()
    mot_intent = mot_intent.sort_values("Interested (%)", ascending=False)
    fig = px.bar(mot_intent, x="purchase_motivation",
                 y=["Interested (%)", "Not Interested (%)"],
                 barmode="stack",
                 title="Purchase Motivation vs Buyer Intent",
                 color_discrete_map={"Interested (%)": "#1D9E75",
                                     "Not Interested (%)": "#E24B4A"},
                 labels={"purchase_motivation": "Motivation", "value": "%"})
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Social influence score distribution ───────────────────────────────────
    st.subheader("Social Influence Score Distribution by Intent")
    fig = px.histogram(df, x="social_influence_score",
                       color=df["purchase_intent_label"].map({1: "Interested", 0: "Not Interested"}),
                       nbins=20, barmode="overlay", opacity=0.75,
                       title="Social Influence Score — Buyers vs Non-Buyers",
                       color_discrete_map={"Interested": "#1D9E75",
                                           "Not Interested": "#E24B4A"},
                       labels={"social_influence_score": "Social Influence Score",
                               "color": "Segment"})
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
