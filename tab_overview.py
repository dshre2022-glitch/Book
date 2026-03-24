"""tab_overview.py – Overview KPIs and data health."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


def render():
    st.title("📚 Book DNA – Overview Dashboard")
    st.markdown("*Founder's morning pulse — key metrics at a glance.*")

    df = load_data()

    # ── KPI row ──────────────────────────────────────────────────────────────
    total = len(df)
    pct_interested = df["purchase_intent_label"].mean() * 100
    avg_spend = df["monthly_book_spend_numeric"].mean()
    top_product = (
        df["products_interested"]
        .dropna()
        .str.split("|")
        .explode()
        .value_counts()
        .index[0]
        if "products_interested" in df.columns else "N/A"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Respondents", f"{total:,}")
    c2.metric("% Likely to Buy", f"{pct_interested:.1f}%")
    c3.metric("Avg Monthly Book Spend", f"₹{avg_spend:.0f}")
    c4.metric("Top Product Interest", top_product)

    st.markdown("---")

    # ── Data health ──────────────────────────────────────────────────────────
    st.subheader("Data Quality Summary")
    col_a, col_b = st.columns(2)

    with col_a:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        health_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values
        }).query("`Missing Count` > 0").head(10)
        if health_df.empty:
            st.success("No missing values detected in dataset.")
        else:
            st.dataframe(health_df, use_container_width=True)

    with col_b:
        label_counts = df["purchase_intent_label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        label_counts["Label"] = label_counts["Label"].map({1: "Interested (1)", 0: "Not Interested (0)"})
        fig = px.pie(label_counts, names="Label", values="Count",
                     title="Classification Label Distribution",
                     color_discrete_sequence=["#1D9E75", "#E24B4A"])
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Respondents by city tier ──────────────────────────────────────────────
    st.subheader("Respondents by City Tier")
    city_counts = df["city_tier"].value_counts().reset_index()
    city_counts.columns = ["City Tier", "Count"]
    fig2 = px.bar(city_counts, x="City Tier", y="Count",
                  color="City Tier",
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  title="Survey Reach by City Tier")
    fig2.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Purchase intent breakdown ──────────────────────────────────────────────
    st.subheader("Purchase Intent Responses (Raw)")
    intent_counts = df["purchase_intent"].value_counts().reset_index()
    intent_counts.columns = ["Response", "Count"]
    order = ["Very likely", "Likely", "Neutral", "Unlikely", "Not interested"]
    intent_counts["Response"] = pd.Categorical(intent_counts["Response"],
                                               categories=order, ordered=True)
    intent_counts = intent_counts.sort_values("Response")
    fig3 = px.bar(intent_counts, x="Response", y="Count",
                  color="Response",
                  color_discrete_sequence=["#1D9E75","#5DCAA5","#EF9F27","#F09595","#E24B4A"],
                  title="Q25 – Purchase Intent Distribution")
    fig3.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.caption(f"Dataset: {total} respondents · {len(df.columns)} features · "
               "Synthetic data generated with Indian market distributions.")
