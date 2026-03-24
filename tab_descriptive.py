"""tab_descriptive.py – Descriptive Analysis."""

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
    st.title("📊 Descriptive Analysis")
    st.markdown("*What does our survey audience look like?*")

    df = load_data()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")
    city_opts = ["All"] + sorted(df["city_tier"].dropna().unique().tolist())
    age_opts  = ["All"] + sorted(df["age_group"].dropna().unique().tolist())
    gen_opts  = ["All"] + sorted(df["gender"].dropna().unique().tolist())

    city_f = st.sidebar.selectbox("City Tier", city_opts, key="d_city")
    age_f  = st.sidebar.selectbox("Age Group", age_opts,  key="d_age")
    gen_f  = st.sidebar.selectbox("Gender",    gen_opts,  key="d_gen")

    fdf = df.copy()
    if city_f != "All": fdf = fdf[fdf["city_tier"] == city_f]
    if age_f  != "All": fdf = fdf[fdf["age_group"]  == age_f]
    if gen_f  != "All": fdf = fdf[fdf["gender"]      == gen_f]

    st.caption(f"Showing {len(fdf):,} of {len(df):,} respondents")

    # ── Row 1: Demographics ───────────────────────────────────────────────────
    st.subheader("Demographics")
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        ag = fdf["age_group"].value_counts().reset_index()
        ag.columns = ["Age Group", "Count"]
        fig = px.bar(ag, x="Age Group", y="Count", title="Age Distribution",
                     color="Age Group", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        gd = fdf["gender"].value_counts().reset_index()
        gd.columns = ["Gender", "Count"]
        fig = px.pie(gd, names="Gender", values="Count", title="Gender Split",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c3:
        inc = fdf["income_band"].value_counts().reset_index()
        inc.columns = ["Income Band", "Count"]
        fig = px.bar(inc, x="Count", y="Income Band", orientation="h",
                     title="Income Distribution",
                     color="Income Band",
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Reading behaviour ───────────────────────────────────────────────
    st.subheader("Reading Behaviour")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        freq_order = ["Daily", "3-5 times/week", "Once a week",
                      "Few times/month", "Rarely", "Never"]
        rf = fdf["reading_frequency"].value_counts().reindex(freq_order, fill_value=0).reset_index()
        rf.columns = ["Frequency", "Count"]
        fig = px.bar(rf, x="Frequency", y="Count", title="Reading Frequency",
                     color="Frequency",
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        genres = (fdf["genres_enjoyed"].dropna().str.split("|").explode()
                  .value_counts().reset_index())
        genres.columns = ["Genre", "Count"]
        fig = px.bar(genres.head(9), x="Count", y="Genre", orientation="h",
                     title="Genre Popularity",
                     color="Count", color_continuous_scale="Teal")
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Personality & Lifestyle ────────────────────────────────────────
    st.subheader("Personality & Lifestyle")
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        pers = fdf["reading_personality"].value_counts().reset_index()
        pers.columns = ["Personality", "Count"]
        fig = px.pie(pers, names="Personality", values="Count",
                     title="Book DNA Personality Types",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        life = (fdf["lifestyle_activities"].dropna().str.split("|").explode()
                .value_counts().reset_index())
        life.columns = ["Activity", "Count"]
        fig = px.bar(life.head(10), x="Count", y="Activity", orientation="h",
                     title="Top Lifestyle Activities",
                     color="Count", color_continuous_scale="Purples")
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Product & Spend ─────────────────────────────────────────────────
    st.subheader("Product Interest & Spending")
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        prods = (fdf["products_interested"].dropna().str.split("|").explode()
                 .value_counts().reset_index())
        prods.columns = ["Product", "Count"]
        fig = px.bar(prods, x="Count", y="Product", orientation="h",
                     title="Product Interest Ranking",
                     color="Count", color_continuous_scale="Greens")
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r4c2:
        spend = fdf["monthly_book_spend"].value_counts().reset_index()
        spend.columns = ["Spend Band", "Count"]
        fig = px.bar(spend, x="Spend Band", y="Count",
                     title="Monthly Book Spend Distribution",
                     color="Spend Band",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 5: Gifting & Occasion ──────────────────────────────────────────────
    st.subheader("Gifting Behaviour")
    r5c1, r5c2 = st.columns(2)

    with r5c1:
        bp = fdf["buying_pattern"].value_counts().reset_index()
        bp.columns = ["Pattern", "Count"]
        fig = px.pie(bp, names="Pattern", values="Count",
                     title="Self-Buy vs Gifting Split",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r5c2:
        occ = (fdf["gifting_occasions"].dropna().str.split("|").explode()
               .value_counts().reset_index())
        occ.columns = ["Occasion", "Count"]
        fig = px.bar(occ, x="Occasion", y="Count",
                     title="Top Gifting Occasions",
                     color="Occasion",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Colour & Snack ─────────────────────────────────────────────────────────
    st.subheader("Aesthetic & Cultural Preferences")
    r6c1, r6c2, r6c3 = st.columns(3)

    with r6c1:
        cp = fdf["colour_palette"].value_counts().reset_index()
        cp.columns = ["Palette", "Count"]
        fig = px.bar(cp, x="Count", y="Palette", orientation="h",
                     title="Colour Palette Preferences",
                     color="Count", color_continuous_scale="Oranges")
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r6c2:
        snack = fdf["snack_preference"].value_counts().reset_index()
        snack.columns = ["Snack", "Count"]
        fig = px.pie(snack, names="Snack", values="Count",
                     title="Snack Preference",
                     color_discrete_sequence=px.colors.qualitative.Antique)
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r6c3:
        disc = fdf["discount_preference"].value_counts().reset_index()
        disc.columns = ["Discount Type", "Count"]
        fig = px.bar(disc, x="Count", y="Discount Type", orientation="h",
                     title="Preferred Discount Type",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
