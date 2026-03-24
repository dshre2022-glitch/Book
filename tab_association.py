"""tab_association.py – Association Rule Mining."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_OK = True
except ImportError:
    MLXTEND_OK = False


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


def build_transactions(df, col):
    return (df[col].fillna("None").str.split("|")
            .apply(lambda x: [i.strip() for i in x if i.strip() and i.strip() != "None"])
            .tolist())


@st.cache_data
def run_apriori(transactions, min_support=0.05, min_confidence=0.3):
    if not MLXTEND_OK:
        return None, None
    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    te_df  = pd.DataFrame(te_arr, columns=te.columns_)
    freq_items = apriori(te_df, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return freq_items, pd.DataFrame()
    rules = association_rules(freq_items, metric="confidence",
                              min_threshold=min_confidence)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return freq_items, rules


def render():
    st.title("🔗 Predictive Analysis — Association Rule Mining")
    st.markdown("*What do customers choose together? Discover product bundles, lifestyle links, and festival patterns.*")

    if not MLXTEND_OK:
        st.error("**mlxtend** library is required for Association Rule Mining. "
                 "Please ensure it is listed in requirements.txt and the environment is rebuilt.")
        st.info("requirements.txt includes `mlxtend>=0.23.0`. "
                "On Streamlit Cloud this installs automatically.")
        return

    df = load_data()

    # ── Controls ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### ARM Parameters")
    min_sup  = st.sidebar.slider("Min Support",    0.02, 0.30, 0.05, 0.01, key="arm_sup")
    min_conf = st.sidebar.slider("Min Confidence", 0.10, 0.90, 0.30, 0.05, key="arm_conf")
    min_lift = st.sidebar.slider("Min Lift",       1.0,  5.0,  1.0,  0.1,  key="arm_lift")
    top_n    = st.sidebar.slider("Show Top N Rules", 5, 50, 20, 5, key="arm_n")

    basket_choice = st.radio(
        "Select Association Basket",
        ["Product Basket (Q17)", "Lifestyle Basket (Q12)", "Genre Basket (Q7)",
         "Past Purchases (Q29)", "Cultural + Festival (Q13 × Q31)"],
        horizontal=True
    )

    # ── Build transactions ────────────────────────────────────────────────────
    if basket_choice == "Product Basket (Q17)":
        transactions = build_transactions(df, "products_interested")
        basket_label = "Products"
    elif basket_choice == "Lifestyle Basket (Q12)":
        transactions = build_transactions(df, "lifestyle_activities")
        basket_label = "Lifestyle Activities"
    elif basket_choice == "Genre Basket (Q7)":
        transactions = build_transactions(df, "genres_enjoyed")
        basket_label = "Genres"
    elif basket_choice == "Past Purchases (Q29)":
        transactions = build_transactions(df, "past_purchases")
        basket_label = "Past Purchases"
    else:
        # Cultural: combine clothing + occasions
        def comb(row):
            parts = []
            if pd.notna(row["clothing_style"]):
                parts.append("Style:" + str(row["clothing_style"]))
            if pd.notna(row["gifting_occasions"]):
                parts += ["Occ:" + o.strip()
                          for o in str(row["gifting_occasions"]).split("|")]
            return parts
        transactions = df.apply(comb, axis=1).tolist()
        basket_label = "Cultural + Festival"

    transactions = [t for t in transactions if len(t) >= 2]
    st.caption(f"Running Apriori on {len(transactions):,} transactions · "
               f"min_support={min_sup} · min_confidence={min_conf}")

    freq_items, rules = run_apriori(transactions, min_support=min_sup,
                                    min_confidence=min_conf)

    if rules is None or rules.empty:
        st.warning("No rules found with current thresholds. "
                   "Try lowering min support or confidence.")
        return

    # Apply lift filter
    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)

    # ── Top rules table ───────────────────────────────────────────────────────
    st.subheader(f"Top {min(top_n, len(rules))} Rules by Lift — {basket_label}")
    display_cols = ["antecedents_str", "consequents_str",
                    "support", "confidence", "lift"]
    show_rules = rules[display_cols].head(top_n).reset_index(drop=True)
    show_rules.columns = ["Antecedent", "Consequent",
                          "Support", "Confidence", "Lift"]
    show_rules["Support"]    = show_rules["Support"].round(4)
    show_rules["Confidence"] = show_rules["Confidence"].round(4)
    show_rules["Lift"]       = show_rules["Lift"].round(3)
    st.dataframe(show_rules, use_container_width=True)

    st.markdown("---")

    # ── Scatter: Support vs Confidence, size=Lift ────────────────────────────
    st.subheader("Support × Confidence × Lift Bubble Chart")
    plot_rules = rules.head(top_n).copy()
    fig_bubble = px.scatter(
        plot_rules,
        x="support", y="confidence",
        size="lift", color="lift",
        hover_data={"antecedents_str": True, "consequents_str": True,
                    "lift": ":.3f", "support": ":.4f", "confidence": ":.4f"},
        color_continuous_scale="Teal",
        title=f"Rules — Support vs Confidence (bubble size = Lift) · {basket_label}",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
    )
    fig_bubble.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Lift distribution histogram ───────────────────────────────────────────
    st.subheader("Lift Distribution Across All Rules")
    fig_lift = px.histogram(rules, x="lift", nbins=30,
                             title="Distribution of Lift Values",
                             color_discrete_sequence=["#EF9F27"])
    fig_lift.add_vline(x=1.0, line_dash="dash", line_color="red",
                       annotation_text="Lift = 1 (random)")
    fig_lift.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_lift, use_container_width=True)

    # ── Confidence distribution ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        fig_conf = px.histogram(rules, x="confidence", nbins=25,
                                title="Confidence Distribution",
                                color_discrete_sequence=["#378ADD"])
        fig_conf.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_b:
        fig_sup = px.histogram(rules, x="support", nbins=25,
                               title="Support Distribution",
                               color_discrete_sequence=["#7F77DD"])
        fig_sup.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_sup, use_container_width=True)

    # ── Frequent itemsets ─────────────────────────────────────────────────────
    st.subheader("Most Frequent Itemsets")
    if freq_items is not None and not freq_items.empty:
        freq_show = freq_items.copy()
        freq_show["itemsets_str"] = freq_show["itemsets"].apply(
            lambda x: ", ".join(sorted(x))
        )
        freq_show = freq_show.sort_values("support", ascending=False).head(20)
        fig_freq = px.bar(freq_show, x="support", y="itemsets_str",
                          orientation="h",
                          title="Top 20 Frequent Itemsets by Support",
                          color="support",
                          color_continuous_scale="Greens")
        fig_freq.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                               coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_freq, use_container_width=True)

    # ── Business interpretation ────────────────────────────────────────────────
    st.subheader("Top 5 Business Insights from Rules")
    top5 = rules.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        st.info(
            f"**Rule {i}:** Customers who choose **{row['antecedents_str']}** "
            f"also tend to choose **{row['consequents_str']}**  \n"
            f"Support: `{row['support']:.3f}` · "
            f"Confidence: `{row['confidence']:.3f}` · "
            f"Lift: `{row['lift']:.3f}`"
        )
