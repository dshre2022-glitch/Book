"""tab_prescriptive.py – Prescriptive Analysis + New Customer Predictor."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import io

from utils import preprocess, TARGET_CLF, TARGET_REG

CLUSTER_NAMES = {
    0: "Urban Gen Z Escapist",
    1: "Aspirational Tier-2 Learner",
    2: "Premium Gifting Buyer",
    3: "Reluctant Non-Reader",
    4: "Traditional Homemaker",
}

PRESCRIPTIONS = {
    "Urban Gen Z Escapist": {
        "priority": "🟢 Priority 1 — Invest",
        "channel": "Instagram Reels, BookTok",
        "lead_product": "Reading Journal + Themed Candle bundle",
        "discount": "20% flat off first order",
        "message": "Your reading personality, finally in a box.",
        "why": "High digital reach, low CAC, escapist identity aligns perfectly with our brand story.",
    },
    "Aspirational Tier-2 Learner": {
        "priority": "🟢 Priority 2 — Grow",
        "channel": "WhatsApp, YouTube BookTube",
        "lead_product": "Book Subscription Box + Reading Life Planner",
        "discount": "Bundle deal — 2 months subscription at 1-month price",
        "message": "The book that changes your life is waiting.",
        "why": "Strong word-of-mouth, loyal once converted, Diwali gifting spike potential.",
    },
    "Premium Gifting Buyer": {
        "priority": "🟡 Priority 3 — Premium",
        "channel": "Email campaigns, Diwali/festival ads",
        "lead_product": "Book DNA Profile Kit + Reading Corner Decor",
        "discount": "Free eco-friendly gift packaging (no price discount needed)",
        "message": "The most thoughtful gift this Diwali.",
        "why": "High AOV, low volume needed for strong revenue, premium eco-packaging closes the sale.",
    },
    "Reluctant Non-Reader": {
        "priority": "🟠 Deprioritise Y1",
        "channel": "Retargeting only (low budget)",
        "lead_product": "Free Book DNA Quiz → Journal as entry product",
        "discount": "30% off first purchase — heavy incentive required",
        "message": "You're not a 'non-reader'. You just haven't met your book yet.",
        "why": "High conversion cost, needs strong hook. Use quiz as free lead magnet.",
    },
    "Traditional Homemaker": {
        "priority": "🟠 Revisit Year 2",
        "channel": "WhatsApp groups, regional language content",
        "lead_product": "Themed Candle + Stationery Set",
        "discount": "Festive bundle — Diwali/Rakhi themed packaging",
        "message": "Stories that feel like home.",
        "why": "Lower digital reach, but strong gifting intent. Festival campaigns work well.",
    },
}


@st.cache_data
def load_data():
    path = "book_dna_clustered.csv" if __import__("os").path.exists("book_dna_clustered.csv") \
        else "book_dna_survey_2000.csv"
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    rf       = joblib.load("models/clf_rf.pkl")
    kmeans   = joblib.load("models/kmeans.pkl")
    ridge    = joblib.load("models/reg_ridge.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return rf, kmeans, ridge, encoders


def render():
    st.title("🎯 Prescriptive Analysis + New Customer Predictor")
    st.markdown("*What should we do for each segment? Upload new survey data to predict customer value.*")

    df = load_data()
    rf, kmeans, ridge, encoders = load_models()

    # ── Section 1: Prescriptive action cards ──────────────────────────────────
    st.subheader("Marketing Action Plan by Segment")

    if "cluster_name" in df.columns:
        seg_summary = df.groupby("cluster_name").agg(
            Count=("purchase_intent_label", "count"),
            Pct_Interested=("purchase_intent_label", "mean"),
            Avg_Spend=("monthly_book_spend_numeric", "mean")
        ).reset_index()
        seg_summary["Pct_Interested"] = (seg_summary["Pct_Interested"] * 100).round(1)
        seg_summary["Avg_Spend"] = seg_summary["Avg_Spend"].round(0)
    else:
        seg_summary = pd.DataFrame()

    for seg_name, rx in PRESCRIPTIONS.items():
        with st.expander(f"{rx['priority']}  |  {seg_name}", expanded=False):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                if not seg_summary.empty:
                    row = seg_summary[seg_summary["cluster_name"] == seg_name]
                    if not row.empty:
                        r = row.iloc[0]
                        st.metric("Respondents", f"{int(r['Count']):,}")
                        st.metric("% Interested", f"{r['Pct_Interested']}%")
                        st.metric("Avg Spend", f"₹{r['Avg_Spend']:.0f}")
            with col_b:
                st.markdown(f"**Channel:** {rx['channel']}")
                st.markdown(f"**Lead Product:** {rx['lead_product']}")
                st.markdown(f"**Discount Strategy:** {rx['discount']}")
                st.markdown(f"**Campaign Message:** *{rx['message']}*")
                st.caption(f"Why: {rx['why']}")

    st.markdown("---")

    # ── Section 2: Summary priority matrix ────────────────────────────────────
    st.subheader("Investment Priority Matrix")
    priority_data = {
        "Segment": list(PRESCRIPTIONS.keys()),
        "Priority": ["1 – Invest", "2 – Grow", "3 – Premium", "4 – Retarget", "5 – Y2"],
        "CAC Estimate": ["Low", "Medium", "Low", "High", "High"],
        "Revenue Potential": ["Medium", "High", "High", "Low", "Low"],
        "Recommended Budget %": [35, 30, 20, 10, 5],
    }
    prio_df = pd.DataFrame(priority_data)
    st.dataframe(prio_df, use_container_width=True, hide_index=True)

    fig_budget = px.pie(prio_df, names="Segment", values="Recommended Budget %",
                        title="Recommended Year 1 Marketing Budget Allocation",
                        color_discrete_sequence=["#1D9E75","#378ADD","#EF9F27",
                                                  "#E24B4A","#7F77DD"])
    fig_budget.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig_budget, use_container_width=True)

    st.markdown("---")

    # ── Section 3: New data upload & prediction ────────────────────────────────
    st.subheader("New Customer Predictor")
    st.markdown(
        "Upload a new CSV of survey responses (same 33-column format) to instantly get "
        "purchase probability, customer segment, and predicted spend for each respondent."
    )

    with st.expander("Download template CSV (first 5 rows of training data)"):
        template = df.drop(
            columns=["cluster", "cluster_name", "purchase_intent_label",
                     "persona_id"], errors="ignore"
        ).head(5)
        csv_bytes = template.to_csv(index=False).encode()
        st.download_button("Download Template CSV", csv_bytes,
                           file_name="book_dna_template.csv",
                           mime="text/csv")

    uploaded = st.file_uploader("Upload new survey CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(new_df):,} new respondents · {len(new_df.columns)} columns")
            st.dataframe(new_df.head(3), use_container_width=True)

            with st.spinner("Running predictions …"):
                X_new, _ = preprocess(new_df, fit_encoders=encoders, return_encoders=True)

                # Classification
                buy_prob = rf.predict_proba(X_new)[:, 1]

                # Clustering
                cluster_ids = kmeans.predict(X_new)
                cluster_names_pred = [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster_ids]

                # Regression
                pred_spend = np.clip(ridge.predict(X_new), 0, None).round(0)

                # Priority flag
                def priority_flag(prob):
                    if prob >= 0.65:
                        return "🟢 High Priority"
                    elif prob >= 0.40:
                        return "🟡 Nurture"
                    else:
                        return "🔴 Low Priority"

                results = new_df.copy()
                results["buy_probability"]   = buy_prob.round(3)
                results["customer_segment"]  = cluster_names_pred
                results["predicted_spend_rs"]= pred_spend
                results["action_flag"]       = [priority_flag(p) for p in buy_prob]

            st.subheader("Prediction Results")

            # Summary metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total New Records",   f"{len(results):,}")
            mc2.metric("High Priority",        f"{(results['action_flag']=='🟢 High Priority').sum():,}")
            mc3.metric("Avg Buy Probability",  f"{buy_prob.mean():.1%}")
            mc4.metric("Avg Predicted Spend",  f"₹{pred_spend.mean():.0f}")

            # Action flag distribution
            flag_counts = results["action_flag"].value_counts().reset_index()
            flag_counts.columns = ["Action", "Count"]
            fig_flag = px.pie(flag_counts, names="Action", values="Count",
                              title="Action Flag Distribution",
                              color_discrete_map={
                                  "🟢 High Priority": "#1D9E75",
                                  "🟡 Nurture": "#EF9F27",
                                  "🔴 Low Priority": "#E24B4A"
                              })
            fig_flag.update_layout(margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_flag, use_container_width=True)

            # Segment distribution for new data
            seg_new = results["customer_segment"].value_counts().reset_index()
            seg_new.columns = ["Segment", "Count"]
            fig_seg_new = px.bar(seg_new, x="Segment", y="Count",
                                 title="New Respondents by Predicted Segment",
                                 color="Segment",
                                 color_discrete_sequence=["#378ADD","#1D9E75",
                                                           "#EF9F27","#E24B4A","#7F77DD"])
            fig_seg_new.update_layout(showlegend=False,
                                      margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_seg_new, use_container_width=True)

            # Full results table
            st.subheader("Full Results Table")
            display_results = results[["buy_probability", "customer_segment",
                                        "predicted_spend_rs", "action_flag"]].copy()
            if "age_group" in results.columns:
                display_results.insert(0, "age_group", results["age_group"])
            if "city_tier" in results.columns:
                display_results.insert(0, "city_tier", results["city_tier"])
            st.dataframe(display_results, use_container_width=True)

            # Download predictions
            out_csv = results.to_csv(index=False).encode()
            st.download_button(
                "Download Predictions as CSV",
                out_csv,
                file_name="book_dna_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure the uploaded CSV matches the template format.")
