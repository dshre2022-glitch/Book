"""
app.py – Book DNA Analytics Dashboard
Entry point for Streamlit.
Auto-generates data and trains models on first run if artefacts are missing.
"""

import os
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Book DNA Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap: generate data + train models if missing ───────────────────────
@st.cache_resource(show_spinner=True)
def bootstrap():
    import subprocess, sys
    if not os.path.exists("book_dna_survey_2000.csv"):
        st.info("Generating synthetic dataset (first run) …")
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    if not os.path.exists("models/clf_rf.pkl"):
        st.info("Training ML models (first run, ~30 s) …")
        subprocess.run([sys.executable, "train_models.py"], check=True)

with st.spinner("Initialising Book DNA Analytics …"):
    bootstrap()

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = {
    "Overview": "tab_overview",
    "Descriptive Analysis": "tab_descriptive",
    "Diagnostic Analysis": "tab_diagnostic",
    "Predictive – Classification": "tab_classification",
    "Predictive – Clustering": "tab_clustering",
    "Predictive – Association Rules": "tab_association",
    "Predictive – Regression": "tab_regression",
    "Prescriptive + New Data Predictor": "tab_prescriptive",
}

st.sidebar.markdown("## 📚 Book DNA")
st.sidebar.markdown("*Data-Driven Business Intelligence*")
st.sidebar.markdown("---")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Book DNA Analytics v1.0  \nBuilt with Streamlit + scikit-learn")

# ── Dynamic page load ─────────────────────────────────────────────────────────
module_name = PAGES[selection]
if module_name == "tab_overview":
    from tab_overview import render
elif module_name == "tab_descriptive":
    from tab_descriptive import render
elif module_name == "tab_diagnostic":
    from tab_diagnostic import render
elif module_name == "tab_classification":
    from tab_classification import render
elif module_name == "tab_clustering":
    from tab_clustering import render
elif module_name == "tab_association":
    from tab_association import render
elif module_name == "tab_regression":
    from tab_regression import render
elif module_name == "tab_prescriptive":
    from tab_prescriptive import render

render()
