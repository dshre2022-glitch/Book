# Book DNA Analytics Dashboard

Data-driven business intelligence for the Book DNA reading-lifestyle brand targeting Gen Z India.

## Features
- **Descriptive Analysis** – Demographics, reading habits, lifestyle preferences
- **Diagnostic Analysis** – Correlations, cross-tabs, purchase drivers
- **Classification** – Random Forest + Logistic Regression, ROC curve, feature importance
- **Clustering** – K-Means (k=5), elbow/silhouette, PCA visualisation, segment profiles
- **Association Rule Mining** – Apriori on products, genres, lifestyle, cultural baskets
- **Regression** – Ridge regression predicting monthly spend (₹)
- **Prescriptive + New Data Predictor** – Segment action plans + upload new CSV for instant predictions

## Local Setup

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/book-dna-analytics.git
cd book-dna-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset and train models
python generate_data.py
python train_models.py

# 4. Launch dashboard
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push all files (flat, no sub-folders) to a GitHub repo
2. Go to https://share.streamlit.io → New app
3. Select repo, set **Main file** to `app.py`
4. Click **Deploy**

The app auto-generates data and trains models on first launch (~30 seconds).
Models are cached via `@st.cache_resource`.

## File Structure
```
app.py                    ← Streamlit entry point
generate_data.py          ← Synthetic dataset generator
train_models.py           ← ML model trainer
utils.py                  ← Shared preprocessing pipeline
tab_overview.py           ← Tab 1: KPIs
tab_descriptive.py        ← Tab 2: Descriptive charts
tab_diagnostic.py         ← Tab 3: Diagnostic analysis
tab_classification.py     ← Tab 4a: Classification
tab_clustering.py         ← Tab 4b: Clustering
tab_association.py        ← Tab 4c: Association rules
tab_regression.py         ← Tab 4d: Regression
tab_prescriptive.py       ← Tab 5: Prescriptive + new data upload
requirements.txt          ← Python dependencies
README.md                 ← This file
```

## Dataset
2,000 synthetic respondents based on Indian market distributions.
33 survey features + 3 derived features + classification and regression targets.
5 built-in customer personas with realistic cross-column correlations and ~4% noise injection.
