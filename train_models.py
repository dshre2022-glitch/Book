"""
train_models.py
Run once before launching the Streamlit app:
    python train_models.py
Produces: clf_rf.pkl, clf_lr.pkl, kmeans.pkl, reg_ridge.pkl, encoders.pkl
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from utils import preprocess, TARGET_CLF, TARGET_REG

DATA_PATH = "book_dna_survey_2000.csv"


def train_all():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH)

    print("Preprocessing …")
    X, encoders = preprocess(df, return_encoders=True)
    y_clf = df[TARGET_CLF].astype(int)
    y_reg = df[TARGET_REG].fillna(0)

    # ── Classification split ──────────────────────────────────────────────────
    X_tr, X_te, yc_tr, yc_te = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Random Forest
    print("Training Random Forest …")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, yc_tr)
    yc_pred = rf.predict(X_te)
    yc_prob = rf.predict_proba(X_te)[:, 1]
    print(f"  RF  Acc={accuracy_score(yc_te, yc_pred):.3f}  "
          f"AUC={roc_auc_score(yc_te, yc_prob):.3f}")

    # Logistic Regression
    print("Training Logistic Regression …")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr_s, yc_tr)
    ylr_pred = lr.predict(X_te_s)
    ylr_prob = lr.predict_proba(X_te_s)[:, 1]
    print(f"  LR  Acc={accuracy_score(yc_te, ylr_pred):.3f}  "
          f"AUC={roc_auc_score(yc_te, ylr_prob):.3f}")

    # ── Clustering ────────────────────────────────────────────────────────────
    print("Training K-Means (k=5) …")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X)
    df["cluster"] = cluster_labels
    print(f"  Cluster sizes: {dict(pd.Series(cluster_labels).value_counts().sort_index())}")

    # ── Regression ────────────────────────────────────────────────────────────
    print("Training Ridge Regression …")
    # drop rows where target is extreme outlier for cleaner training
    mask = y_reg < 2000
    X_reg = X[mask]
    y_reg_clean = y_reg[mask]
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg, y_reg_clean, test_size=0.2, random_state=42
    )
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xr_tr, yr_tr)
    yr_pred = ridge.predict(Xr_te)
    rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
    print(f"  Ridge R²={r2_score(yr_te, yr_pred):.3f}  RMSE=₹{rmse:.1f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf,      "models/clf_rf.pkl")
    joblib.dump(lr,      "models/clf_lr.pkl")
    joblib.dump(scaler,  "models/scaler.pkl")
    joblib.dump(kmeans,  "models/kmeans.pkl")
    joblib.dump(ridge,   "models/reg_ridge.pkl")
    joblib.dump(encoders,"models/encoders.pkl")

    # Save cluster-annotated data for dashboard use
    df.to_csv("book_dna_clustered.csv", index=False)
    print("\nAll models saved to models/")
    print("Clustered data saved to book_dna_clustered.csv")


if __name__ == "__main__":
    train_all()
