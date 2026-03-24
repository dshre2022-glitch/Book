"""
utils.py – shared preprocessing & helper functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Column metadata ────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "age_group", "gender", "city_tier", "occupation", "income_band",
    "reading_frequency", "reading_personality", "reading_time",
    "reading_format", "clothing_style", "saree_preference", "colour_palette",
    "snack_preference", "stationery_spend_band", "subscription_intent",
    "monthly_book_spend", "purchase_motivation", "discount_preference",
    "discovery_channel", "self_identity", "aspiration", "buying_pattern",
    "loyalty_orientation", "purchase_intent",
]

ORDINAL_COLS = [
    "stress_level", "online_comfort", "social_proof_need", "eco_importance",
]

NUMERIC_COLS = [
    "monthly_book_spend_numeric", "aspiration_gap_score",
    "social_influence_score", "past_purchase_count",
]

MULTI_COLS = [
    "genres_enjoyed", "lifestyle_activities", "products_interested",
    "past_purchases", "gifting_occasions",
]

TARGET_CLF  = "purchase_intent_label"
TARGET_REG  = "monthly_book_spend_numeric"

INCOME_MIDPOINTS = {
    "Below 15k": 10000, "15k-30k": 22500, "30k-60k": 45000,
    "60k-1L": 80000, "Above 1L": 125000, "Prefer not to say": 35000,
}

SPEND_MIDPOINTS = {
    "0 (library/borrowing)": 0, "1-200": 100,
    "201-500": 350, "501-1000": 750, "Above 1000": 1200,
}

FREQ_ORDER = {
    "Never": 0, "Rarely": 1, "Few times/month": 2,
    "Once a week": 3, "3-5 times/week": 4, "Daily": 5,
}


def load_and_validate(path_or_df):
    """Load CSV or accept DataFrame; return DataFrame with basic validation."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    return df


def expand_multi(df, col, prefix=None):
    """One-hot expand a pipe-separated multi-select column."""
    pfx = prefix or col
    expanded = df[col].fillna("").str.get_dummies(sep="|")
    expanded.columns = [f"{pfx}__{c.replace(' ', '_')}" for c in expanded.columns]
    return expanded


def preprocess(df, fit_encoders=None, return_encoders=False):
    """
    Full preprocessing pipeline.
    fit_encoders: dict of {col: LabelEncoder} from training – pass for new data.
    Returns (X, encoders_dict) if return_encoders else X.
    """
    df = df.copy()

    # ── Derived numeric ──
    if "income_band" in df.columns:
        df["income_numeric"] = df["income_band"].map(INCOME_MIDPOINTS).fillna(35000)
    if "monthly_book_spend" in df.columns:
        df["book_spend_numeric_raw"] = df["monthly_book_spend"].map(SPEND_MIDPOINTS).fillna(100)
    if "reading_frequency" in df.columns:
        df["reading_freq_ord"] = df["reading_frequency"].map(FREQ_ORDER).fillna(0)

    # ── Ordinal passthrough ──
    for c in ORDINAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(3)

    # ── Numeric passthrough ──
    for c in ["aspiration_gap_score", "social_influence_score", "past_purchase_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ── Label-encode categoricals ──
    encoders = fit_encoders or {}
    cat_feats = [c for c in CATEGORICAL_COLS if c in df.columns and c not in [TARGET_CLF, "purchase_intent"]]
    for c in cat_feats:
        df[c] = df[c].astype(str).fillna("Unknown")
        if fit_encoders and c in encoders:
            le = encoders[c]
            df[c + "_enc"] = df[c].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            le = LabelEncoder()
            df[c + "_enc"] = le.fit_transform(df[c])
            encoders[c] = le

    # ── Expand multi-select (top items only for tractability) ──
    multi_dummies = []
    for col in MULTI_COLS:
        if col in df.columns:
            exp = expand_multi(df, col)
            multi_dummies.append(exp)

    # ── Assemble feature matrix ──
    enc_cols = [c + "_enc" for c in cat_feats if c + "_enc" in df.columns]
    scalar_cols = (ORDINAL_COLS + ["income_numeric", "book_spend_numeric_raw",
                                    "reading_freq_ord", "aspiration_gap_score",
                                    "social_influence_score", "past_purchase_count"])
    scalar_cols = [c for c in scalar_cols if c in df.columns]

    parts = [df[enc_cols].reset_index(drop=True),
             df[scalar_cols].reset_index(drop=True)]
    for exp in multi_dummies:
        parts.append(exp.reset_index(drop=True))

    X = pd.concat(parts, axis=1).fillna(0)

    if return_encoders:
        return X, encoders
    return X


def get_feature_names(df, fit_encoders=None):
    """Return feature names matching preprocess() output."""
    X, _ = preprocess(df, fit_encoders=fit_encoders, return_encoders=True)
    return list(X.columns)
