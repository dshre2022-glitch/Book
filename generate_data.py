"""
Book DNA Survey – Synthetic Dataset Generator
Run once: python generate_data.py
Produces: book_dna_survey_2000.csv
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

N = 2000

# ── Persona weights ──────────────────────────────────────────────────────────
PERSONAS = ["Urban Gen Z Escapist", "Aspirational Tier2 Learner",
            "Premium Gifting Buyer", "Reluctant Non-Reader",
            "Traditional Homemaker"]
PERSONA_WEIGHTS = [0.32, 0.24, 0.18, 0.15, 0.11]

# ── Option lists (matching survey) ───────────────────────────────────────────
AGE_OPTS       = ["Under 18", "18-24", "25-30", "31-40", "41-50", "51+"]
GENDER_OPTS    = ["Female", "Male", "Non-binary", "Prefer not to say"]
CITY_OPTS      = ["Metro", "Tier 2", "Tier 3/Small town", "Rural"]
OCC_OPTS       = ["Student", "Salaried (private)", "Salaried (govt)",
                  "Self-employed/business", "Freelancer/gig", "Homemaker", "Unemployed"]
INCOME_OPTS    = ["Below 15k", "15k-30k", "30k-60k", "60k-1L", "Above 1L", "Prefer not to say"]
FREQ_OPTS      = ["Daily", "3-5 times/week", "Once a week",
                  "Few times/month", "Rarely", "Never"]
GENRE_OPTS     = ["Fiction", "Fantasy/Sci-fi", "Self-help", "Business/Finance",
                  "Biography/Memoir", "Romance", "Mystery/Thriller",
                  "Spirituality/Philosophy", "Comics/Graphic novels"]
PERSONALITY_OPTS = ["Escapist", "Learner", "Social", "Nostalgic", "Explorer", "Reluctant"]
TIME_OPTS      = ["Early morning", "Commute/travel", "Afternoon break",
                  "Late night", "Weekends only", "No fixed time"]
FORMAT_OPTS    = ["Physical books only", "E-books", "Audiobooks",
                  "Mix physical+digital", "Whatever is convenient"]
LIFESTYLE_OPTS = ["Journaling/writing", "Yoga/meditation", "Gym/sports",
                  "Cooking/baking", "Travel", "Art/craft", "Gaming",
                  "Watching OTT", "Music", "Shopping online"]
CLOTHING_OPTS  = ["Traditional/ethnic", "Western/casual", "Indo-western fusion",
                  "Formal/professional", "Streetwear/trendy", "No strong preference"]
SAREE_OPTS     = ["Silk (Kanjivaram/Banarasi)", "Cotton/linen",
                  "Chiffon/georgette", "Synthetic/affordable", "I don't wear sarees"]
COLOUR_OPTS    = ["Earthy/muted", "Pastels", "Bold/vibrant",
                  "Dark/moody", "Neutral/monochrome"]
SNACK_OPTS     = ["Chai/masala tea", "Coffee", "Nimbu pani/cold drinks",
                  "Dry snacks", "Sweet snacks", "Nothing"]
PRODUCT_OPTS   = ["Reading journal", "Book subscription box", "Book DNA profile kit",
                  "Stationery set", "Reading life planner", "Themed candle",
                  "Bookish tote bag", "Reading corner decor"]
SPEND_BAND_OPTS = ["Under 200", "200-400", "401-700", "701-1000", "Above 1000"]
SUB_OPTS       = ["Yes under 500/mo", "Yes 500-800/mo", "Yes up to 1000/mo",
                  "Only as gift", "No"]
BOOK_SPEND_OPTS = ["0 (library/borrowing)", "1-200", "201-500", "501-1000", "Above 1000"]
MOTIVATION_OPTS = ["Personalisation", "Discount/offer", "Gift-worthy packaging",
                   "Social media/influencer", "Subscription saving", "Friend recommendation"]
DISCOUNT_OPTS  = ["Flat % off", "Bundle deal", "Loyalty points/cashback",
                  "Referral discount", "Free gift with purchase"]
DISCOVERY_OPTS = ["Instagram/Reels", "YouTube BookTube", "Friends/family",
                  "Amazon/Flipkart", "Bookstore visit", "Goodreads/book apps",
                  "Rarely discover"]
SELF_ID_OPTS   = ["Ambitious", "Curious", "Grounded", "Creative", "Disciplined", "Drifting"]
ASPIRATION_OPTS = ["Well-read/intellectual", "Calm/mindful", "Financially independent",
                   "More creative", "More disciplined", "More socially connected"]
PAST_PURCH_OPTS = ["Journals/planners/stationery", "Scented candles/home decor",
                   "Physical books", "Tote bags/accessories",
                   "Online courses/subscriptions", "Subscription boxes"]
OCCASION_OPTS  = ["Diwali", "Birthday", "Graduation", "Raksha Bandhan",
                  "New Year", "Valentine's Day"]
BUYING_OPTS    = ["Mostly for myself", "Mostly as gifts", "Equal self+gifts",
                  "Only festival/occasion", "Rarely buy these"]
LOYALTY_OPTS   = ["Stick for years", "Loyal till better option", "Shop around",
                  "Loyal if rewarded", "Follow trends not brands"]
INTENT_OPTS    = ["Very likely", "Likely", "Neutral", "Unlikely", "Not interested"]

# ── Persona-specific probability tables ───────────────────────────────────────
def sample(opts, probs):
    return np.random.choice(opts, p=probs)

def sample_multi(opts, avg_picks, max_picks=None):
    n = max(1, int(np.random.normal(avg_picks, 1)))
    if max_picks:
        n = min(n, max_picks)
    n = min(n, len(opts))
    return "|".join(np.random.choice(opts, size=n, replace=False).tolist())

def gen_row(persona):
    p = PERSONAS.index(persona)
    r = {"persona_id": persona}

    # Q1 Age
    age_probs = [
        [0.05, 0.45, 0.28, 0.14, 0.05, 0.03],  # P1 Urban Gen Z
        [0.03, 0.28, 0.35, 0.24, 0.08, 0.02],  # P2 Aspirational
        [0.01, 0.08, 0.25, 0.38, 0.22, 0.06],  # P3 Premium Gifting
        [0.02, 0.18, 0.30, 0.32, 0.13, 0.05],  # P4 Reluctant
        [0.01, 0.05, 0.18, 0.35, 0.28, 0.13],  # P5 Traditional
    ]
    r["age_group"] = sample(AGE_OPTS, age_probs[p])

    # Q2 Gender
    gender_probs = [
        [0.62, 0.36, 0.02, 0.00],
        [0.48, 0.50, 0.01, 0.01],
        [0.55, 0.43, 0.01, 0.01],
        [0.45, 0.53, 0.01, 0.01],
        [0.78, 0.21, 0.00, 0.01],
    ]
    r["gender"] = sample(GENDER_OPTS, gender_probs[p])

    # Q3 City
    city_probs = [
        [0.58, 0.30, 0.10, 0.02],
        [0.25, 0.48, 0.22, 0.05],
        [0.55, 0.32, 0.10, 0.03],
        [0.35, 0.38, 0.22, 0.05],
        [0.15, 0.40, 0.35, 0.10],
    ]
    r["city_tier"] = sample(CITY_OPTS, city_probs[p])

    # Q4 Occupation
    occ_probs = [
        [0.48, 0.28, 0.05, 0.08, 0.08, 0.02, 0.01],
        [0.22, 0.40, 0.12, 0.14, 0.08, 0.03, 0.01],
        [0.05, 0.42, 0.10, 0.30, 0.08, 0.04, 0.01],
        [0.20, 0.35, 0.10, 0.18, 0.10, 0.05, 0.02],
        [0.05, 0.12, 0.08, 0.15, 0.05, 0.52, 0.03],
    ]
    r["occupation"] = sample(OCC_OPTS, occ_probs[p])

    # Q5 Income
    inc_probs = [
        [0.22, 0.35, 0.28, 0.10, 0.03, 0.02],
        [0.12, 0.30, 0.35, 0.16, 0.05, 0.02],
        [0.04, 0.10, 0.28, 0.35, 0.20, 0.03],
        [0.20, 0.32, 0.28, 0.13, 0.04, 0.03],
        [0.25, 0.35, 0.25, 0.10, 0.03, 0.02],
    ]
    r["income_band"] = sample(INCOME_OPTS, inc_probs[p])

    # Q6 Reading frequency
    freq_probs = [
        [0.10, 0.20, 0.22, 0.25, 0.18, 0.05],
        [0.08, 0.15, 0.20, 0.28, 0.22, 0.07],
        [0.06, 0.12, 0.20, 0.30, 0.25, 0.07],
        [0.02, 0.05, 0.10, 0.22, 0.38, 0.23],
        [0.05, 0.10, 0.18, 0.28, 0.28, 0.11],
    ]
    r["reading_frequency"] = sample(FREQ_OPTS, freq_probs[p])

    # Q7 Genres (multi-select)
    genre_weights = [
        [0.7, 0.75, 0.4, 0.3, 0.4, 0.55, 0.5, 0.3, 0.4],
        [0.4, 0.3, 0.80, 0.75, 0.55, 0.3, 0.4, 0.45, 0.2],
        [0.5, 0.3, 0.6, 0.65, 0.60, 0.4, 0.45, 0.35, 0.15],
        [0.3, 0.35, 0.4, 0.35, 0.3, 0.35, 0.4, 0.25, 0.3],
        [0.45, 0.2, 0.35, 0.2, 0.3, 0.65, 0.35, 0.60, 0.2],
    ]
    selected = [g for g, w in zip(GENRE_OPTS, genre_weights[p]) if random.random() < w]
    r["genres_enjoyed"] = "|".join(selected) if selected else "Fiction"

    # Q8 Reading personality
    pers_probs = [
        [0.45, 0.15, 0.20, 0.08, 0.10, 0.02],
        [0.08, 0.45, 0.15, 0.05, 0.20, 0.07],
        [0.12, 0.30, 0.18, 0.15, 0.15, 0.10],
        [0.08, 0.10, 0.12, 0.10, 0.08, 0.52],
        [0.15, 0.18, 0.10, 0.30, 0.08, 0.19],
    ]
    r["reading_personality"] = sample(PERSONALITY_OPTS, pers_probs[p])

    # Q9 Reading time
    time_probs = [
        [0.10, 0.15, 0.10, 0.40, 0.15, 0.10],
        [0.15, 0.25, 0.15, 0.20, 0.18, 0.07],
        [0.12, 0.20, 0.20, 0.18, 0.20, 0.10],
        [0.08, 0.18, 0.12, 0.18, 0.28, 0.16],
        [0.18, 0.10, 0.20, 0.22, 0.20, 0.10],
    ]
    r["reading_time"] = sample(TIME_OPTS, time_probs[p])

    # Q10 Format
    fmt_probs = [
        [0.30, 0.28, 0.12, 0.22, 0.08],
        [0.35, 0.22, 0.08, 0.25, 0.10],
        [0.42, 0.20, 0.08, 0.22, 0.08],
        [0.25, 0.28, 0.15, 0.20, 0.12],
        [0.50, 0.15, 0.08, 0.18, 0.09],
    ]
    r["reading_format"] = sample(FORMAT_OPTS, fmt_probs[p])

    # Q11 Stress level (1-5)
    stress_means = [3.8, 3.2, 2.8, 3.5, 2.9]
    r["stress_level"] = int(np.clip(round(np.random.normal(stress_means[p], 0.9)), 1, 5))

    # Q12 Lifestyle (multi)
    life_weights = [
        [0.65, 0.50, 0.40, 0.35, 0.45, 0.55, 0.40, 0.70, 0.60, 0.65],
        [0.55, 0.45, 0.50, 0.40, 0.50, 0.35, 0.25, 0.55, 0.50, 0.55],
        [0.35, 0.55, 0.55, 0.60, 0.70, 0.40, 0.20, 0.50, 0.45, 0.60],
        [0.25, 0.30, 0.35, 0.30, 0.35, 0.25, 0.50, 0.65, 0.50, 0.55],
        [0.45, 0.50, 0.20, 0.65, 0.30, 0.55, 0.10, 0.55, 0.40, 0.30],
    ]
    selected_life = [l for l, w in zip(LIFESTYLE_OPTS, life_weights[p]) if random.random() < w]
    r["lifestyle_activities"] = "|".join(selected_life) if selected_life else "Music"

    # Q13 Clothing
    cloth_probs = [
        [0.12, 0.38, 0.28, 0.10, 0.10, 0.02],
        [0.28, 0.30, 0.25, 0.10, 0.05, 0.02],
        [0.25, 0.28, 0.28, 0.14, 0.03, 0.02],
        [0.18, 0.35, 0.22, 0.15, 0.07, 0.03],
        [0.55, 0.15, 0.18, 0.05, 0.02, 0.05],
    ]
    r["clothing_style"] = sample(CLOTHING_OPTS, cloth_probs[p])

    # Q14 Saree type
    saree_probs = [
        [0.05, 0.08, 0.12, 0.08, 0.67],
        [0.10, 0.18, 0.10, 0.12, 0.50],
        [0.28, 0.15, 0.15, 0.08, 0.34],
        [0.08, 0.15, 0.10, 0.10, 0.57],
        [0.22, 0.30, 0.15, 0.18, 0.15],
    ]
    r["saree_preference"] = sample(SAREE_OPTS, saree_probs[p])

    # Q15 Colour palette
    colour_probs = [
        [0.15, 0.28, 0.12, 0.35, 0.10],
        [0.30, 0.20, 0.18, 0.15, 0.17],
        [0.25, 0.22, 0.15, 0.20, 0.18],
        [0.20, 0.22, 0.18, 0.22, 0.18],
        [0.28, 0.18, 0.28, 0.10, 0.16],
    ]
    r["colour_palette"] = sample(COLOUR_OPTS, colour_probs[p])

    # Q16 Snack
    snack_probs = [
        [0.40, 0.25, 0.12, 0.08, 0.10, 0.05],
        [0.35, 0.22, 0.15, 0.12, 0.10, 0.06],
        [0.25, 0.30, 0.15, 0.12, 0.12, 0.06],
        [0.32, 0.20, 0.15, 0.14, 0.12, 0.07],
        [0.45, 0.10, 0.12, 0.18, 0.12, 0.03],
    ]
    r["snack_preference"] = sample(SNACK_OPTS, snack_probs[p])

    # Q17 Products interested (multi)
    prod_weights = [
        [0.75, 0.65, 0.70, 0.70, 0.65, 0.72, 0.65, 0.60],
        [0.60, 0.70, 0.55, 0.60, 0.72, 0.40, 0.45, 0.35],
        [0.55, 0.75, 0.60, 0.50, 0.55, 0.65, 0.55, 0.65],
        [0.25, 0.30, 0.25, 0.28, 0.30, 0.20, 0.25, 0.18],
        [0.50, 0.45, 0.45, 0.42, 0.48, 0.50, 0.40, 0.38],
    ]
    selected_prod = [pr for pr, w in zip(PRODUCT_OPTS, prod_weights[p]) if random.random() < w]
    r["products_interested"] = "|".join(selected_prod) if selected_prod else "None"

    # Q18 Spend on stationery
    spend_probs = [
        [0.10, 0.28, 0.35, 0.18, 0.09],
        [0.15, 0.35, 0.30, 0.14, 0.06],
        [0.04, 0.12, 0.28, 0.30, 0.26],
        [0.25, 0.38, 0.25, 0.09, 0.03],
        [0.18, 0.35, 0.28, 0.14, 0.05],
    ]
    r["stationery_spend_band"] = sample(SPEND_BAND_OPTS, spend_probs[p])

    # Q19 Subscription intent
    sub_probs = [
        [0.20, 0.30, 0.18, 0.12, 0.20],
        [0.15, 0.28, 0.22, 0.15, 0.20],
        [0.10, 0.22, 0.30, 0.22, 0.16],
        [0.05, 0.10, 0.12, 0.15, 0.58],
        [0.08, 0.15, 0.18, 0.25, 0.34],
    ]
    r["subscription_intent"] = sample(SUB_OPTS, sub_probs[p])

    # Q20 Monthly book spend (target for regression)
    book_spend_probs = [
        [0.08, 0.25, 0.38, 0.22, 0.07],
        [0.10, 0.28, 0.35, 0.20, 0.07],
        [0.06, 0.15, 0.30, 0.30, 0.19],
        [0.28, 0.38, 0.22, 0.09, 0.03],
        [0.20, 0.35, 0.28, 0.12, 0.05],
    ]
    r["monthly_book_spend"] = sample(BOOK_SPEND_OPTS, book_spend_probs[p])

    # numeric midpoints for regression target
    spend_map = {"0 (library/borrowing)": 0, "1-200": 100, "201-500": 350,
                 "501-1000": 750, "Above 1000": 1200}
    r["monthly_book_spend_numeric"] = spend_map[r["monthly_book_spend"]] + np.random.normal(0, 30)
    r["monthly_book_spend_numeric"] = max(0, round(r["monthly_book_spend_numeric"], 0))

    # Q21 Motivation
    mot_probs = [
        [0.38, 0.18, 0.15, 0.18, 0.08, 0.03],
        [0.25, 0.22, 0.12, 0.15, 0.18, 0.08],
        [0.20, 0.10, 0.40, 0.10, 0.12, 0.08],
        [0.10, 0.42, 0.12, 0.20, 0.10, 0.06],
        [0.18, 0.25, 0.28, 0.10, 0.10, 0.09],
    ]
    r["purchase_motivation"] = sample(MOTIVATION_OPTS, mot_probs[p])

    # Q22 Discount type
    disc_probs = [
        [0.28, 0.25, 0.18, 0.15, 0.14],
        [0.25, 0.28, 0.22, 0.15, 0.10],
        [0.15, 0.25, 0.28, 0.12, 0.20],
        [0.35, 0.22, 0.18, 0.15, 0.10],
        [0.22, 0.28, 0.22, 0.12, 0.16],
    ]
    r["discount_preference"] = sample(DISCOUNT_OPTS, disc_probs[p])

    # Q23 Discovery channel
    disc_ch_probs = [
        [0.48, 0.18, 0.15, 0.10, 0.05, 0.02, 0.02],
        [0.22, 0.28, 0.22, 0.15, 0.08, 0.03, 0.02],
        [0.18, 0.15, 0.28, 0.22, 0.10, 0.05, 0.02],
        [0.28, 0.15, 0.28, 0.18, 0.05, 0.02, 0.04],
        [0.18, 0.10, 0.38, 0.20, 0.10, 0.02, 0.02],
    ]
    r["discovery_channel"] = sample(DISCOVERY_OPTS, disc_ch_probs[p])

    # Q24 Online comfort (1-5)
    comfort_means = [4.2, 3.8, 4.0, 3.5, 3.2]
    r["online_comfort"] = int(np.clip(round(np.random.normal(comfort_means[p], 0.8)), 1, 5))

    # Q26 Self identity
    self_probs = [
        [0.18, 0.28, 0.12, 0.22, 0.10, 0.10],
        [0.28, 0.22, 0.10, 0.12, 0.20, 0.08],
        [0.30, 0.18, 0.18, 0.14, 0.15, 0.05],
        [0.10, 0.12, 0.18, 0.12, 0.08, 0.40],
        [0.12, 0.15, 0.30, 0.20, 0.12, 0.11],
    ]
    r["self_identity"] = sample(SELF_ID_OPTS, self_probs[p])

    # Q27 Aspiration
    asp_probs = [
        [0.38, 0.22, 0.15, 0.15, 0.08, 0.02],
        [0.25, 0.15, 0.30, 0.10, 0.18, 0.02],
        [0.22, 0.20, 0.28, 0.12, 0.12, 0.06],
        [0.20, 0.18, 0.22, 0.12, 0.15, 0.13],
        [0.18, 0.28, 0.15, 0.15, 0.12, 0.12],
    ]
    r["aspiration"] = sample(ASPIRATION_OPTS, asp_probs[p])

    # aspiration gap score (0=aligned, 1=moderate gap, 2=large gap)
    self_map = {"Ambitious": 0, "Curious": 1, "Grounded": 2,
                "Creative": 3, "Disciplined": 4, "Drifting": 5}
    asp_map  = {"Well-read/intellectual": 1, "Calm/mindful": 2,
                "Financially independent": 0, "More creative": 3,
                "More disciplined": 4, "More socially connected": 5}
    gap = abs(self_map.get(r["self_identity"], 0) - asp_map.get(r["aspiration"], 0))
    r["aspiration_gap_score"] = min(gap, 5)

    # Q28 Social proof need (1-5)
    soc_means = [3.5, 3.0, 2.8, 3.8, 3.2]
    r["social_proof_need"] = int(np.clip(round(np.random.normal(soc_means[p], 1.0)), 1, 5))

    # social influence score = avg(motivation_is_social, social_proof_need)
    social_mot = 1 if r["purchase_motivation"] in ["Social media/influencer", "Friend recommendation"] else 0
    r["social_influence_score"] = round((social_mot * 5 + r["social_proof_need"]) / 2, 1)

    # Q29 Past purchases (multi)
    past_weights = [
        [0.60, 0.55, 0.70, 0.50, 0.45, 0.35],
        [0.55, 0.35, 0.65, 0.40, 0.55, 0.30],
        [0.45, 0.65, 0.60, 0.55, 0.50, 0.45],
        [0.20, 0.18, 0.30, 0.18, 0.22, 0.10],
        [0.40, 0.45, 0.50, 0.28, 0.20, 0.15],
    ]
    selected_past = [pp for pp, w in zip(PAST_PURCH_OPTS, past_weights[p]) if random.random() < w]
    r["past_purchases"] = "|".join(selected_past) if selected_past else "None"
    r["past_purchase_count"] = len(selected_past)

    # Q30 Buying intent
    buying_probs = [
        [0.55, 0.12, 0.20, 0.10, 0.03],
        [0.45, 0.18, 0.22, 0.10, 0.05],
        [0.18, 0.42, 0.25, 0.12, 0.03],
        [0.30, 0.15, 0.18, 0.15, 0.22],
        [0.25, 0.28, 0.22, 0.18, 0.07],
    ]
    r["buying_pattern"] = sample(BUYING_OPTS, buying_probs[p])

    # Q31 Occasion (multi)
    occ_weights = [
        [0.45, 0.65, 0.50, 0.38, 0.42, 0.45],
        [0.55, 0.60, 0.45, 0.48, 0.38, 0.30],
        [0.70, 0.65, 0.55, 0.55, 0.45, 0.35],
        [0.30, 0.38, 0.25, 0.28, 0.22, 0.18],
        [0.65, 0.55, 0.38, 0.60, 0.30, 0.20],
    ]
    selected_occ = [o for o, w in zip(OCCASION_OPTS, occ_weights[p]) if random.random() < w]
    r["gifting_occasions"] = "|".join(selected_occ) if selected_occ else "Birthday"

    # Q32 Eco importance (1-5)
    eco_means = [3.5, 3.2, 4.2, 2.8, 3.0]
    r["eco_importance"] = int(np.clip(round(np.random.normal(eco_means[p], 0.9)), 1, 5))

    # Q33 Loyalty
    loyalty_probs = [
        [0.22, 0.35, 0.20, 0.15, 0.08],
        [0.28, 0.32, 0.18, 0.16, 0.06],
        [0.38, 0.28, 0.12, 0.16, 0.06],
        [0.10, 0.28, 0.28, 0.18, 0.16],
        [0.30, 0.28, 0.18, 0.18, 0.06],
    ]
    r["loyalty_orientation"] = sample(LOYALTY_OPTS, loyalty_probs[p])

    # Q25 Purchase intent (classification target) – LAST
    intent_probs = [
        [0.32, 0.38, 0.18, 0.08, 0.04],
        [0.25, 0.35, 0.22, 0.12, 0.06],
        [0.28, 0.35, 0.20, 0.12, 0.05],
        [0.04, 0.12, 0.25, 0.35, 0.24],
        [0.10, 0.25, 0.28, 0.25, 0.12],
    ]
    r["purchase_intent"] = sample(INTENT_OPTS, intent_probs[p])
    r["purchase_intent_label"] = 1 if r["purchase_intent"] in ["Very likely", "Likely"] else 0

    return r


# ── Generate base rows ────────────────────────────────────────────────────────
personas_assigned = np.random.choice(PERSONAS, size=N, p=PERSONA_WEIGHTS)
rows = [gen_row(p) for p in personas_assigned]
df = pd.DataFrame(rows)

# ── Inject outliers & noise (~4%) ─────────────────────────────────────────────
outlier_idx = np.random.choice(df.index, size=80, replace=False)

# Type 1: contradictory – never reads but many genres (30 rows)
for i in outlier_idx[:30]:
    df.at[i, "reading_frequency"] = "Never"
    df.at[i, "genres_enjoyed"] = "|".join(np.random.choice(GENRE_OPTS, size=5, replace=False))

# Type 2: income-spend mismatch (20 rows)
for i in outlier_idx[30:50]:
    df.at[i, "income_band"] = "Below 15k"
    df.at[i, "monthly_book_spend"] = "Above 1000"
    df.at[i, "monthly_book_spend_numeric"] = 1200 + np.random.normal(0, 100)

# Type 3: straight-liners – all scales = 3 (15 rows)
for i in outlier_idx[50:65]:
    df.at[i, "stress_level"] = 3
    df.at[i, "online_comfort"] = 3
    df.at[i, "social_proof_need"] = 3
    df.at[i, "eco_importance"] = 3

# Type 4: cross-persona anomalies (15 rows)
for i in outlier_idx[65:80]:
    df.at[i, "city_tier"] = "Rural"
    df.at[i, "income_band"] = "Below 15k"
    df.at[i, "eco_importance"] = 5
    df.at[i, "products_interested"] = "Reading journal|Book DNA profile kit|Themed candle|Stationery set"

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "book_dna_survey_2000.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows × {len(df.columns)} columns → {out_path}")
print(f"Columns: {list(df.columns)}")
print(f"Label distribution:\n{df['purchase_intent_label'].value_counts()}")
