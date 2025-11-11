"""
PrepPredict - Minimal App (College Prediction + Skill Recommender + Exam Info)

HOW TO RUN LOCALLY
1) pip install -U streamlit scikit-learn pandas numpy
2) Save this file as app.py
3) Make sure your dataset file exists as: CET-CUTOFF2025.csv (same folder)
4) streamlit run app.py

ABOUT THE MODEL
- Trains an ExtraTreesRegressor to predict the *closing cutoff rank* for a given
  (College, Location, Branch, Category). Eligibility is then derived by comparing
  the user's rank to the predicted cutoff.
- Dataset is expected with columns:
  CETCode, College, Location, Branch, 1G, 1K, 1R, 2AG, 2AK, 2AR, 2BG, 2BK, 2BR,
  3AG, 3AK, 3AR, 3BG, 3BK, 3BR, GM, GMK, GMR, SCG, SCK, SCR, STG, STK, STR

NOTE
- Only three sections: College Predictor, Skill Recommender, Exam Info.
- No other pages or features.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="PrepPredict - Minimal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "CET-CUTOFF2025.csv"  # <-- your file name

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Clean column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    category_cols = [
        "1G", "1K", "1R", "2AG", "2AK", "2AR", "2BG", "2BK", "2BR",
        "3AG", "3AK", "3AR", "3BG", "3BK", "3BR", "GM", "GMK", "GMR",
        "SCG", "SCK", "SCR", "STG", "STK", "STR",
    ]
    keep_cols = [c for c in ["CETCode", "College", "Location", "Branch"] if c in df.columns]
    exist_cats = [c for c in category_cols if c in df.columns]

    long_df = df.melt(
        id_vars=keep_cols,
        value_vars=exist_cats,
        var_name="Category",
        value_name="CutoffRank",
    )
    # Force numeric ranks; coerce errors to NaN and drop
    long_df["CutoffRank"] = pd.to_numeric(long_df["CutoffRank"], errors="coerce")
    long_df = long_df.dropna(subset=["CutoffRank"]).reset_index(drop=True)
    return long_df

@st.cache_resource(show_spinner=False)
def train_model(long_df: pd.DataFrame):
    # Features & target
    feature_cols = [c for c in ["College", "Location", "Branch", "Category"] if c in long_df.columns]
    X = long_df[feature_cols]
    y = long_df["CutoffRank"].astype(float)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols),
    ])
    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([
        ("prep", pre),
        ("model", model),
    ])
    pipe.fit(X, y)
    return pipe, feature_cols

# ----------------------------
# Domain logic
# ----------------------------
def predict_for_all_colleges(pipe, feature_cols, catalog_df: pd.DataFrame, user_category: str, user_rank: int,
                             location_filter: str | None = None, branch_filter: str | None = None,
                             top_n: int = 50) -> pd.DataFrame:
    # Build candidate grid (unique college/location/branch combos)
    base_cols = [c for c in ["College", "Location", "Branch"] if c in catalog_df.columns]
    cat_uniques = catalog_df[base_cols].drop_duplicates().reset_index(drop=True)
    cat_uniques["Category"] = user_category

    # Optional filters
    if location_filter and "Location" in cat_uniques.columns:
        cat_uniques = cat_uniques[cat_uniques["Location"].str.contains(location_filter, case=False, na=False)]
    if branch_filter and "Branch" in cat_uniques.columns:
        cat_uniques = cat_uniques[cat_uniques["Branch"].str.contains(branch_filter, case=False, na=False)]

    if cat_uniques.empty:
        return pd.DataFrame()

    # Ensure required feature columns exist
    Xq = cat_uniques[[c for c in feature_cols if c in cat_uniques.columns]].copy()

    # Predict cutoff ranks
    preds = pipe.predict(Xq)
    preds = np.clip(preds, a_min=1, a_max=None)  # ranks must be positive

    out = cat_uniques.copy()
    out["PredictedCutoff"] = preds.round(0).astype(int)
    out["YourRank"] = int(user_rank)

    # Status buckets
    def status(row):
        r = row["YourRank"]
        c = row["PredictedCutoff"]
        if r <= 0.8 * c:
            return "Eligible ✅"
        elif r <= 1.1 * c:
            return "Possible 🟡"
        elif r <= 1.3 * c:
            return "Reach 🔴"
        else:
            return "Unlikely ❌"

    out["Status"] = out.apply(status, axis=1)

    # Sort by status priority then by predicted cutoff
    pri = {"Eligible ✅": 1, "Possible 🟡": 2, "Reach 🔴": 3, "Unlikely ❌": 4}
    out = out.sort_values(by=["Status", "PredictedCutoff"], key=lambda s: s.map(pri) if s.name == "Status" else s)

    # Reorder columns
    cols = [c for c in ["College", "Location", "Branch", "Category", "Status", "PredictedCutoff", "YourRank"] if c in out.columns]
    return out[cols].head(top_n).reset_index(drop=True)


def skill_recommendations(interests: list[str], rank: int) -> dict:
    # Lightweight rules mixing interests & rank bands
    buckets = []
    if rank <= 3000:
        buckets.append("Top Rank")
    elif rank <= 15000:
        buckets.append("Strong Rank")
    else:
        buckets.append("Develop & Explore")

    # Map interests to skills/courses
    catalog = {
        "IT / Software": {
            "skills": ["Python", "Data Structures & Algorithms", "Web Dev", "SQL", "Git/GitHub", "Cloud Basics"],
            "courses": ["Full-Stack Projects", "DSA Prep", "Intro to AWS/Azure", "System Design Basics"],
        },
        "AI / Data": {
            "skills": ["NumPy/Pandas", "ML Fundamentals", "Model Deployment", "Statistics", "Data Viz"],
            "courses": ["ML Specialization", "Data Engineering 101", "MLOps Basics"],
        },
        "Electronics / Embedded": {
            "skills": ["C/C++", "Arduino/RPi", "Digital Electronics", "PCB Tools", "RTOS Basics"],
            "courses": ["Embedded Projects", "IoT Systems", "VLSI Intro"],
        },
        "Core (Mech/Civil)": {
            "skills": ["CAD (AutoCAD/SolidWorks)", "Manufacturing Basics", "Statics & Mechanics", "BIM Basics"],
            "courses": ["CAD Mastery", "CAM/CAE Intro", "Construction Mgmt"],
        },
        "Design / UI-UX": {
            "skills": ["Figma", "User Research", "Wireframing", "Prototyping", "Design Systems"],
            "courses": ["UI/UX Foundations", "Portfolio Projects", "Accessibility"],
        },
        "Medical / Life Sciences": {
            "skills": ["Bioinformatics Basics", "Research Methods", "Medical Writing", "Data Viz"],
            "courses": ["Genomics 101", "Public Health Analytics"],
        },
        "Govt / Competitive": {
            "skills": ["Quant Reasoning", "Logical Ability", "General Awareness", "Time Mgmt"],
            "courses": ["Aptitude Crash Course", "Mock Tests"],
        },
    }

    chosen = [c for c in interests if c in catalog]
    merged_skills, merged_courses = [], []
    for c in chosen:
        merged_skills += catalog[c]["skills"]
        merged_courses += catalog[c]["courses"]

    # De-duplicate while preserving order
    def unique(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return {
        "rank_band": ", ".join(buckets),
        "skills": unique(merged_skills)[:12] if merged_skills else [],
        "courses": unique(merged_courses)[:10] if merged_courses else [],
    }

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📚 PrepPredict")
page = st.sidebar.radio(
    "Navigate",
    ["🎯 College Predictor", "💼 Skill Recommender", "📘 Exam Info"],
)

# ----------------------------
# Pages
# ----------------------------
if page == "🎯 College Predictor":
    st.title("🎯 College Predictor (KCET)")
    st.caption("Powered by Extra Trees ML model trained on your cutoff dataset.")

    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found: {DATA_FILE}. Place CET-CUTOFF2025.csv in the app folder.")
        st.stop()

    raw_df = load_dataset(DATA_FILE)
    long_df = melt_to_long(raw_df)

    if long_df.empty:
        st.warning("Dataset appears empty after processing. Check your file and columns.")
        st.stop()

    pipe, feat_cols = train_model(long_df)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        categories = sorted(long_df["Category"].unique().tolist())
        sel_cat = st.selectbox("Select Category", categories, index=categories.index("GM") if "GM" in categories else 0)
    with c2:
        user_rank = st.number_input("Your KCET Rank", min_value=1, max_value=500000, value=12000, step=1)
    with c3:
        topn = st.slider("How many results?", 10, 200, 50, step=10)

    f1, f2 = st.columns(2)
    with f1:
        loc_filter = st.text_input("Filter by Location (optional)")
    with f2:
        br_filter = st.text_input("Filter by Branch (optional)")

    if st.button("🔍 Predict Colleges", type="primary"):
        with st.spinner("Predicting cutoffs and eligibility..."):
            results = predict_for_all_colleges(
                pipe, feat_cols, long_df, sel_cat, int(user_rank),
                location_filter=loc_filter.strip() or None,
                branch_filter=br_filter.strip() or None,
                top_n=topn,
            )

        if results.empty:
            st.warning("No matching results. Try clearing filters or adjust inputs.")
        else:
            st.info(f"Showing predictions for Category **{sel_cat}**, Rank **{user_rank}**.")
            # Color-coded display
            for i, row in results.iterrows():
                colA, colB = st.columns([3, 1])
                title = f"**{i+1}. {row['College']}** — {row['Branch']}"
                meta = f"📍 {row.get('Location', '')}  |  📊 Predicted Cutoff: {row['PredictedCutoff']}  |  🧍 Your Rank: {row['YourRank']}"
                if row["Status"].startswith("Eligible"):
                    with colA:
                        st.success(title)
                        st.write(meta)
                    with colB:
                        st.markdown(f"### {row['Status']}")
                elif row["Status"].startswith("Possible"):
                    with colA:
                        st.warning(title)
                        st.write(meta)
                    with colB:
                        st.markdown(f"### {row['Status']}")
                elif row["Status"].startswith("Reach"):
                    with colA:
                        st.error(title)
                        st.write(meta)
                    with colB:
                        st.markdown(f"### {row['Status']}")
                else:
                    with colA:
                        st.write(title)
                        st.caption(meta)
                    with colB:
                        st.markdown(f"### {row['Status']}")
                st.divider()

            # Summary counts
            counts = results["Status"].value_counts()
            st.success(
                f"✅ Eligible: {counts.get('Eligible ✅', 0)}  |  🟡 Possible: {counts.get('Possible 🟡', 0)}  |  🔴 Reach: {counts.get('Reach 🔴', 0)}  |  ❌ Unlikely: {counts.get('Unlikely ❌', 0)}"
            )

    st.caption("Note: Predictions approximate prior cutoffs using a model; actual cutoffs vary by year.")

elif page == "💼 Skill Recommender":
    st.title("💼 Skill Recommender")
    st.write("Get suggestions based on your interests and KCET rank.")

    interests = st.multiselect(
        "Select your interests",
        [
            "IT / Software",
            "AI / Data",
            "Electronics / Embedded",
            "Core (Mech/Civil)",
            "Design / UI-UX",
            "Medical / Life Sciences",
            "Govt / Competitive",
        ],
        default=["IT / Software", "AI / Data"],
    )
    rnk = st.number_input("Your KCET Rank", min_value=1, max_value=500000, value=15000, step=1)

    if st.button("✨ Get Recommendations", type="primary"):
        rec = skill_recommendations(interests, int(rnk))
        st.subheader(f"🎓 Track: {rec['rank_band']}")

        if rec["skills"]:
            st.markdown("### 🛠️ Skills to Focus")
            for s in rec["skills"]:
                st.write(f"• {s}")

        if rec["courses"]:
            st.markdown("### 📚 Suggested Courses/Paths")
            for c in rec["courses"]:
                st.write(f"• {c}")

        st.success("Build a small portfolio of 2–3 projects aligned to your chosen track.")

elif page == "📘 Exam Info":
    st.title("📘 Exam Eligibility & Info (Quick)")

    exams = {
        "KCET": {
            "eligibility": "PUC/12th with PCM for Engineering; domicile rules apply for Karnataka candidates.",
            "dates": "Expected: April 2025 (Tentative)",
            "site": "https://kea.kar.nic.in",
        },
        "NEET": {
            "eligibility": "12th with PCB/BT; minimum marks as per category for MBBS/BDS/AYUSH programs.",
            "dates": "Expected: May 2025 (Tentative)",
            "site": "https://neet.nta.nic.in",
        },
        "JEE Main": {
            "eligibility": "12th with PCM; used for NIT/IIIT/other CFTIs; Advanced for IITs.",
            "dates": "Jan & Apr 2025 (Tentative)",
            "site": "https://jeemain.nta.nic.in",
        },
    }

    sel = st.selectbox("Select Exam", list(exams.keys()))

    st.subheader(sel)
    st.write(exams[sel]["eligibility"])
    st.info(f"📅 {exams[sel]['dates']}")
    st.markdown(f"🔗 Official: [{exams[sel]['site']}]({exams[sel]['site']})")

    st.markdown("---")
    st.markdown("### Quick Eligibility Self-check (KCET)")
    kcet_rank = st.number_input("Enter your KCET rank (for a rough self-check)", min_value=1, max_value=500000, value=20000)
    cat = st.text_input("Enter your category code (e.g., GM, 2AG, 3BG, SCG)", value="GM")
    st.caption("This is a simple check. For detailed predictions, use the College Predictor tab.")
    if st.button("Check Now"):
        st.success(f"If your category is {cat} and your rank is {kcet_rank}, you can compare against predicted cutoffs in the College Predictor.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 PrepPredict")
