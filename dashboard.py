"""
Amazon Review Classification Dashboard
Before vs After pipeline visualization + live prediction.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from features import load_vectorizer, transform_live  # noqa: E402
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"

# Original pipeline results (236-test holdout, from Report §5.1)
BEFORE_SENTIMENT = pd.DataFrame(
    [
        {"Model": "Naive Bayes", "Accuracy": 0.8347, "Macro_F1": 0.3033, "Weighted_F1": 0.7596},
        {"Model": "Logistic Regression", "Accuracy": 0.8559, "Macro_F1": 0.5116, "Weighted_F1": 0.8152},
        {"Model": "Random Forest", "Accuracy": 0.8390, "Macro_F1": 0.5510, "Weighted_F1": 0.8303},
        {"Model": "LLM (Zero-Shot)", "Accuracy": 0.8475, "Macro_F1": 0.5983, "Weighted_F1": 0.8394},
    ]
)
BEFORE_CATEGORY = pd.DataFrame(
    [
        {"Model": "Naive Bayes", "Accuracy": 0.7203, "Macro_F1": 0.2094, "Weighted_F1": 0.6032},
        {"Model": "Logistic Regression", "Accuracy": 0.7627, "Macro_F1": 0.3272, "Weighted_F1": 0.6896},
        {"Model": "Random Forest", "Accuracy": 0.7712, "Macro_F1": 0.4681, "Weighted_F1": 0.7065},
    ]
)


def clean_review_text(text: str) -> str:
    import re

    text = str(text).lower()
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_reviews() -> pd.DataFrame:
    cols = [
        "review_id", "reviews.rating", "reviews.text", "reviews.title",
        "sentiment", "llm_sentiment", "llm_category",
    ]
    df = pd.read_csv(ROOT / "preprocessed_reviews.csv", usecols=cols, low_memory=False)
    df["word_count"] = df["reviews.text"].astype(str).str.split().str.len()
    df["sentiment_llm_disagree"] = df["sentiment"] != df["llm_sentiment"]
    return df


@st.cache_data
def load_human_gold() -> pd.DataFrame:
    path = ROOT / "data" / "reviews_human_gold.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(
        path,
        usecols=[
            "review_id", "reviews.rating", "reviews.title", "reviews.text",
            "sentiment", "llm_sentiment", "llm_category",
            "human_sentiment", "human_category",
        ],
        low_memory=False,
    )


@st.cache_data
def load_metrics(name: str) -> pd.DataFrame:
    path = METRICS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def load_models():
    vectorizer = load_vectorizer(ROOT)
    with open(ROOT / "sentiment_logistic_regression.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    with open(ROOT / "category_random_forest.pkl", "rb") as f:
        category_model = pickle.load(f)
    return vectorizer, sentiment_model, category_model


def kpi_row(cols, items):
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def metrics_bar_chart(df: pd.DataFrame, title: str):
    if df.empty:
        st.warning("Metrics file not found.")
        return
    melted = df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Macro_F1", "Weighted_F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric", barmode="group",
        title=title, text_auto=".2f",
        color_discrete_sequence=["#2b7bba", "#e68422", "#4caf50"],
    )
    fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-25, height=420)
    st.plotly_chart(fig, use_container_width=True)


def before_after_chart(before: pd.DataFrame, after: pd.DataFrame, model: str, title: str):
    b = before.loc[before["Model"] == model, "Macro_F1"]
    a = after.loc[after["Model"] == model, "Macro_F1"]
    if b.empty or a.empty:
        return
    fig = go.Figure(
        data=[
            go.Bar(name="Before", x=[model], y=[b.iloc[0]], marker_color="#94a3b8"),
            go.Bar(name="After", x=[model], y=[a.iloc[0]], marker_color="#2563eb"),
        ]
    )
    fig.update_layout(title=title, yaxis_range=[0, 1], barmode="group", height=320)
    st.plotly_chart(fig, use_container_width=True)


def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"Chart not found: {path.name}")


def tab_before(df: pd.DataFrame, gold: pd.DataFrame):
    st.subheader("Before Pipeline — Raw Data & Original Evaluation")
    st.caption("Exploratory view of 1,177 reviews and original 236-test results (pre human gold).")

    c1, c2, c3, c4, c5 = st.columns(5)
    disagree_pct = df["sentiment_llm_disagree"].mean() * 100
    kpi_row(
        [c1, c2, c3, c4, c5],
        [
            ("Total reviews", f"{len(df):,}"),
            ("Avg star rating", f"{df['reviews.rating'].mean():.1f}"),
            ("Positive (stars)", f"{(df['sentiment']=='Positive').mean()*100:.0f}%"),
            ("LLM ≠ star sentiment", f"{disagree_pct:.0f}%"),
            ("Human gold set", "100" if not gold.empty else "—"),
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        rating_counts = df["reviews.rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        fig = px.bar(rating_counts, x="Rating", y="Count", title="Star Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig = px.pie(sent_counts, names="Sentiment", values="Count", title="Sentiment (from Stars)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cat_counts = df["llm_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(cat_counts, x="Category", y="Count", title="LLM Theme Labels (Full Dataset)")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(df, x="word_count", nbins=40, title="Review Length (words)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Original model results (236-test holdout)")
    m1, m2 = st.columns(2)
    with m1:
        metrics_bar_chart(BEFORE_SENTIMENT, "Sentiment vs Star Ratings (Before)")
    with m2:
        metrics_bar_chart(BEFORE_CATEGORY, "Theme vs LLM Labels (Before)")

    if not gold.empty:
        st.markdown("#### Label agreement (human gold, n=100)")
        human_star = (gold["human_sentiment"] == gold["sentiment"]).mean() * 100
        human_llm_sent = (gold["human_sentiment"] == gold["llm_sentiment"]).mean() * 100
        human_llm_cat = (gold["human_category"] == gold["llm_category"]).mean() * 100
        a1, a2, a3 = st.columns(3)
        a1.metric("Human vs star sentiment", f"{human_star:.0f}%")
        a2.metric("Human vs LLM sentiment", f"{human_llm_sent:.0f}%")
        a3.metric("Human vs LLM theme", f"{human_llm_cat:.0f}%")

        st.markdown("#### Star sentiment vs LLM sentiment (full dataset)")
        cross = (
            df.groupby(["sentiment", "llm_sentiment"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.density_heatmap(
            cross, x="sentiment", y="llm_sentiment", z="Count",
            title="Star-Mapped Sentiment vs LLM Sentiment",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Human gold preview (label disagreements)")
        gold = gold.copy()
        gold["theme_disagree"] = gold["human_category"] != gold["llm_category"]
        gold["sent_disagree"] = gold["human_sentiment"] != gold["llm_sentiment"]
        filter_opt = st.selectbox(
            "Filter gold reviews",
            ["All", "Human ≠ LLM theme", "Human ≠ star sentiment", "Human ≠ LLM sentiment"],
        )
        view = gold.copy()
        if filter_opt == "Human ≠ LLM theme":
            view = view[view["theme_disagree"]]
        elif filter_opt == "Human ≠ star sentiment":
            view = view[view["human_sentiment"] != view["sentiment"]]
        elif filter_opt == "Human ≠ LLM sentiment":
            view = view[view["human_sentiment"] != view["llm_sentiment"]]
        view = view.copy()
        view["snippet"] = view["reviews.text"].astype(str).str.slice(0, 120) + "..."
        st.dataframe(
            view[
                ["review_id", "reviews.rating", "sentiment", "llm_sentiment", "human_sentiment",
                 "llm_category", "human_category", "snippet"]
            ],
            use_container_width=True,
            height=280,
        )


def tab_after(df: pd.DataFrame):
    st.subheader("After Pipeline — Human Gold & Retrained Models")
    st.caption("Primary evaluation on 100 human-labeled reviews; validation tracks for comparison.")

    track_a_sent = load_metrics("track_a_sentiment_gold.csv")
    track_a_cat = load_metrics("track_a_category_gold.csv")
    track_b = load_metrics("track_b_sentiment_stars.csv")
    track_c_cat = load_metrics("track_c_category_llm.csv")
    cv_gold = load_metrics("cv_human_gold.csv")

    c1, c2, c3, c4 = st.columns(4)
    lr_f1 = track_a_sent.loc[track_a_sent["Model"] == "Logistic Regression", "Macro_F1"]
    llm_theme_f1 = track_a_cat.loc[track_a_cat["Model"] == "LLM (Zero-Shot)", "Macro_F1"]
    kpi_row(
        [c1, c2, c3, c4],
        [
            ("Best sentiment Macro F1", f"{lr_f1.iloc[0]*100:.1f}%" if len(lr_f1) else "—"),
            ("Best theme Macro F1 (human)", f"{llm_theme_f1.iloc[0]*100:.1f}%" if len(llm_theme_f1) else "—"),
            ("Sentiment vs stars (val)", f"{track_b.loc[track_b['Model']=='Logistic Regression','Macro_F1'].iloc[0]*100:.1f}%" if not track_b.empty else "—"),
            ("Theme vs LLM (val)", f"{track_c_cat.loc[track_c_cat['Model']=='Random Forest','Macro_F1'].iloc[0]*100:.1f}%" if not track_c_cat.empty else "—"),
        ],
    )

    st.markdown("#### Track A — Primary (Human Gold, n=100)")
    a1, a2 = st.columns(2)
    with a1:
        metrics_bar_chart(track_a_sent, "Sentiment — Human Labels as Ground Truth")
    with a2:
        metrics_bar_chart(track_a_cat, "Theme — Human Labels as Ground Truth")

    if not cv_gold.empty:
        st.markdown("#### 5-Fold CV on human gold (human-label training target)")
        st.dataframe(cv_gold, use_container_width=True, hide_index=True)

    st.markdown("#### Before vs After — Macro F1 lift")
    b1, b2, b3 = st.columns(3)
    with b1:
        before_after_chart(
            BEFORE_SENTIMENT, track_b if not track_b.empty else track_a_sent,
            "Logistic Regression", "Sentiment Macro F1 (LR): Before vs After (val/stars)",
        )
    with b2:
        before_after_chart(
            BEFORE_CATEGORY, track_c_cat,
            "Random Forest", "Theme Macro F1 (RF): Before vs After (val/LLM)",
        )
    with b3:
        # Human gold only — no before equivalent
        if not track_a_cat.empty:
            fig = px.bar(
                track_a_cat.sort_values("Macro_F1", ascending=True),
                x="Macro_F1", y="Model", orientation="h",
                title="Theme Macro F1 on Human Gold (new metric)",
                text_auto=".2f",
            )
            fig.update_layout(xaxis_range=[0, 1], height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Confusion matrices & static charts")
    g1, g2 = st.columns(2)
    with g1:
        show_image(FIGURES / "sentiment_confusion_gold.png", "Sentiment confusion (human gold)")
        show_image(FIGURES / "sentiment_comparison_gold.png", "Sentiment model comparison (human gold)")
    with g2:
        show_image(FIGURES / "category_confusion_gold.png", "Theme confusion (human gold)")
        show_image(FIGURES / "category_comparison_gold.png", "Theme model comparison (human gold)")

    with st.expander("Original evaluation charts (236-test holdout, before retraining)"):
        o1, o2 = st.columns(2)
        with o1:
            show_image(ROOT / "sentiment_confusion.png", "Sentiment confusion (before)")
            show_image(ROOT / "sentiment_comparison.png", "Sentiment comparison (before)")
        with o2:
            show_image(ROOT / "category_confusion.png", "Theme confusion (before)")
            show_image(ROOT / "category_comparison.png", "Theme comparison (before)")

    st.markdown("#### Baseline vs best model (Macro F1, human gold)")
    if not track_a_sent.empty and not track_a_cat.empty:
        lift = pd.DataFrame(
            [
                {"Task": "Sentiment", "Baseline": track_a_sent.loc[track_a_sent["Model"] == "Majority Class", "Macro_F1"].iloc[0],
                 "Best": track_a_sent.loc[track_a_sent["Model"] == "Logistic Regression", "Macro_F1"].iloc[0],
                 "Best model": "Logistic Regression"},
                {"Task": "Theme", "Baseline": track_a_cat.loc[track_a_cat["Model"] == "Majority Class", "Macro_F1"].iloc[0],
                 "Best": track_a_cat.loc[track_a_cat["Model"] == "LLM (Zero-Shot)", "Macro_F1"].iloc[0],
                 "Best model": "LLM (Zero-Shot)"},
            ]
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Majority baseline", x=lift["Task"], y=lift["Baseline"], marker_color="#94a3b8"))
        fig.add_trace(go.Bar(name="Best model", x=lift["Task"], y=lift["Best"], marker_color="#2563eb",
                             text=lift["Best model"], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_range=[0, 1], height=340, title="Macro F1 lift over majority class")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Pipeline funnel")
    funnel_data = pd.DataFrame(
        {
            "Stage": [
                "Raw reviews", "Train split", "Validation", "Human gold (held out)",
                "Word+char TF-IDF + rating features", "Models trained",
            ],
            "Count": [1177, 861, 216, 100, 13000, 3],
        }
    )
    st.dataframe(funnel_data, hide_index=True, use_container_width=True)


def tab_predict():
    st.subheader("Live Prediction")
    st.caption("Logistic Regression (sentiment, text+rating) + Random Forest (theme, text).")

    try:
        vectorizer, sent_model, cat_model = load_models()
    except FileNotFoundError:
        st.error("Models not found. Run: python scripts/run_post_labeling_pipeline.py")
        return

    default = "Great product but shipping took two weeks and the box was damaged."
    title = st.text_input("Review title (optional)", value="Mixed feelings")
    text = st.text_area("Enter review text", value=default, height=120)
    rating = st.slider("Star rating", min_value=1, max_value=5, value=3)

    if st.button("Classify", type="primary"):
        text_x, sent_x = transform_live(text, title, float(rating), vectorizer)
        sent_pred = sent_model.predict(sent_x)[0]
        cat_pred = cat_model.predict(text_x)[0]
        sent_proba = sent_model.predict_proba(sent_x)[0]
        cat_proba = cat_model.predict_proba(text_x)[0]
        sent_conf = float(sent_proba.max())
        cat_conf = float(cat_proba.max())

        c1, c2 = st.columns(2)
        c1.metric("Sentiment", sent_pred, f"confidence {sent_conf:.0%}")
        c2.metric("Theme", cat_pred, f"confidence {cat_conf:.0%}")

        if sent_conf < 0.55 or cat_conf < 0.55:
            st.warning("Low confidence — consider human review or LLM fallback.")

        with st.expander("Sentiment probabilities"):
            st.dataframe(
                pd.DataFrame({"Label": sent_model.classes_, "Probability": sent_proba}),
                hide_index=True,
            )
        with st.expander("Theme probabilities"):
            st.dataframe(
                pd.DataFrame({"Label": cat_model.classes_, "Probability": cat_proba}),
                hide_index=True,
            )


def main():
    st.set_page_config(
        page_title="Amazon Review Classification Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Amazon Review Classification Dashboard")
    st.markdown(
        "Visualize **before** (raw data & original eval) and **after** "
        "(human gold validation & retrained models) the ML pipeline."
    )

    df = load_reviews()
    gold = load_human_gold()

    tab1, tab2, tab3 = st.tabs(["Before Pipeline", "After Pipeline", "Live Prediction"])
    with tab1:
        tab_before(df, gold)
    with tab2:
        tab_after(df)
    with tab3:
        tab_predict()


if __name__ == "__main__":
    main()
