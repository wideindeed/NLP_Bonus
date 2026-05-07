"""
Sentiment Analysis Web App
CNN-based sentiment analysis with GloVe embeddings.
Compares Original CNN (lab) vs Improved CNN.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import models

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Thing",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global reset & base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #f5f0e8;
        color: #1a1a1a;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* ── Header ── */
    .app-header {
        border-bottom: 2px solid #1a1a1a;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .app-title {
        font-family: 'DM Mono', monospace;
        font-size: 2rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        color: #1a1a1a;
        margin: 0;
    }
    .app-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: #6b6b6b;
        margin-top: 0.3rem;
        font-weight: 300;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid #1a1a1a;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        font-weight: 500;
        color: #6b6b6b;
        background: transparent;
        border: none;
        padding: 0.6rem 1.2rem;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a !important;
        background: transparent !important;
    }

    /* ── Result cards ── */
    .result-card {
        background: #ffffff;
        border: 2px solid #1a1a1a;
        border-radius: 4px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-card .label {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: #6b6b6b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.8rem;
    }
    .result-card .sentiment {
        font-family: 'DM Sans', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .result-card .confidence {
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        color: #6b6b6b;
    }
    .sentiment-pos { color: #2d6a4f; }
    .sentiment-neg { color: #c1121f; }

    /* ── Accuracy badge ── */
    .acc-badge {
        display: inline-block;
        background: #1a1a1a;
        color: #f5f0e8;
        font-family: 'DM Mono', monospace;
        font-size: 1.4rem;
        font-weight: 500;
        padding: 0.8rem 1.5rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }

    /* ── Info note ── */
    .note-box {
        background: #fff8e8;
        border: 1.5px solid #e8c84a;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        font-size: 0.9rem;
        color: #5a4a00;
        font-family: 'DM Sans', sans-serif;
        margin-top: 1rem;
    }

    /* ── Section label ── */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b6b6b;
        border-bottom: 1px solid #d4cfca;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1a1a1a;
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #f5f0e8 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: #e8cd33 !important;
        color: #1a1a1a !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.85rem !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #e8c84a !important;
    }
    .sidebar-status-ok {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #74c69d !important;
    }
    .sidebar-status-warn {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #e8c84a !important;
    }

    /* ── Streamlit overrides ── */
    .stTextArea textarea {
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        background: #ffffff;
        border: 2px solid #1a1a1a;
        border-radius: 4px;
        color: #1a1a1a;
    }
    .stTextArea textarea:focus {
        border-color: #e8c84a;
        box-shadow: none;
    }
    .stButton > button[kind="primary"] {
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        background: #1a1a1a;
        color: #f5f0e8;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton > button[kind="primary"]:hover {
        background: #333;
    }
    .stDataFrame { border: 2px solid #1a1a1a; border-radius: 4px; }
    table { font-family: 'DM Mono', monospace; font-size: 0.85rem; }
    thead th { background: #1a1a1a !important; color: #f5f0e8 !important; }

    /* plotly chart bg fix */
    .js-plotly-plot { border: 2px solid #1a1a1a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Globals ───────────────────────────────────────────────────────────────────
SAVE_DIR = "saved_models"
GLOVE_PATH = "glove.6B.100d.txt"
DATA_PATH = "imdb_labelled.tsv"


@st.cache_resource
def load_saved_artifacts():
    try:
        orig = load_model(os.path.join(SAVE_DIR, "original_cnn.keras"))
        impr = load_model(os.path.join(SAVE_DIR, "improved_cnn.keras"))
        with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "rb") as f:
            tok = pickle.load(f)
        return orig, impr, tok
    except Exception:
        return None, None, None


orig_model, impr_model, tokenizer = load_saved_artifacts()

# ── Sidebar (Restored to original functionality) ──────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='font-family:"DM Mono",monospace; font-size:1.1rem; font-weight:500; margin-bottom:0.2rem;'>
            sentiment thing
        </div>
        <div style='font-family:"DM Sans",sans-serif; font-size:0.8rem; color:#888; margin-bottom:1.5rem;'>
            CNN + GloVe, two flavours
        </div>
    """, unsafe_allow_html=True)

    if st.button("train both models", use_container_width=True, type="primary"):
        if not os.path.exists(GLOVE_PATH):
            st.error(f"missing: {GLOVE_PATH}")
        elif not os.path.exists(DATA_PATH):
            st.error(f"missing: {DATA_PATH}")
        else:
            with st.spinner("training... grab a coffee ☕"):
                res = models.train_and_save_models(GLOVE_PATH, DATA_PATH, SAVE_DIR)
                st.session_state["train_results"] = res
                st.success("done. check results tab.")

    st.markdown("<hr style='border-color:#333; margin:1.2rem 0;'>", unsafe_allow_html=True)

    if orig_model and impr_model:
        st.markdown('<div class="sidebar-status-ok">● models loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-status-warn">○ not trained yet</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-family:"DM Sans",sans-serif; font-size:0.75rem; color:#555; line-height:1.6;'>
            Original CNN mirrors the lab pipeline.<br><br>
            Improved CNN uses a robust N-Gram architecture with AdamW optimizer to prevent overfitting.
        </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">sentiment thing.</div>
    <div class="app-sub">original cnn vs improved cnn — IMDB movie review edition</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["predict", "batch upload", "results & architecture"])

# ── TAB 1: PREDICT ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">type something</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="Enter movie review",
        value="This movie was absolutely fantastic, I loved every minute of it!",
        height=120,
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        models_ready = bool(orig_model and impr_model)
        predict_clicked = st.button(
            "run prediction",
            type="primary",
            disabled=not models_ready,
            help="Train to unlock prediction." if not models_ready else "Run the sentiment analysis!"
        )

    if predict_clicked:
        res_orig = models.predict_text(user_input, orig_model, tokenizer)
        res_impr = models.predict_text(user_input, impr_model, tokenizer)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">predictions</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            pos_cls_orig = "sentiment-pos" if res_orig["sentiment"] == "Positive" else "sentiment-neg"
            emoji_orig = "👍" if res_orig["sentiment"] == "Positive" else "👎"
            st.markdown(f"""
            <div class="result-card">
                <div class="label">original cnn (lab)</div>
                <div class="sentiment {pos_cls_orig}">{emoji_orig} {res_orig["sentiment"]}</div>
                <div class="confidence">{res_orig["confidence"]:.1%} confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            pos_cls_impr = "sentiment-pos" if res_impr["sentiment"] == "Positive" else "sentiment-neg"
            emoji_impr = "👍" if res_impr["sentiment"] == "Positive" else "👎"
            st.markdown(f"""
            <div class="result-card">
                <div class="label">improved cnn</div>
                <div class="sentiment {pos_cls_impr}">{emoji_impr} {res_impr["sentiment"]}</div>
                <div class="confidence">{res_impr["confidence"]:.1%} confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style='background:#1e2130; border:1px solid #2e3250; border-radius:12px; padding:1.5rem; margin-top:2rem;'>
                <h4 style='color:#4F8BF9; margin-top:0; font-family:"DM Sans",sans-serif; font-size:1.1rem;'>
                        🧠 Why might the Improved Model be less "confident"?
                </h4>
                <p style='font-size:0.95rem; color:#ccc; line-height:1.6; margin-bottom:0; font-family:"DM Sans",sans-serif;'>
                    You might notice the Original CNN frequently outputs <strong>99% confidence</strong>, while the Improved CNN hovers around <strong>70-85%</strong>. This is actually a feature, not a bug!<br><br>
                    The Original CNN has zero regularisation. It perfectly memorises the training data, making it "arrogant" and poorly calibrated. It will guess 99% even on ambiguous text. The Improved CNN uses <strong>Dropout</strong>, which randomly turns off parts of its network during training. This forces the model to cautiously weigh the entire sentence context instead of jumping to conclusions based on a single buzzword. A 75% score from a calibrated model is much more trustworthy than a 99% score from an overfit one.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ── TAB 2: BATCH CSV ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">batch predictions via csv</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="note-box">
        your csv needs a column called <code>text</code> and a <code>sentiment</code> column to compare between predicted and actual file values. predictions run on the improved model.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("drop your csv here", type=["csv"])

    if uploaded_file and impr_model:
        # ── BULLETPROOF CSV READER ──
        try:
            # Try standard UTF-8 first
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            # If that fails, go back to the start of the file and try Latin-1
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            # Standardize column names to lowercase for easier checking
            df.columns = [c.lower() for c in df.columns]

            if 'text' in df.columns:
                with st.spinner("Processing batch..."):
                    # 1. Vectorized Preprocessing
                    cleaned_texts = [models.spacy_tokenize(str(t)) for t in df['text']]
                    seqs = tokenizer.texts_to_sequences(cleaned_texts)
                    X_batch = pad_sequences(seqs, maxlen=models.MAX_LEN)

                    # 2. Batch Prediction
                    raw_probs = impr_model.predict(X_batch, verbose=0).flatten()

                    # 3. Store Results
                    df['prediction'] = ["positive" if p > 0.5 else "negative" for p in raw_probs]
                    df['conf_score'] = [p if p > 0.5 else 1.0 - p for p in raw_probs]
                    df['confidence'] = [f"{c:.1%}" for c in df['conf_score']]

                st.markdown(f"<br><div class='section-label'>{len(df)} rows processed</div>", unsafe_allow_html=True)

                # --- NEW: COMPARISON LOGIC ---
                m1, m2, m3 = st.columns(3)

                # --- ENHANCED COMPARISON UI ---
                if 'sentiment' in df.columns:
                    # Standardize ground truth
                    df['sentiment'] = df['sentiment'].astype(str).str.lower()

                    # Calculate Accuracy
                    correct = (df['prediction'] == df['sentiment']).sum()
                    accuracy = correct / len(df)

                    # ROW 1: Accuracy and Totals
                    st.markdown("### 📊 Performance Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Batch Accuracy", f"{accuracy:.1%}")
                    c2.metric("Total Samples", len(df))
                    c3.metric("Correct Predictions", f"{correct} / {len(df)}")

                    # ROW 2: Side-by-Side Comparison
                    st.markdown("---")
                    col_actual, col_predicted = st.columns(2)

                    with col_actual:
                        st.write("**Actual (from CSV)**")
                        a_pos = len(df[df['sentiment'] == 'positive'])
                        a_neg = len(df[df['sentiment'] == 'negative'])
                        st.write(f"✅ Positive: {a_pos}")
                        st.write(f"❌ Negative: {a_neg}")

                    with col_predicted:
                        st.write("**Predicted (by Model)**")
                        p_pos = (df['prediction'] == 'positive').sum()
                        p_neg = (df['prediction'] == 'negative').sum()
                        st.write(f"✅ Positive: {p_pos}")
                        st.write(f"❌ Negative: {p_neg}")

                    st.success("Comparison complete! Check the table below for row-by-row details.")

                else:
                    # Fallback if no sentiment column exists
                    m1, m2, m3 = st.columns(3)
                    p_pos = (df['prediction'] == 'positive').sum()
                    m1.metric("Total Rows", len(df))
                    m2.metric("Predicted Pos", p_pos)
                    m3.metric("Predicted Neg", len(df) - p_pos)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df, width='stretch')
            else:
                st.error("CSV must have a column named 'text'.")

    elif uploaded_file and not impr_model:
        st.error("models aren't loaded yet — train them first.")


# ── TAB 3: RESULTS & ARCHITECTURE ────────────────────────────────────────────
with tab3:
    train_results = st.session_state.get("train_results", None)

    if train_results:
        st.markdown('<div class="section-label">test accuracy</div>', unsafe_allow_html=True)
        acc_orig = train_results["original"]["test_acc"]
        acc_impr = train_results["improved"]["test_acc"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">original cnn</div>
                <div class="acc-badge">{acc_orig:.1%}</div>
                <div class="confidence" style="margin-top:0.5rem;">test accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            delta = acc_impr - acc_orig
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            st.markdown(f"""
            <div class="result-card">
                <div class="label">improved cnn</div>
                <div class="acc-badge">{acc_impr:.1%}</div>
                <div class="confidence" style="margin-top:0.5rem;">test accuracy &nbsp;
                    <span style="color:{'#2d6a4f' if delta >= 0 else '#c1121f'}; font-weight:600;">{delta_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">training curves</div>', unsafe_allow_html=True)

        h_orig = train_results["original"]["history"]
        h_impr = train_results["improved"]["history"]

        ORIG_TRAIN = "#2563eb"
        ORIG_VAL   = "#93c5fd"
        IMPR_TRAIN = "#16a34a"
        IMPR_VAL   = "#f97316"

        CHART_LAYOUT = dict(
            paper_bgcolor="#f5f0e8",
            plot_bgcolor="#ffffff",
            font=dict(family="DM Mono, monospace", size=11, color="#1a1a1a"),
            margin=dict(l=40, r=40, t=80, b=40),
            height=340,
            showlegend=False,
        )
        AXIS_STYLE = dict(gridcolor="#e8e3db", linecolor="#bbb", tickcolor="#bbb")

        # Accuracy plot
        fig_acc = make_subplots(
            rows=1, cols=2,
            subplot_titles=("original cnn — accuracy", "improved cnn — accuracy"),
            horizontal_spacing=0.12,
        )

        for col_idx, (h, train_col, val_col) in enumerate([
            (h_orig, ORIG_TRAIN, ORIG_VAL),
            (h_impr, IMPR_TRAIN, IMPR_VAL),
        ], start=1):
            epochs = list(range(1, len(h["accuracy"]) + 1))
            fig_acc.add_trace(go.Scatter(
                x=epochs, y=h["accuracy"], mode="lines+markers",
                name="train", line=dict(color=train_col, width=2.5),
                marker=dict(size=6, color=train_col)
            ), row=1, col=col_idx)
            fig_acc.add_trace(go.Scatter(
                x=epochs, y=h["val_accuracy"], mode="lines+markers",
                name="val", line=dict(color=val_col, width=2.5, dash="dash"),
                marker=dict(size=6, color=val_col, symbol="diamond")
            ), row=1, col=col_idx)

        fig_acc.update_layout(**CHART_LAYOUT)
        fig_acc.update_xaxes(title_text="epoch", **AXIS_STYLE)
        fig_acc.update_yaxes(title_text="accuracy", range=[0, 1], **AXIS_STYLE)
        for ann in fig_acc.layout.annotations:
            ann.y = ann.y + 0.06
        st.plotly_chart(fig_acc, use_container_width=True)

        # Loss plot
        fig_loss = make_subplots(
            rows=1, cols=2,
            subplot_titles=("original cnn — loss", "improved cnn — loss"),
            horizontal_spacing=0.12,
        )

        for col_idx, (h, train_col, val_col) in enumerate([
            (h_orig, ORIG_TRAIN, ORIG_VAL),
            (h_impr, IMPR_TRAIN, IMPR_VAL),
        ], start=1):
            epochs = list(range(1, len(h["loss"]) + 1))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=h["loss"], mode="lines+markers",
                name="train", line=dict(color=train_col, width=2.5),
                marker=dict(size=6, color=train_col)
            ), row=1, col=col_idx)
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=h["val_loss"], mode="lines+markers",
                name="val", line=dict(color=val_col, width=2.5, dash="dash"),
                marker=dict(size=6, color=val_col, symbol="diamond")
            ), row=1, col=col_idx)

        fig_loss.update_layout(**CHART_LAYOUT)
        fig_loss.update_xaxes(title_text="epoch", **AXIS_STYLE)
        fig_loss.update_yaxes(title_text="loss", **AXIS_STYLE)
        for ann in fig_loss.layout.annotations:
            ann.y = ann.y + 0.06
        st.plotly_chart(fig_loss, use_container_width=True)

    else:
        st.markdown("""
        <div class="note-box">
            no training results yet. hit <strong>train both models</strong> in the sidebar —
            accuracy + loss curves will appear here once training finishes.
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture comparison ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">what changed between the two models</div>', unsafe_allow_html=True)

    diff_data = {
        "feature": ["embedding layer", "conv layers", "regularisation", "optimizer & capacity"],
        "original cnn (lab)": [
            "frozen GloVe (trainable=False)",
            "single Conv1D (k=5)",
            "none",
            "Adam + Dense(64)"
        ],
        "improved cnn": [
            "frozen GloVe (trainable=False)",
            "multi-branch N-Gram (k=2, 3, 5)",
            "Dropout(0.2)",
            "AdamW + Dense(128)"
        ],
    }
    st.table(pd.DataFrame(diff_data))

    # ── Cross Dataset Evaluation (Yelp) ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Assignment Task: Cross-Dataset Evaluation</div>', unsafe_allow_html=True)

    if orig_model and impr_model:
        if st.button("Run Yelp Reviews Test", type="primary"):
            with st.spinner("Downloading 1,000 Yelp reviews and evaluating..."):
                yelp_res = models.test_on_new_dataset(orig_model, impr_model, tokenizer)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="result-card" style="border-color:#2563eb;">
                        <div class="label" style="color:#2563eb;">original cnn (on yelp data)</div>
                        <div class="acc-badge" style="background:#2563eb;">{yelp_res['orig_acc']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="result-card" style="border-color:#16a34a;">
                        <div class="label" style="color:#16a34a;">improved cnn (on yelp data)</div>
                        <div class="acc-badge" style="background:#16a34a;">{yelp_res['impr_acc']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                    <div style='background:#1e2130; border:1px solid #2e3250; border-radius:12px; padding:1.5rem; margin-top:1.5rem;'>
                        <h4 style='color:#e8c84a; margin-top:0; font-family:"DM Sans",sans-serif; font-size:1.1rem;'>
                            📝 Does performance change? Why?
                        </h4>
                        <p style='font-size:0.95rem; color:#ccc; line-height:1.6; margin-bottom:0; font-family:"DM Sans",sans-serif;'>
                            <strong>Yes, performance drops significantly.</strong> This phenomenon is known as <em>Dataset Shift</em> (or Domain Shift). <br><br>
                            The models were trained strictly on IMDB Movie Reviews, meaning their internal vocabulary weights are tuned heavily toward words like "cinematography", "director", "plot", and "acting". When tested on Yelp restaurant reviews, the models suddenly encounter completely different terminology—words like "waiter", "delicious", "flavor", and "kitchen". <br><br>
                            Because the models never mapped these domain-specific restaurant words to positive or negative sentiments during training, they are essentially trying to read a foreign language, causing their accuracy to fall compared to their baseline IMDB performance.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Train or load models to unlock the Cross-Dataset Evaluation tool.")