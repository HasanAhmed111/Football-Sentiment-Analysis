import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.ensemble import predict_ensemble
from utils.sentiment_explainer import explain_sentiment

st.set_page_config(
    page_title="Sports Sentiment Intelligence System",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("🏆 Sports Sentiment Intelligence Dashboard")
st.markdown("Advanced Multi-Model NLP System for Sports Analytics")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Ensemble", "Logistic Regression", "SVM", "LSTM", "BiLSTM", "DistilBERT"]
)

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Basic", "Advanced Contextual Analysis"]
)

# ---------------- INPUT ----------------
st.subheader("Enter Sports Text")

user_input = st.text_area("Type or paste sports commentary")

# ---------------- ANALYSIS BUTTON ----------------
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter text")
        st.stop()

    # PREDICTION
    result = predict_ensemble(user_input, model_choice)

    sentiment = result["sentiment"]
    confidence = result["confidence"]

    # ---------------- KPI CARDS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Sentiment", sentiment)
    col2.metric("Confidence", f"{confidence:.2f}")
    col3.metric("Model", model_choice)

    # ---------------- GAUGE CHART ----------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Sentiment Strength"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "blue"},
        }
    ))

    st.plotly_chart(fig)

    # ---------------- CONTEXT ANALYSIS ----------------
    if analysis_mode == "Advanced Contextual Analysis":

        explanation = explain_sentiment(user_input)

        st.subheader("🧠 Context Analysis")

        st.write(explanation["summary"])

        st.write("Key Signals:")
        for item in explanation["signals"]:
            st.write("-", item)

    # ---------------- MODEL OUTPUT ----------------
    st.subheader("Model Output Details")
    st.json(result)