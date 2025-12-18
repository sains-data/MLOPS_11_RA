import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mushroom Classifier")

st.title("🍄 Mushroom Classification")

pipeline = joblib.load("models/model.pkl")

feature_names = pipeline.named_steps["preprocess"].transformers_[0][2]

inputs = {}
for col in feature_names:
    inputs[col] = st.text_input(col)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    pred = pipeline.predict(df)[0]

    label = "Edible 🍄" if pred == 0 else "Poisonous ☠️"
    st.success(f"Prediction: {label}")
