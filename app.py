import streamlit as st
import pandas as pd
import joblib

# ======================
# Load model & encoder
# ======================
artifact = joblib.load("models/model.pkl")
model = artifact["model"]
encoder = artifact["encoder"]

LABEL_MAP = {
    0: "Edible 🍄",
    1: "Poisonous ☠️"
}

st.set_page_config(
    page_title="Mushroom Classification",
    layout="centered"
)

st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom is **edible** or **poisonous**")

# ======================
# Input Form
# ======================
with st.form("mushroom_form"):
    inputs = {}

    for col in encoder.feature_names_in_:
        inputs[col] = st.text_input(col)

    submitted = st.form_submit_button("Predict")

# ======================
# Prediction
# ======================
if submitted:
    df = pd.DataFrame([inputs])

    try:
        X = encoder.transform(df)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X).max()

        st.success(f"Prediction: **{LABEL_MAP[pred]}**")
        st.info(f"Confidence: **{prob:.2%}**")

    except Exception as e:
        st.error("Input tidak valid. Pastikan semua field diisi.")
        st.exception(e)
