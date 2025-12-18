import streamlit as st
import pandas as pd
import joblib

# ======================
# Load Model Artifact
# ======================
artifact = joblib.load("models/model.pkl")
model = artifact["model"]
columns = artifact["columns"]

st.set_page_config(page_title="Mushroom Classifier", layout="centered")

st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom is **Edible** or **Poisonous**")

# ======================
# Input Form
# ======================
input_data = {}

for col in columns:
    input_data[col] = st.text_input(col)

# ======================
# Predict
# ======================
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])

        # Ensure column order
        input_df = input_df[columns]

        prediction = model.predict(input_df)[0]
        label = "Edible" if prediction == 0 else "Poisonous"

        st.success(f"Prediction: **{label}**")

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)
