import streamlit as st
import pandas as pd
import joblib

# Load model
model_data = joblib.load("model/model.pkl")
model = model_data["model"]
columns = model_data["columns"]

st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom is **Edible** or **Poisonous**")

st.subheader("Input Mushroom Features")

# User input
user_input = {}
for col in columns:
    user_input[col] = st.selectbox(col, [0, 1])

# Convert to dataframe
input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    label_map = {0: "Edible", 1: "Poisonous"}
    st.success(f"Prediction: **{label_map[prediction]}**")
