import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# 1. SETUP HALAMAN
# ---------------------------------------------------------
st.set_page_config(page_title="Mushroom AI", layout="centered")

# ---------------------------------------------------------
# 2. FUNGSI LOAD MODEL (DENGAN GPS / ABSOLUTE PATH)
# ---------------------------------------------------------
@st.cache_resource
def load_model_smart():
    # Ini adalah Trik GPS-nya:
    # Cari tahu di mana file app.py ini berada di dalam server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Gabungkan alamat folder app.py dengan folder models
    # Hasilnya akan seperti: /mount/src/.../models/model.pkl
    model_path = os.path.join(current_dir, 'models', 'model.pkl')
    
    try:
        loaded_artifact = joblib.load(model_path)
        return loaded_artifact
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan di lokasi: {model_path}")
        st.info("Pastikan folder 'models' sejajar (satu folder) dengan app.py")
        return None

# Load data
artifact = load_model_smart()

# Cek jika gagal load, berhenti disini
if artifact is None:
    st.stop()

model = artifact["model"]
encoder = artifact["encoder"]

# ---------------------------------------------------------
# 3. TAMPILAN APLIKASI
# ---------------------------------------------------------
st.title("🍄 Mushroom Classification App")
st.write("Apakah jamur ini aman dimakan?")

with st.form("input_form"):
    inputs = {}
    
    # Deteksi fitur otomatis dari encoder
    if hasattr(encoder, 'feature_names_in_'):
        for col in encoder.feature_names_in_:
            # Coba ambil opsi kategori jika ada (untuk Dropdown)
            if hasattr(encoder, 'categories_'):
                # Cari index kolom
                idx = list(encoder.feature_names_in_).index(col)
                options = encoder.categories_[idx]
                inputs[col] = st.selectbox(f"Pilih {col}", options)
            else:
                # Jika tidak ada info kategori, pakai text input biasa
                inputs[col] = st.text_input(col)
    
    submitted = st.form_submit_button("Cek Jamur")

if submitted:
    try:
        df = pd.DataFrame([inputs])
        # Transformasi data
        X_transform = encoder.transform(df)
        # Prediksi
        prediction = model.predict(X_transform)[0]
        
        if prediction == 0:
            st.success("✅ EDIBLE (Bisa Dimakan)")
        else:
            st.error("☠️ POISONOUS (Beracun!)")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
