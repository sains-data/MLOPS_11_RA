import streamlit as st
import pandas as pd
import joblib
import os

# --- FUNGSI LOAD MODEL PINTAR ---
@st.cache_resource
def load_model():
    # 1. Cari lokasi file app.py ini berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Gabungkan path folder ini dengan folder models
    model_path = os.path.join(current_dir, "models", "model.pkl")
    
    # 3. Cek apakah file benar-benar ada?
    if not os.path.exists(model_path):
        st.error(f"❌ File tidak ditemukan di: {model_path}")
        st.warning("Pastikan file 'model.pkl' ada di dalam folder 'models' di GitHub Anda.")
        return None
    
    return joblib.load(model_path)

# Load Model
artifact = load_model()

# Jika gagal, berhenti
if artifact is None:
    st.stop()

model = artifact["model"]
encoder = artifact["encoder"]

# ... Lanjutkan dengan kode tampilan aplikasi di bawah ...

# ==========================================
# 4. KONFIGURASI TAMPILAN (UI)
# ==========================================
st.set_page_config(
    page_title="Mushroom AI Detector",
    page_icon="🍄",
    layout="wide"
)

# Mapping hasil prediksi (Sesuaikan jika label model Anda terbalik)
# 0 biasanya Edible, 1 biasanya Poisonous (tapi tergantung training Anda)
LABEL_MAP = {
    0: "Edible (Aman Dimakan) 🥗",
    1: "Poisonous (Beracun) ☠️"
}

st.title("🍄 Mushroom Classification AI")
st.markdown("""
Aplikasi ini menggunakan Machine Learning untuk memprediksi apakah jamur aman atau berbahaya.
Silakan pilih ciri-ciri jamur di menu sebelah kiri (Sidebar).
""")
st.markdown("---")

# ==========================================
# 5. INPUT FORM OTOMATIS (DI SIDEBAR)
# ==========================================
st.sidebar.header("📝 Masukkan Ciri-Ciri Jamur")
st.sidebar.write("Sesuaikan parameter berikut:")

input_data = {}

with st.sidebar.form("mushroom_form"):
    # Cek apakah encoder memiliki nama fitur (agar input dinamis)
    if hasattr(encoder, 'feature_names_in_'):
        for i, col_name in enumerate(encoder.feature_names_in_):
            
            # LOGIKA PINTAR:
            # Jika encoder punya daftar kategori (seperti OneHot/Ordinal), buat Dropdown.
            # Jika tidak, buat Text Input biasa.
            if hasattr(encoder, 'categories_'):
                # Ambil opsi yang valid untuk kolom ini
                options = list(encoder.categories_[i])
                input_data[col_name] = st.selectbox(f"Pilih {col_name}", options)
            else:
                input_data[col_name] = st.text_input(f"Isi {col_name}")
    else:
        st.error("Encoder tidak memiliki informasi fitur. Pastikan model disimpan dengan benar.")

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Analisis Jamur")

# ==========================================
# 6. LOGIKA PREDIKSI
# ==========================================
if submitted:
    # 1. Ubah input dictionary menjadi DataFrame
    df_input = pd.DataFrame([input_data])

    try:
        # Tampilkan data yang diinput (opsional, agar user yakin)
        with st.expander("Lihat Data yang Anda Input"):
            st.dataframe(df_input)

        # 2. Transformasi data menggunakan Encoder
        # (Mengubah teks seperti 'convex' menjadi angka yang dimengerti model)
        X_transformed = encoder.transform(df_input)

        # 3. Prediksi
        prediction = model.predict(X_transformed)[0]
        
        # 4. Hitung Probabilitas (Keyakinan Model)
        # Mengambil nilai probabilitas tertinggi
        probability = model.predict_proba(X_transformed).max()

        # ==========================================
        # 7. TAMPILKAN HASIL
        # ==========================================
        st.subheader("Hasil Analisis AI:")
        
        # Buat kolom agar tampilan rapi
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 0:
                st.success(f"### {LABEL_MAP[0]}")
                st.image("https://cdn-icons-png.flaticon.com/512/753/753938.png", width=150, caption="Safe to Eat")
            else:
                st.error(f"### {LABEL_MAP[1]}")
                st.image("https://cdn-icons-png.flaticon.com/512/753/753345.png", width=150, caption="Danger!")

        with col2:
            st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{probability:.2%}")
            
            if probability < 0.6:
                st.warning("⚠️ Model kurang yakin dengan prediksi ini. Harap berhati-hati!")
            else:
                st.info("ℹ️ Model cukup yakin dengan prediksi ini.")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses prediksi.")
        st.error(f"Error detail: {e}")
