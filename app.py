import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 1. Konfigurasi Halaman (Harus Paling Atas)
# ==========================================
st.set_page_config(
    page_title="Mushroom AI Detector",
    page_icon="🍄",
    layout="wide"
)

# ==========================================
# 2. Load Model dengan Jalur "Anti-Nyasar"
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # Trik Ajaib: Cari file berdasarkan lokasi app.py berada
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'model.pkl')
        
        artifact = joblib.load(model_path)
        return artifact["model"], artifact["encoder"]
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan di: {model_path}")
        return None, None
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat load model: {e}")
        return None, None

model, encoder = load_assets()

# Jika model gagal load, hentikan aplikasi
if model is None or encoder is None:
    st.stop()

# Mapping Label
LABEL_MAP = {
    0: "Edible (Bisa Dimakan) 🥗",
    1: "Poisonous (Beracun) ☠️"
}

# ==========================================
# 3. Tampilan Utama (UI)
# ==========================================
st.title("🍄 Mushroom Classification AI")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://img.freepik.com/free-vector/colorful-mushrooms-collection_1284-13134.jpg", caption="Safety First!")

with col2:
    st.info("Aplikasi ini menggunakan Machine Learning untuk memprediksi apakah jamur aman dimakan atau beracun berdasarkan ciri-cirinya.")

# ==========================================
# 4. Input Form "Ajaib" (Di Sidebar)
# ==========================================
st.sidebar.header("📝 Masukkan Ciri-Ciri Jamur")
st.sidebar.write("Pilih opsi di bawah ini:")

inputs = {}

# Form Input
with st.sidebar.form("mushroom_form"):
    # LOOP AJAIB: 
    # Mendeteksi fitur secara otomatis dan membuat Dropdown (Selectbox)
    # Ini mencegah user salah ketik (Typo)
    
    if hasattr(encoder, 'feature_names_in_') and hasattr(encoder, 'categories_'):
        for i, col_name in enumerate(encoder.feature_names_in_):
            # Ambil opsi valid dari encoder langsung
            options = list(encoder.categories_[i])
            inputs[col_name] = st.selectbox(f"Pilih {col_name}", options)
    else:
        # Fallback jika encoder tidak punya info kategori (pakai text biasa)
        st.warning("Mode Input Manual (Encoder tidak menyimpan kategori)")
        for col_name in getattr(encoder, 'feature_names_in_', []):
            inputs[col_name] = st.text_input(col_name)

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# ==========================================
# 5. Proses Prediksi
# ==========================================
if submitted:
    # Buat DataFrame dari input
    df_input = pd.DataFrame([inputs])

    try:
        # Transformasi data
        X_transformed = encoder.transform(df_input)
        
        # Prediksi
        prediction = model.predict(X_transformed)[0]
        probability = model.predict_proba(X_transformed).max()

        # Tampilkan Hasil
        st.markdown("### Hasil Analisis:")
        
        if prediction == 0: # Edible
            st.success(f"## {LABEL_MAP[0]}")
            st.balloons()
        else: # Poisonous
            st.error(f"## {LABEL_MAP[1]}")
            
        st.metric(label="Tingkat Kepercayaan (Confidence)", value=f"{probability:.2%}")

        # Tampilkan data yang diinput user (untuk debug/konfirmasi)
        with st.expander("Lihat Data Input"):
            st.dataframe(df_input)

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses data.")
        st.code(e)
