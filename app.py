import streamlit as st
import pandas as pd
import joblib
import io

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Sistem Deteksi Stres Siswa SMA", page_icon="🧠", layout="wide")

@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model_stacking.pkl')
        return scaler, model
    except FileNotFoundError:
        st.error("⚠️ File scaler.pkl atau model_stacking.pkl tidak ditemukan.")
        return None, None

scaler, model = load_models()
EXPECTED_COLUMNS = ["Kualitas Tidur", "Sakit Kepala", "Kinerja Akademis", "Beban Belajar", "Ekstrakurikuler"]
TARGET_MAP = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

# ==========================================
# 2. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("Navigasi Aplikasi")
menu = st.sidebar.radio("Menu", ["🏠 Beranda / Deskripsi", "📊 Deteksi Stres Massal (Upload File)"])
st.sidebar.markdown("---")
st.sidebar.info("🎓 **Skripsi:** Pengembangan Sistem Untuk Mendeteksi Tingkat Stress Siswa SMA Pada Penerapan Kurikulum AI Koding Menggunakan Multi Model.")

# ==========================================
# 3. LOGIKA HALAMAN
# ==========================================
if menu == "🏠 Beranda / Deskripsi":
    st.title("Selamat Datang di Sistem Deteksi Stres Siswa 🧠")
    st.markdown("""
    Aplikasi web ini dikembangkan sebagai bagian dari penelitian skripsi untuk mendeteksi tingkat stres siswa SMA, khususnya dalam menghadapi penerapan kurikulum **AI & Koding**.
    
    ### ⚙️ Metode: *Stacking Ensemble Learning*
    1. **Base Learners:** Decision Tree (DT), Support Vector Machine (SVM), K-Nearest Neighbors (KNN).
    2. **Meta Learner:** Logistic Regression.
    """)

elif menu == "📊 Deteksi Stres Massal (Upload File)":
    st.title("Deteksi Stres Siswa Massal 📁")
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.write("### 👁️ Pratinjau Data yang Diunggah:")
            st.dataframe(df.head())
            
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ **Error:** Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Prediksi Tingkat Stres"):
                    with st.spinner('Sistem sedang memproses data...'):
                        X_unscaled = df[EXPECTED_COLUMNS]
                        X_scaled = scaler.transform(X_unscaled)
                        predictions = model.predict(X_scaled)
                        df_result = df.copy()
                        df_result["Hasil Deteksi Stres"] = [TARGET_MAP[pred] for pred in predictions]
                        
                    st.success("✅ Prediksi berhasil dilakukan!")
                    
                    def color_stress(val):
                        if val == 'Tinggi': return 'background-color: #ffcccc'
                        elif val == 'Sedang': return 'background-color: #ffffcc'
                        elif val == 'Rendah': return 'background-color: #ccffcc'
                        return ''
                    
                    st.dataframe(df_result.style.applymap(color_stress, subset=['Hasil Deteksi Stres']))
                    
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Unduh Hasil Prediksi (CSV)", data=csv, file_name='Hasil_Deteksi_Stres.csv', mime='text/csv')
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
