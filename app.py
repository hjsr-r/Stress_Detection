import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="MindfulAI - Deteksi Stres Siswa",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS (Tema Cream & Soft Blue sesuai referensi UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Warna Latar Belakang Utama (Cream/Beige Soft) */
    .stApp {
        background-color: #F8F6F0; 
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #EAE6DF;
    }
    
    /* Hero Section - Landing Page */
    .hero-section {
        background-color: #FFFFFF;
        padding: 5rem 3rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        border: 1px solid #EAE6DF;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.03);
    }
    .hero-title {
        color: #2D3748; /* Dark Slate / Navy accent */
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #718096; /* Muted Slate */
        font-size: 1.25rem;
        max-width: 800px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    
    /* Feature Cards */
    .feature-card {
        background-color: #FFFFFF;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        border: 1px solid #EAE6DF;
        border-top: 5px solid #6BA4D8; /* Soft Blue Accent */
        height: 100%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        color: #2D3748;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Tombol Utama (Soft Blue) */
    .stButton>button {
        background-color: #6BA4D8; 
        color: white;
        border-radius: 50px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #558CBF;
        color: white;
        box-shadow: 0 4px 12px rgba(107, 164, 216, 0.3);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #718096;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #2D3748;
        border-bottom-color: #6BA4D8;
    }
    
    /* Container Box untuk Tab Content */
    .content-box {
        background-color: #FFFFFF;
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid #EAE6DF;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI SKALA INPUT (PRE-PROCESSING)
# ==========================================
SCALE_QUALITY = {"Sangat Buruk": 1, "Buruk": 2, "Cukup": 3, "Baik": 4, "Sangat Baik": 5}
SCALE_FREQUENCY = {"Tidak Pernah": 1, "Jarang": 2, "Kadang-kadang": 3, "Sering": 4, "Sangat Sering": 5}
SCALE_PERFORMANCE = {"Sangat Rendah": 1, "Rendah": 2, "Rata-rata": 3, "Tinggi": 4, "Sangat Tinggi": 5}
SCALE_LOAD = {"Sangat Ringan": 1, "Ringan": 2, "Sedang": 3, "Berat": 4, "Sangat Berat": 5}
SCALE_ACTIVE = {"Tidak Aktif": 1, "Kurang Aktif": 2, "Cukup Aktif": 3, "Aktif": 4, "Sangat Aktif": 5}

EXPECTED_FEATURES = {
    "Kualitas Tidur": SCALE_QUALITY,
    "Sakit Kepala": SCALE_FREQUENCY,
    "Kinerja Akademis": SCALE_PERFORMANCE,
    "Beban Belajar": SCALE_LOAD,
    "Ekstrakurikuler": SCALE_ACTIVE
}

TARGET_MAP = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

# ==========================================
# 3. FUNGSI MUAT MODEL & PREDIKSI
# ==========================================
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model_stacking.pkl')
        return scaler, model
    except FileNotFoundError:
        st.error("File scaler.pkl atau model_stacking.pkl tidak ditemukan di sistem.")
        return None, None

scaler, model = load_models()

def predict_stress(data_numeric_df):
    X_scaled = scaler.transform(data_numeric_df)
    predictions = model.predict(X_scaled)
    labels = [TARGET_MAP[pred] for pred in predictions]
    return labels

def convert_excel_to_numeric(df, col_mapping):
    df_numeric = pd.DataFrame()
    errors = []
    for model_col, excel_col in col_mapping.items():
        if excel_col in df.columns:
            scale = EXPECTED_FEATURES[model_col]
            try:
                df_numeric[model_col] = df[excel_col].map(scale)
                if df_numeric[model_col].isnull().any():
                    errors.append(f"Kolom '{excel_col}' mengandung teks yang tidak dikenali sistem.")
            except Exception as e:
                errors.append(f"Error mengonversi kolom '{excel_col}': {e}")
        else:
            errors.append(f"Kolom '{model_col}' tidak ditemukan di Excel.")
    return df_numeric, errors

# ==========================================
# 4. NAVIGASI SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color: #2D3748; font-weight: 800; text-align: center; margin-bottom: 2rem;'>MindfulAI</h2>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigasi Utama",
        ["Beranda", "Analisis & Deteksi", "Informasi Sistem"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #F8F9FA; padding: 1.5rem; border-radius: 12px; border: 1px solid #EAE6DF;'>
        <p style='color: #718096; font-size: 0.85rem; margin: 0; line-height: 1.5;'>
        <strong>Sistem Deteksi Stres Siswa</strong><br><br>
        Didukung oleh pendekatan Multi-Model Stacking Ensemble Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. LOGIKA HALAMAN 1: BERANDA
# ==========================================
if menu == "Beranda":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Helping Your Student Shine Through Life.</div>
        <div class="hero-subtitle">Platform analisis cerdas yang dirancang khusus untuk memetakan tingkat stres siswa SMA menghadapi tantangan akademis modern. Cepat, akurat, dan berbasis data.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">Penilaian Mandiri</div>
            <p style="color:#718096; font-size:0.95rem; line-height: 1.6;">Evaluasi kondisi akademis dan fisik siswa secara individual melalui instrumen kuesioner digital yang terintegrasi langsung dengan model pemrosesan.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">Pemrosesan Massal</div>
            <p style="color:#718096; font-size:0.95rem; line-height: 1.6;">Fasilitas khusus tenaga pendidik untuk mengunggah rekapitulasi data siswa dalam format spreadsheet guna pemetaan kondisi psikologis satu sekolah.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">Algoritma Ensemble</div>
            <p style="color:#718096; font-size:0.95rem; line-height: 1.6;">Menggabungkan ketajaman analisis dari algoritma Decision Tree, SVM, dan KNN yang disatukan melalui Logistic Regression untuk hasil presisi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("<br><br>", unsafe_allow_html=True)

# ==========================================
# 6. LOGIKA HALAMAN 2: DETEKSI
# ==========================================
elif menu == "Analisis & Deteksi":
    st.markdown("<h2 style='color: #2D3748; font-weight: 800;'>Modul Analisis Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #718096; margin-bottom: 2rem;'>Pilih metode input data yang sesuai dengan kebutuhan evaluasi Anda.</p>", unsafe_allow_html=True)
    
    tab_form, tab_excel = st.tabs(["Evaluasi Individu", "Impor Data Massal"])
    
    # --- TAB 1: FORMULIR INDIVIDU ---
    with tab_form:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #2D3748; margin-bottom: 1.5rem;'>Instrumen Asesmen Mandiri</h4>", unsafe_allow_html=True)
        
        with st.form("form_individu"):
            col1, col2 = st.columns(2)
            with col1:
                in_tidur = st.selectbox("Bagaimana kualitas tidur Anda?", list(SCALE_QUALITY.keys()), index=3)
                in_kepala = st.selectbox("Seberapa sering Anda mengalami sakit kepala?", list(SCALE_FREQUENCY.keys()), index=1)
                in_ekskul = st.selectbox("Seberapa aktif Anda dalam kegiatan ekstrakurikuler?", list(SCALE_ACTIVE.keys()), index=2)
                
            with col2:
                in_akademis = st.selectbox("Bagaimana kinerja akademis Anda saat ini?", list(SCALE_PERFORMANCE.keys()), index=2)
                in_beban = st.selectbox("Bagaimana beban belajar Anda dalam kurikulum saat ini?", list(SCALE_LOAD.keys()), index=2)
                in_nama = st.text_input("Identitas Siswa (Opsional)", "Anonim")

            st.write("<br>", unsafe_allow_html=True)
            btn_analyze = st.form_submit_button("Proses Analisis")

        if btn_analyze:
            if scaler is None or model is None:
                st.error("Kegagalan sistem: Model utama tidak ditemukan.")
            else:
                with st.spinner('Menjalankan inferensi model...'):
                    data_individu = {
                        "Kualitas Tidur": [SCALE_QUALITY[in_tidur]],
                        "Sakit Kepala": [SCALE_FREQUENCY[in_kepala]],
                        "Kinerja Akademis": [SCALE_PERFORMANCE[in_akademis]],
                        "Beban Belajar": [SCALE_LOAD[in_beban]],
                        "Ekstrakurikuler": [SCALE_ACTIVE[in_ekskul]]
                    }
                    df_individu_numeric = pd.DataFrame(data_individu)
                    hasil = predict_stress(df_individu_numeric)[0]
                    
                    st.write("---")
                    
                    # Warna hasil disesuaikan agar rapi
                    color_map = {"Rendah": "#F0FDF4", "Sedang": "#FEFCE8", "Tinggi": "#FEF2F2"}
                    text_color_map = {"Rendah": "#166534", "Sedang": "#854D0E", "Tinggi": "#991B1B"}
                    border_color_map = {"Rendah": "#BBF7D0", "Sedang": "#FEF08A", "Tinggi": "#FECACA"}
                    
                    st.markdown(f"""
                    <div style="background-color:{color_map[hasil]}; padding: 2.5rem; border-radius: 16px; text-align: center; border: 1px solid {border_color_map[hasil]};">
                        <p style="color:{text_color_map[hasil]}; margin:0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Status Terdeteksi</p>
                        <h2 style="color:{text_color_map[hasil]}; font-size: 3.5rem; font-weight: 800; margin:0.5rem 0;">Stres {hasil}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: UPLOAD MASSAL ---
    with tab_excel:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #2D3748; margin-bottom: 1.5rem;'>Fasilitas Impor Spreadsheet</h4>", unsafe_allow_html=True)
        
        # Template Data
        template_data = {
            "Nama Lengkap": ["Siswa A", "Siswa B"],
            "Kualitas Tidur": ["Sangat Baik", "Buruk"],
            "Sakit Kepala": ["Jarang", "Sering"],
            "Kinerja Akademis": ["Tinggi", "Rendah"],
            "Beban Belajar": ["Sedang", "Sangat Berat"],
            "Aktivitas Ekskul": ["Aktif", "Tidak Aktif"]
        }
        towrite = io.BytesIO()
        pd.DataFrame(template_data).to_excel(towrite, index=False, header=True)
        towrite.seek(0)
        
        st.download_button(
            label="Unduh Format Template",
            data=towrite,
            file_name="Template_Data_Asesmen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.write("<hr style='border: 1px dashed #EAE6DF; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih file dataset (.csv, .xlsx)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.markdown("<p style='font-weight:700; color:#2D3748; margin-top: 2rem;'>Konfigurasi Pemetaan Variabel</p>", unsafe_allow_html=True)
                col_mapping = {}
                actual_cols = list(df.columns)
                
                c1, c2 = st.columns(2)
                with c1:
                    col_mapping["Kualitas Tidur"] = st.selectbox("Variabel Kualitas Tidur:", actual_cols, index=1 if len(actual_cols)>1 else 0)
                    col_mapping["Sakit Kepala"] = st.selectbox("Variabel Sakit Kepala:", actual_cols, index=2 if len(actual_cols)>2 else 0)
                    col_mapping["Ekstrakurikuler"] = st.selectbox("Variabel Ekstrakurikuler:", actual_cols, index=5 if len(actual_cols)>5 else 0)
                with c2:
                    col_mapping["Kinerja Akademis"] = st.selectbox("Variabel Kinerja Akademis:", actual_cols, index=3 if len(actual_cols)>3 else 0)
                    col_mapping["Beban Belajar"] = st.selectbox("Variabel Beban Belajar:", actual_cols, index=4 if len(actual_cols)>4 else 0)
                
                st.write("<br>", unsafe_allow_html=True)
                if st.button("Jalankan Pemrosesan Massal"):
                    if scaler is None or model is None:
                        st.error("Model tidak tersedia.")
                    else:
                        with st.spinner('Menghitung parameter...'):
                            df_numeric_mass, errors = convert_excel_to_numeric(df, col_mapping)
                            
                            if errors:
                                for err in errors:
                                    st.error(err)
                                st.stop()
                            
                            hasil_massal = predict_stress(df_numeric_mass)
                            df_result = df.copy()
                            df_result["Klasifikasi Stres"] = hasil_massal
                        
                        st.write("<hr style='border: 1px solid #EAE6DF; margin: 2rem 0;'>", unsafe_allow_html=True)
                        st.markdown("<h4 style='color: #2D3748;'>Distribusi Data Populasi</h4>", unsafe_allow_html=True)
                        
                        counts = df_result["Klasifikasi Stres"].value_counts().reset_index()
                        counts.columns = ['Tingkat', 'Jumlah']
                        
                        # Palet warna grafik disesuaikan referensi Soft Blue / Navy
                        chart_colors = {"Tinggi": "#E598A4", "Sedang": "#F6E0B5", "Rendah": "#A8C7FA"}
                        
                        v1, v2 = st.columns(2)
                        with v1:
                            fig_bar = px.bar(counts, x='Tingkat', y='Jumlah', color='Tingkat',
                                             color_discrete_map=chart_colors, text_auto=True)
                            fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_bar, use_container_width=True)
                            
                        with v2:
                            fig_pie = px.pie(counts, values='Jumlah', names='Tingkat', color='Tingkat',
                                             color_discrete_map=chart_colors, hole=0.5)
                            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        st.markdown("<h4 style='color: #2D3748; margin-top: 2rem;'>Tabel Rekapitulasi</h4>", unsafe_allow_html=True)
                        st.dataframe(df_result, use_container_width=True)
                        
                        csv = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Laporan Akhir (CSV)",
                            data=csv,
                            file_name='Rekapitulasi_Stres_Siswa.csv',
                            mime='text/csv',
                        )
            except Exception as e:
                st.error(f"Inkompatibilitas struktur file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 7. LOGIKA HALAMAN 3: INFORMASI METODE
# ==========================================
elif menu == "Informasi Sistem":
    st.markdown("<h2 style='color: #2D3748; font-weight: 800;'>Spesifikasi Teknis Sistem</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #718096; margin-bottom: 2rem;'>Dokumentasi metodologi pemrosesan data yang diterapkan.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-box'>
        <h4 style='color: #2D3748; margin-bottom: 1rem;'>Topologi Stacking Ensemble Learning</h4>
        <p style='color: #718096; line-height: 1.6;'>
        Sistem pemrosesan klasifikasi pada aplikasi ini tidak bergantung pada satu algoritma tunggal. Pendekatan <i>Stacking Ensemble</i> digunakan untuk menutupi kelemahan satu model dengan kekuatan model lainnya guna menghasilkan akurasi yang lebih konsisten.
        </p>
        
        <div style='display: flex; gap: 2rem; margin-top: 2rem;'>
            <div style='flex: 1; padding: 2rem; background-color: #F8F9FA; border-radius: 16px; border: 1px solid #EAE6DF;'>
                <strong style='color: #2D3748; font-size: 1.1rem;'>Layer 1: Base Estimators</strong>
                <ul style='color: #718096; margin-top: 1rem; line-height: 1.8;'>
                    <li><b>Decision Tree:</b> Pemetaan kondisional bersyarat.</li>
                    <li><b>SVM:</b> Penentuan margin batas klasifikasi.</li>
                    <li><b>KNN:</b> Klasifikasi jarak antar titik data.</li>
                </ul>
            </div>
            <div style='flex: 1; padding: 2rem; background-color: #FFFFFF; border-radius: 16px; border: 1px solid #6BA4D8;'>
                <strong style='color: #2D3748; font-size: 1.1rem;'>Layer 2: Meta Learner</strong>
                <ul style='color: #718096; margin-top: 1rem; line-height: 1.8;'>
                    <li><b>Logistic Regression:</b> Agregator probabilistik yang mengambil keputusan akhir berdasarkan hasil output dari Layer 1.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
