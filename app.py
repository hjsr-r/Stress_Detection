import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px # Tambahan untuk visualisasi yang lebih bagus

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="MindfulAI - Deteksi Stres Siswa SMAN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS (Untuk membuat tampilan mirip gambar/modern) ---
st.markdown("""
<style>
    /* Mengubah font global */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Mengubah warna background utama */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Mengubah tampilan Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Mengubah warna teks sidebar */
    [data-testid="stSidebar"] .stMarkdown {
        color: #333333;
    }
    /* Hero Section Styling (Halaman Beranda) */
    .hero-section {
        background-color: #ffffff;
        padding: 4rem 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title {
        color: #1e3a8a; /* Biru gelap */
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        color: #4b5563; /* Abu-abu */
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    /* Card styling untuk info */
    .info-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    /* Button styling (Primer) */
    .stButton>button {
        background-color: #2563eb; /* Biru */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        border: none;
        color: white;
    }
    /* Styling untuk visualisasi */
    .plot-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI SKALA INPUT (PRE-PROCESSING)
# ==========================================
# Sangat Penting: Model Anda membutuhkan angka. Kode ini mengubah teks menjadi angka.
# Sesuaikan skala ini dengan cara Anda melatih model.

# Skala 1-5 (Ordinat)
SCALE_QUALITY = {"Sangat Buruk": 1, "Buruk": 2, "Cukup": 3, "Baik": 4, "Sangat Baik": 5}
SCALE_FREQUENCY = {"Tidak Pernah": 1, "Jarang": 2, "Kadang-kadang": 3, "Sering": 4, "Sangat Sering": 5}
SCALE_PERFORMANCE = {"Sangat Rendah": 1, "Rendah": 2, "Rata-rata": 3, "Tinggi": 4, "Sangat Tinggi": 5}
SCALE_LOAD = {"Sangat Ringan": 1, "Ringan": 2, "Sedang": 3, "Berat": 4, "Sangat Berat": 5}
SCALE_ACTIVE = {"Tidak Aktif": 1, "Kurang Aktif": 2, "Cukup Aktif": 3, "Aktif": 4, "Sangat Aktif": 5}

# Mapping fitur yang diwajibkan oleh model (X)
EXPECTED_FEATURES = {
    "Kualitas Tidur": SCALE_QUALITY,
    "Sakit Kepala": SCALE_FREQUENCY,
    "Kinerja Akademis": SCALE_PERFORMANCE,
    "Beban Belajar": SCALE_LOAD,
    "Ekstrakurikuler": SCALE_ACTIVE
}

# Mapping hasil prediksi (Y)
TARGET_MAP = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

# ==========================================
# 3. FUNGSI MUAT MODEL & PREDIKSI
# ==========================================
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model_stacking.pkl')
        # Pengecekan versi scikit-learn opsional (aktifkan jika error pkl)
        # import sklearn
        # if sklearn.__version__ != '1.x.x': # sesuaikan versi
        #     st.warning(f"Versi sklearn berbeda. Model: [dari pkl], Server: {sklearn.__version__}")
        return scaler, model
    except FileNotFoundError:
        st.error("⚠️ File scaler.pkl atau model_stacking.pkl tidak ditemukan.")
        return None, None

scaler, model = load_models()

def predict_stress(data_numeric_df):
    """Menerima DataFrame berisi angka murni, melakukan scaling, lalu memprediksi."""
    X_scaled = scaler.transform(data_numeric_df)
    predictions = model.predict(X_scaled)
    labels = [TARGET_MAP[pred] for pred in predictions]
    return labels

# ==========================================
# 4. KONVERSI DATA EXCEL (TEXT -> NUMERIC)
# ==========================================
def convert_excel_to_numeric(df, col_mapping):
    """
    Mengubah DataFrame Excel berisi teks deskriptif menjadi angka 
    berdasarkan mapping skala yang ditentukan di poin #2.
    """
    df_numeric = pd.DataFrame()
    errors = []

    for model_col, excel_col in col_mapping.items():
        if excel_col in df.columns:
            # Dapatkan skala untuk fitur ini (misal: SCALE_QUALITY)
            scale = EXPECTED_FEATURES[model_col]
            
            # Buat kolom baru dengan nilai yang sudah dikonversi
            try:
                # .map() akan mengubah teks menjadi angka berdasarkan dict skala
                df_numeric[model_col] = df[excel_col].map(scale)
                
                # Cek jika ada data yang gagal dikonversi (NaN)
                if df_numeric[model_col].isnull().any():
                    errors.append(f"Kolom '{excel_col}' (untuk {model_col}) mengandung teks yang tidak dikenali sistem. Pastikan isinya sesuai template.")
            except Exception as e:
                errors.append(f"Error mengonversi kolom '{excel_col}': {e}")
        else:
            errors.append(f"Kolom wajib '{model_col}' tidak ditemukan di Excel.")
            
    return df_numeric, errors

# ==========================================
# 5. NAVIGASI SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3079/3079020.png", width=80) # Icon dummy
    st.title("MindfulAI")
    st.markdown("Sistem Deteksi Stres Siswa SMA")
    
    # Menu Navigasi (Meningkat menjadi 3 halaman)
    menu = st.radio(
        "Menu Utama",
        ["🏠 Beranda", "📊 Analisis & Deteksi", "ℹ️ Informasi Metode"]
    )
    
    st.markdown("---")
    st.info("🎓 **Skripsi:** Pengembangan Sistem Untuk Mendeteksi Tingkat Stress Siswa SMA Pada Penerapan Kurikulum AI Koding Menggunakan Multi Model.")

# ==========================================
# 6. LOGIKA HALAMAN 1: BERANDA (Landing Page)
# ==========================================
if menu == "🏠 Beranda":
    # Hero Section mirip contoh gambar
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">MindfulAI Students</div>
        <div class="hero-subtitle">The smartest AI-powered stress detection platform for high school students facing modern coding curriculum challenges.</div>
        <p style="color: #6b7280; font-size: 0.9rem;">Dikembangkan untuk mendeteksi tingkat stres siswa secara dini berdasarkan pola tidur, keluhan fisik, dan beban akademis.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#1e3a8a;">🔍 Deteksi Cepat</h4>
            <p style="color:#4b5563; font-size:0.9rem;">Isi formulir singkat mengenai kondisi Anda, dan biarkan AI kami menganalisis tingkat stres Anda dalam hitungan detik.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#1e3a8a;">📁 Upload Massal (BK)</h4>
            <p style="color:#4b5563; font-size:0.9rem;">Guru BK dapat mengunggah data siswa satu sekolah menggunakan format Excel untuk analisis massal dan visualisasi distribusi.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#1e3a8a;">🧠 Teknologi Stacking</h4>
            <p style="color:#4b5563; font-size:0.9rem;">Sistem menggunakan gabungan algoritma terbaik (Decision Tree, SVM, KNN) untuk akurasi prediksi yang lebih optimal.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tombol Call to Action
    st.write("---")
    st.write("### Siap mengetahui tingkat stres Anda?")
    if st.button("🚀 Mulai Analisis Sekarang"):
        st.warning("Silakan pilih menu '📊 Analisis & Deteksi' di sidebar sebelah kiri.")

# ==========================================
# 7. LOGIKA HALAMAN 2: DETEKSI (Gabungan Form & Excel)
# ==========================================
elif menu == "📊 Analisis & Deteksi":
    st.title("Analisis Tingkat Stres Siswa 📊")
    st.write("Silakan pilih metode input data di bawah ini:")
    
    # Menggunakan TABS untuk memisahkan Form Individu dan Upload Massal
    tab_form, tab_excel = st.tabs(["📝 Formulir Individu (Siswa)", "📁 Upload Massal Excel (Guru BK)"])
    
    # --- TAB 1: FORMULIR INDIVIDU (Mirip Gambar 2) ---
    with tab_form:
        st.subheader("Formulir Penilaian Mandiri Siswa")
        st.write("Isilah formulir di bawah ini sesuai dengan kondisi Anda yang sebenarnya dalam 1 bulan terakhir.")
        
        # Peringatan mengenai numeric inputs
        st.info("💡 **Informasi:** Model AI membutuhkan input angka. Di sini Anda memilih teks, tetapi sistem akan otomatis mengubahnya menjadi angka (misal: Sangat Baik = 5) sebelum dianalisis.")

        # Input menggunakan selectbox (Dropdown Teks)
        with st.form("form_individu"):
            col1, col2 = st.columns(2)
            with col1:
                # Perhatikan: Kunci dictionary (Sangat Baik, Baik) yang ditampilkan ke user
                in_tidur = st.selectbox("1. Bagaimana kualitas tidur Anda?", list(SCALE_QUALITY.keys()), index=3) # Default: Baik
                in_kepala = st.selectbox("2. Seberapa sering Anda mengalami sakit kepala?", list(SCALE_FREQUENCY.keys()), index=1) # Default: Jarang
                in_ekskul = st.selectbox("3. Seberapa aktif Anda dalam kegiatan ekstrakurikuler?", list(SCALE_ACTIVE.keys()), index=2) # Default: Cukup Aktif
                
            with col2:
                in_akademis = st.selectbox("4. Bagaimana kinerja akademis Anda saat ini?", list(SCALE_PERFORMANCE.keys()), index=2) # Default: Rata-rata
                in_beban = st.selectbox("5. Bagaimana beban belajar Anda dalam kurikulum AI/Koding?", list(SCALE_LOAD.keys()), index=2) # Default: Sedang
                in_nama = st.text_input("Nama Siswa (Opsional)", "Siswa Demo")

            btn_analyze = st.form_submit_button("🚀 Analisis Tingkat Stres Saya")

        if btn_analyze:
            if scaler is None or model is None:
                st.error("Model gagal dimuat. Tidak dapat melakukan prediksi.")
            else:
                with st.spinner('AI sedang menganalisis data Anda...'):
                    # --- PRE-PROCESSING INDIVIDU (Teks -> Angka) ---
                    # Mengambil nilai angka dari dictionary skala
                    data_individu = {
                        "Kualitas Tidur": [SCALE_QUALITY[in_tidur]],
                        "Sakit Kepala": [SCALE_FREQUENCY[in_kepala]],
                        "Kinerja Akademis": [SCALE_PERFORMANCE[in_akademis]],
                        "Beban Belajar": [SCALE_LOAD[in_beban]],
                        "Ekstrakurikuler": [SCALE_ACTIVE[in_ekskul]]
                    }
                    df_individu_numeric = pd.DataFrame(data_individu)
                    
                    # --- PREDIKSI ---
                    hasil = predict_stress(df_individu_numeric)[0] # Ambil hasil pertama
                    
                    # --- MENAMPILKAN HASIL ---
                    st.write("---")
                    st.success(f"Analisis berhasil untuk **{in_nama}**!")
                    
                    # Menampilkan hasil dengan warna berbeda
                    color_map = {"Rendah": "#ccffcc", "Sedang": "#ffffcc", "Tinggi": "#ffcccc"}
                    text_color_map = {"Rendah": "#166534", "Sedang": "#854d0e", "Tinggi": "#991b1b"}
                    
                    st.markdown(f"""
                    <div style="background-color:{color_map[hasil]}; padding: 2rem; border-radius: 10px; text-align: center; border: 1px solid {text_color_map[hasil]};">
                        <h4 style="color:{text_color_map[hasil]}; margin:0;">Tingkat Stres Terdeteksi:</h4>
                        <h1 style="color:{text_color_map[hasil]}; font-size: 4rem; margin:0.5rem 0;">{hasil}</h1>
                        <p style="color:{text_color_map[hasil]}; font-size:1rem;">Berdasarkan pola data yang Anda masukkan.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Menampilkan data angka yang dikirim ke model (buat skripsi)
                    with st.expander("Lihat Data Angka (Input Model)"):
                        st.dataframe(df_individu_numeric)

    # --- TAB 2: UPLOAD MASSAL EXCEL (Untuk Guru BK) ---
    with tab_excel:
        st.subheader("Analisis Massal via Excel/CSV")
        st.write("Guru BK dapat mengunggah file data siswa untuk melihat distribusi tingkat stres satu sekolah.")
        
        # Penjelasan Robustness
        st.info("""
        💡 **Robustness Feature:** 1. File Excel Anda **TIDAK HARUS** menggunakan nama kolom persis ("Kualitas Tidur", dll). Anda bisa mencocokkannya nanti.
        2. Isi data Excel **TIDAK HARUS** berupa angka. Anda bisa menggunakan teks deskriptif (misal: "Sangat Baik", "Jarang") sesuai template.
        """)
        
        # Template Download
        st.markdown("### 📥 1. Unduh Template Excel")
        st.write("Gunakan template ini agar pengisian data teks deskriptif tidak salah ketik.")
        
        template_data = {
            "No": [1, 2],
            "Nama Siswa": ["Siswa A", "Siswa B"],
            "Kolom_Tidur_Siswa": ["Sangat Baik", "Buruk"], # Nama kolom beda buat demo
            "Kolom_Pusing_Siswa": ["Jarang", "Sering"],
            "Kolom_Nilai_Siswa": ["Tinggi", "Rendah"],
            "Kolom_Beban_Siswa": ["Sedang", "Sangat Berat"],
            "Kolom_Ekskul_Siswa": ["Aktif", "Tidak Aktif"]
        }
        template_df = pd.DataFrame(template_data)
        
        # Buat tombol download excel
        towrite = io.BytesIO()
        template_df.to_excel(towrite, index=False, header=True)
        towrite.seek(0)
        
        st.download_button(
            label="Download Template Excel (Contoh Teks)",
            data=towrite,
            file_name="Template_Data_Stres_Teks.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.write("---")
        
        # File Uploader
        st.markdown("### 📤 2. Unggah Data Siswa")
        uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # Membaca file berdasarkan ekstensinya
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.write("### 👁️ Pratinjau Data Asli:")
                st.dataframe(df.head()) # Menampilkan 5 baris pertama
                
                # --- FITUR ROBUSTNESS: PENCOCOKAN KOLOM ---
                st.markdown("### 🧩 3. Cocokkan Kolom Excel")
                st.write("Sistem mendeteksi kolom yang sesuai. Jika sistem salah, silakan pilih kolom manual dari Excel Anda yang berisi data tersebut.")
                
                # Dictionary untuk menyimpan pemetaan kolom (Model Col -> Excel Col)
                col_mapping = {}
                
                col_match1, col_match2 = st.columns(2)
                
                # List nama kolom asli dari Excel
                actual_excel_cols = list(df.columns)
                
                with col_match1:
                    # Mencoba tebak otomatis, jika gagal default ke kolom pertama
                    col_mapping["Kualitas Tidur"] = st.selectbox("Kolom 'Kualitas Tidur' adalah:", actual_excel_cols, index=2 if len(actual_excel_cols)>2 else 0)
                    col_mapping["Sakit Kepala"] = st.selectbox("Kolom 'Sakit Kepala' adalah:", actual_excel_cols, index=3 if len(actual_excel_cols)>3 else 0)
                    col_mapping["Ekstrakurikuler"] = st.selectbox("Kolom 'Ekstrakurikuler' adalah:", actual_excel_cols, index=6 if len(actual_excel_cols)>6 else 0)
                    
                with col_match2:
                    col_mapping["Kinerja Akademis"] = st.selectbox("Kolom 'Kinerja Akademis' adalah:", actual_excel_cols, index=4 if len(actual_excel_cols)>4 else 0)
                    col_mapping["Beban Belajar"] = st.selectbox("Kolom 'Beban Belajar' adalah:", actual_excel_cols, index=5 if len(actual_excel_cols)>5 else 0)
                
                # Tombol untuk memicu prediksi massal
                st.write("---")
                if st.button("🚀 Prediksi Massal & Tampilkan Visualisasi"):
                    if scaler is None or model is None:
                        st.error("Model gagal dimuat. Tidak dapat melakukan prediksi.")
                    else:
                        with st.spinner('Sistem sedang memproses data massal...'):
                            
                            # 1. PRE-PROCESSING MASSAL (Ubah Excel Teks -> Angka sesuai mapping kolom)
                            df_numeric_mass, errors = convert_excel_to_numeric(df, col_mapping)
                            
                            # Cek jika ada error saat konversi (misal salah ketik teks skala)
                            if errors:
                                for err in errors:
                                    st.error(f"❌ Error Data: {err}")
                                st.stop() # Hentikan proses jika ada data rusak
                            
                            # 2. PROSES PREDDIKSI MASSAL
                            hasil_massal = predict_stress(df_numeric_mass)
                            
                            # 3. TAMBAHKAN HASIL KE DATAFRAME ASLI
                            df_result = df.copy()
                            df_result["Hasil Deteksi Stres"] = hasil_massal
                            
                        st.success("✅ Prediksi massal berhasil dilakukan!")
                        
                        # --- VISUALISASI HASIL (Requirement Visualisasi) ---
                        st.markdown("### 📊 Visualisasi Distribusi Tingkat Stres")
                        
                        # Menghitung jumlah setiap kelas
                        counts = df_result["Hasil Deteksi Stres"].value_counts().reset_index()
                        counts.columns = ['Tingkat Stres', 'Jumlah Siswa']
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                            st.write("#### Diagram Batang")
                            # Diagram batang menggunakan plotly express
                            fig_bar = px.bar(
                                counts, 
                                x='Tingkat Stres', 
                                y='Jumlah Siswa', 
                                color='Tingkat Stres',
                                color_discrete_map={"Tinggi": "#ef4444", "Sedang": "#eab308", "Rendah": "#22c55e"}, # Merah, Kuning, Hijau modern
                                text_auto=True
                            )
                            fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig_bar, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        with viz_col2:
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                            st.write("#### Diagram Lingkaran (Persentase)")
                            # Diagram lingkaran menggunakan plotly express
                            fig_pie = px.pie(
                                counts, 
                                values='Jumlah Siswa', 
                                names='Tingkat Stres',
                                color='Tingkat Stres',
                                color_discrete_map={"Tinggi": "#ef4444", "Sedang": "#eab308", "Rendah": "#22c55e"},
                                hole=0.4
                            )
                            fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig_pie, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        # --- TABEL HASIL DETEKSI (Highlight) ---
                        st.write("---")
                        st.markdown("### 🎯 Tabel Hasil Deteksi Massal:")
                        
                        # Highlight tabel
                        def color_stress(val):
                            if val == 'Tinggi': return 'background-color: #ffcccc; color: #991b1b; font-weight:bold;'
                            elif val == 'Sedang': return 'background-color: #ffffcc; color: #854d0e;'
                            elif val == 'Rendah': return 'background-color: #ccffcc; color: #166534;'
                            return ''
                        
                        st.dataframe(df_result.style.applymap(color_stress, subset=['Hasil Deteksi Stres']), use_container_width=True)
                        
                        # --- FITUR DOWNLOAD HASIL ---
                        st.write("### 📥 Unduh Hasil")
                        # Konversi DataFrame hasil ke CSV
                        csv = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Hasil Prediksi (CSV)",
                            data=csv,
                            file_name='Hasil_Deteksi_Stres_Massal.csv',
                            mime='text/csv',
                        )
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")

# ==========================================
# 8. LOGIKA HALAMAN 3: INFORMASI METODE
# ==========================================
elif menu == "ℹ️ Informasi Metode":
    st.title("Informasi Metode Penelitian 🧠")
    st.write("Halaman ini menjelaskan aspek teknis dari model AI yang digunakan dalam skripsi ini.")
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown(f"""
        <div class="info-card">
            <h3 style="color:#1e3a8a;">Arsitektur Multi Model (Stacking Ensemble)</h3>
            <p>Sistem ini menggunakan teknik Stacking Ensemble Learning. Ini adalah metode canggih yang menggabungkan beberapa algoritma Machine Learning berbeda (Base Learners) untuk meningkatkan akurasi dan stabilitas prediksi dibandingkan hanya menggunakan satu model saja.</p>
            
            <h4>1. Base Learners (Model Tingkat 0)</h4>
            <p>Data input dipelajari secara bersamaan oleh tiga algoritma berbeda:</p>
            <ul>
                <li><b>Decision Tree (DT):</b> Membuat aturan keputusan berdasarkan pola data.</li>
                <li><b>Support Vector Machine (SVM):</b> Mencari hyperplane (batasan) terbaik untuk memisahkan kelas stres.</li>
                <li><b>K-Nearest Neighbors (KNN):</b> Memprediksi berdasarkan kemiripan dengan data tetangga terdekat.</li>
            </ul>
            
            <h4>2. Meta Learner (Model Tingkat 1)</h4>
            <ul>
                <li><b>Logistic Regression:</b> Prediksi dari ketiga base learner di atas kemudian digabungkan dan dipelajari kembali oleh Logistic Regression untuk menentukan hasil prediksi final (Rendah / Sedang / Tinggi).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#1e3a8a;">📊 Atribut Penelitian</h4>
            <p style="font-size:0.9rem;">Model dilatih menggunakan 5 fitur utama:</p>
            <ol style="font-size:0.9rem; color:#4b5563;">
                <li>Kualitas Tidur (Skala 1-5)</li>
                <li>Frekuensi Sakit Kepala (Skala 1-5)</li>
                <li>Kinerja Akademis (Skala 1-5)</li>
                <li>Beban Belajar AI/Koding (Skala 1-5)</li>
                <li>Keaktifan Ekskul (Skala 1-5)</li>
            </ol>
            <p style="font-size:0.9rem;">Target Prediksi: Rendah (0), Sedang (1), Tinggi (2).</p>
        </div>
        <div class="info-card">
            <h4 style="color:#1e3a8a;">📞 Kontak Peneliti</h4>
            <p style="font-size:0.9rem;">[Nama Anda/NIM]<br>Program Studi [Nama Prodi]<br>[Nama Universitas]</p>
        </div>
        """, unsafe_allow_html=True)
