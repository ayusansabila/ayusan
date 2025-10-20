import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Path Model dan Dataset ---
MODEL_PATH = "models/linear_regression_los.pkl"
DATA_PATH = "LengthOfStay.csv"

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Load Dataset ---
if not os.path.exists(DATA_PATH):
    st.error(f"❌ File dataset tidak ditemukan di: {DATA_PATH}")
    st.stop()

data = pd.read_csv(DATA_PATH)

# --- Aplikasi Utama ---
def main():
    st.sidebar.title("🏥 Menu Navigasi")
    page = st.sidebar.radio("Pilih Halaman:", ["Prediksi", "Tentang Dataset"])

    # ==========================================================
    # 🧮 HALAMAN PREDIKSI
    # ==========================================================
    if page == "Prediksi":
        st.title("📊 Prediksi Lama Rawat Pasien")
        st.write(
            "Masukkan data pasien untuk memperkirakan berapa lama pasien akan dirawat di rumah sakit "
            "berdasarkan model regresi linear."
        )

        st.header("🧩 Masukkan Data Pasien")
        rcount = st.number_input("Jumlah Diagnosa (rcount)", min_value=0, max_value=50, value=3)
        gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        asthma = st.radio("Asma?", ["Ya", "Tidak"])
        pneum = st.radio("Pneumonia?", ["Ya", "Tidak"])
        depress = st.radio("Depresi?", ["Ya", "Tidak"])

        if st.button("🔍 Prediksi Lama Rawat"):
            # Validasi input
            if rcount > 20:
                st.warning("⚠️ Angka diagnosa terlalu besar, periksa kembali input Anda (biasanya < 20).")
                return

            # Siapkan data untuk prediksi
            data_input = np.array([[
                rcount,
                1 if gender == "Laki-laki" else 0,
                1 if asthma == "Ya" else 0,
                1 if pneum == "Ya" else 0,
                1 if depress == "Ya" else 0
            ]])

            # Prediksi
            prediksi = model.predict(data_input)[0]

            st.success(f"⏱️ Perkiraan lama rawat pasien adalah **{prediksi:.2f} hari**")

    # ==========================================================
    # 📊 HALAMAN TENTANG DATASET
    # ==========================================================
    elif page == "Tentang Dataset":
        st.title("📘 Tentang Dataset")
        st.write("""
        Dataset ini digunakan untuk melatih model **Linear Regression** dalam memprediksi *Length of Stay (LOS)* pasien.

        **Fitur yang digunakan:**
        - `rcount`: Jumlah diagnosa pasien (biasanya 1–10)
        - `gender`: Jenis kelamin pasien
        - `asthma`: Apakah pasien memiliki riwayat asma
        - `pneum`: Apakah pasien memiliki pneumonia
        - `depress`: Apakah pasien memiliki depresi

        **Tujuan:**
        Model ini membantu memperkirakan lama rawat pasien untuk mendukung perencanaan kapasitas rumah sakit.
        """)

        st.subheader("🧾 Cuplikan Data CSV")
        st.dataframe(data.head())

        st.subheader("📈 Statistik Deskriptif")
        st.write(data.describe())

        # Korelasi antar variabel
        st.subheader("🔗 Korelasi Antar Variabel")
        corr = data.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Grafik rata-rata lama rawat berdasarkan gender
        st.subheader("🧍‍♂️ Rata-rata Lama Rawat Berdasarkan Jenis Kelamin")
        if 'gender' in data.columns and 'lengthofstay' in data.columns:
            avg_stay = data.groupby('gender')['lengthofstay'].mean().reset_index()
            st.bar_chart(avg_stay.set_index('gender'))
        else:
            st.warning("Kolom 'gender' atau 'lengthofstay' tidak ditemukan di dataset.")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
