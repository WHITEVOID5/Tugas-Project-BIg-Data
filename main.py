import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt

# === Akuisisi Data ===
url = "https://ckan.jombangkab.go.id/dataset/34f6fad9-ea0b-4a66-acae-7c5ebe551a3a/resource/306eee6e-4c30-48cd-852d-66064211b54d/download/angka-kejadian-malaria-kabupaten-jombang-2018-s.d.-2022.csv"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = StringIO(response.text)
    df = pd.read_csv(data, sep=';')

    # === Preprocessing ===
    df.dropna(inplace=True)
    df['Angka Kejadian Malaria'] = df['Angka Kejadian Malaria'].str.replace(',', '.', regex=False)
    df['Angka Kejadian Malaria'] = df['Angka Kejadian Malaria'].astype(float)
    df['Tahun'] = df['Tahun'].astype(int)
    df.sort_values('Tahun', inplace=True)
    df['Periode'] = range(1, len(df) + 1)

    # === Model Regresi Linear ===
    X = df[['Periode']]
    y = df['Angka Kejadian Malaria']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # === Streamlit App ===
    st.title("📊 Prediksi Angka Kejadian Malaria di Kabupaten Jombang")
    st.markdown("""
    Dataset dari [data.go.id](https://data.go.id).  
    Model: Regresi Linier  
    Satuan: Kasus per 1000 penduduk  
    """)

    st.subheader("📈 Data Aktual vs Prediksi")
    fig, ax = plt.subplots()
    ax.plot(df['Tahun'], y, marker='o', label='Aktual')
    ax.plot(df['Tahun'], y_pred, linestyle='--', color='red', label='Prediksi')
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Angka Kejadian Malaria")
    ax.set_title("Tren Angka Kejadian Malaria 2018–2022")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📉 Nilai RMSE")
    st.write(f"Root Mean Squared Error (RMSE): `{rmse:.4f}`")

    st.subheader("🔮 Prediksi Tahun Berikutnya")
    tahun_terakhir = df['Tahun'].max()
    tahun_prediksi = st.slider("Pilih Tahun", min_value=tahun_terakhir + 1, max_value=tahun_terakhir + 5)
    periode_baru = df['Periode'].max() + (tahun_prediksi - tahun_terakhir)
    prediksi_baru = model.predict([[periode_baru]])[0]
    st.success(f"Prediksi angka kejadian malaria pada tahun {tahun_prediksi}: **{prediksi_baru:.4f}** kasus per 1000 penduduk")

else:
    st.error(f"Gagal mengunduh data. Kode status: {response.status_code}")
