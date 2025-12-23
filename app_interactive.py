import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# --- Cek file model & dataset ---
model_file = 'car_price_model.pkl'
data_file = 'used_cars.csv'

if not os.path.exists(model_file):
    st.error(f"Model '{model_file}' tidak ditemukan. Jalankan train_model.py dulu.")
    st.stop()
if not os.path.exists(data_file):
    st.error(f"Dataset '{data_file}' tidak ditemukan.")
    st.stop()

# --- Load model ---
model = joblib.load(model_file)

# --- Load dataset ---
df = pd.read_csv(data_file)

# --- Bersihkan kolom milage ---
df['milage'] = df['milage'].astype(str).str.extract(r'(\d+)')
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

# --- Bersihkan kolom model_year & price ---
df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

# --- Drop baris NaN ---
df = df[['brand','model','fuel_type','transmission','model_year','milage','price']].dropna()

st.title("Prediksi Harga Mobil Bekas Interaktif")

# --- Tampilkan info dataset ---
st.subheader("Informasi Dataset")
st.write(f"Jumlah data valid: {len(df)}")
st.write(df[['price','milage','model_year']].describe())

# --- Histogram harga ---
st.subheader("Distribusi Harga Mobil")
fig1, ax1 = plt.subplots()
ax1.hist(df['price'], bins=20, color='skyblue', edgecolor='black')
ax1.set_xlabel("Harga")
ax1.set_ylabel("Jumlah Mobil")
st.pyplot(fig1)

# --- Histogram milage ---
st.subheader("Distribusi Jarak Tempuh (Milage)")
fig2, ax2 = plt.subplots()
ax2.hist(df['milage'], bins=20, color='lightgreen', edgecolor='black')
ax2.set_xlabel("Jarak Tempuh")
ax2.set_ylabel("Jumlah Mobil")
st.pyplot(fig2)

# --- Input pengguna ---
st.subheader("Prediksi Harga Mobil")
brand = st.selectbox("Merek Mobil (brand)", df['brand'].unique())
filtered_models = df[df['brand']==brand]['model'].unique()
model_name = st.selectbox("Model Mobil (model)", filtered_models)

model_year = st.number_input(
    "Tahun Mobil (model_year)",
    min_value=int(df['model_year'].min()),
    max_value=int(df['model_year'].max()),
    value=int(df['model_year'].median())
)

milage = st.number_input(
    "Jarak Tempuh (milage)",
    min_value=0,
    max_value=int(df['milage'].max()),
    value=int(df['milage'].median())
)

fuel = st.selectbox("Jenis Bahan Bakar (fuel_type)", df['fuel_type'].dropna().unique())
trans = st.selectbox("Transmisi", df['transmission'].dropna().unique())

# --- Prediksi ---
if st.button("Prediksi Harga"):
    input_df = pd.DataFrame([{
        'brand': brand,
        'model': model_name,
        'model_year': model_year,
        'milage': milage,
        'fuel_type': fuel,
        'transmission': trans
    }])
    
    pred_price = model.predict(input_df)
    st.success(f"Estimasi Harga Mobil Bekas: ${pred_price[0]:,.2f}")
