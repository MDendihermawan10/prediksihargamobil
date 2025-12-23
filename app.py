import streamlit as st
import pandas as pd
import joblib
import os

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

# --- Load dataset untuk dropdown ---
df = pd.read_csv(data_file)

# --- Bersihkan kolom milage ---
df['milage'] = df['milage'].astype(str).str.extract(r'(\d+)')
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

# --- Bersihkan kolom model_year & price ---
df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

# --- Drop baris NaN di kolom penting ---
df = df[['brand','model','fuel_type','transmission','model_year','milage']].dropna()

st.title("Prediksi Harga Mobil Bekas")

# --- Input pengguna ---
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
    value=int(df['milage'].median())  # default aman
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
