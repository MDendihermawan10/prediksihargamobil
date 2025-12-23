import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- Cek file dataset ---
file_path = 'used_cars.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' tidak ditemukan. Pastikan ada di folder project.")

# --- Load dataset ---
df = pd.read_csv(file_path)
print("Kolom dataset:", df.columns.tolist())

# --- Bersihkan kolom price ---
df['price'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# --- Bersihkan kolom milage (ambil angka pertama dari teks) ---
df['milage'] = df['milage'].astype(str).str.extract(r'(\d+)')
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

# --- Bersihkan kolom model_year ---
df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')

# --- Hapus baris yang invalid ---
df = df.dropna(subset=['price','milage','model_year'])

print("5 data pertama setelah preprocessing:")
print(df[['brand','model','model_year','milage','price']].head())
print(f"Jumlah data valid: {len(df)}")

if len(df) == 0:
    raise ValueError("Tidak ada data valid setelah preprocessing. Cek datasetmu.")

# --- Pilih fitur & target ---
X = df[['brand','model','model_year','milage','fuel_type','transmission']]
y = df['price']

# --- Preprocessing kategorikal ---
categorical = ['brand','model','fuel_type','transmission']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
)

# --- Pipeline Random Forest ---
rf = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Training ---
print("Training model...")
rf.fit(X_train, y_train)

# --- Evaluasi ---
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# --- Simpan model ---
joblib.dump(rf, 'car_price_model.pkl')
print("Model tersimpan sebagai 'car_price_model.pkl'")
