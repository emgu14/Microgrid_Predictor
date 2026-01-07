import pandas as pd
import numpy as np
import joblib
import os
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("🚀 ÉTAPE 1 : ENTRAINEMENT DE L'IA")

# 1. TÉLÉCHARGEMENT
print("📡 Téléchargement du dataset UCI...")
try:
    dataset = fetch_ucirepo(id=235)
    df = dataset.data.features.copy()
except Exception as e:
    print(f"❌ Erreur internet : {e}")
    exit()

# 2. NETTOYAGE
print("🧹 Nettoyage des données...")
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('datetime', inplace=True)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Remplissage des trous (NaN)
df['Global_active_power'] = df['Global_active_power'].fillna(method='ffill')

# Moyenne par heure
data_hourly = df['Global_active_power'].resample('H').mean().to_frame()
data_hourly = data_hourly.fillna(method='ffill').fillna(0)

print(f"✅ Données prêtes : {len(data_hourly)} heures.")

# 3. PRÉPARATION IA
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_hourly.values)

def create_dataset(dataset, look_back=24):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 24
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split Simulation (On garde les 200 dernières heures)
split = len(X) - 200
X_train, y_train = X[:split], y[:split]
X_simu = X[split:]

# 4. ENTRAINEMENT
print("🧠 Entraînement LSTM (Patience...)...")
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=2, batch_size=64, verbose=1)

# 5. SAUVEGARDE
model.save('my_brain.h5')
joblib.dump(scaler, 'my_scaler.pkl')
np.save('simulation_data.npy', X_simu)

print("💾 Fichiers sauvegardés : my_brain.h5, my_scaler.pkl, simulation_data.npy")
print("✅ TERMINE. Passe au fichier 3 (Dashboard) puis 2 (Capteur).")