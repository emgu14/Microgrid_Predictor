import time
import paho.mqtt.client as mqtt
import numpy as np
import json
import os

print("🚀 ÉTAPE 2 : SIMULATION CAPTEUR IoT")

# Configuration
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "projet/tunisie/microgrid/v1"

if not os.path.exists('simulation_data.npy'):
    print("❌ ERREUR : Lance le fichier 1_train.py d'abord !")
    exit()

# Chargement des données
X_simu = np.load('simulation_data.npy')

client = mqtt.Client(client_id="Sensor_Sim_2024")

try:
    print(f"🔌 Connexion au broker {BROKER}...")
    client.connect(BROKER, PORT, 60)
except Exception as e:
    print(f"❌ Erreur connexion : {e}")
    exit()

print("📡 Envoi des données en cours...")

for i in range(len(X_simu)):
    # Convertit les données numpy en liste simple pour l'envoi
    sequence = X_simu[i].tolist()
    
    payload = json.dumps({"data": sequence})
    client.publish(TOPIC, payload)
    
    print(f"📤 Paquet envoyé {i+1}/{len(X_simu)}")
    time.sleep(2) # Pause de 2 secondes