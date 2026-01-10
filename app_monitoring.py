import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Cache les logs rouges

import streamlit as st
import pandas as pd
import numpy as np
import time
import paho.mqtt.client as mqtt
import json
import joblib
from tensorflow.keras.models import load_model
import queue

# --- 1. CONFIGURATION PAGE ---
st.set_page_config(
    page_title="Microgrid France - Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. STYLE CSS (Fond sombre + Cartes) ---
st.markdown("""
    <style>
    .stApp {background-color: #0E1117;}
    /* Style des cartes KPIs */
    div[data-testid="metric-container"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    h1, h2, h3, p {color: white;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT IA ---
BROKER = "broker.emqx.io"
TOPIC = "projet/tunisie/microgrid/v1"
MODEL_FILE = 'my_brain.h5'
SCALER_FILE = 'my_scaler.pkl'
data_queue = queue.Queue()

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE): return None, None
    return load_model(MODEL_FILE), joblib.load(SCALER_FILE)

model, scaler = load_resources()

# Session State
if 'history_real' not in st.session_state: st.session_state.history_real = []
if 'history_pred' not in st.session_state: st.session_state.history_pred = []

# --- 4. SIDEBAR (Paramètres) ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    # Prix moyen en France env. 0.20€ / kWh
    prix_kwh = st.number_input("Prix kWh (€)", value=0.22) 
    seuil_max = st.slider("Seuil Alerte (kW)", 1.0, 5.0, 2.0)
    if st.button("Effacer l'historique"):
        st.session_state.history_real = []
        st.session_state.history_pred = []

# --- 5. MISE EN PAGE (PLACEHOLDERS) ---
# On crée les "cases vides" ICI, AVANT la boucle pour éviter l'empilement

# En-tête
c1, c2 = st.columns([0.5, 6])
with c1: st.image("https://cdn-icons-png.flaticon.com/512/2933/2933864.png", width=60)
with c2: 
    st.title("Smart Microgrid - France (Sceaux)")
    st.caption("Système de prédiction de charge par Deep Learning (LSTM)")

st.markdown("---")

# Ligne 1 : Les 4 KPIs alignés
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
# On crée les placeholders (les boîtes vides fixes)
box_kpi1 = col_kpi1.empty()
box_kpi2 = col_kpi2.empty()
box_kpi3 = col_kpi3.empty()
box_kpi4 = col_kpi4.empty()

# Ligne 2 : Graphique (Gauche) et Alertes (Droite)
col_main, col_side = st.columns([2.5, 1])

with col_main:
    st.subheader("📉 Consommation vs Prédiction")
    chart_box = st.empty() # Le graphique sera toujours dessiné ici

with col_side:
    st.subheader("🛡️ État du Réseau")
    gauge_box = st.empty() # La jauge
    st.write("")
    alert_box = st.empty() # Le message d'alerte
    rec_box = st.empty()   # Le conseil IA

# Ligne 3 : Stats (en bas)
st.markdown("---")
st.markdown("##### 📊 Statistiques de la session")
c_stat1, c_stat2, c_stat3 = st.columns(3)
stat_box1 = c_stat1.empty()
stat_box2 = c_stat2.empty()
stat_box3 = c_stat3.empty()


# --- 6. GESTION MQTT ---
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        data_queue.put(np.array(payload["data"]))
    except: pass

client = mqtt.Client(client_id=f"Viewer_{time.time()}")
client.on_message = on_message
try:
    client.connect(BROKER, 1883, 60)
    client.subscribe(TOPIC)
    client.loop_start()
except: st.error("Erreur connexion MQTT")


# --- 7. BOUCLE PRINCIPALE ---
while True:
    # Traitement des données en attente
    while not data_queue.empty():
        sequence = data_queue.get()
        # Calcul IA
        val_reelle = scaler.inverse_transform([[sequence[-1][0]]])[0][0]
        input_data = sequence.reshape(1, 24, 1)
        pred = scaler.inverse_transform(model.predict(input_data, verbose=0))[0][0]
        
        st.session_state.history_real.append(val_reelle)
        st.session_state.history_pred.append(pred)
        
        # On garde 50 points max pour que le graph reste lisible
        if len(st.session_state.history_real) > 50:
            st.session_state.history_real.pop(0)
            st.session_state.history_pred.pop(0)

    # MISE A JOUR DE L'INTERFACE (Rafraîchissement des cases)
    if len(st.session_state.history_real) > 0:
        # On force le type float pour éviter les erreurs "float32"
        curr = float(st.session_state.history_real[-1])
        futur = float(st.session_state.history_pred[-1])
        delta = futur - curr
        cout = curr * prix_kwh

        # 1. Remplissage des KPIs (Update, pas Append)
        box_kpi1.metric("Conso Actuelle", f"{curr:.2f} kW")
        box_kpi2.metric("Prédiction (1h)", f"{futur:.2f} kW", delta=f"{delta:+.2f} kW", delta_color="inverse")
        box_kpi3.metric("Coût Instantané", f"{cout:.3f} €/h") # En EUROS
        
        if futur > seuil_max:
            box_kpi4.metric("Sécurité", "CRITIQUE", "Délestage !", delta_color="inverse")
        else:
            box_kpi4.metric("Sécurité", "OK", "Stable")

        # 2. Remplissage Graphique
        df = pd.DataFrame({"Réel": st.session_state.history_real, "IA (Prédiction)": st.session_state.history_pred})
        chart_box.line_chart(df, height=350)

        # 3. Remplissage Jauge & Alertes
        ratio = min(futur / seuil_max, 1.0)
        gauge_box.progress(float(ratio), text=f"Charge: {int(ratio*100)}%")

        if futur > seuil_max:
            alert_box.error(f"🚨 PIC DÉTECTÉ : {futur:.2f} kW")
            rec_box.info("👉 **Action :** Couper le chauffage électrique et lancer le groupe électrogène.")
        elif futur > (seuil_max * 0.8):
            alert_box.warning("⚠️ Charge Élevée")
            rec_box.info("👉 **Action :** Réduire l'éclairage.")
        else:
            alert_box.success("✅ Réseau Nominal")
            rec_box.markdown("*Aucune action requise.*")

        # 4. Remplissage Stats bas de page
        vals = st.session_state.history_real
        stat_box1.markdown(f"**Max:** {float(max(vals)):.2f} kW")
        stat_box2.markdown(f"**Moyenne:** {float(np.mean(vals)):.2f} kW")
        stat_box3.markdown(f"**Coût session:** {float(sum(vals)/60 * prix_kwh):.4f} €")

    # Petite pause pour ne pas surchauffer le processeur
    time.sleep(1)