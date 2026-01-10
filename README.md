# Smart Microgrid Predictor
> **Un système intelligent de gestion énergétique pour Microgrids, combinant Deep Learning (LSTM) et simulation IoT temps réel.**

---

## Description

La gestion des pics de consommation (**Peak Shaving**) est un enjeu critique pour les réseaux électriques modernes (Microgrids).

Ce projet propose une solution complète "End-to-End" capable de :
1.  **Collecter** des données de consommation simulées via un protocole IoT (**MQTT**).
2.  **Prédire** la demande énergétique future (T+1 heure) grâce à un réseau de neurones **LSTM**.
3.  **Visualiser** et **Alerter** en temps réel via un tableau de bord Web interactif.

Le système est basé sur le dataset réel de consommation électrique d'un foyer situé à **Sceaux (France)** (Dataset UCI).

---

##  Architecture du Projet

Le projet est modulaire et respecte une architecture industrielle :

| Fichier | Rôle | Description Technique |
| :--- | :--- | :--- |
| **`main.py`** |  **Lanceur** | Script d'orchestration qui lance automatiquement le Dashboard et le Capteur dans des processus séparés. |
| **`1_train.py`** |  **Cerveau (IA)** | Script de Data Science : Nettoyage des données, normalisation et entraînement du modèle LSTM. Génère le fichier `my_brain.h5`. |
| **`2_sensor.py`** | **Capteur (IoT)** | Simulateur de compteur intelligent (Linky). Lit les données de test et les envoie via MQTT toutes les secondes. |
| **`app_monitoring.py`** |  **Dashboard** | Application Web **Streamlit**. Reçoit les flux MQTT, exécute les prédictions et affiche les KPIs/Alertes. |

---

## Fonctionnalités Clés

*    **Deep Learning :** Modèle LSTM (Long Short-Term Memory) optimisé pour les séries temporelles.
*    **IoT Temps Réel :** Communication via broker MQTT public (`broker.emqx.io`).
*    **Dashboard Interactif :** Interface moderne avec graphiques dynamiques et jauges de charge.
*    **Gestion d'Alerte :** Détection automatique des pics de surcharge et recommandations d'actions (Délestage).
*    **Analyse Économique :** Calcul du coût en temps réel basé sur le tarif électrique français (€).

---

## Installation & Démarrage

### 1. Cloner le projet
  bash
   git clone https://github.com/votre-user/Microgrid_Predictor.git
   cd Microgrid_Predictor
### 2. Installer les dépendances
  bash
   pip install -r requirements.txt
### 3. Lancer l'application
 Pas besoin de lancer les fichiers un par un. Le lanceur s'occupe de tout :
 
 bash
   python main.py
   
Le système va automatiquement :
Vérifier si le modèle IA existe (sinon, il lance l'entraînement).
Ouvrir le Dashboard Streamlit dans votre navigateur.
Lancer le Simulateur de Capteur dans une nouvelle fenêtre console.

### Dataset Utilisé
Nom : Individual Household Electric Power Consumption
Source : UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
Détails : 2 075 259 mesures collectées dans une maison en France entre 2006 et 2010.
### Stack Technique
Langage : Python 3.12
Machine Learning : TensorFlow / Scikit-learn
Data Processing : Pandas, NumPy
IoT & Réseau : MQTT
Visualisation : Streamlit

