import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fpdf import FPDF
from datetime import datetime
import numpy as np

# Configuration de la page
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", page_icon="🔬", layout="wide")

# --- CSS POUR DESIGN MEDICAL ET TRANSPARENCE ---
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.92)), 
        url("https://img.freepik.com/free-photo/medical-technology-concept-smart-health-care_53876-104440.jpg");
        background-size: cover;
        background-attachment: fixed;
    }}
    .main-title {{
        color: #004d99;
        text-shadow: 1px 1px 2px #cccccc;
        font-family: 'serif';
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_csv("cardiovascular_risk_numeric.csv")
    mapping = {'Never': 0, 'Former': 1, 'Current': 2}
    df['smoking_status'] = df['smoking_status'].map(mapping)
    X = df.drop(['Patient_ID', 'heart_disease_risk_score', 'risk_category'], axis=1)
    y = df['risk_category']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return model, X.columns

model, feat_cols = train_model()

st.markdown("<h1 class='main-title' style='text-align: center;'>🔬 LABORATOIRE HOUBAD DOUAA</h1>", unsafe_allow_html=True)

# Affichage de l'heure
heure_actuelle = datetime.now().strftime("%H:%M:%S")
st.write(f"🕒 Session ouverte à : **{heure_actuelle}**")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Patient")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        age = st.number_input("Âge (Minimum 10 ans)", min_value=10, max_value=115, value=40)
    with col2:
        st.subheader("🩺 Mesures")
        sys_bp = st.number_input("Tension Systolique", 80, 220, 120)
        dia_bp = st.number_input("Tension Diastolique", 40, 140, 80)
        chol = st.number_input("Cholestérol (mg/dL)", 100, 450, 200)
    with col3:
        st.subheader("🏃 Vie & Stress")
        smoke = st.selectbox("Tabac", ["Jamais", "Ex-fumeur", "Fumeur"])
        stress = st.slider("Niveau de Stress (1-10)", 1, 10, 5)
        family = st.radio("Hérédité Cardiaque", ["Non", "Oui"])

    submitted = st.form_submit_button("🚀 LANCER L'ANALYSE IA")

if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    # Données d'entrée (BMI fixé à 25 par défaut ici pour la prédiction)
    input_data = pd.DataFrame([[age, 25.0, sys_bp, dia_bp, chol, 72, m_smoke[smoke], 7000, stress, 3, 7, (1 if family=="Oui" else 0), 7, 0]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_percent = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    # --- TEXTES DÉTAILLÉS ---
    cats = ["PROFIL CARDIOVASCULAIRE OPTIMAL", "VIGILANCE ET AJUSTEMENT", "ALERTE PRIORITAIRE"]
    colors = ["#28a745", "#ff8800", "#cc0000"]
    
    instrs = [
        "Vos résultats indiquent un risque cardiovasculaire faible. Continuez à privilégier une alimentation riche en fibres, fruits et légumes. Poursuivez une activité physique régulière (min 150 min/semaine) et effectuez un bilan de routine annuel.",
        "Votre score indique une probabilité modérée. Réduisez votre consommation de sel et de sucres. Intégrez des techniques de gestion du stress et prenez rendez-vous avec votre médecin généraliste pour un bilan lipidique complet.",
        "URGENT : Le système détecte un niveau de risque élevé. Vous devez consulter un CARDIOLOGUE dans les plus brefs délais. Arrêtez tout effort intense et surveillez quotidiennement votre tension artérielle."
    ]

    st.markdown(f"<h2 style='text-align:center; color:{colors[res_idx]};'>{cats[res_idx]} ({risk_percent:.1f}%)</h2>", unsafe_allow_html=True)
    
    # Graphique de probabilité
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(["Niveau de Risque"], [risk_percent], color=colors[res_idx])
    ax.set_xlim(0, 100)
    st.pyplot(fig)
    
    st.warning(instrs[res_idx])

    # --- GÉNÉRATION DU PDF ---
    pdf = FPDF()
    pdf.add_page()
    
    # Filigrane
    pdf.set_font("Times", 'B', 40)
    pdf.set_text_color(245, 245, 245)
    pdf.rotate(45, 100, 100)
    pdf.text(10, 190, "LABORATOIRE HOUBAD DOUAA - IA SYSTEM")
    pdf.rotate(0)

    # Entête
    pdf.set_font("Times", 'B', 24)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 20, "LABORATOIRE HOUBAD DOUAA", ln=True, align='C')
    pdf.set_font("Times", 'I', 11)
    pdf.cell(190, 5, f"Rapport medical genere le : {datetime.now().strftime('%d/%m/%Y a %H:%M')}", ln=True, align='C')
    pdf.ln(15)

    # Patient Data Section
    pdf.set_font("Times", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, f"BILAN DE : {nom.upper()} {prenom.upper()}", ln=True)
    pdf.set_font("Times", '', 12)
    pdf.cell(95, 10, f"Age : {age} ans")
    pdf.cell(95, 10, f"Tension : {sys_bp}/{dia_bp} mmHg", ln=True)
    pdf.cell(95, 10, f"Cholesterol : {chol} mg/dL")
    pdf.cell(95, 10, f"Statut Tabac : {smoke}", ln=True)
    pdf.ln(10)

    # Résultat IA
    pdf.set_font("Times", 'B', 16)
    if res_idx == 0: pdf.set_text_color(40, 167, 69)
    elif res_idx == 1: pdf.set_text_color(255, 136, 0)
    else: pdf.set_text_color(204, 0, 0)
    
    pdf.cell(190, 15, f"RESULTAT IA : {cats[res_idx]} ({risk_percent:.1f}%)", border=1, ln=True, align='C')
    
    # Instructions détaillées
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 13)
    pdf.cell(190, 10, "INSTRUCTIONS MEDICALES ET CONSEILS :", ln=True)
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(190, 10, instrs[res_idx], border=1)
    
    pdf.ln(20)
    pdf.set_font("Times", 'I', 10)
    pdf.cell(190, 10, "Ce document est genere par Intelligence Artificielle et ne remplace pas une expertise clinique.", align='C')

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    st.download_button("📩 TELECHARGER LE RAPPORT PDF COMPLET", data=pdf_bytes, file_name=f"Rapport_{nom}.pdf", mime="application/pdf")
