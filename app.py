import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pytz

# Configuration de la page
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", page_icon="🔬", layout="wide")

# --- CONFIGURATION DE L'HEURE ALGERIE ---
timezone_dz = pytz.timezone('Africa/Algiers')
heure_algerie = datetime.now(timezone_dz).strftime("%d/%m/%Y %H:%M:%S")

# --- CSS AVANCE POUR DESIGN VIF ET BACKGROUND ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
        url("https://img.freepik.com/free-photo/doctor-working-with-digital-tablet-virtual-interface-screen_1150-19401.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1 { color: #003366; font-weight: bold; border-bottom: 4px solid #cc0000; padding-bottom: 10px; }
    .stButton>button {
        background-color: #cc0000; color: white; border-radius: 12px;
        font-weight: bold; width: 100%; border: none; height: 3.5em;
        font-size: 18px; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff0000; transform: scale(1.02); }
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

st.markdown("<h1 style='text-align: center;'>🔬 LABORATOIRE HOUBAD DOUAA - IA CARDIAQUE</h1>", unsafe_allow_html=True)
st.write(f"📍 **Lieu : Algérie** | 🕒 **Heure locale : {heure_algerie}**")

# --- FORMULAIRE COMPLET ---
with st.form("expert_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("👤 Infos Patient")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        age = st.number_input("Âge (Min 10 ans)", min_value=10, max_value=115, value=45)
        family = st.radio("Hérédité Cardiaque", ["Non", "Oui"])
        
    with c2:
        st.subheader("🩺 Paramètres Cliniques")
        sys_bp = st.number_input("Tension Systolique (mmHg)", 80, 220, 125)
        dia_bp = st.number_input("Tension Diastolique (mmHg)", 40, 140, 85)
        chol = st.number_input("Cholestérol (mg/dL)", 100, 450, 195)
        pulse = st.number_input("Pouls (BPM)", 40, 160, 72)
        
    with c3:
        st.subheader("🏃 Mode de Vie")
        smoke = st.selectbox("Tabagisme", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("Nombre de pas / jour", 0, 30000, 6000)
        sleep = st.slider("Sommeil (heures/nuit)", 3, 12, 7)
        stress = st.slider("Niveau de Stress (1-10)", 1, 10, 5)
        alcohol = st.number_input("Alcool (verres/semaine)", 0, 40, 0)
        diet = st.slider("Qualité Alimentation (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("💥 ANALYSE FINALE & GÉNÉRATION RAPPORT")

if submitted:
    # Prédiction avec XGBoost
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    input_data = pd.DataFrame([[
        age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke], 
        steps, stress, 3, sleep, (1 if family=="Oui" else 0), diet, alcohol
    ]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    colors = ["#00E676", "#FFEA00", "#FF1744"] # Vert, Jaune, Rouge (vifs)
    labels = ["RISQUE FAIBLE", "RISQUE MODÉRÉ", "RISQUE ÉLEVÉ"]
    
    # --- GRAPHIQUE EN DEMI-CERCLE (GAUGE) ---
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    val = (risk_score / 100) * np.pi
    ax.barh(0, val, color=colors[res_idx], align='center')
    ax.set_xlim(0, np.pi)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    plt.text(0, -0.1, "0%", ha='center', fontsize=10, fontweight='bold')
    plt.text(np.pi, -0.1, "100%", ha='center', fontsize=10, fontweight='bold')
    plt.text(np.pi/2, 0.1, f"{risk_score:.1f}%", ha='center', fontsize=20, fontweight='bold', color=colors[res_idx])
    st.pyplot(fig)

    st.markdown(f"<h2 style='text-align:center; color:{colors[res_idx]};'>{labels[res_idx]}</h2>", unsafe_allow_html=True)

    # --- INSTRUCTIONS LONGUES ET DÉTAILLÉES ---
    instrs = [
        "✅ Votre profil est optimal. Continuez ce mode de vie sain. Maintenez une alimentation riche en fibres, pratiquez une activité physique régulière (150 min/semaine) et veillez à conserver un cycle de sommeil régulier.",
        "⚠️ Vigilance requise. Un ajustement du mode de vie est nécessaire : réduisez significativement le sel et le sucre, améliorez votre qualité de sommeil et gérez votre stress. Une consultation de routine avec votre médecin est conseillée.",
        "🚨 DANGER : VEUILLEZ VISITER UN CARDIOLOGUE LE PLUS TÔT POSSIBLE. Risque critique détecté. Arrêtez tout effort physique intense immédiatement et surveillez votre tension quotidiennement en attendant l'avis d'un spécialiste."
    ]
    st.info(instrs[res_idx])

    # --- GÉNÉRATION DU PDF PROFESSIONNEL ---
    pdf = FPDF()
    pdf.add_page()
    
    # Filigrane
    pdf.set_font("Times", 'B', 32)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(10, 190, "LABORATOIRE HOUBAD DOUAA - ALGERIE")
    pdf.rotate(0)
    
    # Header
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Times", 'B', 22)
    pdf.cell(190, 15, "RAPPORT MEDICAL DE PREDICTION CARDIAQUE", ln=True, align='C')
    pdf.set_font("Times", 'I', 11)
    pdf.cell(190, 8, f"Genere le : {heure_algerie} (Heure d'Algerie)", ln=True, align='C')
    pdf.ln(10)
    
    # Infos Patient
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 13)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(190, 10, f"  PATIENT : {nom.upper()} {prenom.upper()}", ln=True, fill=True)
    pdf.set_font("Times", '', 11)
    pdf.multi_cell(190, 8, f"Age: {age} ans | Tension: {sys_bp}/{dia_bp} mmHg | Cholesterol: {chol} mg/dL\nStatut Tabac: {smoke} | Pas/jour: {steps} | Stress: {stress}/10 | Sommeil: {sleep}h\nAlcool: {alcohol} v/sem | Alimentation: {diet}/10 | Pouls: {pulse} BPM", border=1)
    pdf.ln(5)
    
    # Verdict
    pdf.set_font("Times", 'B', 16)
    if res_idx == 0: pdf.set_text_color(40, 167, 69)
    elif res_idx == 1: pdf.set_text_color(210, 150, 0)
    else: pdf.set_text_color(220, 53, 69)
    
    pdf.cell(190, 15, f"VERDICT IA : {labels[res_idx]} ({risk_score:.1f}%)", border=1, ln=True, align='C')
    
    # Instructions PDF
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 12)
    pdf.cell(190, 10, "CONSEILS ET INSTRUCTIONS MEDICALES :", ln=True)
    pdf.set_font("Times", '', 12)
    pdf.multi_cell(190, 10, instrs[res_idx], border=1)
    
    pdf.ln(15)
    pdf.set_font("Times", 'I', 9)
    pdf.cell(190, 10, "Document certifie par le Laboratoire Houbad Douaa - Expertise IA.", align='C')

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    st.download_button("📩 TELECHARGER LE RAPPORT PDF OFFICIEL", data=pdf_bytes, file_name=f"Rapport_Cardio_{nom}.pdf")
