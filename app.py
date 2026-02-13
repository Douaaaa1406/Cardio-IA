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

# --- CSS POUR DESIGN VIF ET BACKGROUND ---
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
st.write(f"📍 **Lieu : Algerie** | 🕒 **Heure locale : {heure_algerie}**")

with st.form("expert_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("👤 Infos Patient")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prenom")
        age = st.number_input("Age (Min 10 ans)", min_value=10, max_value=115, value=45)
        family = st.radio("Heredite Cardiaque", ["Non", "Oui"])
    with c2:
        st.subheader("🩺 Parametres Cliniques")
        sys_bp = st.number_input("Tension Systolique (mmHg)", 80, 220, 125)
        dia_bp = st.number_input("Tension Diastolique (mmHg)", 40, 140, 85)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 450, 195)
        pulse = st.number_input("Pouls (BPM)", 40, 160, 72)
    with c3:
        st.subheader("🏃 Mode de Vie")
        smoke = st.selectbox("Tabagisme", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("Nombre de pas / jour", 0, 30000, 6000)
        sleep = st.slider("Sommeil (heures/nuit)", 3, 12, 7)
        stress = st.slider("Niveau de Stress (1-10)", 1, 10, 5)
        alcohol = st.number_input("Alcool (verres/semaine)", 0, 40, 0)
        diet = st.slider("Qualite Alimentation (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("💥 ANALYSE FINALE & GENERATION RAPPORT")

if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    input_data = pd.DataFrame([[age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke], steps, stress, 3, sleep, (1 if family=="Oui" else 0), diet, alcohol]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    colors = ["#00E676", "#FFEA00", "#FF1744"]
    labels = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]
    
    # --- GAUGE CHART ---
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

    # Textes pour l'interface (avec accents)
    instrs_ui = [
        "✅ Votre profil est optimal. Continuez ce mode de vie sain.",
        "⚠️ Vigilance requise. Un ajustement du mode de vie est nécessaire.",
        "🚨 DANGER : VEUILLEZ VISITER UN CARDIOLOGUE LE PLUS TÔT POSSIBLE."
    ]
    # Textes pour le PDF (SANS ACCENTS pour éviter l'erreur Unicode)
    instrs_pdf = [
        "Votre profil est optimal. Continuez ce mode de vie sain. Maintenez une alimentation riche en fibres et pratiquez une activite physique.",
        "Vigilance requise. Un ajustement du mode de vie est necessaire : reduisez le sel, ameliorez votre sommeil et gerez votre stress.",
        "DANGER : VEUILLEZ VISITER UN CARDIOLOGUE LE PLUS TOT POSSIBLE. Risque critique detecte. Arretez tout effort physique."
    ]
    st.info(instrs_ui[res_idx])

    # --- PDF ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 30)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(10, 190, "LABORATOIRE HOUBAD DOUAA - ALGERIE")
    pdf.rotate(0)
    
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(190, 15, "RAPPORT MEDICAL - IA CARDIAQUE", ln=True, align='C')
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(190, 8, f"Genere le : {heure_algerie}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, f"PATIENT : {nom.upper()} {prenom.upper()}", ln=True)
    pdf.set_font("Arial", '', 10)
    params_text = f"Age: {age} | Tension: {sys_bp}/{dia_bp} | Chol: {chol} | Pas: {steps} | Stress: {stress} | Sommeil: {sleep}h | Alcool: {alcohol} | Diet: {diet} | Pouls: {pulse}"
    pdf.multi_cell(190, 8, params_text, border=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 16)
    if res_idx == 0: pdf.set_text_color(40, 167, 69)
    elif res_idx == 1: pdf.set_text_color(210, 150, 0)
    else: pdf.set_text_color(220, 53, 69)
    pdf.cell(190, 15, f"VERDICT IA : {labels[res_idx]} ({risk_score:.1f}%)", border=1, ln=True, align='C')
    
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(190, 10, instrs_pdf[res_idx], border=1)

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore') # 'ignore' securise l'encodage
    st.download_button("📩 TELECHARGER LE RAPPORT PDF", data=pdf_bytes, file_name=f"Rapport_Cardio_{nom}.pdf", mime="application/pdf")
