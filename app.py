import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pytz

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", page_icon="🔬", layout="wide")

# --- HEURE ALGERIE ---
timezone_dz = pytz.timezone('Africa/Algiers')
heure_algerie = datetime.now(timezone_dz).strftime("%d/%m/%Y %H:%M:%S")

# --- DESIGN & FOND D'ECRAN ---
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
        url("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-attachment: fixed;
    }}
    .main-title {{ color: #003366; font-family: 'Cambria', serif; text-align: center; border-bottom: 3px solid #cc0000; }}
    .stButton>button {{
        background-color: #cc0000; color: white; border-radius: 10px;
        font-weight: bold; height: 3.5em; width: 100%; border: none;
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

st.markdown("<h1 class='main-title'>🔬 LABORATOIRE HOUBAD DOUAA - SYSTEME IA CARDIAQUE</h1>", unsafe_allow_html=True)
st.write(f"🕒 **Heure d'Algerie : {heure_algerie}**")

# --- FORMULAIRE AVEC TOUS LES PARAMETRES ---
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("👤 Patient")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prenom")
        age = st.number_input("Age", 10, 110, 45)
        family = st.radio("Heredite Cardiaque", ["Non", "Oui"])
        pulse = st.number_input("Pouls (BPM)", 40, 160, 72)
    with c2:
        st.subheader("🩺 Clinique")
        sys_bp = st.number_input("Tension Systolique", 80, 220, 120)
        dia_bp = st.number_input("Tension Diastolique", 40, 140, 80)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 450, 200)
        bmi = st.number_input("IMC (BMI)", 10.0, 50.0, 25.0)
    with c3:
        st.subheader("🏃 Mode de Vie")
        smoke = st.selectbox("Tabac", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("Pas/jour", 0, 30000, 6000)
        alcohol = st.number_input("Alcool (verres/sem)", 0, 50, 0)
        sleep = st.slider("Sommeil (h/nuit)", 3, 12, 7)
        stress = st.slider("Stress (1-10)", 1, 10, 5)
        diet = st.slider("Qualite Alimentation (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("🚀 ANALYSE COMPLETE DU RISQUE")

if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    # Envoi de TOUS les paramètres au modèle
    input_data = pd.DataFrame([[age, bmi, sys_bp, dia_bp, chol, pulse, m_smoke[smoke], steps, stress, 3, sleep, (1 if family=="Oui" else 0), diet, alcohol]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"] # Vert, Jaune, Rouge
    labels = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]

    # --- JAUGE AVEC FLECHE (DEMI-CERCLE) ---
    st.write("---")
    col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
    with col_g2:
        fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, np.pi, 100)
        ax.barh(0, np.pi, color='lightgrey', alpha=0.3) # Fond
        ax.barh(0, (risk_score/100)*np.pi, color=colors[res_idx]) # Risque
        # Flèche
        ax.annotate('', xy=((risk_score/100)*np.pi, 0), xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        ax.set_xlim(0, np.pi)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        plt.text(np.pi/2, 0.3, f"{risk_score:.1f}%", ha='center', fontsize=20, fontweight='bold', color=colors[res_idx])
        st.pyplot(fig)

    st.markdown(f"<h2 style='text-align:center; color:{colors[res_idx]};'>{labels[res_idx]}</h2>", unsafe_allow_html=True)

    # --- RECOMMENDATIONS DETAILLEES ---
    recos = [
        "FELICITATIONS : Votre profil est optimal. Maintenez une alimentation riche en fibres (legumes, fruits). Continuez une activite physique de 150 min/semaine. Votre hygiene de vie est votre meilleur atout.",
        "ATTENTION : Risque modere detecte. Nous recommandons de reduire la consommation de sel et de graisses saturees. Augmentez votre nombre de pas quotidiens a 10.000 et surveillez votre tension chaque mois.",
        "URGENT : RISQUE ELEVE. Une consultation chez un CARDIOLOGUE est indispensable sous 48h. Arretez tout effort physique violent. Un bilan complet (ECG, test d effort) doit etre realise immediatement."
    ]
    st.info(recos[res_idx])

    # --- GENERATION PDF DETAILLE ---
    pdf = FPDF()
    pdf.add_page()
    
    # Filigrane & Header
    pdf.set_font("Times", 'B', 30)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(10, 190, "LABORATOIRE HOUBAD DOUAA")
    pdf.rotate(0)
    
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Times", 'B', 22)
    pdf.cell(190, 15, "RAPPORT MEDICAL DE PREDICTION CARDIAQUE", ln=True, align='C')
    pdf.set_font("Times", 'I', 11)
    pdf.cell(190, 8, f"Date d'analyse : {heure_algerie} (Algerie)", ln=True, align='C')
    pdf.ln(10)
    
    # Section Patient (Tableau)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(190, 10, f"  PATIENT : {nom.upper()} {prenom.upper()}", ln=True, fill=True)
    pdf.set_font("Times", '', 11)
    
    # Liste de tous les paramètres dans le PDF
    data_list = [
        f"Age: {age} ans | IMC: {bmi} | Heredite: {family}",
        f"Tension: {sys_bp}/{dia_bp} mmHg | Chol: {chol} mg/dL | Pouls: {pulse} BPM",
        f"Tabac: {smoke} | Pas/jour: {steps} | Alcool: {alcohol} verres/sem",
        f"Sommeil: {sleep}h | Stress: {stress}/10 | Qualite Diete: {diet}/10"
    ]
    for line in data_list:
        pdf.cell(190, 8, line, border=1, ln=True)
    
    pdf.ln(10)
    
    # Verdict & Score
    pdf.set_font("Times", 'B', 16)
    if res_idx == 0: pdf.set_text_color(40, 167, 69)
    elif res_idx == 1: pdf.set_text_color(210, 150, 0)
    else: pdf.set_text_color(220, 53, 69)
    pdf.cell(190, 15, f"VERDICT IA : {labels[res_idx]} ({risk_score:.1f}%)", border=1, ln=True, align='C')
    
    # Recommandations détaillées sans accents pour éviter UnicodeError
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Times", 'B', 12)
    pdf.cell(190, 10, "RECOMMENDATIONS MEDICALES :", ln=True)
    pdf.set_font("Times", '', 12)
    
    recos_clean = [
        "Profil optimal. Maintenez une alimentation equilibree. Sport regulier conseille.",
        "Vigilance. Reduisez le sel et le sucre. Augmentez l activite physique. Bilan annuel requis.",
        "ALERTE : Consultez un cardiologue d urgence. Risque critique. Repos total recommande."
    ]
    pdf.multi_cell(190, 10, recos_clean[res_idx], border=1)
    
    pdf.ln(20)
    pdf.set_font("Times", 'I', 10)
    pdf.cell(190, 10, "Ce document est une aide au diagnostic issue de l intelligence artificielle.", align='C')

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button("📩 TELECHARGER LE BILAN COMPLET (PDF)", data=pdf_bytes, file_name=f"Bilan_Cardio_{nom}.pdf")
