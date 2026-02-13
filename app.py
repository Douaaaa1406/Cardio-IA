import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pytz

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", layout="wide")

# --- HEURE ALGERIE ---
timezone_dz = pytz.timezone('Africa/Algiers')
heure_algerie = datetime.now(timezone_dz).strftime("%d/%m/%Y %H:%M:%S")

# --- DESIGN : IMAGE DE FOND & CONTRASTE ---
# Note : Pour l'image, j'utilise un lien direct compatible pour le CSS
IMAGE_URL = "https://thumbs.dreamstime.com/b/coeur-anatomique-tube-don-sang-enroul%C3%A9-compteur-laboratoire-en-acier-inoxydable-honorer-journ%C3%A9e-mondiale-382446848.jpg"

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
        url("{IMAGE_URL}");
        background-size: cover;
        background-attachment: fixed;
    }}
    /* Bloc de formulaire très visible */
    [data-testid="stVerticalBlock"] > div:nth-child(1) {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 15px;
        border: 2px solid #003366;
    }}
    .main-title {{ 
        color: #FFFFFF; 
        text-align: center; 
        font-family: 'Arial Black';
        text-shadow: 3px 3px 5px #000000;
        background-color: rgba(0, 51, 102, 0.7);
        padding: 10px;
        border-radius: 10px;
    }}
    label {{ color: #001f3f !important; font-weight: bold !important; font-size: 16px !important; }}
    .stButton>button {{
        background-color: #cc0000 !important;
        color: white !important;
        font-weight: bold !important;
        height: 4em;
        width: 100%;
        border-radius: 10px;
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

st.markdown("<h1 class='main-title'>🔬 LABORATOIRE HOUBAD DOUAA - BILAN CARDIAQUE IA</h1>", unsafe_allow_html=True)
st.write(f"🕒 **HEURE LOCALE (ALGERIE) : {heure_algerie}**")

# --- FORMULAIRE ---
with st.form("form_labo"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("👤 Infos Personnelles")
        nom = st.text_input("NOM")
        prenom = st.text_input("PRENOM")
        age = st.number_input("AGE", 10, 110, 45)
        family = st.radio("HEREDITE CARDIAQUE", ["Non", "Oui"])
    with c2:
        st.subheader("🩺 Parametres Cliniques")
        sys_bp = st.number_input("TENSION SYSTOLIQUE", 80, 220, 120)
        dia_bp = st.number_input("TENSION DIASTOLIQUE", 40, 140, 80)
        chol = st.number_input("CHOLESTEROL (mg/dL)", 100, 450, 200)
        pulse = st.number_input("POULS (BPM)", 40, 160, 72)
    with c3:
        st.subheader("🏃 Mode de Vie")
        smoke = st.selectbox("TABAC", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("PAS / JOUR", 0, 30000, 5000)
        sleep = st.slider("SOMMEIL (Heures)", 3, 12, 7)
        stress = st.slider("STRESS (1-10)", 1, 10, 5)
        alcohol = st.number_input("ALCOOL (verres/sem)", 0, 50, 0)
        diet = st.slider("DIETE (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("ANALYSER ET GENERER LE RAPPORT")

if submitted:
    # Prediction
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    input_data = pd.DataFrame([[age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke], steps, stress, 3, sleep, (1 if family=="Oui" else 0), diet, alcohol]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    cats = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]
    colors_hex = ["#00FF00", "#FFFF00", "#FF0000"]
    
    st.markdown(f"<h2 style='text-align:center; color:{colors_hex[res_idx]}; background-color:rgba(0,0,0,0.7); padding:10px;'>RESULTAT : {cats[res_idx]} ({risk_score:.1f}%)</h2>", unsafe_allow_html=True)

    # --- GENERATION PDF PROFESSIONNEL ---
    pdf = FPDF()
    pdf.add_page()
    
    # Filigrane
    pdf.set_font("Arial", 'B', 40)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(20, 190, "LABORATOIRE HOUBAD DOUAA")
    pdf.rotate(0)

    # HEADER
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(190, 15, "RAPPORT D'ANALYSE CARDIAQUE - IA", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 10, f"Date: {heure_algerie} | Algerie", ln=True, align='C')
    pdf.ln(5)

    # PARTIE 1 : INFORMATIONS PERSONNELLES & PARAMETRES
    pdf.set_fill_color(230, 240, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, " 1. INFORMATIONS DU PATIENT ET CONSTANTES", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    
    col_w = 95
    pdf.cell(col_w, 10, f" Nom complet: {nom.upper()} {prenom.upper()}", border=1)
    pdf.cell(col_w, 10, f" Age: {age} ans", border=1, ln=True)
    pdf.cell(col_w, 10, f" Tension: {sys_bp}/{dia_bp} mmHg", border=1)
    pdf.cell(col_w, 10, f" Cholesterol: {chol} mg/dL", border=1, ln=True)
    pdf.cell(col_w, 10, f" Tabagisme: {smoke}", border=1)
    pdf.cell(col_w, 10, f" Pouls: {pulse} BPM", border=1, ln=True)
    pdf.cell(col_w, 10, f" Pas/jour: {steps}", border=1)
    pdf.cell(col_w, 10, f" Stress: {stress}/10", border=1, ln=True)
    pdf.ln(5)

    # PARTIE 2 : DECISION DE L'IA
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(255, 230, 230)
    pdf.cell(190, 10, " 2. ANALYSE ET DECISION DE L'IA", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 14)
    if res_idx == 0: pdf.set_text_color(0, 128, 0)
    elif res_idx == 1: pdf.set_text_color(200, 150, 0)
    else: pdf.set_text_color(200, 0, 0)
    
    pdf.cell(190, 15, f" VERDICT : {cats[res_idx]} | TAUX DE CERTITUDE : {risk_score:.1f}%", border=1, ln=True, align='C')
    pdf.ln(5)

    # PARTIE 3 : RECOMMENDATIONS MEDICALES
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(230, 255, 230)
    pdf.cell(190, 10, " 3. RECOMMENDATIONS MEDICALES", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    
    if res_idx == 0:
        reco = "PROFIL SAIN. Maintenez votre mode de vie actuel. Continuez une activite physique de 30 min/jour. Gardez une alimentation riche en fibres et faible en sel. Un controle annuel est suffisant."
    elif res_idx == 1:
        reco = "VIGILANCE. Il est imperatif de modifier votre hygene de vie. Reduisez le sucre et les graisses saturees. Augmentez votre activite physique. Pratiquez des exercices de respiration pour gerer le stress. Prenez rendez-vous pour un bilan lipidique complet."
    else:
        reco = "ALERTE CRITIQUE. Vous devez consulter un CARDIOLOGUE d'urgence. Arretez tout effort physique intense. Surveillez votre tension arterielle matin et soir. Ce resultat indique une surcharge cardiovasculaire necessitant une expertise clinique immediate."
    
    pdf.multi_cell(190, 10, reco, border=1)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(190, 10, "Document genere electroniquement par le systeme expert Houbad Douaa.", align='C')

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button("📩 TELECHARGER LE RAPPORT PDF OFFICIEL", data=pdf_bytes, file_name=f"Rapport_Houbad_{nom}.pdf", mime="application/pdf")
