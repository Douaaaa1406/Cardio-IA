import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pytz

# --- CONFIGURATION ---
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", layout="wide")

# Heure Algerie
timezone_dz = pytz.timezone('Africa/Algiers')
heure_algerie = datetime.now(timezone_dz).strftime("%d/%m/%Y %H:%M:%S")

# --- DESIGN "SOFT GRAY" (NI CLAIR NI SOMBRE) ---
IMAGE_URL = "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"

st.markdown(f"""
    <style>
    .stApp {{
        /* Fond Gris Perle avec transparence ajustée pour voir l'image sans être ébloui */
        background: linear-gradient(rgba(235, 235, 235, 0.82), rgba(235, 235, 235, 0.82)), 
        url("{IMAGE_URL}");
        background-size: cover;
        background-attachment: fixed;
    }}
    
    .main-title {{ 
        color: #002244; 
        text-align: center; 
        font-family: 'Arial Black';
        border-bottom: 5px solid #b30000;
        margin-bottom: 25px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }}

    /* Zones de saisie foncées pour un contraste net sur fond gris */
    input, select, textarea, [data-baseweb="select"], [data-baseweb="input"] {{
        background-color: #d1d8e0 !important; 
        border: 2px solid #003366 !important;
        color: #000000 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }}

    /* Style des sous-titres et labels */
    h3, label {{ 
        color: #001f3f !important; 
        font-weight: 800 !important; 
    }}

    /* Bouton d'action vif */
    .stButton>button {{
        background-color: #cc0000 !important;
        color: white !important;
        font-weight: bold !important;
        height: 3.8em;
        width: 100%;
        border-radius: 12px;
        font-size: 19px !important;
        box-shadow: 3px 3px 12px rgba(0,0,0,0.3);
        border: 1px solid white;
    }}
    
    /* Conteneur des colonnes pour les rendre plus "solides" */
    [data-testid="stVerticalBlock"] > div {{
        background-color: rgba(255, 255, 255, 0.4);
        padding: 15px;
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

st.markdown("<h1 class='main-title'>BILAN DE SANTÉ CARDIAQUE IA</h1>", unsafe_allow_html=True)
st.write(f"📍 **Laboratoire Houbad Douaa - Algerie** | 🕒 **{heure_algerie}**")

with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("👤 Identité")
        nom = st.text_input("NOM")
        prenom = st.text_input("PRÉNOM")
        age = st.number_input("ÂGE", 10, 110, 45)
        family = st.radio("HÉRÉDITÉ CARDIAQUE", ["Non", "Oui"])
    with c2:
        st.subheader("🩺 Clinique")
        sys_bp = st.number_input("TENSION SYSTOLIQUE", 80, 220, 120)
        dia_bp = st.number_input("TENSION DIASTOLIQUE", 40, 140, 80)
        chol = st.number_input("CHOLESTÉROL (mg/dL)", 100, 450, 200)
        pulse = st.number_input("POULS (BPM)", 40, 160, 72)
    with c3:
        st.subheader("🏃 Mode de Vie")
        smoke = st.selectbox("TABAGISME", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("STEPS / JOUR", 0, 30000, 5000)
        sleep = st.slider("SOMMEIL (H/nuit)", 3, 12, 7)
        stress = st.slider("STRESS (1-10)", 1, 10, 5)
        alcohol = st.number_input("ALCOOL (v/sem)", 0, 50, 0)
        diet = st.slider("ALIMENTATION (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("ANALYSER ET GÉNÉRER RAPPORT")

if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    input_data = pd.DataFrame([[age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke], steps, stress, 3, sleep, (1 if family=="Oui" else 0), diet, alcohol]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    cats = ["RISQUE FAIBLE", "RISQUE MODÉRÉ", "RISQUE ÉLEVÉ"]
    colors = ["#28a745", "#ffc107", "#dc3545"]
    
    st.markdown(f"<h2 style='text-align:center; color:white; background-color:{colors[res_idx]}; padding:15px; border-radius:10px; border:2px solid black;'>RÉSULTAT : {cats[res_idx]} ({risk_score:.1f}%)</h2>", unsafe_allow_html=True)

    # --- PDF GÉNÉRATION (FORMAT FINAL) ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 15, "BILAN DE SANTE CARDIAQUE IA", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 10, f"Laboratoire Houbad Douaa - Algerie | {heure_algerie}", ln=True, align='C')
    pdf.ln(5)

    # PARTIE 1
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, " 1. DONNEES DU PATIENT ET MODE DE VIE", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    
    data = [
        ["NOM", nom.upper(), "ALCOOL", f"{alcohol} v/sem"],
        ["PRENOM", prenom.upper(), "PAS / JOUR", f"{steps}"],
        ["AGE", f"{age} ans", "SOMMEIL", f"{sleep} h/nuit"],
        ["TENSION", f"{sys_bp}/{dia_bp}", "STRESS", f"{stress}/10"],
        ["CHOLESTEROL", f"{chol} mg/dL", "DIETE", f"{diet}/10"],
        ["POULS", f"{pulse} BPM", "TABAC", smoke],
        ["HEREDITE", family, "", ""]
    ]
    for row in data:
        pdf.cell(45, 8, row[0], border=1)
        pdf.cell(50, 8, row[1], border=1)
        pdf.cell(45, 8, row[2], border=1)
        pdf.cell(50, 8, row[3], border=1, ln=True)
    pdf.ln(10)

    # PARTIE 2
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, " 2. ANALYSE ET DECISION DE L'IA", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 15, f" RESULTAT : {cats[res_idx]} ({risk_score:.1f}%)", border=1, ln=True, align='C')
    pdf.ln(10)

    # PARTIE 3
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, " 3. RECOMMENDATIONS MEDICALES", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    
    reco = {
        0: "Profil optimal. Maintenez une alimentation riche en fibres et pratiquez une activite physique reguliere (30 min/jour). Bilan annuel conseille.",
        1: "Vigilance requise. Reduisez la consommation de sel et de graisses saturees. Augmentez votre activite physique. Un bilan sanguin complet est recommande.",
        2: "ALERTE : Risque eleve detecte. Une consultation chez un cardiologue est imperatve d urgence. Arretez tout effort physique intense."
    }
    pdf.multi_cell(190, 10, reco[res_idx], border=1)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 10, "Document genere electroniquement par le systeme expert Houbad Douaa", align='C')

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button("📩 TÉLÉCHARGER LE BILAN PDF", data=pdf_bytes, file_name=f"Bilan_Houbad_{nom}.pdf")
