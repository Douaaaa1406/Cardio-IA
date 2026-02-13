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

# --- HEURE ALGERIE ---
timezone_dz = pytz.timezone('Africa/Algiers')
heure_algerie = datetime.now(timezone_dz).strftime("%d/%m/%Y %H:%M:%S")

# --- CSS AVANCÉ : BACKGROUND MEDICAL & DESIGN ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.88)), 
        url("https://images.unsplash.com/photo-1576091160550-2173dad99901?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-title { color: #003366; font-weight: bold; text-align: center; border-bottom: 3px solid #cc0000; }
    .stButton>button {
        background-color: #003366; color: white; border-radius: 8px;
        font-weight: bold; width: 100%; border: none; height: 3em;
    }
    /* Style pour les colonnes de saisie */
    [data-testid="stVerticalBlock"] { background-color: rgba(255, 255, 255, 0.5); border-radius: 10px; padding: 10px; }
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

st.markdown("<h1 class='main-title'>🔬 LABORATOIRE HOUBAD DOUAA - IA CARDIAQUE</h1>", unsafe_allow_html=True)
st.write(f"📍 **Algerie** | 🕒 **{heure_algerie}**")

with st.form("expert_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("👤 Identite")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prenom")
        age = st.number_input("Age (Min 10)", min_value=10, max_value=115, value=45)
        family = st.radio("Heredite", ["Non", "Oui"])
    with c2:
        st.subheader("🩺 Clinique")
        sys_bp = st.number_input("Tension Systolique", 80, 220, 125)
        dia_bp = st.number_input("Tension Diastolique", 40, 140, 85)
        chol = st.number_input("Cholesterol", 100, 450, 200)
    with c3:
        st.subheader("🏃 Lifestyle")
        smoke = st.selectbox("Tabac", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("Pas/jour", 0, 30000, 6000)
        stress = st.slider("Stress (1-10)", 1, 10, 5)

    submitted = st.form_submit_button("💥 ANALYSER LE RISQUE")

if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    # Données avec valeurs par défaut pour les champs masqués
    input_data = pd.DataFrame([[age, 25.0, sys_bp, dia_bp, chol, 72, m_smoke[smoke], steps, stress, 3, 7, (1 if family=="Oui" else 0), 7, 0]], columns=feat_cols)
    
    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]
    
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    labels = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]

    # --- PETIT GRAPHIQUE DEMI-CERCLE ---
    col_left, col_mid, col_right = st.columns([1, 1, 1])
    with col_mid:
        fig, ax = plt.subplots(figsize=(3, 1.5), subplot_kw={'projection': 'polar'})
        val = (risk_score / 100) * np.pi
        ax.barh(0, val, color=colors[res_idx], align='center')
        ax.set_xlim(0, np.pi)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        plt.text(np.pi/2, 0.2, f"{risk_score:.1f}%", ha='center', fontsize=12, fontweight='bold', color=colors[res_idx])
        st.pyplot(fig)

    st.markdown(f"<h3 style='text-align:center; color:{colors[res_idx]};'>{labels[res_idx]}</h3>", unsafe_allow_html=True)

    # --- PDF SANS ACCENTS ---
    instr_pdf = [
        "Profil optimal. Continuez ce mode de vie sain. Sport et dietetique recommandes.",
        "Vigilance requise. Ameliorez votre hygene de vie et gerez votre stress.",
        "DANGER : CONSULTEZ UN CARDIOLOGUE D URGENCE. Arret de tout effort intense."
    ]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 30)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(10, 190, "LABORATOIRE HOUBAD DOUAA - ALGERIE")
    pdf.rotate(0)
    
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(190, 15, "RAPPORT IA CARDIAQUE", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, f"PATIENT : {nom.upper()} {prenom.upper()}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(190, 10, f"Age: {age} | Tension: {sys_bp}/{dia_bp} | Risque: {risk_score:.1f}%", border=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 12, f"CONCLUSION : {labels[res_idx]}", border=1, ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(190, 10, instr_pdf[res_idx])

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button("📩 TELECHARGER LE RAPPORT PDF", data=pdf_bytes, file_name=f"Rapport_{nom}.pdf")
