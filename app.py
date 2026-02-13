import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fpdf import FPDF
from datetime import datetime
import pytz

# Configuration de la page
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", page_icon="🔬", layout="wide")

@st.cache_resource
def train_model():
    df = pd.read_csv("cardiovascular_risk_numeric.csv")
    mapping = {'Never': 0, 'Former': 1, 'Current': 2}
    df['smoking_status'] = df['smoking_status'].map(mapping)
    X = df.drop(['Patient_ID', 'heart_disease_risk_score', 'risk_category'], axis=1)
    y = df['risk_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, acc

model, feat_cols, acc = train_model()

st.markdown("<h1 style='text-align: center; color: #003366;'>🔬 LABORATOIRE HOUBAD DOUAA</h1>", unsafe_allow_html=True)
st.write(f"**Fiabilite du systeme : {acc*100:.2f}%**")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Identite")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prenom")
        age = st.number_input("Age", 18, 100, 45)
        family = st.radio("Heredite Cardiaque", ["Non", "Oui"])
    with col2:
        st.subheader("🩺 Constantes")
        sys_bp = st.number_input("Tension Systolique", 80, 200, 120)
        dia_bp = st.number_input("Tension Diastolique", 40, 130, 80)
        chol = st.number_input("Cholesterol", 100, 400, 200)
        pulse = st.number_input("Pouls (BPM)", 40, 150, 72)
    with col3:
        st.subheader("🏃 Mode de Vie")
        smoke = st.selectbox("Tabac", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("Pas par jour", 0, 30000, 7000)
        sleep = st.slider("Sommeil (h/nuit)", 4, 12, 7)
        stress = st.slider("Stress (1-10)", 1, 10, 5)

    submitted = st.form_submit_button("🔍 ANALYSER ET GENERER LE RAPPORT")

if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    # Prédiction (complétée avec valeurs par défaut pour les champs cachés)
    input_data = pd.DataFrame([[age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke], steps, stress, 3, sleep, (1 if family=="Oui" else 0), 7, 0]], columns=feat_cols)
    res_idx = model.predict(input_data)[0]
    
    cats = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]
    colors = ["#28a745", "#ffc107", "#dc3545"] # Vert, Jaune, Rouge
    instructions = [
        "Continuez ce mode de vie sain.",
        "Essayer de corriger votre mode de vie (alimentation, sport).",
        "VEUILLEZ VISITER UN CARDIOLOGUE LE PLUS TOT POSSIBLE."
    ]
    
    res_text = cats[res_idx]
    instr_text = instructions[res_idx]

    st.markdown(f"<h2 style='text-align:center; color:{colors[res_idx]};'>{res_text}</h2>", unsafe_allow_html=True)
    st.info(instr_text)

    # --- GENERATION DU PDF PROFESSIONNEL ---
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Filigrane (Watermark)
    pdf.set_font("Arial", 'B', 35)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(15, 190, "LABORATOIRE HOUBAD DOUAA - IA")
    pdf.rotate(0)

    # 2. Logo et Entête
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(0, 51, 102) # Bleu Marine
    pdf.cell(190, 15, "LABORATOIRE HOUBAD DOUAA", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 5, f"Date de l'analyse : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # 3. Informations Patient (Cadre Bleu)
    pdf.set_fill_color(230, 240, 250)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, f" RAPPORT DU PATIENT : {nom.upper()} {prenom.upper()}", ln=True, fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(95, 8, f" Age : {age} ans", border=1)
    pdf.cell(95, 8, f" Heredite : {family}", border=1, ln=True)
    pdf.cell(95, 8, f" Tension : {sys_bp}/{dia_bp} mmHg", border=1)
    pdf.cell(95, 8, f" Cholesterol : {chol} mg/dL", border=1, ln=True)
    pdf.cell(95, 8, f" Tabagisme : {smoke}", border=1)
    pdf.cell(95, 8, f" Pouls : {pulse} BPM", border=1, ln=True)
    pdf.ln(10)

    # 4. RESULTAT IA (Couleur selon le risque)
    if res_idx == 0: pdf.set_text_color(40, 167, 69) # Vert
    elif res_idx == 1: pdf.set_text_color(210, 150, 0) # Orange
    else: pdf.set_text_color(220, 53, 69) # Rouge
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 15, f"RESULTAT IA : {res_text}", border=1, ln=True, align='C')
    
    # 5. Instructions Médicales
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "INSTRUCTIONS ET CONSEILS :", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(190, 10, instr_text, border=1)

    pdf_output = pdf.output(dest='S').encode('latin-1')
    st.download_button("📩 TELECHARGER LE RAPPORT PDF OFFICIEL", data=pdf_output, file_name=f"Rapport_{nom}.pdf", mime="application/pdf")
