import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from fpdf import FPDF
from datetime import datetime
import pytz
import io

# Configuration de la page
st.set_page_config(page_title="LABORATOIRE HOUBAD DOUAA", layout="wide")

# --- CHARGEMENT ET ENTRAÎNEMENT ---
@st.cache_resource
def train_model():
    # Chargement du fichier que tu as mis sur GitHub
    df = pd.read_csv("cardiovascular_risk_numeric.csv")
    mapping = {'Never': 0, 'Former': 1, 'Current': 2}
    df['smoking_status'] = df['smoking_status'].map(mapping)
    
    X = df.drop(['Patient_ID', 'heart_disease_risk_score', 'risk_category'], axis=1)
    y = df['risk_category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, X.columns, acc, kappa, cm

model, feat_cols, acc, kappa, cm = train_model()

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center; color: #003366;'>🔬 LABORATOIRE HOUBAD DOUAA</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Système Expert de Prédiction Cardiaque (XGBoost)</h3>", unsafe_allow_html=True)

# Barre latérale pour la fiabilité
with st.sidebar:
    st.header("📊 Fiabilité du Modèle")
    st.metric("Précision Globale", f"{acc*100:.2f}%")
    st.metric("Score Kappa", f"{kappa:.3f}")
    
    st.write("---")
    st.write("### Importance des Paramètres")
    fig_imp, ax_imp = plt.subplots()
    importances = model.feature_importances_
    pd.Series(importances, index=feat_cols).sort_values().plot(kind='barh', color='#0077b6', ax=ax_imp)
    st.pyplot(fig_imp)

# Formulaire de saisie
with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        nom = st.text_input("Nom du Patient")
        prenom = st.text_input("Prénom du Patient")
        age = st.number_input("Âge", 18, 100, 45)
        bmi = st.number_input("IMC (BMI)", 10.0, 50.0, 25.0)
        smoke = st.selectbox("Statut Tabagique", ["Jamais", "Ex-fumeur", "Fumeur"])
    with col2:
        sys_bp = st.number_input("Tension Systolique (mmHg)", 80, 200, 120)
        dia_bp = st.number_input("Tension Diastolique (mmHg)", 40, 130, 80)
        chol = st.number_input("Cholestérol (mg/dL)", 100, 400, 190)
        family = st.radio("Antécédents Familiaux", ["Non", "Oui"])
        stress = st.slider("Niveau de Stress (1-10)", 1, 10, 5)

    submitted = st.form_submit_button("🔍 LANCER L'ANALYSE IA")

if submitted:
    # Préparation des données pour XGBoost
    mapping_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    # On complète avec des valeurs moyennes pour les champs restants
    input_df = pd.DataFrame([[
        age, bmi, sys_bp, dia_bp, chol, 72, mapping_smoke[smoke], 
        7000, stress, 3, 7, (1 if family == "Oui" else 0), 7, 0
    ]], columns=feat_cols)
    
    res_idx = model.predict(input_df)[0]
    categories = ["✅ RISQUE FAIBLE", "⚠️ RISQUE MODÉRÉ", "🚨 RISQUE ÉLEVÉ"]
    resultat = categories[res_idx]
    
    st.success(f"Résultat de l'IA : {resultat}")

    # GÉNÉRATION DU PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Filigrane
    pdf.set_font("Arial", 'B', 30)
    pdf.set_text_color(240, 240, 240)
    pdf.rotate(45, 100, 100)
    pdf.text(20, 190, "LABORATOIRE HOUBAD DOUAA")
    pdf.rotate(0)
    
    # Contenu
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, "RAPPORT MEDICAL DE PREDICTION CARDIAQUE", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient : {nom.upper()} {prenom.upper()}", ln=True)
    pdf.cell(100, 10, f"Date : {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 12, f"VERDICT IA : {resultat}", border=1, ln=True, align='C')
    
    pdf_output = f"Rapport_{nom}.pdf"
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    
    st.download_button("📩 Télécharger le Rapport Médical PDF", data=pdf_bytes, file_name=pdf_output, mime="application/pdf")
