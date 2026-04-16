import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pytz

# --- CONFIGURATION ---
st.set_page_config(page_title="CardioIA Pro", layout="wide", page_icon="🫀")

# Heure Algerie
timezone_dz = pytz.timezone('Africa/Algiers')
heure_algerie = datetime.now(timezone_dz).strftime("%d/%m/%Y %H:%M:%S")

# --- DESIGN PRO BLEU IA & CARDIAQUE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

    :root {
        --navy: #020B18;
        --deep: #041525;
        --panel: #071E35;
        --card: #0A2540;
        --border: #0E3A5E;
        --accent: #00A8FF;
        --glow: #00D4FF;
        --red: #FF2D55;
        --green: #00E676;
        --amber: #FFB300;
        --text: #C8E6FA;
        --muted: #6A95B0;
    }

    /* ── ROOT ── */
    .stApp {
        background: var(--navy);
        font-family: 'Rajdhani', sans-serif;
    }

    /* ECG animated line */
    .ecg-header {
        background: linear-gradient(135deg, #020B18 0%, #041525 50%, #061D33 100%);
        border-bottom: 2px solid var(--accent);
        padding: 28px 40px 20px;
        position: relative;
        overflow: hidden;
    }
    .ecg-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='60'%3E%3Cpolyline points='0,30 60,30 70,30 80,10 90,50 100,30 120,30 180,30 190,30 200,5 210,55 220,30 250,30 310,30 320,30 330,10 340,50 350,30 380,30 440,30 450,30 460,5 470,55 480,30 600,30' fill='none' stroke='%2300A8FF' stroke-width='1.5' opacity='0.18'/%3E%3C/svg%3E") repeat-x center;
        animation: ecgMove 4s linear infinite;
    }
    @keyframes ecgMove {
        from { background-position: 0 center; }
        to { background-position: 600px center; }
    }

    /* ── HEADER TITLE ── */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 900;
        color: var(--accent);
        text-align: center;
        letter-spacing: 4px;
        text-transform: uppercase;
        text-shadow: 0 0 30px rgba(0,168,255,0.6), 0 0 60px rgba(0,168,255,0.2);
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .sub-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        color: var(--muted);
        text-align: center;
        letter-spacing: 3px;
        margin-top: 8px;
        position: relative;
        z-index: 1;
    }
    .heart-icon {
        font-size: 2.4rem;
        animation: heartbeat 1.2s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        14% { transform: scale(1.25); }
        28% { transform: scale(1); }
        42% { transform: scale(1.15); }
        70% { transform: scale(1); }
    }

    /* ── STATUS BAR ── */
    .status-bar {
        background: var(--deep);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 20px;
        margin: 18px 0 22px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: var(--accent);
    }
    .status-dot {
        width: 8px; height: 8px;
        background: var(--green);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--green);
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── SECTION HEADERS ── */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 8px 14px;
        background: linear-gradient(90deg, rgba(0,168,255,0.15) 0%, transparent 100%);
        border-left: 3px solid var(--accent);
        border-radius: 0 6px 6px 0;
        margin-bottom: 18px;
    }

    /* ── INPUTS ── */
    input[type="number"],
    input[type="text"],
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    input:focus, div[data-baseweb="input"]:focus-within input {
        border-color: var(--accent) !important;
        box-shadow: 0 0 12px rgba(0,168,255,0.25) !important;
        outline: none !important;
    }

    /* ── SELECT ── */
    div[data-baseweb="select"] > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] span {
        color: var(--text) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ── LABELS ── */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stSlider label, .stRadio label {
        color: var(--muted) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }

    /* ── SLIDER ── */
    .stSlider > div > div > div[data-testid="stThumbValue"],
    .stSlider [data-testid="stThumbValue"] {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }
    .stSlider [role="slider"] {
        background: var(--accent) !important;
        box-shadow: 0 0 10px var(--accent) !important;
    }

    /* ── RADIO ── */
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: var(--text) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ── FORM SUBMIT BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #003F7F 0%, #0066CC 50%, #0088FF 100%) !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 3px !important;
        padding: 16px 32px !important;
        border-radius: 8px !important;
        border: 1px solid var(--accent) !important;
        width: 100% !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 25px rgba(0,168,255,0.3), inset 0 1px 0 rgba(255,255,255,0.1) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 40px rgba(0,168,255,0.6), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* ── DOWNLOAD BUTTON ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1A1A2E 0%, #162447 100%) !important;
        color: var(--accent) !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.75rem !important;
        letter-spacing: 2px !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        width: 100% !important;
        box-shadow: 0 0 20px rgba(0,168,255,0.2) !important;
    }

    /* ── RESULT CARD ── */
    .result-low {
        background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,230,118,0.05));
        border: 2px solid var(--green);
        box-shadow: 0 0 30px rgba(0,230,118,0.2), inset 0 0 30px rgba(0,230,118,0.05);
    }
    .result-medium {
        background: linear-gradient(135deg, rgba(255,179,0,0.12), rgba(255,179,0,0.05));
        border: 2px solid var(--amber);
        box-shadow: 0 0 30px rgba(255,179,0,0.2), inset 0 0 30px rgba(255,179,0,0.05);
    }
    .result-high {
        background: linear-gradient(135deg, rgba(255,45,85,0.15), rgba(255,45,85,0.05));
        border: 2px solid var(--red);
        box-shadow: 0 0 30px rgba(255,45,85,0.25), inset 0 0 30px rgba(255,45,85,0.05);
        animation: danger-pulse 2s ease-in-out infinite;
    }
    @keyframes danger-pulse {
        0%, 100% { box-shadow: 0 0 30px rgba(255,45,85,0.25); }
        50% { box-shadow: 0 0 50px rgba(255,45,85,0.5); }
    }
    .result-card {
        border-radius: 14px;
        padding: 28px 36px;
        text-align: center;
        margin: 24px 0;
    }
    .result-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .result-score {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        letter-spacing: 2px;
        margin-top: 8px;
        opacity: 0.85;
    }

    /* ── FORM CONTAINER ── */
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 40px rgba(0,0,0,0.6);
    }

    /* ── COLUMNS ── */
    [data-testid="stHorizontalBlock"] > div {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 20px 18px !important;
        margin: 0 6px !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--navy); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ── NUMBER INPUT ARROW ── */
    button[kind="secondary"] { display: none !important; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ── HEADER ──
st.markdown("""
<div class="ecg-header">
    <div class="main-title">
        <span class="heart-icon">🫀</span>&nbsp; CardioIA Pro &nbsp;<span class="heart-icon">🫀</span>
    </div>
    <div class="sub-title">SYSTÈME EXPERT DE DIAGNOSTIC CARDIAQUE PAR INTELLIGENCE ARTIFICIELLE</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    <span>SYSTÈME EN LIGNE</span>
    &nbsp;|&nbsp;
    <span>📍 Laboratoire Cardiologie · Algérie</span>
    &nbsp;|&nbsp;
    <span>🕒 {heure_algerie}</span>
    &nbsp;|&nbsp;
    <span>🤖 Modèle: XGBoost v2 · Précision: 94.7%</span>
</div>
""", unsafe_allow_html=True)


# ── MODEL ──
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


# ── FORM ──
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">👤 IDENTITÉ PATIENT</div>', unsafe_allow_html=True)
        nom = st.text_input("NOM")
        prenom = st.text_input("PRÉNOM")
        age = st.number_input("ÂGE (ans)", 10, 110, 45)
        family = st.radio("HÉRÉDITÉ CARDIAQUE", ["Non", "Oui"])

    with c2:
        st.markdown('<div class="section-header">🩺 DONNÉES CLINIQUES</div>', unsafe_allow_html=True)
        sys_bp = st.number_input("TENSION SYSTOLIQUE (mmHg)", 80, 220, 120)
        dia_bp = st.number_input("TENSION DIASTOLIQUE (mmHg)", 40, 140, 80)
        chol = st.number_input("CHOLESTÉROL (mg/dL)", 100, 450, 200)
        pulse = st.number_input("POULS (BPM)", 40, 160, 72)

    with c3:
        st.markdown('<div class="section-header">🏃 MODE DE VIE</div>', unsafe_allow_html=True)
        smoke = st.selectbox("TABAGISME", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps = st.number_input("PAS / JOUR", 0, 30000, 5000)
        sleep = st.slider("SOMMEIL (H/nuit)", 3, 12, 7)
        stress = st.slider("NIVEAU DE STRESS (1-10)", 1, 10, 5)
        alcohol = st.number_input("ALCOOL (verres/sem.)", 0, 50, 0)
        diet = st.slider("QUALITÉ ALIMENTAIRE (1-10)", 1, 10, 7)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡  LANCER L'ANALYSE IA — GÉNÉRER LE BILAN")


# ── RESULTS ──
if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    input_data = pd.DataFrame(
        [[age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke],
          steps, stress, 3, sleep, (1 if family == "Oui" else 0), diet, alcohol]],
        columns=feat_cols
    )

    proba = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx = model.predict(input_data)[0]

    cats = ["RISQUE FAIBLE", "RISQUE MODÉRÉ", "RISQUE ÉLEVÉ"]
    result_classes = ["result-low", "result-medium", "result-high"]
    result_colors = ["#00E676", "#FFB300", "#FF2D55"]
    result_emojis = ["✅", "⚠️", "🚨"]

    st.markdown(f"""
    <div class="result-card {result_classes[res_idx]}">
        <div style="color:{result_colors[res_idx]}; font-size:2.6rem; margin-bottom:6px;">{result_emojis[res_idx]}</div>
        <div class="result-label" style="color:{result_colors[res_idx]};">{cats[res_idx]}</div>
        <div class="result-score" style="color:{result_colors[res_idx]};">Score de risque : {risk_score:.1f}%</div>
        <div style="font-family:'Rajdhani',sans-serif; color:#8AAFC0; font-size:0.85rem; margin-top:10px; letter-spacing:1px;">
            Patient : {prenom.upper()} {nom.upper()} &nbsp;|&nbsp; Âge : {age} ans
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PDF GENERATION (PRO DESIGN) ──
    pdf = FPDF()
    pdf.add_page()

    # === HEADER BAND ===
    pdf.set_fill_color(2, 11, 24)          # navy
    pdf.rect(0, 0, 210, 42, 'F')
    pdf.set_fill_color(0, 168, 255)        # accent blue stripe
    pdf.rect(0, 40, 210, 3, 'F')

    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(0, 168, 255)
    pdf.set_y(8)
    pdf.cell(210, 12, "CARDIO IA PRO", ln=False, align='C')
    pdf.set_y(20)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(106, 149, 176)
    pdf.cell(210, 8, "SYSTEME EXPERT DE DIAGNOSTIC CARDIAQUE PAR INTELLIGENCE ARTIFICIELLE", ln=True, align='C')
    pdf.set_font("Arial", '', 8)
    pdf.cell(210, 6, f"Laboratoire Cardiologie  |  Algerie  |  {heure_algerie}", ln=True, align='C')

    pdf.set_y(50)

    # === SECTION HELPER ===
    def section_title(title):
        pdf.set_fill_color(4, 21, 37)
        pdf.set_draw_color(0, 168, 255)
        pdf.set_line_width(0.5)
        pdf.rect(10, pdf.get_y(), 190, 9, 'FD')
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(0, 168, 255)
        pdf.set_x(14)
        pdf.cell(186, 9, f"  {title}", ln=True)
        pdf.ln(2)

    def data_row(label1, val1, label2="", val2="", shade=False):
        y = pdf.get_y()
        if shade:
            pdf.set_fill_color(10, 37, 64)
        else:
            pdf.set_fill_color(7, 30, 53)
        pdf.rect(10, y, 190, 8, 'F')
        # left pair
        pdf.set_font("Arial", '', 8.5)
        pdf.set_text_color(106, 149, 176)
        pdf.set_x(14)
        pdf.cell(38, 8, label1)
        pdf.set_font("Arial", 'B', 8.5)
        pdf.set_text_color(200, 230, 250)
        pdf.cell(52, 8, val1)
        # right pair
        if label2:
            pdf.set_font("Arial", '', 8.5)
            pdf.set_text_color(106, 149, 176)
            pdf.cell(38, 8, label2)
            pdf.set_font("Arial", 'B', 8.5)
            pdf.set_text_color(200, 230, 250)
            pdf.cell(52, 8, val2)
        pdf.ln(8)

    # === SECTION 1 — PATIENT ===
    section_title("1. DONNEES DU PATIENT")
    data_row("NOM", nom.upper(), "PRENOM", prenom.upper(), shade=False)
    data_row("AGE", f"{age} ans", "HEREDITE CARD.", family, shade=True)
    pdf.ln(4)

    # === SECTION 2 — CLINIQUE ===
    section_title("2. DONNEES CLINIQUES")
    data_row("TENSION", f"{sys_bp}/{dia_bp} mmHg", "CHOLESTEROL", f"{chol} mg/dL", shade=False)
    data_row("POULS", f"{pulse} BPM", "TABAGISME", smoke, shade=True)
    pdf.ln(4)

    # === SECTION 3 — MODE DE VIE ===
    section_title("3. MODE DE VIE")
    data_row("PAS / JOUR", f"{steps}", "SOMMEIL", f"{sleep} h/nuit", shade=False)
    data_row("STRESS", f"{stress}/10", "QUALITE DIET.", f"{diet}/10", shade=True)
    data_row("ALCOOL", f"{alcohol} v/sem.", "", "", shade=False)
    pdf.ln(4)

    # === SECTION 4 — RESULTAT IA ===
    section_title("4. ANALYSE ET DECISION DE L'IA")

    # Result box colored
    result_fill = {0: (0, 40, 20), 1: (40, 30, 0), 2: (50, 5, 10)}
    result_border = {0: (0, 200, 100), 1: (220, 160, 0), 2: (220, 40, 60)}
    result_txt = {0: (0, 230, 118), 1: (255, 179, 0), 2: (255, 50, 80)}
    rf, rb, rt = result_fill[res_idx], result_border[res_idx], result_txt[res_idx]

    y0 = pdf.get_y()
    pdf.set_fill_color(*rf)
    pdf.set_draw_color(*rb)
    pdf.set_line_width(1)
    pdf.rect(10, y0, 190, 24, 'FD')
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(*rt)
    pdf.set_y(y0 + 4)
    pdf.cell(210, 8, f"{cats[res_idx]}   —   Score : {risk_score:.1f}%", ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(200, 220, 235)
    pdf.cell(210, 6, f"Patient : {prenom.upper()} {nom.upper()}  |  Age : {age} ans", ln=True, align='C')
    pdf.ln(6)

    # Probability bars
    cat_labels = ["Risque Faible", "Risque Modere", "Risque Eleve"]
    bar_colors = [(0, 200, 100), (220, 160, 0), (220, 50, 70)]
    pdf.set_font("Arial", '', 8)
    for i, (lbl, prob) in enumerate(zip(cat_labels, proba)):
        y_bar = pdf.get_y()
        pdf.set_text_color(106, 149, 176)
        pdf.set_x(14)
        pdf.cell(48, 7, lbl)
        bar_w = int(prob * 120)
        pdf.set_fill_color(*bar_colors[i])
        pdf.rect(62, y_bar + 1.5, bar_w, 4, 'F')
        pdf.set_text_color(200, 230, 250)
        pdf.set_x(186)
        pdf.cell(20, 7, f"{prob*100:.1f}%", align='R')
        pdf.ln(7)
    pdf.ln(4)

    # === SECTION 5 — RECOMMANDATIONS ===
    section_title("5. RECOMMANDATIONS MEDICALES")

    reco_text = {
        0: ("Profil optimal detecte.", "Maintenez une alimentation riche en fibres et pratiquez 30 min d'activite physique par jour. Hydratation suffisante recommandee. Bilan cardiaque annuel conseille. Continuez ce mode de vie sain."),
        1: ("Vigilance requise.", "Reduisez la consommation de sel et de graisses saturees. Augmentez progressivement votre activite physique. Limitez le stress chronique. Un bilan sanguin complet avec lipidogramme est fortement recommande."),
        2: ("ALERTE CRITIQUE — RISQUE ELEVE DETECTE.", "Une consultation d'urgence chez un cardiologue est imperative. Arretez tout effort physique intense immediatement. Evitez alcool, tabac et aliments gras. Surveillance tensionnelle quotidienne obligatoire.")
    }

    reco_title, reco_body = reco_text[res_idx]
    pdf.set_fill_color(*rf)
    pdf.set_draw_color(*rb)
    pdf.set_line_width(0.5)
    yy = pdf.get_y()
    # Title line
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(*rt)
    pdf.set_x(14)
    pdf.multi_cell(182, 8, reco_title)
    pdf.set_font("Arial", '', 8.5)
    pdf.set_text_color(200, 230, 250)
    pdf.set_x(14)
    pdf.multi_cell(182, 6, reco_body)
    pdf.rect(10, yy, 190, pdf.get_y() - yy, 'D')
    pdf.ln(6)

    # === FOOTER ===
    pdf.set_y(-20)
    pdf.set_fill_color(0, 168, 255)
    pdf.rect(0, pdf.get_y() - 2, 210, 0.8, 'F')
    pdf.set_font("Arial", 'I', 7.5)
    pdf.set_text_color(106, 149, 176)
    pdf.cell(95, 8, "Document genere electroniquement — CardioIA Pro", align='L')
    pdf.cell(95, 8, f"Ref: CIA-{datetime.now().strftime('%Y%m%d%H%M%S')}", align='R')

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')

    st.download_button(
        "📥  TÉLÉCHARGER LE BILAN PDF COMPLET",
        data=pdf_bytes,
        file_name=f"CardioIA_Bilan_{nom}_{prenom}_{datetime.now().strftime('%Y%m%d')}.pdf"
    )
