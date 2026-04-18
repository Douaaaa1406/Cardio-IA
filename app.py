import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pytz
import time
import unicodedata
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="CardioIA Pro", layout="wide", page_icon="🫀")

# --- HELPER: strip accents for PDF ---
def safe(text):
    """Convert any string to latin-1 safe ASCII for FPDF."""
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- LIGHT ELEGANT THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Montserrat:wght@300;400;500;600;700&family=Playfair+Display:wght@700;900&display=swap');

    :root {
        --cream: #F8F5F0;
        --white: #FFFFFF;
        --light-gray: #F2EEE9;
        --border: #D8CFC4;
        --soft: #EBE5DC;
        --navy: #0B1F3A;
        --blue-mid: #1A3A5C;
        --blue-light: #2E6DA4;
        --accent: #C8102E;
        --text-dark: #1A1A2E;
        --text-mid: #3D4A5C;
        --text-muted: #8A97A8;
        --gold: #B8972A;
    }

    .stApp {
        background: var(--cream);
        font-family: 'Montserrat', sans-serif;
    }

    /* FLOATING HEARTS */
    .hearts-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .heart-float {
        position: absolute;
        bottom: -60px;
        font-size: 18px;
        opacity: 0;
        animation: floatUp linear infinite;
        color: #C8102E;
    }
    .heart-float:nth-child(1)  { left: 5%;  animation-duration: 13s; animation-delay: 0s;   font-size: 14px; }
    .heart-float:nth-child(2)  { left: 13%; animation-duration: 16s; animation-delay: 2s;   font-size: 20px; }
    .heart-float:nth-child(3)  { left: 22%; animation-duration: 11s; animation-delay: 4s;   font-size: 11px; }
    .heart-float:nth-child(4)  { left: 33%; animation-duration: 14s; animation-delay: 1s;   font-size: 16px; }
    .heart-float:nth-child(5)  { left: 45%; animation-duration: 13s; animation-delay: 6s;   font-size: 22px; }
    .heart-float:nth-child(6)  { left: 55%; animation-duration: 17s; animation-delay: 3s;   font-size: 10px; }
    .heart-float:nth-child(7)  { left: 65%; animation-duration: 12s; animation-delay: 5s;   font-size: 18px; }
    .heart-float:nth-child(8)  { left: 76%; animation-duration: 15s; animation-delay: 7s;   font-size: 14px; }
    .heart-float:nth-child(9)  { left: 85%; animation-duration: 10s; animation-delay: 2s;   font-size: 20px; }
    .heart-float:nth-child(10) { left: 93%; animation-duration: 14s; animation-delay: 9s;   font-size: 13px; }
    .heart-float:nth-child(11) { left: 18%; animation-duration: 18s; animation-delay: 11s;  font-size: 16px; }
    .heart-float:nth-child(12) { left: 38%; animation-duration: 15s; animation-delay: 8s;   font-size: 9px; }

    @keyframes floatUp {
        0%   { transform: translateY(0) rotate(-12deg); opacity: 0; }
        8%   { opacity: 0.20; }
        92%  { opacity: 0.10; }
        100% { transform: translateY(-115vh) rotate(12deg); opacity: 0; }
    }

    /* HEADER */
    .main-header {
        background: linear-gradient(135deg, #0B1F3A 0%, #1A3A5C 55%, #1E4976 100%);
        border-radius: 20px;
        padding: 36px 40px 28px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 50px rgba(11,31,58,0.22);
    }
    .main-header::before {
        content: '\u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661';
        position: absolute;
        top: 9px; left: 0; right: 0;
        font-size: 10px; color: rgba(255,255,255,0.07);
        letter-spacing: 6px; text-align: center; font-family: serif;
    }
    .main-header::after {
        content: '\u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665  \u2661  \u2665';
        position: absolute;
        bottom: 9px; left: 0; right: 0;
        font-size: 10px; color: rgba(255,255,255,0.07);
        letter-spacing: 6px; text-align: center; font-family: serif;
    }
    .header-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.55rem;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 2px 20px rgba(255,255,255,0.12);
        margin: 0;
        position: relative; z-index: 1;
    }
    .header-subtitle {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.73rem;
        color: rgba(255,255,255,0.50);
        text-align: center;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 10px;
        font-weight: 300;
        position: relative; z-index: 1;
    }
    .beat-heart {
        font-size: 2.6rem;
        display: inline-block;
        animation: heartbeat 1.3s ease-in-out infinite;
        filter: drop-shadow(0 0 14px rgba(220,50,70,0.85));
    }
    @keyframes heartbeat {
        0%   { transform: scale(1.0); }
        14%  { transform: scale(1.32); }
        28%  { transform: scale(1.0); }
        42%  { transform: scale(1.18); }
        70%  { transform: scale(1.0); }
        100% { transform: scale(1.0); }
    }

    /* CLOCK BAR */
    .clock-bar {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 50px;
        padding: 11px 30px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    }
    .clock-left {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.70rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 2px;
        text-transform: uppercase;
        display: flex; align-items: center; gap: 8px;
    }
    .clock-dot {
        width: 8px; height: 8px;
        background: #28C76F;
        border-radius: 50%;
        box-shadow: 0 0 10px #28C76F;
        animation: blink 1s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .clock-time {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--navy);
        letter-spacing: 3px;
    }
    .clock-right {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.70rem;
        color: var(--text-muted);
        letter-spacing: 1px;
    }

    /* SECTION LABELS */
    .section-label {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--blue-light);
        border-bottom: 2px solid var(--soft);
        padding-bottom: 10px;
        margin-bottom: 16px;
        display: flex; align-items: center; gap: 8px;
    }
    .section-label::before { content: '\u2665'; color: var(--accent); font-size: 11px; }

    /* INPUTS */
    input[type="number"], input[type="text"],
    div[data-baseweb="input"] input, textarea {
        background: var(--light-gray) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-dark) !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        transition: all 0.2s !important;
    }
    input:focus {
        border-color: var(--blue-light) !important;
        background: var(--white) !important;
        box-shadow: 0 0 0 3px rgba(46,109,164,0.12) !important;
    }

    /* SELECT */
    div[data-baseweb="select"] > div {
        background: var(--light-gray) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] span {
        color: var(--text-dark) !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
    }

    /* LABELS */
    label, .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stSlider label, .stRadio label {
        color: var(--text-mid) !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: var(--text-dark) !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    /* SUBMIT BUTTON */
    .stButton > button {
        background: linear-gradient(135deg, #0B1F3A 0%, #1A3A5C 50%, #2E6DA4 100%) !important;
        color: white !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.80rem !important;
        letter-spacing: 3px !important;
        padding: 18px 36px !important;
        border-radius: 50px !important;
        border: none !important;
        width: 100% !important;
        text-transform: uppercase !important;
        box-shadow: 0 8px 30px rgba(11,31,58,0.28) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 14px 40px rgba(11,31,58,0.40) !important;
    }

    /* DOWNLOAD BUTTON */
    .stDownloadButton > button {
        background: var(--white) !important;
        color: var(--navy) !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 2px !important;
        border: 2px solid var(--navy) !important;
        border-radius: 50px !important;
        padding: 16px 32px !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(11,31,58,0.10) !important;
        transition: all 0.3s !important;
    }
    .stDownloadButton > button:hover {
        background: var(--navy) !important;
        color: white !important;
        transform: translateY(-2px) !important;
    }

    /* RESULT CARD */
    .result-wrap {
        border-radius: 20px;
        padding: 36px 40px;
        text-align: center;
        margin: 28px 0;
    }
    .result-low    { background: linear-gradient(135deg,#E8FAF0,#D0F4E3); border:2px solid #28C76F; box-shadow:0 12px 40px rgba(40,199,111,0.18); }
    .result-medium { background: linear-gradient(135deg,#FFF8E6,#FFF0C0); border:2px solid #FFB300; box-shadow:0 12px 40px rgba(255,179,0,0.18); }
    .result-high   { background: linear-gradient(135deg,#FEE8EC,#FDD0D8); border:2px solid #C8102E; box-shadow:0 12px 40px rgba(200,16,46,0.22); animation:result-pulse 2s ease-in-out infinite; }
    @keyframes result-pulse {
        0%,100% { box-shadow:0 12px 40px rgba(200,16,46,0.22); }
        50%     { box-shadow:0 18px 60px rgba(200,16,46,0.42); }
    }
    .result-main-label {
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 900; letter-spacing: 2px;
    }
    .result-score-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.95rem; font-weight: 600; letter-spacing: 1px;
        margin-top: 8px; opacity: 0.82;
    }
    .result-patient {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.1rem; color: var(--text-mid);
        margin-top: 12px; font-style: italic;
    }

    /* FORM & COLUMNS */
    div[data-testid="stForm"] { background:transparent; border:none; padding:0; }
    [data-testid="column"] {
        background: var(--white) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 24px 20px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
    }

    /* MISC */
    #MainMenu, footer, header { visibility: hidden; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--cream); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--blue-light); }
    </style>

    <div class="hearts-bg">
        <div class="heart-float">&#9829;</div>
        <div class="heart-float">&#9825;</div>
        <div class="heart-float">&#9829;</div>
        <div class="heart-float">&#9825;</div>
        <div class="heart-float">&#9829;</div>
        <div class="heart-float">&#9825;</div>
        <div class="heart-float">&#9829;</div>
        <div class="heart-float">&#9825;</div>
        <div class="heart-float">&#9829;</div>
        <div class="heart-float">&#9825;</div>
        <div class="heart-float">&#9829;</div>
        <div class="heart-float">&#9825;</div>
    </div>
""", unsafe_allow_html=True)


# ── HEADER ──
st.markdown("""
<div class="main-header">
    <div class="header-title">
        <span class="beat-heart">&#129706;</span>&nbsp; CardioIA Pro &nbsp;<span class="beat-heart">&#129706;</span>
    </div>
    <div class="header-subtitle">Plateforme de Diagnostic Cardiaque par Intelligence Artificielle</div>
</div>
""", unsafe_allow_html=True)


# ── LIVE CLOCK ──
clock_ph = st.empty()

def show_clock():
    tz = pytz.timezone('Africa/Algiers')
    now = datetime.now(tz)
    date_str = now.strftime("%A %d %B %Y").capitalize()
    time_str = now.strftime("%H : %M : %S")
    clock_ph.markdown(f"""
    <div class="clock-bar">
        <div class="clock-left">
            <span class="clock-dot"></span>Heure Algerie &mdash; En Direct
        </div>
        <div class="clock-time">{time_str}</div>
        <div class="clock-right">&#128205; {date_str}</div>
    </div>
    """, unsafe_allow_html=True)
    return now

current_time = show_clock()


# ── MODEL ──
@st.cache_resource
def train_model():
    df = pd.read_csv("cardiovascular_risk_numeric.csv")
    df['smoking_status'] = df['smoking_status'].map({'Never': 0, 'Former': 1, 'Current': 2})
    X = df.drop(['Patient_ID', 'heart_disease_risk_score', 'risk_category'], axis=1)
    y = df['risk_category']
    mdl = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    mdl.fit(X, y)
    return mdl, X.columns

model, feat_cols = train_model()


# ── FORM ──
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-label">Identite Patient</div>', unsafe_allow_html=True)
        nom    = st.text_input("Nom")
        prenom = st.text_input("Prenom")
        age    = st.number_input("Age (ans)", 10, 110, 45)
        family = st.radio("Heredite Cardiaque", ["Non", "Oui"])
    with c2:
        st.markdown('<div class="section-label">Donnees Cliniques</div>', unsafe_allow_html=True)
        sys_bp = st.number_input("Tension Systolique (mmHg)", 80, 220, 120)
        dia_bp = st.number_input("Tension Diastolique (mmHg)", 40, 140, 80)
        chol   = st.number_input("Cholesterol (mg/dL)", 100, 450, 200)
        pulse  = st.number_input("Pouls (BPM)", 40, 160, 72)
    with c3:
        st.markdown('<div class="section-label">Mode de Vie</div>', unsafe_allow_html=True)
        smoke   = st.selectbox("Tabagisme", ["Jamais", "Ex-fumeur", "Fumeur"])
        steps   = st.number_input("Pas / Jour", 0, 30000, 5000)
        sleep   = st.slider("Sommeil (H/nuit)", 3, 12, 7)
        stress  = st.slider("Niveau de Stress (1-10)", 1, 10, 5)
        alcohol = st.number_input("Alcool (verres/sem.)", 0, 50, 0)
        diet    = st.slider("Qualite Alimentaire (1-10)", 1, 10, 7)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Lancer l'Analyse IA - Generer le Bilan Cardiaque")


# ── RESULTS ──
if submitted:
    m_smoke = {'Jamais': 0, 'Ex-fumeur': 1, 'Fumeur': 2}
    input_data = pd.DataFrame(
        [[age, 25.0, sys_bp, dia_bp, chol, pulse, m_smoke[smoke],
          steps, stress, 3, sleep, (1 if family == "Oui" else 0), diet, alcohol]],
        columns=feat_cols
    )
    proba      = model.predict_proba(input_data)[0]
    risk_score = np.max(proba) * 100
    res_idx    = model.predict(input_data)[0]

    cats   = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]
    cls    = ["result-low", "result-medium", "result-high"]
    colors = ["#28C76F", "#E6A000", "#C8102E"]
    emojis = ["&#9989;", "&#9888;", "&#128680;"]

    st.markdown(f"""
    <div class="result-wrap {cls[res_idx]}">
        <div style="font-size:2.8rem;margin-bottom:8px;">{emojis[res_idx]}</div>
        <div class="result-main-label" style="color:{colors[res_idx]};">{cats[res_idx]}</div>
        <div class="result-score-text" style="color:{colors[res_idx]};">Score de risque IA : {risk_score:.1f}%</div>
        <div class="result-patient">Patient : {safe(prenom).capitalize()} {safe(nom).upper()} &nbsp;&#9829;&nbsp; Age : {age} ans</div>
    </div>
    """, unsafe_allow_html=True)

    # ── PRO PDF ──
    tz  = pytz.timezone('Africa/Algiers')
    now = datetime.now(tz)
    heure_pdf = now.strftime("%d/%m/%Y  %H:%M:%S")
    ref_num   = f"CIA-{now.strftime('%Y%m%d%H%M%S')}"

    # Safe versions for PDF (latin-1 only)
    nom_pdf    = safe(nom).upper()
    prenom_pdf = safe(prenom).upper()
    smoke_pdf  = safe(smoke)
    family_pdf = safe(family)

    cats_pdf = ["RISQUE FAIBLE", "RISQUE MODERE", "RISQUE ELEVE"]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── TOP BAND ──
    pdf.set_fill_color(11, 31, 58)
    pdf.rect(0, 0, 210, 50, 'F')
    pdf.set_fill_color(46, 109, 164)
    pdf.rect(0, 50, 210, 3.5, 'F')
    pdf.set_fill_color(184, 151, 42)
    pdf.rect(0, 53.5, 210, 0.8, 'F')

    pdf.set_y(8)
    pdf.set_font("Arial", 'B', 23)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 11, "CARDIO IA PRO", align='C', ln=True)

    pdf.set_font("Arial", 'I', 8.5)
    pdf.set_text_color(140, 180, 220)
    pdf.cell(210, 7, "RAPPORT MEDICAL DE DIAGNOSTIC CARDIAQUE PAR INTELLIGENCE ARTIFICIELLE", align='C', ln=True)

    pdf.set_font("Arial", '', 7.5)
    pdf.set_text_color(100, 140, 175)
    pdf.cell(210, 6, f"Laboratoire de Cardiologie  |  Algerie  |  {heure_pdf}", align='C', ln=True)

    pdf.set_font("Arial", '', 7)
    pdf.set_text_color(80, 115, 155)
    pdf.cell(180, 6, f"Reference : {ref_num}", align='R', ln=True)

    pdf.set_y(60)

    # ── HELPERS ──
    def section_hdr(label):
        pdf.set_fill_color(11, 31, 58)
        pdf.set_draw_color(46, 109, 164)
        pdf.set_line_width(0.5)
        y = pdf.get_y()
        pdf.rect(10, y, 190, 10, 'FD')
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_x(15)
        pdf.cell(185, 10, f"  {label}", ln=True)
        pdf.ln(1)

    def data_row(l1, v1, l2="", v2="", shade=False):
        y = pdf.get_y()
        if shade:
            pdf.set_fill_color(232, 241, 252)
        else:
            pdf.set_fill_color(244, 248, 254)
        pdf.set_draw_color(210, 222, 238)
        pdf.set_line_width(0.15)
        pdf.rect(10, y, 190, 9, 'FD')
        pdf.set_font("Arial", '', 8.5)
        pdf.set_text_color(80, 100, 130)
        pdf.set_x(14)
        pdf.cell(42, 9, safe(l1))
        pdf.set_font("Arial", 'B', 8.5)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(48, 9, safe(str(v1)))
        if l2:
            pdf.set_font("Arial", '', 8.5)
            pdf.set_text_color(80, 100, 130)
            pdf.cell(42, 9, safe(l2))
            pdf.set_font("Arial", 'B', 8.5)
            pdf.set_text_color(11, 31, 58)
            pdf.cell(48, 9, safe(str(v2)))
        pdf.ln(9)

    # ── S1 — PATIENT ──
    section_hdr("I.   INFORMATIONS DU PATIENT")
    data_row("Nom",  nom_pdf,       "Prenom",             prenom_pdf)
    data_row("Age",  f"{age} ans",  "Heredite Cardiaque", family_pdf, shade=True)
    pdf.ln(5)

    # ── S2 — CLINIQUE ──
    section_hdr("II.  DONNEES CLINIQUES")
    data_row("Tension Arterielle", f"{sys_bp}/{dia_bp} mmHg", "Cholesterol", f"{chol} mg/dL")
    data_row("Pouls",              f"{pulse} BPM",             "Tabagisme",   smoke_pdf, shade=True)
    pdf.ln(5)

    # ── S3 — MODE DE VIE ──
    section_hdr("III. MODE DE VIE & HABITUDES")
    data_row("Pas / Jour",       str(steps),         "Sommeil",             f"{sleep} h/nuit")
    data_row("Niveau de Stress", f"{stress} / 10",   "Qualite Alimentaire", f"{diet} / 10", shade=True)
    data_row("Alcool",           f"{alcohol} v/sem.", "",                   "")
    pdf.ln(5)

    # ── S4 — RESULTAT IA ──
    section_hdr("IV.  ANALYSE PAR INTELLIGENCE ARTIFICIELLE")

    risk_fills   = {0: (215, 248, 228), 1: (255, 248, 215), 2: (255, 228, 234)}
    risk_borders = {0: (40, 199, 111),  1: (220, 155, 0),   2: (200, 16, 46)}
    risk_texts   = {0: (15, 110, 60),   1: (130, 85, 0),    2: (155, 10, 28)}

    rf = risk_fills[res_idx]
    rb = risk_borders[res_idx]
    rt = risk_texts[res_idx]

    y0 = pdf.get_y()
    pdf.set_fill_color(*rf)
    pdf.set_draw_color(*rb)
    pdf.set_line_width(1.2)
    pdf.rect(10, y0, 190, 28, 'FD')

    pdf.set_font("Arial", 'B', 17)
    pdf.set_text_color(*rt)
    pdf.set_y(y0 + 5)
    pdf.cell(210, 9, f"{cats_pdf[res_idx]}   |   Score IA : {risk_score:.1f}%", align='C', ln=True)
    pdf.set_font("Arial", 'I', 8.5)
    pdf.set_text_color(70, 85, 110)
    pdf.cell(210, 7, f"Patient : {prenom_pdf} {nom_pdf}   |   Age : {age} ans", align='C', ln=True)
    pdf.ln(5)

    # probability bars
    cat_lbl = ["Risque Faible", "Risque Modere", "Risque Eleve"]
    bar_col = [(40, 199, 111), (220, 155, 0), (200, 16, 46)]
    for i, (lbl, prob) in enumerate(zip(cat_lbl, proba)):
        yb = pdf.get_y()
        pdf.set_font("Arial", '', 8)
        pdf.set_text_color(70, 90, 120)
        pdf.set_x(14)
        pdf.cell(48, 8, lbl)
        pdf.set_fill_color(225, 234, 246)
        pdf.rect(62, yb+2, 118, 4, 'F')
        pdf.set_fill_color(*bar_col[i])
        pdf.rect(62, yb+2, int(prob*118), 4, 'F')
        pdf.set_font("Arial", 'B', 8)
        pdf.set_text_color(11, 31, 58)
        pdf.set_x(184)
        pdf.cell(22, 8, f"{prob*100:.1f}%", align='R')
        pdf.ln(8)
    pdf.ln(4)

    # ── S5 — RECOMMANDATIONS ──
    section_hdr("V.   RECOMMANDATIONS MEDICALES")

    reco_titre = {
        0: "Profil Cardiovasculaire Optimal",
        1: "Vigilance Cardiovasculaire Requise",
        2: "ALERTE - Risque Cardiaque Eleve Detecte"
    }
    reco_corps = {
        0: ("L'analyse par intelligence artificielle indique un profil cardiaque favorable. "
            "Poursuivez vos bonnes habitudes de vie : alimentation equilibree riche en fibres, "
            "activite physique reguliere d'au moins 30 min/jour, hydratation suffisante. "
            "Bilan cardiaque annuel conseille. Surveillez tension et cholesterol periodiquement."),
        1: ("Des facteurs de risque moderes ont ete detectes. Reduisez la consommation de sel, "
            "de graisses saturees et de sucres raffines. Augmentez progressivement l'activite physique "
            "(150 min/semaine). Limitez l'alcool et gerez le stress chronique. "
            "Bilan sanguin complet recommande. Consultez votre medecin dans les 4 semaines."),
        2: ("L'IA identifie un risque cardiovasculaire significativement eleve. "
            "Consultation urgente chez un cardiologue imperative. Evitez tout effort physique intense. "
            "Arretez le tabac immediatement, eliminez l'alcool, regime pauvre en sel et graisses. "
            "Surveillance tensionnelle quotidienne obligatoire. En cas de douleur thoracique, "
            "essoufflement ou palpitations, contactez les urgences sans delai.")
    }

    yr = pdf.get_y()
    pdf.set_fill_color(*rf)
    pdf.set_draw_color(*rb)
    pdf.set_line_width(0.5)
    pdf.set_font("Arial", 'B', 9.5)
    pdf.set_text_color(*rt)
    pdf.set_x(14)
    pdf.multi_cell(182, 8, safe(reco_titre[res_idx]))
    pdf.set_font("Arial", '', 8.5)
    pdf.set_text_color(38, 52, 72)
    pdf.set_x(14)
    pdf.multi_cell(182, 6, safe(reco_corps[res_idx]))
    pdf.rect(10, yr, 190, pdf.get_y() - yr + 2, 'D')
    pdf.ln(5)

    # ── S6 — SIGNATURE ──
    section_hdr("VI.  VALIDATION & SIGNATURE MEDICALE")
    ys = pdf.get_y()
    pdf.set_fill_color(244, 248, 254)
    pdf.set_draw_color(210, 222, 238)
    pdf.set_line_width(0.2)
    pdf.rect(10, ys, 190, 28, 'FD')
    pdf.set_y(ys + 4)
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(80, 100, 130)
    pdf.set_x(14)
    pdf.cell(90, 6, "Medecin / Cardiologue Responsable :", ln=False)
    pdf.set_x(120)
    pdf.cell(80, 6, "Cachet & Signature :", ln=True)
    pdf.set_draw_color(46, 109, 164)
    pdf.set_line_width(0.4)
    pdf.line(14, pdf.get_y() + 11, 104, pdf.get_y() + 11)
    pdf.line(120, pdf.get_y() + 11, 200, pdf.get_y() + 11)
    pdf.ln(18)
    pdf.set_font("Arial", 'I', 7)
    pdf.set_text_color(140, 155, 175)
    pdf.set_x(14)
    pdf.cell(182, 6, f"Ce rapport est genere electroniquement le {heure_pdf}  --  Ref. {ref_num}", align='C', ln=True)

    # ── FOOTER BAND ──
    pdf.set_y(-18)
    pdf.set_fill_color(11, 31, 58)
    pdf.rect(0, pdf.get_y()-1, 210, 22, 'F')
    pdf.set_fill_color(184, 151, 42)
    pdf.rect(0, pdf.get_y()-1, 210, 0.8, 'F')
    pdf.set_font("Arial", '', 7.5)
    pdf.set_text_color(100, 140, 180)
    pdf.set_y(pdf.get_y() + 3)
    pdf.cell(95, 6, "  CardioIA Pro  |  Laboratoire de Cardiologie  |  Algerie", align='L')
    pdf.cell(95, 6, f"Page 1 / 1   |   {ref_num}  ", align='R')

    # ── OUTPUT — BytesIO (seule méthode garantissant un vrai fichier PDF) ──
  # ── OUTPUT — BytesIO ──
    # Get the PDF content as a byte-string directly
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    # Wrap it in a BytesIO object for the download button
    buf = io.BytesIO(pdf_output)
    buf.seek(0)

    st.download_button(
        label="Telecharger le Bilan PDF Professionnel",
        data=buf,
        file_name=f"CardioIA_Bilan_{safe(nom)}_{safe(prenom)}_{now.strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

# ── AUTO-REFRESH CLOCK ──
time.sleep(1)
show_clock()
st.rerun()
