"""
Dashboard: Prediksi Tingkat Kecemasan
Mata Kuliah: Pembelajaran Mesin — EAS Project
Streamlit App — 4 halaman: Beranda, EDA, Prediksi, Tentang
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnxietyScan · Prediksi Kecemasan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

.metric-card {
    background: #1a1f2e; border: 1px solid #252b3b;
    border-radius: 14px; padding: 18px 20px; text-align: left;
    display: flex; align-items: flex-start; gap: 14px;
    transition: border-color .15s;
}
.metric-card:hover { border-color: #334155; }
.metric-icon {
    width: 42px; height: 42px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 1.3rem;
}
.metric-icon.blue   { background: rgba(59,130,246,0.12); }
.metric-icon.purple { background: rgba(167,139,250,0.12); }
.metric-icon.amber  { background: rgba(245,158,11,0.12); }
.metric-icon.green  { background: rgba(74,222,128,0.12); }
.metric-body { flex: 1; }
.metric-card .val {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem;
    font-weight: 700; color: #e2e8f0; line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.74rem; color: #64748b; margin-top: 5px;
    text-transform: uppercase; letter-spacing: .06em;
}
.metric-card .sub { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }

.section-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem;
    font-weight: 700; color: #e2e8f0; margin: 2rem 0 1rem;
    padding-bottom: 8px; border-bottom: 2px solid #334155;
}

.pred-card { border-radius: 16px; padding: 28px 32px; margin: 1.5rem 0; text-align: center; }
.pred-card.low    { background: #0f2027; border: 2px solid #16a34a; }
.pred-card.medium { background: #1c1500; border: 2px solid #d97706; }
.pred-card.high   { background: #1c0a0a; border: 2px solid #dc2626; }
.pred-label { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; margin-bottom: 6px; }
.pred-card.low    .pred-label { color: #4ade80; }
.pred-card.medium .pred-label { color: #fbbf24; }
.pred-card.high   .pred-label { color: #f87171; }
.pred-desc { font-size: 0.95rem; color: #94a3b8; max-width: 500px; margin: 0 auto; line-height: 1.6; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 1rem 0; }
.chip { padding: 5px 14px; border-radius: 20px; font-size: 0.80rem; font-weight: 500; }
.chip.risk    { background: #3b1515; color: #f87171; border: 1px solid #7f1d1d; }
.chip.protect { background: #0f2027; color: #4ade80; border: 1px solid #14532d; }
.chip.neutral { background: #1e2130; color: #94a3b8; border: 1px solid #334155; }

.hero {
    background: linear-gradient(135deg, #0f1117 0%, #131a2a 60%, #0f1a24 100%);
    border: 1px solid #1e2a3a; border-radius: 20px; padding: 48px 40px; margin-bottom: 2rem;
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    overflow: hidden;
}
.hero-text { flex: 1.3; min-width: 280px; }
.hero-visual { flex: 1; min-width: 240px; max-width: 360px; display: flex; justify-content: center; }
.hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700;
    color: #e2e8f0; line-height: 1.2; margin-bottom: 12px;
}
.hero p { font-size: 1.05rem; color: #64748b; max-width: 560px; line-height: 1.7; }
.hero-accent { color: #38bdf8; }
@media (max-width: 900px) { .hero-visual { display: none; } }

.info-box {
    background: #0c1929; border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8; border-radius: 8px;
    padding: 14px 18px; font-size: 0.88rem; color: #94a3b8;
    margin: 1rem 0; line-height: 1.6;
}
hr { border-color: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────
PLOT_BG  = "#0f1117"
PAPER_BG = "#0f1117"
GRID_COL = "#1e2130"
TEXT_COL = "#94a3b8"
FONT_FAM = "Inter, sans-serif"

def apply_theme(fig, height=360):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=TEXT_COL, size=12),
        height=height, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COL, font=dict(size=11)),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickfont=dict(size=11)),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────
# DATASET LOADING (DINAMIS — dibaca langsung dari file Excel)
# ─────────────────────────────────────────────────────────────────────
DATASET_FILENAME = "enhanced_anxiety_dataset.xlsx"

TARGET_COL  = "Anxiety Level (1-10)"
NUMERIC_COLS = [
    "Age","Sleep Hours","Physical Activity (hrs/week)",
    "Caffeine Intake (mg/day)","Alcohol Consumption (drinks/week)",
    "Stress Level (1-10)","Heart Rate (bpm)",
    "Breathing Rate (breaths/min)","Sweating Level (1-5)",
    "Therapy Sessions (per month)","Diet Quality (1-10)"
]
BINARY_COLS = ["Smoking","Dizziness","Medication",
               "Family History of Anxiety","Recent Major Life Event"]
BINARY_LABELS = {
    "Smoking":"Smoking","Dizziness":"Dizziness","Medication":"Medication",
    "Family History of Anxiety":"Family History","Recent Major Life Event":"Life Event"
}
CORR_LABELS = {
    "Age":"Age","Sleep Hours":"Sleep Hours",
    "Physical Activity (hrs/week)":"Physical Activity",
    "Caffeine Intake (mg/day)":"Caffeine Intake",
    "Alcohol Consumption (drinks/week)":"Alcohol Consumption",
    "Stress Level (1-10)":"Stress Level","Heart Rate (bpm)":"Heart Rate",
    "Breathing Rate (breaths/min)":"Breathing Rate",
    "Sweating Level (1-5)":"Sweating Level",
    "Therapy Sessions (per month)":"Therapy Sessions",
    "Diet Quality (1-10)":"Diet Quality",
}

@st.cache_data
def load_dataset():
    """Baca dataset langsung dari file Excel di folder app. Return None kalau tidak ada."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, DATASET_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_excel(path)
        return df
    except Exception:
        return None

@st.cache_data
def compute_eda(df: pd.DataFrame):
    """Hitung semua statistik EDA secara dinamis dari dataframe."""
    eda = {}

    # Distribusi anxiety level mentah (1-10)
    eda["anxiety_dist"] = df[TARGET_COL].value_counts().sort_index().to_dict()

    # Distribusi kelas 3-class
    cls = pd.cut(df[TARGET_COL], bins=[0,4,7,10], labels=["Low (1–4)","Medium (5–7)","High (8–10)"])
    eda["class_dist"] = cls.value_counts().reindex(["Low (1–4)","Medium (5–7)","High (8–10)"]).to_dict()

    # Korelasi fitur numerik
    corr = df[NUMERIC_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    eda["correlations"] = {CORR_LABELS.get(k,k): round(v,3) for k,v in corr.items()}

    # Stress vs anxiety
    eda["stress_anx"] = df.groupby("Stress Level (1-10)")[TARGET_COL].mean().round(2).to_dict()

    # Sleep bin vs anxiety
    bins = [0,5,6,7,8,15]
    labels = ["< 5 jam","5–6 jam","6–7 jam","7–8 jam","> 8 jam"]
    sleep_bin = pd.cut(df["Sleep Hours"], bins=bins, labels=labels)
    eda["sleep_anx"] = df.groupby(sleep_bin, observed=True)[TARGET_COL].mean().round(2).reindex(labels).to_dict()

    # Occupation vs anxiety
    if "Occupation" in df.columns:
        eda["occ_anx"] = df.groupby("Occupation")[TARGET_COL].mean().round(2).sort_values(ascending=False).to_dict()
    else:
        eda["occ_anx"] = {}

    # Age group vs anxiety
    age_bins = [17,25,35,45,55,200]
    age_labels = ["18–25","26–35","36–45","46–55","56+"]
    age_bin = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    eda["age_anx"] = df.groupby(age_bin, observed=True)[TARGET_COL].mean().round(2).reindex(age_labels).to_dict()

    # Binary factors vs anxiety
    binary_anx = {}
    for col in BINARY_COLS:
        if col in df.columns:
            vals = df.groupby(col)[TARGET_COL].mean().round(2)
            binary_anx[BINARY_LABELS.get(col,col)] = {
                "No": vals.get("No", float('nan')),
                "Yes": vals.get("Yes", float('nan')),
            }
    eda["binary_anx"] = binary_anx

    # Summary stats
    eda["n_rows"]    = len(df)
    eda["n_cols"]    = df.shape[1]
    eda["n_missing"] = int(df.isnull().sum().sum())
    eda["mean_anxiety"] = round(df[TARGET_COL].mean(), 2)

    return eda

_df_raw = load_dataset()
DATA_LOADED = _df_raw is not None
EDA = compute_eda(_df_raw) if DATA_LOADED else None

# Model performance & feature importance dicatat dari hasil evaluasi
# notebook Colab (training ulang setiap reload dashboard tidak praktis),
# ditampilkan sebagai referensi performa model yang sudah dilatih.
MODEL_RESULTS = {
    "Random Forest": {"Accuracy":0.8721,"F1-Score":0.8718,"ROC-AUC":0.9512},
    "XGBoost":       {"Accuracy":0.8834,"F1-Score":0.8831,"ROC-AUC":0.9601},
    "MLP":           {"Accuracy":0.8563,"F1-Score":0.8558,"ROC-AUC":0.9387},
}

FEATURE_IMPORTANCE = {
    "Stress Level (1-10)":             0.1823,
    "Sleep Hours":                     0.1354,
    "Therapy Sessions (per month)":    0.1198,
    "Caffeine Intake (mg/day)":        0.0891,
    "Age":                             0.0742,
    "Physical Activity (hrs/week)":    0.0631,
    "Diet Quality (1-10)":             0.0587,
    "Heart Rate (bpm)":                0.0521,
    "Breathing Rate (breaths/min)":    0.0478,
    "Alcohol Consumption (drinks/week)":0.0412,
}

# ─────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    base = os.path.dirname(os.path.abspath(__file__))
    for name, fname in [
        ("Random Forest","best_model.pkl"),
        ("XGBoost","xgb_model.pkl"),
        ("MLP","mlp_model.pkl"),
    ]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            try: models[name] = joblib.load(path)
            except Exception: pass
    scaler_path  = os.path.join(base, "scaler.pkl")
    feat_path    = os.path.join(base, "feature_columns.pkl")
    scaler       = joblib.load(scaler_path)  if os.path.exists(scaler_path) else None
    feat_columns = joblib.load(feat_path)    if os.path.exists(feat_path)   else None
    return models, scaler, feat_columns

try:
    MODELS, SCALER, FEAT_COLS = load_models()
    MODEL_LOADED = len(MODELS) > 0
except Exception:
    MODELS, SCALER, FEAT_COLS = {}, None, None
    MODEL_LOADED = False

def demo_predict(inputs):
    score = 0.0
    score += inputs["stress"] * 0.35
    score -= inputs["sleep"]  * 0.20
    score += inputs["therapy"] * 0.15
    score += (inputs["caffeine"] / 599) * 0.10
    score -= (inputs["phys_act"] / 10)  * 0.05
    score -= (inputs["diet"] / 10)      * 0.05
    score += 0.08 if inputs["family_hist"] else 0
    score += 0.05 if inputs["smoking"]     else 0
    score += 0.04 if inputs["dizziness"]   else 0
    score += 0.04 if inputs["life_event"]  else 0
    score += 0.03 if inputs["medication"]  else 0
    score_norm = np.clip(score / 10, 0, 1)
    if score_norm < 0.35:
        return 0, np.array([0.75, 0.20, 0.05])
    elif score_norm < 0.65:
        return 1, np.array([0.20, 0.65, 0.15])
    else:
        return 2, np.array([0.05, 0.20, 0.75])

def run_prediction(inputs, model_name):
    if not MODEL_LOADED or model_name not in MODELS:
        return demo_predict(inputs)
    NUMERIC_COLS = [
        'Age','Sleep Hours','Physical Activity (hrs/week)',
        'Caffeine Intake (mg/day)','Alcohol Consumption (drinks/week)',
        'Stress Level (1-10)','Heart Rate (bpm)',
        'Breathing Rate (breaths/min)','Sweating Level (1-5)',
        'Therapy Sessions (per month)','Diet Quality (1-10)'
    ]
    row = {
        'Age': inputs['age'], 'Sleep Hours': inputs['sleep'],
        'Physical Activity (hrs/week)': inputs['phys_act'],
        'Caffeine Intake (mg/day)': inputs['caffeine'],
        'Alcohol Consumption (drinks/week)': inputs['alcohol'],
        'Stress Level (1-10)': inputs['stress'],
        'Heart Rate (bpm)': inputs['heart_rate'],
        'Breathing Rate (breaths/min)': inputs['breathing'],
        'Sweating Level (1-5)': inputs['sweating'],
        'Therapy Sessions (per month)': inputs['therapy'],
        'Diet Quality (1-10)': inputs['diet'],
        'Smoking': 1 if inputs['smoking'] else 0,
        'Family History of Anxiety': 1 if inputs['family_hist'] else 0,
        'Dizziness': 1 if inputs['dizziness'] else 0,
        'Medication': 1 if inputs['medication'] else 0,
        'Recent Major Life Event': 1 if inputs['life_event'] else 0,
    }
    for g in ['Female','Male','Other']:
        row[f'Gender_{g}'] = 1 if inputs['gender'] == g else 0
    for occ in ['Artist','Athlete','Chef','Doctor','Engineer','Freelancer',
                'Lawyer','Musician','Nurse','Other','Scientist','Student','Teacher']:
        row[f'Occupation_{occ}'] = 1 if inputs['occupation'] == occ else 0
    df_input = pd.DataFrame([row])
    if FEAT_COLS:
        for c in FEAT_COLS:
            if c not in df_input.columns: df_input[c] = 0
        df_input = df_input[FEAT_COLS]
    if SCALER:
        num_in = [c for c in NUMERIC_COLS if c in df_input.columns]
        df_input[num_in] = SCALER.transform(df_input[num_in])
    model = MODELS[model_name]
    label = int(model.predict(df_input)[0])
    proba = model.predict_proba(df_input)[0]
    return label, proba

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px;'>
      <div style='font-family:"Space Grotesk",sans-serif;font-size:1.3rem;
                  font-weight:700;color:#e2e8f0;line-height:1.1;'>
        🧠 AnxietyScan
      </div>
      <div style='font-size:0.75rem;color:#475569;margin-top:4px;'>
        Prediksi Tingkat Kecemasan
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠  Beranda","📊  Analisis Data","🔮  Prediksi","ℹ️   Tentang"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#334155;line-height:1.6;padding:8px 0;'>
        <b style='color:#475569'>Dataset</b><br>
        Enhanced Anxiety Dataset<br>
        11.000 baris · 19 fitur<br><br>
        <b style='color:#475569'>Model</b><br>
        Random Forest · XGBoost · MLP<br><br>
        <b style='color:#475569'>MK Pembelajaran Mesin</b><br>
        EAS Project · 2026
    </div>
    """, unsafe_allow_html=True)
    if MODEL_LOADED:
        st.markdown("""<div style='margin-top:12px;padding:8px 12px;background:#0c2a1a;
            border:1px solid #166534;border-radius:8px;font-size:0.75rem;color:#4ade80;'>
            ✅ Model berhasil dimuat</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='margin-top:12px;padding:8px 12px;background:#1c1500;
            border:1px solid #713f12;border-radius:8px;font-size:0.75rem;color:#fbbf24;'>
            ⚠️ Mode demo — letakkan .pkl di folder app</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ═════════════════════════════════════════════════════════════════════
if page == "🏠  Beranda":
    if not DATA_LOADED:
        st.markdown(f"""<div class="info-box" style="border-left-color:#ef4444;">
            ❌ <b>Dataset tidak ditemukan.</b> Letakkan file <code>{DATASET_FILENAME}</code>
            di folder yang sama dengan <code>app.py</code>, lalu refresh halaman ini.
            Semua chart EDA dihitung langsung dari file Excel tersebut.
        </div>""", unsafe_allow_html=True)
        st.stop()

    import streamlit.components.v1 as components

    hero_left, hero_right = st.columns([1.4, 1])
    with hero_left:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f1117 0%,#131a2a 60%,#0f1a24 100%);
                    border:1px solid #1e2a3a;border-radius:20px;padding:40px 36px;height:100%;
                    min-height:280px;">
          <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;
                     color:#e2e8f0;line-height:1.2;margin:0 0 12px;">
            Prediksi Tingkat<br><span style="color:#38bdf8;">Kecemasan</span> Anda
          </h1>
          <p style="font-size:1rem;color:#64748b;line-height:1.7;margin:0 0 20px;max-width:440px;">
            Dashboard berbasis Machine Learning untuk menganalisis dan memprediksi
            tingkat kecemasan berdasarkan faktor gaya hidup, kondisi klinis,
            dan faktor psikososial — dihitung langsung dari dataset Excel.
          </p>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#0c2a1a;border:1px solid #166534;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;color:#4ade80;font-weight:500;">
              ✅ Random Forest
            </span>
            <span style="background:#0c1929;border:1px solid #1e3a5f;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;color:#38bdf8;font-weight:500;">
              ⚡ XGBoost
            </span>
            <span style="background:#1a0f2e;border:1px solid #4c1d95;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;color:#c084fc;font-weight:500;">
              🧠 MLP Neural Network
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        low_h  = round(EDA["class_dist"].get("Low (1–4)",  0) / EDA["n_rows"] * 100)
        mid_h  = round(EDA["class_dist"].get("Medium (5–7)",0) / EDA["n_rows"] * 100)
        high_h = round(EDA["class_dist"].get("High (8–10)",0) / EDA["n_rows"] * 100)
        low_bar  = max(2, round(low_h  * 0.55))
        mid_bar  = max(2, round(mid_h  * 0.55))
        high_bar = max(2, round(high_h * 0.55))

        SVG_HTML = f"""<!DOCTYPE html>
<html><head><style>
  body{{margin:0;padding:0;background:#0f1117;overflow:hidden;}}
  @keyframes pulse1{{0%,100%{{opacity:.85;r:2.5px}}50%{{opacity:.2;r:4px}}}}
  @keyframes pulse2{{0%,100%{{opacity:.7;r:2px}} 50%{{opacity:.15;r:3.5px}}}}
  @keyframes pulse3{{0%,100%{{opacity:.75;r:2.2px}}50%{{opacity:.2;r:3.8px}}}}
  @keyframes wavepulse{{0%,100%{{r:4px;opacity:.9}}50%{{r:7px;opacity:.3}}}}
  .p1{{animation:pulse1 2.8s ease-in-out infinite}}
  .p2{{animation:pulse2 2.1s ease-in-out infinite 0.7s}}
  .p3{{animation:pulse3 3.2s ease-in-out infinite 1.2s}}
  .pw{{animation:wavepulse 2s ease-in-out infinite}}
</style></head><body>
<svg viewBox="0 0 360 295" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:100%;display:block;">
  <defs>
    <radialGradient id="bg" cx="50%" cy="46%" r="55%">
      <stop offset="0%" stop-color="#1e3a5f" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#0f1117" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="bf" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#60a5fa" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#a78bfa" stop-opacity="0.22"/>
    </linearGradient>
    <linearGradient id="bs" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#60a5fa"/>
      <stop offset="100%" stop-color="#a78bfa"/>
    </linearGradient>
    <linearGradient id="wg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="50%" stop-color="#818cf8"/>
      <stop offset="100%" stop-color="#a78bfa"/>
    </linearGradient>
  </defs>

  <ellipse cx="180" cy="135" rx="145" ry="128" fill="url(#bg)"/>
  <circle cx="180" cy="130" r="116" fill="none" stroke="#1e2a3a" stroke-width="1.2" stroke-dasharray="4 6"/>
  <circle cx="180" cy="130" r="78"  fill="none" stroke="#1e2a3a" stroke-width="0.8"/>

  <circle cx="180" cy="14"  r="5"   fill="#38bdf8" opacity="0.9"/>
  <circle cx="296" cy="130" r="4"   fill="#a78bfa" opacity="0.85"/>
  <circle cx="64"  cy="130" r="4"   fill="#fbbf24" opacity="0.85"/>
  <circle cx="213" cy="234" r="4.5" fill="#4ade80" opacity="0.85"/>
  <circle cx="70"  cy="58"  r="3"   fill="#f472b6" opacity="0.7"/>

  <line x1="180" y1="14"  x2="180" y2="52"  stroke="#38bdf8" stroke-width="0.8" stroke-opacity="0.35"/>
  <line x1="296" y1="130" x2="258" y2="130" stroke="#a78bfa" stroke-width="0.8" stroke-opacity="0.35"/>
  <line x1="64"  y1="130" x2="102" y2="130" stroke="#fbbf24" stroke-width="0.8" stroke-opacity="0.35"/>

  <g transform="translate(180,126)">
    <path d="M-55-10 C-60-42,-30-62,-2-56 C16-66,44-56,50-32 C68-26,70 4,56 16 C60 36,42 56,20 52 C10 62,-12 62,-22 50 C-46 56,-64 36,-56 14 C-70 4,-68-20,-55-10Z"
          fill="url(#bf)" stroke="url(#bs)" stroke-width="2.2" stroke-opacity="0.75"/>
    <path d="M0-56 C2-30,-2 0,0 52" fill="none" stroke="#a78bfa" stroke-width="1.2" stroke-opacity="0.4" stroke-linecap="round" stroke-dasharray="2 4"/>
    <path d="M-38-22 C-26-12,-24 4,-36 14" fill="none" stroke="#a78bfa" stroke-width="1.6" stroke-opacity="0.6" stroke-linecap="round"/>
    <path d="M-50 8 C-40 14,-40 26,-48 32"  fill="none" stroke="#818cf8" stroke-width="1.3" stroke-opacity="0.5" stroke-linecap="round"/>
    <path d="M-20-44 C-12-30,-14-10,-22 2"  fill="none" stroke="#60a5fa" stroke-width="1.5" stroke-opacity="0.6" stroke-linecap="round"/>
    <path d="M22-38 C32-26,30-8,20 4"        fill="none" stroke="#38bdf8" stroke-width="1.6" stroke-opacity="0.6" stroke-linecap="round"/>
    <path d="M38-16 C48-6,46 10,36 18"       fill="none" stroke="#60a5fa" stroke-width="1.3" stroke-opacity="0.5" stroke-linecap="round"/>
    <path d="M14 20 C24 28,24 38,16 44"      fill="none" stroke="#a78bfa" stroke-width="1.3" stroke-opacity="0.5" stroke-linecap="round"/>
    <line x1="-18" y1="-8"  x2="20"  y2="-16" stroke="#38bdf8" stroke-width="0.8" stroke-opacity="0.25"/>
    <line x1="20"  y1="-16" x2="8"   y2="28"  stroke="#a78bfa" stroke-width="0.8" stroke-opacity="0.25"/>
    <circle class="p1" cx="-18" cy="-8"  r="2.5" fill="#38bdf8"/>
    <circle class="p2" cx="20"  cy="-16" r="2"   fill="#a78bfa"/>
    <circle class="p3" cx="8"   cy="28"  r="2.2" fill="#4ade80"/>
    <circle           cx="-34" cy="28"  r="1.8" fill="#fbbf24" opacity="0.65"/>
  </g>

  <g transform="translate(18,222)">
    <text x="0" y="-6" fill="#64748b" font-size="7" font-family="Inter,sans-serif">Distribusi Kelas</text>
    <rect x="0"  y="{55-low_bar}"  width="18" height="{low_bar}"  rx="3" fill="#3b82f6" opacity="0.9"/>
    <rect x="24" y="{55-mid_bar}"  width="18" height="{mid_bar}"  rx="3" fill="#818cf8" opacity="0.75"/>
    <rect x="48" y="{55-high_bar}" width="18" height="{high_bar}" rx="3" fill="#ef4444" opacity="0.7"/>
    <text x="1"  y="66" fill="#94a3b8" font-size="6.5" font-family="Inter,sans-serif">Low</text>
    <text x="24" y="66" fill="#94a3b8" font-size="6.5" font-family="Inter,sans-serif">Mid</text>
    <text x="47" y="66" fill="#94a3b8" font-size="6.5" font-family="Inter,sans-serif">High</text>
  </g>

  <path d="M105 255 L126 255 L136 240 L146 268 L156 226 L166 255 L188 255 L198 244 L208 262 L218 234 L228 255 L253 255 L261 248 L270 262 L280 242 L290 255 L310 255"
        fill="none" stroke="url(#wg)" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
  <circle class="pw" cx="156" cy="226" r="4" fill="#38bdf8"/>
  <text x="105" y="278" fill="#475569" font-size="7" font-family="Inter,sans-serif">EEG / Anxiety Waveform</text>

  <g transform="translate(242,20)">
    <rect width="100" height="22" rx="6" fill="#1e2a3a" stroke="#334155" stroke-width="0.8"/>
    <text x="10" y="15" fill="#38bdf8" font-size="8.5" font-weight="600" font-family="Inter,sans-serif">{EDA['n_rows']:,} Observasi</text>
  </g>
  <g transform="translate(248,48)">
    <rect width="86" height="22" rx="6" fill="#1e2a3a" stroke="#334155" stroke-width="0.8"/>
    <text x="10" y="15" fill="#a78bfa" font-size="8.5" font-weight="600" font-family="Inter,sans-serif">{EDA['n_cols']} Fitur Data</text>
  </g>
  <g transform="translate(254,76)">
    <rect width="78" height="22" rx="6" fill="#1e2a3a" stroke="#334155" stroke-width="0.8"/>
    <text x="10" y="15" fill="#4ade80" font-size="8.5" font-weight="600" font-family="Inter,sans-serif">3 Model ML</text>
  </g>
</svg>
</body></html>"""
        components.html(SVG_HTML, height=300, scrolling=False)

    # ── Metric cards dengan progress bar visual ──
    cls_dist   = EDA["class_dist"]
    total_n    = EDA["n_rows"]
    low_pct    = cls_dist.get("Low (1–4)", 0) / total_n * 100
    med_pct    = cls_dist.get("Medium (5–7)", 0) / total_n * 100
    high_pct_v = cls_dist.get("High (8–10)", 0) / total_n * 100

    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,
         f"{EDA['n_rows']:,}", "Total Observasi", "baris data dari file Excel",
         "#3b82f6", "rgba(59,130,246,0.12)", "📊",
         f'<div style="margin-top:10px;"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#64748b;margin-bottom:4px;"><span>Low</span><span>Med</span><span>High</span></div><div style="height:5px;border-radius:3px;background:#1e2130;overflow:hidden;display:flex;"><div style="width:{low_pct:.0f}%;background:#3b82f6;"></div><div style="width:{med_pct:.0f}%;background:#f59e0b;"></div><div style="width:{high_pct_v:.0f}%;background:#ef4444;"></div></div></div>'),
        (c2,
         f"{EDA['n_cols']}", "Total Fitur", "12 numerik · 7 kategorik",
         "#a78bfa", "rgba(167,139,250,0.12)", "🧬",
         '<div style="margin-top:10px;display:flex;gap:4px;flex-wrap:wrap;">'
         + ''.join([f'<div style="height:5px;width:10px;border-radius:2px;background:#a78bfa;opacity:{0.4+i*0.05:.2f};"></div>' for i in range(12)])
         + '</div>'),
        (c3,
         "3 Model", "Algoritma ML", "RF · XGBoost · MLP",
         "#fbbf24", "rgba(245,158,11,0.12)", "🤖",
         '<div style="margin-top:10px;display:flex;gap:6px;">'
         '<div style="flex:1;background:#0c2a1a;border:1px solid #166534;border-radius:6px;padding:3px 6px;font-size:0.68rem;color:#4ade80;text-align:center;">RF</div>'
         '<div style="flex:1;background:#0c1929;border:1px solid #1e3a5f;border-radius:6px;padding:3px 6px;font-size:0.68rem;color:#38bdf8;text-align:center;">XGB</div>'
         '<div style="flex:1;background:#1a0f2e;border:1px solid #4c1d95;border-radius:6px;padding:3px 6px;font-size:0.68rem;color:#c084fc;text-align:center;">MLP</div>'
         '</div>'),
        (c4,
         "88.3%", "Best Accuracy", "XGBoost · 3-class",
         "#4ade80", "rgba(74,222,128,0.12)", "🎯",
         '<div style="margin-top:10px;"><div style="height:6px;border-radius:3px;background:#1e2130;overflow:hidden;">'
         '<div style="width:88.3%;height:100%;background:linear-gradient(90deg,#4ade80,#38bdf8);border-radius:3px;"></div>'
         '</div><div style="font-size:0.7rem;color:#64748b;margin-top:3px;">RF 87.2% · MLP 85.6%</div></div>'),
    ]
    for col, val, lbl, sub, accent, bg, icon, extra in cards:
        col.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid #252b3b;border-radius:14px;
                    padding:18px 18px 14px;transition:border-color .15s;
                    border-top:3px solid {accent};">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <div style="width:36px;height:36px;border-radius:10px;background:{bg};
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem;flex-shrink:0;">{icon}</div>
            <div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                          font-weight:700;color:#e2e8f0;line-height:1;">{val}</div>
              <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                          letter-spacing:.06em;margin-top:2px;">{lbl}</div>
            </div>
          </div>
          <div style="font-size:0.8rem;color:#94a3b8;">{sub}</div>
          {extra}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Anxiety Level dalam Dataset</div>',
                unsafe_allow_html=True)

    levels = list(EDA["anxiety_dist"].keys())
    counts = list(EDA["anxiety_dist"].values())
    colors = ['#3b82f6' if l<=4 else '#f59e0b' if l<=7 else '#ef4444' for l in levels]
    fig = go.Figure(go.Bar(
        x=levels, y=counts, marker_color=colors, marker_line_width=0,
        text=counts, textposition='outside', textfont=dict(size=10,color=TEXT_COL),
        hovertemplate="Level %{x}: %{y:,} data<extra></extra>",
    ))
    fig.update_xaxes(tickvals=levels, title_text="Anxiety Level (1–10)")
    fig.update_yaxes(title_text="Jumlah Observasi")
    total_n = EDA["n_rows"]
    cls_vals = EDA["class_dist"]
    max_count = max(counts) if counts else 1
    ann_specs = [
        (2.5, cls_vals.get("Low (1–4)",0), "#3b82f6"),
        (6,   cls_vals.get("Medium (5–7)",0), "#f59e0b"),
        (9,   cls_vals.get("High (8–10)",0), "#ef4444"),
    ]
    for lv, cnt, col in ann_specs:
        pct = (cnt/total_n*100) if total_n else 0
        fig.add_annotation(x=lv, y=max_count*0.9, text=f"<b>{pct:.1f}%</b>",
                           showarrow=False, font=dict(color=col,size=11))
    apply_theme(fig, 300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Mulai dari sini</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    for col, icon, title, desc in [
        (a,"📊","Analisis Data","Eksplorasi dataset: distribusi, korelasi, faktor risiko, dan profil pekerjaan."),
        (b,"🔮","Prediksi","Masukkan data individu dan dapatkan prediksi tingkat kecemasan secara instan."),
        (c,"ℹ️","Tentang Proyek","Metodologi, arsitektur model, dan perbandingan performa ketiga algoritma ML."),
    ]:
        col.markdown(f"""<div class="metric-card" style="text-align:left;padding:20px;">
          <div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                      font-weight:700;color:#e2e8f0;margin-bottom:6px;">{title}</div>
          <div style="font-size:0.83rem;color:#64748b;line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: ANALISIS DATA
# ═════════════════════════════════════════════════════════════════════
elif page == "📊  Analisis Data":
    if not DATA_LOADED:
        st.markdown(f"""<div class="info-box" style="border-left-color:#ef4444;">
            ❌ <b>Dataset tidak ditemukan.</b> Letakkan file <code>{DATASET_FILENAME}</code>
            di folder yang sama dengan <code>app.py</code>, lalu refresh halaman ini.
        </div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="section-title" style="margin-top:0">Analisis Eksplorasi Data (EDA)</div>',
                unsafe_allow_html=True)
    st.caption(f"📂 Dihitung langsung dari `{DATASET_FILENAME}` · {EDA['n_rows']:,} baris · "
               f"{EDA['n_missing']} missing values · rata-rata anxiety {EDA['mean_anxiety']}")

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distribusi","🔗 Korelasi","⚠️ Faktor Risiko","👔 Pekerjaan & Usia"])

    with tab1:
        cl, cr = st.columns([3,2])
        with cl:
            st.markdown("**Distribusi Anxiety Level (1–10)**")
            levels = list(EDA["anxiety_dist"].keys())
            counts = list(EDA["anxiety_dist"].values())
            colors = ['#3b82f6' if l<=4 else '#f59e0b' if l<=7 else '#ef4444' for l in levels]
            fig = go.Figure(go.Bar(
                x=levels,y=counts,marker_color=colors,marker_line_width=0,
                text=counts,textposition='outside',textfont=dict(size=10,color=TEXT_COL),
                hovertemplate="Level %{x}: %{y:,}<extra></extra>",
            ))
            fig.update_xaxes(tickvals=levels,title_text="Anxiety Level")
            fig.update_yaxes(title_text="Jumlah")
            apply_theme(fig,300)
            st.plotly_chart(fig,use_container_width=True)
        with cr:
            st.markdown("**Proporsi Kelas (3-class)**")
            cls_dist = EDA["class_dist"]
            fig2 = go.Figure(go.Pie(
                labels=list(cls_dist.keys()),values=list(cls_dist.values()),
                marker=dict(colors=['#3b82f6','#f59e0b','#ef4444'],
                            line=dict(color=PLOT_BG,width=2)),
                textinfo='label+percent',textfont=dict(size=11),hole=0.45,
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            fig2.add_annotation(text=f"<b>{EDA['n_rows']:,}</b><br>data",x=0.5,y=0.5,
                                font_size=13,showarrow=False,font_color="#e2e8f0")
            apply_theme(fig2,300)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2,use_container_width=True)

        high_pct = (cls_dist.get("High (8–10)",0) / EDA["n_rows"] * 100) if EDA["n_rows"] else 0
        st.markdown(f"""<div class="info-box">
            Dataset {'<b>imbalanced</b>' if high_pct < 20 else 'cukup seimbang'} — kelas High anxiety
            sebesar <b>{high_pct:.1f}%</b> dari total data.
            {"Notebook menggunakan <b>SMOTE</b> sebelum training untuk mengatasi ketidakseimbangan ini." if high_pct < 20 else ""}
        </div>""", unsafe_allow_html=True)

    with tab2:
        corr_items = sorted(EDA["correlations"].items(), key=lambda x: x[1])
        feats  = [k for k,v in corr_items]
        corrs  = [v for k,v in corr_items]
        bar_colors = ['#3b82f6' if v<0 else '#ef4444' for v in corrs]
        fig = go.Figure(go.Bar(
            x=corrs,y=feats,orientation='h',marker_color=bar_colors,marker_line_width=0,
            text=[f"{v:+.3f}" for v in corrs],textposition='outside',
            textfont=dict(size=10,color=TEXT_COL),
            hovertemplate="%{y}: r = %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=0,line_width=1,line_color="#334155")
        x_max = max(abs(min(corrs, default=0)), abs(max(corrs, default=0))) + 0.15
        fig.update_xaxes(title_text="Korelasi Pearson (r)",range=[-x_max,x_max])
        apply_theme(fig,400)
        st.plotly_chart(fig,use_container_width=True)

        strongest_pos = max(EDA["correlations"].items(), key=lambda x: x[1])
        strongest_neg = min(EDA["correlations"].items(), key=lambda x: x[1])
        col1,col2 = st.columns(2)
        col1.markdown(f"""<div class="info-box">
            🔴 <b>Korelasi Positif Terkuat</b><br>
            <b>{strongest_pos[0]} (r = {strongest_pos[1]:.3f})</b> — fitur ini paling
            berkaitan dengan naiknya anxiety level.
        </div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="info-box">
            🔵 <b>Korelasi Negatif Terkuat</b><br>
            <b>{strongest_neg[0]} (r = {strongest_neg[1]:.3f})</b> — semakin tinggi fitur ini,
            anxiety cenderung semakin rendah.
        </div>""", unsafe_allow_html=True)

    with tab3:
        cl,cr = st.columns(2)
        with cl:
            st.markdown("**Stress Level vs Rata-rata Anxiety**")
            sl = list(EDA["stress_anx"].keys())
            sa = list(EDA["stress_anx"].values())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sl,y=sa,mode='lines+markers',
                line=dict(color='#ef4444',width=2.5),
                marker=dict(size=7,color='#ef4444'),
                fill='tozeroy',fillcolor='rgba(239,68,68,0.08)',
                hovertemplate="Stress %{x}: anxiety %{y:.2f}<extra></extra>",
            ))
            fig.update_xaxes(title_text="Stress Level (1–10)",tickvals=sl)
            fig.update_yaxes(title_text="Mean Anxiety Level")
            apply_theme(fig,280)
            st.plotly_chart(fig,use_container_width=True)
        with cr:
            st.markdown("**Durasi Tidur vs Rata-rata Anxiety**")
            slp_l = list(EDA["sleep_anx"].keys())
            slp_v = list(EDA["sleep_anx"].values())
            fig2 = go.Figure(go.Bar(
                x=slp_l,y=slp_v,
                marker_color=['#ef4444','#f59e0b','#f59e0b','#3b82f6','#3b82f6'],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in slp_v],textposition='outside',
                textfont=dict(size=11,color=TEXT_COL),
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ))
            fig2.update_yaxes(title_text="Mean Anxiety Level")
            apply_theme(fig2,280)
            st.plotly_chart(fig2,use_container_width=True)

        if EDA["binary_anx"]:
            st.markdown("**Faktor Biner: Pengaruh terhadap Anxiety Level**")
            bin_labs = list(EDA["binary_anx"].keys())
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name='Tidak (No)',x=bin_labs,
                y=[EDA["binary_anx"][k]["No"] for k in bin_labs],
                marker_color='#3b82f6',marker_line_width=0,
                hovertemplate="%{x} No: %{y:.2f}<extra></extra>",
            ))
            fig3.add_trace(go.Bar(
                name='Ya (Yes)',x=bin_labs,
                y=[EDA["binary_anx"][k]["Yes"] for k in bin_labs],
                marker_color='#ef4444',marker_line_width=0,
                hovertemplate="%{x} Yes: %{y:.2f}<extra></extra>",
            ))
            fig3.update_layout(barmode='group')
            fig3.update_yaxes(title_text="Mean Anxiety Level")
            apply_theme(fig3,300)
            st.plotly_chart(fig3,use_container_width=True)

            biggest_gap = max(EDA["binary_anx"].items(),
                              key=lambda x: abs(x[1]["Yes"]-x[1]["No"]))
            st.markdown(f"""<div class="info-box">
                <b>{biggest_gap[0]}</b> menunjukkan perbedaan terbesar — rata-rata anxiety
                <b>{biggest_gap[1]['Yes']:.2f}</b> (Yes) vs <b>{biggest_gap[1]['No']:.2f}</b> (No).
            </div>""", unsafe_allow_html=True)

    with tab4:
        cl,cr = st.columns(2)
        with cl:
            if EDA["occ_anx"]:
                st.markdown("**Rata-rata Anxiety per Pekerjaan**")
                occ_sorted = sorted(EDA["occ_anx"].items(), key=lambda x: x[1])
                fig = go.Figure(go.Bar(
                    x=[v for k,v in occ_sorted],y=[k for k,v in occ_sorted],
                    orientation='h',marker_color='#6366f1',marker_line_width=0,
                    text=[f"{v:.2f}" for k,v in occ_sorted],textposition='outside',
                    textfont=dict(size=10,color=TEXT_COL),
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                ))
                fig.update_xaxes(title_text="Mean Anxiety Level")
                apply_theme(fig,380)
                st.plotly_chart(fig,use_container_width=True)
        with cr:
            st.markdown("**Rata-rata Anxiety per Kelompok Usia**")
            age_items = {k:v for k,v in EDA["age_anx"].items() if pd.notna(v)}
            fig2 = go.Figure(go.Bar(
                x=list(age_items.keys()),y=list(age_items.values()),
                marker_color=['#64748b','#f59e0b','#ef4444','#3b82f6','#6366f1'][:len(age_items)],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in age_items.values()],textposition='outside',
                textfont=dict(size=11,color=TEXT_COL),
                hovertemplate="Usia %{x}: %{y:.2f}<extra></extra>",
            ))
            fig2.update_yaxes(title_text="Mean Anxiety Level")
            apply_theme(fig2,280)
            st.plotly_chart(fig2,use_container_width=True)

            if EDA["occ_anx"]:
                st.markdown("**3 Pekerjaan dengan Anxiety Tertinggi**")
                top3 = sorted(EDA["occ_anx"].items(), key=lambda x: -x[1])[:3]
                top3_colors = ["#ef4444","#f59e0b","#6366f1"]
                for (occ_name, val), clr in zip(top3, top3_colors):
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;
                        align-items:center;padding:8px 14px;margin-bottom:6px;
                        background:#1a1f2e;border-radius:8px;border-left:3px solid {clr};">
                      <span style="color:#e2e8f0;font-size:0.9rem;">{occ_name}</span>
                      <span style="color:{clr};font-weight:600;font-size:1rem;">{val:.2f}</span>
                    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI
# ═════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediksi":
    st.markdown('<div class="section-title" style="margin-top:0">Prediksi Tingkat Kecemasan</div>',
                unsafe_allow_html=True)
    if not MODEL_LOADED:
        st.markdown("""<div class="info-box">
            ⚠️ <b>Mode Demo:</b> Prediksi menggunakan rule-based scoring dari hasil EDA.
            Letakkan <code>best_model.pkl</code>, <code>scaler.pkl</code>, dan
            <code>feature_columns.pkl</code> di folder yang sama dengan app.py untuk prediksi nyata.
        </div>""", unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("##### Data Demografi")
        c1,c2,c3 = st.columns(3)
        age        = c1.number_input("Usia (tahun)", 18, 64, 30)
        gender     = c2.selectbox("Jenis Kelamin", ["Female","Male","Other"])
        occupation = c3.selectbox("Pekerjaan", [
            "Artist","Athlete","Chef","Doctor","Engineer","Freelancer",
            "Lawyer","Musician","Nurse","Other","Scientist","Student","Teacher"
        ])
        st.markdown("##### Gaya Hidup")
        c4,c5,c6 = st.columns(3)
        sleep    = c4.slider("Jam Tidur per Malam", 2.0, 12.0, 7.0, 0.1)
        phys_act = c5.slider("Aktivitas Fisik (jam/minggu)", 0.0, 10.0, 3.0, 0.1)
        diet     = c6.slider("Kualitas Diet (1–10)", 1, 10, 5)
        c7,c8 = st.columns(2)
        caffeine = c7.slider("Asupan Kafein (mg/hari)", 0, 600, 200)
        alcohol  = c8.slider("Konsumsi Alkohol (minuman/minggu)", 0, 20, 3)

        st.markdown("##### Kondisi Klinis")
        c9,c10,c11 = st.columns(3)
        stress     = c9.slider("Level Stres (1–10)", 1, 10, 5)
        heart_rate = c10.slider("Detak Jantung (bpm)", 60, 120, 80)
        breathing  = c11.slider("Laju Napas (napas/menit)", 12, 30, 16)
        c12,c13 = st.columns(2)
        sweating = c12.slider("Level Berkeringat (1–5)", 1, 5, 2)
        therapy  = c13.slider("Sesi Terapi per Bulan", 0, 12, 0)

        st.markdown("##### Faktor Psikososial")
        cc1,cc2,cc3,cc4,cc5 = st.columns(5)
        smoking     = cc1.checkbox("Merokok")
        family_hist = cc2.checkbox("Riwayat Keluarga")
        dizziness   = cc3.checkbox("Pusing")
        medication  = cc4.checkbox("Sedang Obat")
        life_event  = cc5.checkbox("Kejadian Besar")

        st.markdown("##### Pilih Model")
        model_choice = st.selectbox(
            "Algoritma Prediksi", ["XGBoost","Random Forest","MLP"],
            help="XGBoost menunjukkan akurasi terbaik (88.3%) pada dataset ini."
        )
        submitted = st.form_submit_button("🔮  Prediksi Sekarang",
                                          use_container_width=True, type="primary")

    if submitted:
        inputs = dict(
            age=age, gender=gender, occupation=occupation,
            sleep=sleep, phys_act=phys_act, diet=diet,
            caffeine=caffeine, alcohol=alcohol, stress=stress,
            heart_rate=heart_rate, breathing=breathing,
            sweating=sweating, therapy=therapy,
            smoking=smoking, family_hist=family_hist,
            dizziness=dizziness, medication=medication, life_event=life_event,
        )
        label, proba = run_prediction(inputs, model_choice)

        label_info = {
            0: ("low","Low","Rendah",
                "Tingkat kecemasan Anda tergolong rendah. Gaya hidup dan kondisi saat ini "
                "relatif mendukung kesehatan mental. Pertahankan pola tidur yang baik dan "
                "manajemen stres yang sehat."),
            1: ("medium","Medium","Sedang",
                "Tingkat kecemasan Anda berada di level menengah. Beberapa faktor risiko perlu "
                "mendapat perhatian. Pertimbangkan untuk meningkatkan kualitas tidur, mengurangi "
                "kafein, dan meningkatkan aktivitas fisik."),
            2: ("high","High","Tinggi",
                "Tingkat kecemasan Anda tergolong tinggi. Sangat disarankan untuk berkonsultasi "
                "dengan profesional kesehatan mental. Perhatikan manajemen stres, pola tidur, "
                "dan pertimbangkan sesi terapi."),
        }
        css_cls, eng_lbl, ind_lbl, desc = label_info[label]

        st.markdown(f"""<div class="pred-card {css_cls}">
          <div style="font-size:0.85rem;color:#64748b;margin-bottom:6px;
                      text-transform:uppercase;letter-spacing:.08em;">
            Hasil Prediksi · {model_choice}</div>
          <div class="pred-label">{ind_lbl} ({eng_lbl})</div>
          <div class="pred-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Probabilitas per Kelas**")
        fig = go.Figure(go.Bar(
            x=["Low (1–4)","Medium (5–7)","High (8–10)"],
            y=[round(p*100,1) for p in proba],
            marker_color=["#3b82f6","#f59e0b","#ef4444"],marker_line_width=0,
            text=[f"{p*100:.1f}%" for p in proba],textposition='outside',
            textfont=dict(size=12,color=TEXT_COL),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_yaxes(range=[0,105],title_text="Probabilitas (%)")
        apply_theme(fig,250)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("**Analisis Faktor Risiko & Pelindung**")
        risk_factors, protect_factors = [], []
        if stress >= 8:   risk_factors.append(f"Stres sangat tinggi (level {stress})")
        elif stress >= 6: risk_factors.append(f"Stres tinggi (level {stress})")
        if sleep < 5:     risk_factors.append(f"Tidur sangat kurang ({sleep:.1f} jam)")
        elif sleep < 6:   risk_factors.append(f"Tidur kurang ({sleep:.1f} jam)")
        elif sleep >= 7:  protect_factors.append(f"Tidur cukup ({sleep:.1f} jam)")
        if caffeine > 400: risk_factors.append(f"Kafein sangat tinggi ({caffeine} mg)")
        elif caffeine > 250: risk_factors.append(f"Kafein tinggi ({caffeine} mg)")
        if family_hist: risk_factors.append("Riwayat kecemasan dalam keluarga")
        if smoking:     risk_factors.append("Perokok aktif")
        if dizziness:   risk_factors.append("Mengalami pusing/dizziness")
        if life_event:  risk_factors.append("Kejadian besar dalam hidup baru-baru ini")
        if phys_act >= 3: protect_factors.append(f"Aktif fisik ({phys_act:.1f} jam/minggu)")
        if diet >= 7:     protect_factors.append(f"Diet berkualitas baik ({diet}/10)")
        if not smoking:   protect_factors.append("Tidak merokok")
        if therapy > 0:   protect_factors.append(f"Sesi terapi aktif ({therapy}x/bulan)")

        chip_html = '<div class="chip-row">'
        for f in risk_factors:    chip_html += f'<span class="chip risk">⚠ {f}</span>'
        for f in protect_factors: chip_html += f'<span class="chip protect">✓ {f}</span>'
        if not risk_factors and not protect_factors:
            chip_html += '<span class="chip neutral">Profil risiko netral</span>'
        chip_html += '</div>'
        st.markdown(chip_html, unsafe_allow_html=True)

        st.markdown("""<div class="info-box" style="border-left-color:#f59e0b;">
            ⚠️ <b>Disclaimer:</b> Prediksi ini merupakan output model Machine Learning berbasis data
            statistik dan <b>bukan diagnosis medis</b>. Untuk penilaian kesehatan mental yang akurat,
            konsultasikan dengan psikolog atau psikiater berlisensi.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: TENTANG
# ═════════════════════════════════════════════════════════════════════
elif page == "ℹ️   Tentang":
    st.markdown('<div class="section-title" style="margin-top:0">Tentang Proyek</div>',
                unsafe_allow_html=True)
    tab_m, tab_p, tab_c = st.tabs(["📐 Metodologi","📊 Performa Model","💻 Cara Menjalankan"])

    with tab_m:
        col1,col2 = st.columns(2)
        col1.markdown("""
**Dataset**
- Sumber: Enhanced Anxiety Dataset
- Ukuran: 11.000 baris × 19 kolom
- Missing values: 0 (data lengkap)
- Target: Anxiety Level 1–10

**Preprocessing**
- Label 3-kelas (Low / Medium / High)
- Binary encoding kolom Yes/No
- One-Hot Encoding Gender & Occupation
- StandardScaler untuk fitur numerik
- SMOTE untuk class imbalance
        """)
        col2.markdown("""
**Split Data**
- 80% training — 20% testing
- Stratified split

**Evaluasi**
- Accuracy, F1-Score (weighted), ROC-AUC (OvR)
- Confusion Matrix per model
- ROC Curve per kelas (One-vs-Rest)
- Feature Importance (RF + XGBoost)

**Framework**
- scikit-learn · XGBoost · imbalanced-learn
- Dashboard: Streamlit + Plotly
        """)
        st.markdown("**Arsitektur MLP (Neural Network)**")
        st.markdown("""<div class="info-box">
            Input Layer → <b>Dense 256</b> (ReLU) → <b>Dense 128</b> (ReLU) →
            <b>Dense 64</b> (ReLU) → Output (3 kelas, Softmax)<br>
            Optimizer: Adam · L2 α=0.001 · Early stopping (patience=15)
        </div>""", unsafe_allow_html=True)

    with tab_p:
        model_names  = list(MODEL_RESULTS.keys())
        metrics_keys = ["Accuracy","F1-Score","ROC-AUC"]
        colors_m     = ["#3b82f6","#f59e0b","#6366f1"]
        fig = go.Figure()
        for metric,color in zip(metrics_keys,colors_m):
            fig.add_trace(go.Bar(
                name=metric, x=model_names,
                y=[MODEL_RESULTS[m][metric] for m in model_names],
                marker_color=color, marker_line_width=0,
                text=[f"{MODEL_RESULTS[m][metric]:.3f}" for m in model_names],
                textposition='outside', textfont=dict(size=10,color=TEXT_COL),
                hovertemplate=f"{metric}: %{{y:.4f}}<extra></extra>",
            ))
        fig.update_layout(barmode='group')
        fig.update_yaxes(range=[0.78,0.99],title_text="Score")
        apply_theme(fig,320)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("**Tabel Performa Detail**")
        rows = [{"Model":m,"Accuracy":f"{v['Accuracy']:.4f}",
                 "F1-Score":f"{v['F1-Score']:.4f}","ROC-AUC":f"{v['ROC-AUC']:.4f}",
                 "Rekomendasi":"✅ Terbaik" if m=="XGBoost" else ""}
                for m,v in MODEL_RESULTS.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("**Top 10 Feature Importance (Random Forest)**")
        fi_items = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1])
        fig2 = go.Figure(go.Bar(
            x=[v for k,v in fi_items], y=[k for k,v in fi_items],
            orientation='h', marker_color='#3b82f6', marker_line_width=0,
            text=[f"{v:.4f}" for k,v in fi_items], textposition='outside',
            textfont=dict(size=10,color=TEXT_COL),
        ))
        fig2.update_xaxes(title_text="Importance Score")
        apply_theme(fig2,380)
        st.plotly_chart(fig2,use_container_width=True)

    with tab_c:
        st.markdown("**Cara Mengakses Dashboard**")
        st.markdown("""<div class="info-box">
            Dashboard <b>AnxietyScan</b> dapat diakses secara langsung melalui browser tanpa perlu instalasi apapun.
            Kunjungi tautan berikut untuk mulai menggunakan dashboard:<br><br>
            🔗 <b>https://anxiety-dashboard-fadzli098.streamlit.app</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Panduan Penggunaan Dashboard**")
        st.markdown("""
**1. Halaman Beranda**
Halaman pertama yang tampil saat membuka dashboard. Berisi ringkasan proyek, jumlah data, model yang digunakan, dan akurasi terbaik yang dicapai. Gunakan halaman ini untuk memahami gambaran umum sistem sebelum melanjutkan ke halaman lain.

**2. Halaman Analisis Data**
Berisi visualisasi eksplorasi data secara interaktif. Terdapat empat tab yang dapat dipilih:
- **Distribusi** — melihat sebaran tingkat kecemasan dan proporsi kelas
- **Korelasi** — melihat hubungan antar fitur dengan tingkat kecemasan
- **Faktor Risiko** — melihat pengaruh gaya hidup dan faktor biner terhadap kecemasan
- **Pekerjaan & Usia** — melihat profil kecemasan berdasarkan jenis pekerjaan

Setiap grafik bersifat interaktif — arahkan kursor ke grafik untuk melihat nilai detail, atau gunakan tombol di pojok kanan grafik untuk memperbesar atau mengunduh grafik.

**3. Halaman Prediksi**
Halaman utama untuk melakukan prediksi tingkat kecemasan. Langkah-langkahnya adalah sebagai berikut:
- Isi data demografi (usia, jenis kelamin, pekerjaan)
- Atur slider gaya hidup (jam tidur, aktivitas fisik, kualitas diet, asupan kafein, konsumsi alkohol)
- Atur slider kondisi klinis (level stres, detak jantung, laju napas, level berkeringat, sesi terapi)
- Centang faktor psikososial yang sesuai dengan kondisi Anda (merokok, riwayat keluarga, pusing, sedang obat, kejadian besar)
- Pilih model algoritma yang ingin digunakan (XGBoost, Random Forest, atau MLP)
- Klik tombol **Prediksi Sekarang** untuk melihat hasil

Hasil prediksi akan menampilkan kelas kecemasan (Rendah, Sedang, atau Tinggi), probabilitas per kelas, serta analisis faktor risiko dan faktor pelindung secara otomatis berdasarkan data yang dimasukkan.

**4. Halaman Tentang**
Berisi informasi lengkap mengenai metodologi penelitian, perbandingan performa model, dan panduan teknis proyek bagi yang ingin mengetahui lebih lanjut.
        """)

        st.markdown("""<div class="info-box" style="border-left-color:#f59e0b;">
            ⚠️ <b>Perhatian:</b> Hasil prediksi yang ditampilkan merupakan output model <i>Machine Learning</i>
            dan <b>bukan merupakan diagnosis medis</b>. Untuk penilaian kesehatan mental yang akurat,
            konsultasikan dengan psikolog atau psikiater berlisensi.
        </div>""", unsafe_allow_html=True)
