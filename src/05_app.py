"""
KORAK 5: STREAMLIT DEPLOYMENT v4 - Ensemble OR voting (XGBoost + LR)
Pokretanje: streamlit run src/05_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ==================== KONFIGURACIJA ====================
st.set_page_config(
    page_title="MotoGP DNF Prediktor",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #0d0d0d 100%); }

    .main-header {
        background: linear-gradient(90deg, #d32f2f 0%, #b71c1c 50%, #d32f2f 100%);
        padding: 25px 20px; border-radius: 16px; text-align: center; margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(211, 47, 47, 0.3);
    }
    .main-header h1 { color: white !important; font-weight: 900; font-size: 2.2em; letter-spacing: 2px; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85) !important; font-size: 1em; margin-top: 6px; }

    .result-card {
        background: linear-gradient(135deg, #1e1e30 0%, #23233a 100%);
        border: 2px solid #3a3a50; border-radius: 20px; padding: 30px;
        text-align: center; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .result-card.dnf { border-color: #d32f2f; box-shadow: 0 8px 40px rgba(211, 47, 47, 0.35); }
    .result-card.finish { border-color: #2ecc71; box-shadow: 0 8px 40px rgba(46, 204, 113, 0.3); }
    .result-text { font-size: 2.2em; font-weight: 900; letter-spacing: 2px; margin: 10px 0; }
    .result-text.dnf { color: #e74c3c; }
    .result-text.finish { color: #2ecc71; }

    .indicator-card {
        background: linear-gradient(135deg, #1e1e30 0%, #252540 100%);
        border: 1px solid #3a3a50; border-radius: 14px; padding: 18px;
        text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .indicator-card .value { font-size: 2em; font-weight: 700; color: #fff; }
    .indicator-card .label { font-size: 0.7em; color: #999; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
    .indicator-card.warn .value { color: #e74c3c; }
    .indicator-card.good .value { color: #2ecc71; }

    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #12121f 100%); }
    div[data-testid="stSidebar"] .stMarkdown, div[data-testid="stSidebar"] label { color: #ccc !important; }

    .stButton > button {
        background: linear-gradient(90deg, #d32f2f 0%, #b71c1c 100%) !important;
        color: white !important; font-weight: 700 !important; font-size: 1.1em !important;
        border: none !important; border-radius: 10px !important; padding: 14px 30px !important;
        letter-spacing: 1px !important; box-shadow: 0 4px 15px rgba(211, 47, 47, 0.4) !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(211, 47, 47, 0.6) !important; }
    .preset-btn > button {
        background: linear-gradient(135deg, #1e1e30 0%, #2a2a40 100%) !important;
        border: 1px solid #3a3a50 !important; font-size: 0.85em !important;
        padding: 12px 8px !important; box-shadow: none !important;
    }
    .preset-btn > button:hover { border-color: #d32f2f !important; background: #252540 !important; }

    footer { visibility: hidden; }
    .custom-footer { text-align: center; color: #555; font-size: 0.8em; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# ==================== UCITAVANJE ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(os.path.join(MODELS_DIR, "best_model_xgb.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "best_model_lr.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl")),
    )

@st.cache_data
def load_data():
    df = pd.read_csv(CLEANED_PATH)
    # Mapa: circuit_name -> pravi shortname iz dataseta
    shortname_map = df.groupby("circuit_name")["shortname"].first().to_dict()
    # Mapa: circuit_name -> prosečan sequence count po godini
    circuit_defaults = {
        "circuits": sorted(df["circuit_name"].unique()),
        "riders": sorted(df["rider_name"].unique()),
        "teams": sorted(df["team_name"].unique()),
        "bikes": sorted(df["bike_name"].unique()),
        "shortname_map": shortname_map
    }

    # Qualifying circuit features (MotoGP Q2, 2022-2025)
    qual_path = os.path.join(BASE_DIR, "data", "raw", "Qualifying.csv")
    dfq = pd.read_csv(qual_path)
    dfq = dfq[dfq["class"] == "MotoGP"].copy()
    dfq = dfq.drop_duplicates()
    dfq["time_sec"] = dfq["time"].apply(
        lambda t: None if ":" not in str(t)
        else float(str(t).split(":")[0]) * 60 + float(str(t).split(":")[1])
    )
    q2 = dfq[dfq["session"] == "Q2"].dropna(subset=["time_sec"])
    cy = q2.groupby(["event", "year"]).apply(lambda g: pd.Series({
        "Quali_Spread":   g["time_sec"].max() - g["time_sec"].min(),
        "Quali_Top6_Gap": g.nsmallest(6, "time_sec")["time_sec"].max()
                          - g["time_sec"].min(),
        "Quali_Time_Std": g["time_sec"].std()
    })).reset_index()
    cf = (cy.groupby("event")[["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]]
          .median().reset_index()
          .set_index("event"))
    # Fallback: global median
    fallback = {c: cf[c].median() for c in ["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]}
    circuit_features = {"features": cf, "fallback": fallback}

    return df, circuit_defaults, circuit_features

model_xgb, model_lr, preprocessor = load_artifacts()
df_full, defaults, quali_features = load_data()

# ==================== PREDIKCIJA ====================
def quali_lookup(defaults, quali_features, circuit_name, col):
    """Vrati qualifying feature za stazu, ili globalni median ako staza nije u qualifying podacima."""
    short = defaults["shortname_map"].get(circuit_name, circuit_name[:3].upper())
    if short in quali_features["features"].index:
        return float(quali_features["features"].loc[short, col])
    return float(quali_features["fallback"][col])

def calculate_features(circuit_name, rider_name, team_name, bike_name, year, sequence):
    hist = df_full[df_full["year"] < year]
    if len(hist) == 0:
        return {"Historical_DNF_Rate": 0.0, "Rider_Experience": 0,
                "Team_DNF_Rate": 0.0, "Bike_DNF_Rate": 0.0,
                "Track_DNF_Rate": 0.0, "Season_Phase": 0.5,
                "Quali_Spread":   round(quali_lookup(defaults, quali_features, circuit_name, "Quali_Spread"), 3),
                "Quali_Top6_Gap": round(quali_lookup(defaults, quali_features, circuit_name, "Quali_Top6_Gap"), 3),
                "Quali_Time_Std": round(quali_lookup(defaults, quali_features, circuit_name, "Quali_Time_Std"), 3)}

    rc = hist[(hist["rider_name"] == rider_name) & (hist["circuit_name"] == circuit_name)]
    hist_dnf = rc["Status_Zavrsetka"].mean() if len(rc) > 0 else (
        hist[hist["rider_name"] == rider_name]["Status_Zavrsetka"].mean()
        if len(hist[hist["rider_name"] == rider_name]) > 0 else 0.0
    )
    rider_exp = len(hist[hist["rider_name"] == rider_name])
    team_dnf = (hist[hist["team_name"] == team_name]["Status_Zavrsetka"].mean()
                if len(hist[hist["team_name"] == team_name]) > 0 else 0.0)
    bike_dnf = (hist[hist["bike_name"] == bike_name]["Status_Zavrsetka"].mean()
                if len(hist[hist["bike_name"] == bike_name]) > 0 else 0.0)
    track_dnf = (hist[hist["circuit_name"] == circuit_name]["Status_Zavrsetka"].mean()
                 if len(hist[hist["circuit_name"] == circuit_name]) > 0 else 0.0)
    yd = df_full[df_full["year"] == year]
    season_phase = (sequence - 1) / max(int(yd["sequence"].max()) - 1, 1) if len(yd) > 0 else 0.5

    return {"Historical_DNF_Rate": round(hist_dnf, 4), "Rider_Experience": rider_exp,
            "Team_DNF_Rate": round(team_dnf, 4), "Bike_DNF_Rate": round(bike_dnf, 4),
            "Track_DNF_Rate": round(track_dnf, 4), "Season_Phase": round(season_phase, 4),
            "Quali_Spread":   round(quali_lookup(defaults, quali_features, circuit_name, "Quali_Spread"), 3),
            "Quali_Top6_Gap": round(quali_lookup(defaults, quali_features, circuit_name, "Quali_Top6_Gap"), 3),
            "Quali_Time_Std": round(quali_lookup(defaults, quali_features, circuit_name, "Quali_Time_Std"), 3)}

def run_prediction(circuit, rider, team, bike, year, sequence):
    features = calculate_features(circuit, rider, team, bike, year, sequence)
    real_shortname = defaults["shortname_map"].get(circuit, circuit[:3].upper())
    input_data = pd.DataFrame([{
        "shortname": real_shortname, "circuit_name": circuit,
        "rider_name": rider, "team_name": team, "bike_name": bike,
        "year": year, "sequence": sequence, **features
    }])
    input_processed = preprocessor.transform(input_data)
    proba_xgb = float(model_xgb.predict_proba(input_processed)[0, 1])
    proba_lr = float(model_lr.predict_proba(input_processed)[0, 1])
    return proba_xgb, proba_lr, features

# ==================== GAUGE ====================
def create_gauge(probability, threshold):
    steps = [
        {"range": [0, 33], "color": "#2ecc71"},
        {"range": [33, 66], "color": "#f1c40f"},
        {"range": [66, 100], "color": "#e74c3c"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"font": {"size": 55, "color": "#fff", "family": "Montserrat"}, "suffix": "%", "valueformat": ".1f"},
        delta={"reference": threshold * 100, "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#2ecc71"}, "position": "bottom"},
        title={"text": "VEROVATNOĆA ODUSTAJANJA", "font": {"size": 12, "color": "#999", "family": "Montserrat"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666", "tickfont": {"color": "#999", "size": 11}},
            "bar": {"color": "#d32f2f" if probability >= threshold else "#2ecc71", "thickness": 0.25},
            "bgcolor": "#1a1a2e", "borderwidth": 0,
            "steps": steps,
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.8, "value": threshold * 100}
        }
    ))
    fig.update_layout(height=320, margin=dict(t=50, b=20, l=30, r=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#fff", "family": "Montserrat"})
    return fig

# ==================== SESSION STATE INIT ====================
if "params" not in st.session_state:
    st.session_state.params = {
        "circuit": defaults["circuits"][0],
        "rider": defaults["riders"][0],
        "team": defaults["teams"][0],
        "bike": defaults["bikes"][0],
        "year": 2025,
        "sequence": 4
    }
if "run_flag" not in st.session_state:
    st.session_state.run_flag = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "threshold_xgb" not in st.session_state:
    st.session_state.threshold_xgb = 0.35
if "threshold_lr" not in st.session_state:
    st.session_state.threshold_lr = 0.30

p = st.session_state.params

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 15px 0;">
        <span style="font-size:2em;">🏍️</span>
        <h3 style="color:#d32f2f; margin:0; font-weight:900;">PARAMETRI TRKE</h3>
        <p style="color:#777; font-size:0.8em;">Podaci poznati pre starta</p>
    </div>
    """, unsafe_allow_html=True)

    # Selectboxovi citaju i pisu u session_state
    p["circuit"] = st.selectbox("🏟️ Staza", defaults["circuits"],
                                 index=defaults["circuits"].index(p["circuit"])
                                 if p["circuit"] in defaults["circuits"] else 0)
    p["rider"] = st.selectbox("🏍️ Vozač", defaults["riders"],
                               index=defaults["riders"].index(p["rider"])
                               if p["rider"] in defaults["riders"] else 0)
    p["team"] = st.selectbox("🔧 Tim", defaults["teams"],
                              index=defaults["teams"].index(p["team"])
                              if p["team"] in defaults["teams"] else 0)
    p["bike"] = st.selectbox("⚙️ Konstruktor", defaults["bikes"],
                              index=defaults["bikes"].index(p["bike"])
                              if p["bike"] in defaults["bikes"] else 0)

    c1, c2 = st.columns(2)
    with c1:
        p["year"] = st.number_input("📅 Godina", 1950, 2030, p["year"])
    with c2:
        p["sequence"] = st.number_input("🔢 Trka br.", 1, 22, p["sequence"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 PREDVIDI ISHOD", type="primary", use_container_width=True):
        st.session_state.run_flag = True

    st.markdown("---")
    st.markdown("### 🎚️ Pragovi odluke (OR voting)")
    
    new_thresh_xgb = st.slider("XGBoost threshold", 0.0, 1.0, st.session_state.threshold_xgb, 0.01,
                                help="XGBoost: verovatnoca iznad koje se predvidja DNF")
    if new_thresh_xgb != st.session_state.threshold_xgb:
        st.session_state.threshold_xgb = new_thresh_xgb
        if st.session_state.last_result is not None:
            st.session_state.run_flag = True

    new_thresh_lr = st.slider("Logistic Regression threshold", 0.0, 1.0, st.session_state.threshold_lr, 0.01,
                               help="LR: verovatnoca iznad koje se predvidja DNF")
    if new_thresh_lr != st.session_state.threshold_lr:
        st.session_state.threshold_lr = new_thresh_lr
        if st.session_state.last_result is not None:
            st.session_state.run_flag = True

    st.caption("OR logika: ako XGBoost ILI LR predvidi DNF → DNF")

    st.markdown("---")
    with st.expander("ℹ️ O modelu"):
        n_features = len(joblib.load(os.path.join(MODELS_DIR, "feature_info.pkl"))["all_cols"])
        st.markdown(f"""
        - **Algoritam**: XGBoost + Logistic Regression (ensemble OR)
        - **Feature-a**: {n_features} (TargetEncoded)
        - **Threshold-i**: XGB={st.session_state.threshold_xgb:.0%}, LR={st.session_state.threshold_lr:.0%}
        - **Recall DNF**: 96.3% | **Precision**: 12.8% | **FN (test)**: 9/244
        - **Podaci**: MotoGP 1949-2021
        """)

# ==================== MAIN AREA ====================
st.markdown("""
<div class="main-header">
    <h1>🏍️ MotoGP DNF PREDIKTOR</h1>
    <p>Predviđanje ishoda trke — Završetak ili Odustajanje (DNF)</p>
</div>
""", unsafe_allow_html=True)

# ---- PRESETS ----
st.markdown("### ⚡ Brzi test scenariji")
st.caption("Klikni za automatsko popunjavanje i predikciju")

presets = [
    {"icon": "🔥", "title": "Marquez\nJerez", "circuit": "Jerez de la Frontera",
     "rider": "Marquez, Marc", "team": "Repsol Honda Team", "bike": "Honda", "year": 2025, "seq": 4},
    {"icon": "👑", "title": "Rossi\nMugello", "circuit": "Mugello",
     "rider": "Rossi, Valentino", "team": "Yamaha Factory Racing", "bike": "Yamaha", "year": 2025, "seq": 6},
    {"icon": "🆕", "title": "Debitant\nteška staza", "circuit": "Sepang",
     "rider": "Binder, Brad", "team": "Red Bull KTM Factory Racing", "bike": "KTM", "year": 2025, "seq": 1},
    {"icon": "⭐", "title": "Šampion\nlaka staza", "circuit": "Assen",
     "rider": "Lorenzo, Jorge", "team": "Ducati Team", "bike": "Ducati", "year": 2025, "seq": 8},
    {"icon": "🌧️", "title": "Kišni\nvikend", "circuit": "Phillip Island",
     "rider": "Crutchlow, Cal", "team": "LCR Honda", "bike": "Honda", "year": 2025, "seq": 3},
]

cols = st.columns(len(presets))
for i, (col, preset) in enumerate(zip(cols, presets)):
    with col:
        btn_key = f"preset_{i}"
        st.markdown(f'<div class="preset-btn">', unsafe_allow_html=True)
        if st.button(f"{preset['icon']} {preset['title']}", key=btn_key,
                     help=f"{preset['circuit']} · {preset['rider']}", use_container_width=True):
            p["circuit"] = preset["circuit"]
            p["rider"] = preset["rider"]
            p["team"] = preset["team"]
            p["bike"] = preset["bike"]
            p["year"] = preset["year"]
            p["sequence"] = preset["seq"]
            st.session_state.run_flag = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---- REZULTAT ----
if st.session_state.run_flag:
    proba_xgb, proba_lr, features = run_prediction(
        p["circuit"], p["rider"], p["team"], p["bike"],
        p["year"], p["sequence"]
    )
    # OR voting
    dnf_xgb = proba_xgb >= st.session_state.threshold_xgb
    dnf_lr = proba_lr >= st.session_state.threshold_lr
    is_dnf = dnf_xgb or dnf_lr
    max_proba = max(proba_xgb, proba_lr)
    st.session_state.last_result = {
        "proba_xgb": proba_xgb, "proba_lr": proba_lr,
        "max_proba": max_proba, "features": features,
        "is_dnf": is_dnf, "dnf_xgb": dnf_xgb, "dnf_lr": dnf_lr,
    }
    st.session_state.run_flag = False

if st.session_state.last_result is not None:
    r = st.session_state.last_result
    proba_xgb = r["proba_xgb"]
    proba_lr = r["proba_lr"]
    max_proba = r["max_proba"]
    features = r["features"]
    is_dnf = r["is_dnf"]

    card_class = "dnf" if is_dnf else "finish"
    text_class = "dnf" if is_dnf else "finish"
    label = "RISK - VOZAC CE ODUSTATI" if is_dnf else "VOZAC CE ZAVRSITI TRKU"
    
    voter_info = ""
    if r["dnf_xgb"] and r["dnf_lr"]:
        voter_info = "(oba modela: DNF)"
    elif r["dnf_xgb"]:
        voter_info = "(XGBoost: DNF, LR: Zavrsio)"
    elif r["dnf_lr"]:
        voter_info = "(LR: DNF, XGBoost: Zavrsio)"

    # Rezultat kartica
    st.markdown(f"""
    <div class="result-card {card_class}">
        <span class="result-text {text_class}">{label}</span>
        <span style="color:#999; font-size:0.9em;">{voter_info}</span>
    </div>
    """, unsafe_allow_html=True)

    avg_thresh = (st.session_state.threshold_xgb + st.session_state.threshold_lr) / 2
    st.plotly_chart(create_gauge(max_proba, avg_thresh), use_container_width=True)

    st.markdown("---")
    st.markdown("### Detalji predikcije (OR voting)")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.metric("Staza", p["circuit"])
    with d2:
        st.metric("Vozac", p["rider"])
    with d3:
        st.metric(f"XGBoost ({st.session_state.threshold_xgb:.0%})",
                  f"{proba_xgb:.1%} {'DNF' if r['dnf_xgb'] else 'OK'}")
    with d4:
        st.metric(f"LR ({st.session_state.threshold_lr:.0%})",
                  f"{proba_lr:.1%} {'DNF' if r['dnf_lr'] else 'OK'}")
    with d5:
        st.metric("Godina / Trka", f"{p['year']} / #{p['sequence']}")

    st.markdown("---")
    st.markdown("### Indikatori rizika")

    indicators = [
        ("Istorijski DNF\n(vozač + staza)", features['Historical_DNF_Rate'],
         "warn" if features['Historical_DNF_Rate'] > 0.15 else "good"),
        ("DNF rate\ntima", features['Team_DNF_Rate'],
         "warn" if features['Team_DNF_Rate'] > 0.12 else "good"),
        ("DNF rate\nstaze", features['Track_DNF_Rate'],
         "warn" if features['Track_DNF_Rate'] > 0.10 else "good"),
        ("DNF rate\nkonstruktora", features['Bike_DNF_Rate'],
         "warn" if features['Bike_DNF_Rate'] > 0.12 else "good"),
        ("Iskustvo\nvozača", features['Rider_Experience'],
         "good" if features['Rider_Experience'] > 30 else "warn"),
    ]

    cols = st.columns(len(indicators))
    for col, (label, value, mood) in zip(cols, indicators):
        with col:
            fmt = f"{value:.1%}" if isinstance(value, float) and value < 1 else f"{value}"
            st.markdown(f"""
            <div class="indicator-card {mood}">
                <div class="value">{fmt}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#555;">
        <span style="font-size:4em;">🏍️</span>
        <h3 style="color:#888; margin-top:20px;">Popunite parametre u sidebar-u i kliknite</h3>
        <h2 style="color:#d32f2f; font-weight:900;">PREDVIDI ISHOD</h2>
        <p style="color:#666;">ili izaberite jedan od brzih scenarija iznad ⚡</p>
    </div>
    """, unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("---")
st.markdown(f"""
<div class="custom-footer">
    MotoGP DNF Prediktor &nbsp;|&nbsp; ML Projekat SAUSAU 2026 &nbsp;|&nbsp;
    Ensemble OR (XGBoost + LR) · thresh={st.session_state.threshold_xgb:.0%}/{st.session_state.threshold_lr:.0%} · FN(test)=10/244
</div>
""", unsafe_allow_html=True)
