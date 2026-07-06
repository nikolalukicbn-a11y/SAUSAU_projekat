# MotoGP DNF Prediktor

ML projekat za predmet **SAUSAU 2026** — predikcija ishoda MotoGP trke (Završio / DNF).  
Student: **Nikola Lukić RA47-2023**

---

## Struktura projekta

```
├── README.md
├── requirements.txt
├── docs/                          # Dokumentacija (PDF)
├── data/
│   ├── raw/                       # Sirovi podaci (FILTERED_ROWS.csv, Qualifying.csv, Race.csv)
│   └── processed/                 # Procesirani podaci (cleaned, encoded, train/val/test split)
├── dataset/                       # Originalne arhive podataka (zip)
├── src/
│   ├── data_load.py               # Korak 1: Učitavanje podataka
│   ├── data_cleaning.py           # Korak 2: Čišćenje i priprema target varijable
│   └── preprocessing.py           # Korak 3: Feature engineering, TargetEncoder, SMOTE, split
├── models/
│   ├── model1_xgboost/            # Model 1: XGBoost Standalone
│   ├── model2_or/                 # Model 2: OR Ensemble Calibrated
│   ├── model3_bagging/            # Model 3: Bagging Ensemble (deployovan)
│   └── experiments/               # Modeli iz eksperimenata
├── results/
│   ├── model1_xgboost/            # Metrike i plotovi za Model 1
│   ├── model2_or/                 # Metrike i plotovi za Model 2
│   ├── model3_bagging/            # Metrike i plotovi za Model 3
│   └── experiments/               # Rezultati svih eksperimenata
├── production/
│   ├── model1_xgboost/train.py    # Trenira i evaluira Model 1
│   ├── model2_or/train.py         # Trenira i evaluira Model 2
│   └── model3_bagging/train.py    # Trenira i evaluira Model 3
├── app/
│   └── app.py                     # Streamlit deployment aplikacija
└── experiments/                   # Svi eksperimentalni pokušaji
    ├── 01_cost_sensitive/
    ├── 02_anomaly_detection/
    ├── 03_stacking/
    └── ...
```

---

## Pipeline

1. **`src/data_load.py`** — Učitava `FILTERED_ROWS.csv` (1949-2022) i dopunske datasetove, snima u `data/processed/`
2. **`src/data_cleaning.py`** — Filtrira MotoGP + 500cc, kreira `Status_Zavrsetka` (0=Završio, 1=DNF), uklanja leak/redundantne kolone
3. **`src/preprocessing.py`** — Feature engineering (16 feature-a), TargetEncoder, StandardScaler, SMOTE 60/40, 70/15/15 split, snima `X_train`/`y_train` i preprocessor

### Feature-i (16):
- **Kategorički** (5): `shortname`, `circuit_name`, `rider_name`, `team_name`, `bike_name`
- **Numerički** (11): `year`, `sequence`, `Historical_DNF_Rate`, `Rider_Experience`, `Team_DNF_Rate`, `Bike_DNF_Rate`, `Track_DNF_Rate`, `Season_Phase`, `Quali_Spread`, `Quali_Top6_Gap`, `Quali_Time_Std`

---

## 3 Produkcijska modela

Svi modeli evaluirani na test skupu (2242 reda, 244 DNF-a)

### Model 1: XGBoost Standalone (SMOTE 60/40)

| Metrika | Vrednost |
|---|---|
| Algoritam | XGBoost (lr=0.01, depth=6, n=200, scale_pos=8) |
| Threshold | 0.80 |
| TN | 462 |
| FP | 1536 |
| **FN** | **20** |
| TP | 224 |
| **Recall** | **91.8%** |
| **Precision** | 12.7% |
| **F1** | 0.224 |

```
Pokretanje: python production/model1_xgboost/train.py
```

### Model 2: OR Ensemble Calibrated (Platt scaling)

| Metrika | Vrednost |
|---|---|
| Algoritam | XGBoost + LogisticRegression (oba kalibrisana Platt-om) |
| Logika | DNF ako XGBoost ≥ 0.35 **ILI** LR ≥ 0.30 |
| TN | 398 |
| FP | 1600 |
| **FN** | **9** |
| TP | 235 |
| **Recall** | **96.3%** |
| **Precision** | 12.8% |
| **F1** | 0.226 |

```
Pokretanje: python production/model2_or/train.py
```

### Model 3: Bagging Ensemble (K-voting) ★ Deployovan

| Metrika | Vrednost |
|---|---|
| Algoritam | 5× XGBoost + 5× LR (bootstrap, kalibrisano) |
| Logika | DNF ako **K ≥ 2** od 10 modela predvidi DNF (thr=0.35 po modelu) |
| TN | 373 |
| FP | 1625 |
| **FN** | **6** |
| TP | 238 |
| **Recall** | **97.5%** |
| **Precision** | 12.8% |
| **F1** | 0.226 |

```
Pokretanje: python production/model3_bagging/train.py
```

---

## Poređenje modela

| Model | FN | FP | Recall | Prec | F1 |
|---|---|---|---|---|---|
| XGBoost Standalone | 20 | **1536** | 91.8% | 12.7% | 0.224 |
| OR Calibrated | **9** | 1600 | **96.3%** | 12.8% | 0.226 |
| Bagging K=2 | **6** | 1625 | **97.5%** | 12.8% | 0.226 |

---

## Pokretanje aplikacije

```bash
# Instaliraj zavisnosti
pip install -r requirements.txt

# Pokreni celokupan pipeline (jednom)
cd src
python data_load.py
python data_cleaning.py
python preprocessing.py

# Treniraj produkcijske modele
cd ../production/model3_bagging
python train.py

# Pokreni Streamlit app
cd ../../app
streamlit run app.py
```

App: `http://localhost:8501`

### Parametri u app-u:
- **K** (1-10): minimalan broj glasova za DNF predikciju
- **Threshold**: prag verovatnoće po modelu (svih 10 modela koristi isti prag)

---

## Datum
Jul 2026
