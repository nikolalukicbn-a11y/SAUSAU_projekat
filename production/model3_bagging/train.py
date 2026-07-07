"""
Model 3: Bagging Ensemble sa K-voting-om (5x XGBoost + 5x LR)
================================================================
Arhitektura: 10 nezavisnih kalibrisanih modela + K-voting konsenzus.

Bootstrap aggregating (bagging):
  - Iz trening skupa (15539 redova) generise se 5 bootstrap uzoraka
  - Bootstrap = izvlacenje SA VRACANJEM (isti red moze biti izvucen vise puta)
  - Svaki bag ima istu velicinu (15539), ali ~37% originalnih redova FALI
  - Na svakom bag-u se trenira jedan XGBoost + jedan LR (oba kalibrisana)
  - Ukupno: 5 x 2 = 10 modela

K-voting:
  - Svaki od 10 modela glasa: DNF (1) ili Zavrsio (0)
  - K = minimalan broj glasova potreban za DNF predikciju
  - K=2 (default): bar 2 modela moraju da se sloze — filtrira random greske

Zasto bagging pomaze:
  - Pojedinacni model pravi random greske
  - 10 modela na razlicitim podskupovima => njihove greske su RAZLICITE
  - Konsenzus (K>=2) ponistava random greske, zadrzava pravi signal

NAPOMENA: Podaci (X_train.csv) su vec TargetEncoded + StandardScaled + SMOTE 60/40
iz koraka 3 (preprocessing.py). Nema potrebe za ponovnim preprocesiranjem.

Hiperparametri:
  XGBoost:  lr=0.01, depth=6, n=200, scale_pos_weight=8
  LR:       C=10, class_weight="balanced"
  Kal.:     cv=3, method="sigmoid" (unutar svakog bag-a)
  Bagovi:   5
  Threshold: 0.35 (isti za svih 10 modela)
  Default K: 2

Ulaz:
  - data/processed/X_train.csv, y_train.csv  (vec obradjeni)
  - data/processed/X_test.csv,  y_test.csv

Izlaz:
  - models/model3_bagging/bagged_models.pkl    (lista 10 tuple-ova)
  - models/model3_bagging/bagging_k.pkl        (default K = 2)
  - models/model3_bagging/thresholds.pkl       (dict: {"threshold": 0.35, "K": 2})
  - models/model3_bagging/preprocessor.pkl     (kopija iz models/)
  - results/model3_bagging/figures/            (confusion matrix + K vs performance)
  - results/model3_bagging/metrics/            (metrics.csv + metrics_by_K.csv)

Metrike na test skupu (K=2, thr=0.35):
  TN=373, FP=1625, FN=6, TP=238
  Accuracy=0.273 | Precision=0.128 | Recall=0.975 | F1=0.226
  Propuštenih DNF-ova: 6/244 (2.5%)
"""

import os
import shutil
import time

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

matplotlib.use("Agg")

# ============================================================
# 1. PUTANJE I FOLDERI
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, "models", "model3_bagging")
RESULTS_DIR = os.path.join(BASE, "results", "model3_bagging")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# ============================================================
# 2. UCITAVANJE PODATAKA (vec enkodirani i skalirani iz koraka 3)
# ============================================================
X_train = pd.read_csv(os.path.join(BASE, "data", "processed", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "processed", "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(BASE, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(BASE, "data", "processed", "y_test.csv")).values.ravel()

# ============================================================
# 3. KONFIGURACIJA
# ============================================================
N_BAGS = 5        # broj bootstrap uzoraka (svaki generise XGBoost + LR)
THRESHOLD = 0.35  # prag verovatnoce DNF po modelu
BEST_K = 2        # default K — minimalan broj glasova za DNF

print("=" * 55)
print(f"MODEL 3: Bagging Ensemble ({N_BAGS}x XGBoost + {N_BAGS}x LR)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Threshold po modelu: {THRESHOLD}, Default K: {BEST_K}")
print("=" * 55)

# ============================================================
# 4. BOOTSTRAP UZORKOVANJE + TRENIRANJE 10 MODELA
# ============================================================
# Za svaki od 5 bagova:
#   1) Generisemo bootstrap uzorak (nasumicno izvlacenje SA VRACANJEM)
#   2) Treniramo XGBoost + Platt kalibracija (3-fold CV unutar bag-a)
#   3) Treniramo LR + Platt kalibracija (3-fold CV unutar bag-a)
# Rezultat: 5 XGBoost + 5 LR = 10 modela

xgb_models = []  # lista od 5 kalibrisanih XGBoost modela
lr_models = []   # lista od 5 kalibrisanih LR modela

np.random.seed(42)  # fiksni seed za ponovljivost

for i in range(N_BAGS):
    print(f"\nBag {i+1}/{N_BAGS}...")

    # Bootstrap uzorak: izvlacimo 15539 redova SA VRACANJEM
    # Neki redovi se pojavljuju 2x-3x, neki 0x (~37% fali)
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bag = X_train.iloc[idx].values
    y_bag = y_train[idx]
    print(f"  Uzorak: {len(X_bag)} redova, DNF={y_bag.sum()}")

    # XGBoost — isti hiperparametri kao Model 1 i 2
    # random_state se menja (42+i) => svaki bag ima drugaciju inicijalizaciju
    xgb = XGBClassifier(
        learning_rate=0.01, max_depth=6, n_estimators=200,
        scale_pos_weight=8, random_state=42 + i,
        n_jobs=-1, eval_metric="logloss",
    )
    xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="sigmoid", n_jobs=-1)
    xgb_cal.fit(X_bag, y_bag)
    xgb_models.append(xgb_cal)

    # LogisticRegression
    lr = LogisticRegression(
        C=10, class_weight="balanced", max_iter=2000, random_state=42 + i,
    )
    lr_cal = CalibratedClassifierCV(lr, cv=3, method="sigmoid", n_jobs=-1)
    lr_cal.fit(X_bag, y_bag)
    lr_models.append(lr_cal)

    print(f"  XGBoost + LR istrenirani i kalibrisani")

# ============================================================
# 5. K-VOTING — EVALUACIJA ZA SVE K VREDNOSTI
# ============================================================
print(f"\n--- Evaluacija K-voting (thr={THRESHOLD}) ---")
print(f"  {'K':>4} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5} {'Rec':>7} {'Prec':>7} {'F1':>7}")

# Generisemo glasove svih 10 modela za svaki test uzorak
# Svaki model: verovatnoca DNF >= THRESHOLD => glas 1 (DNF), inace 0
votes = np.zeros(len(y_test), dtype=int)

for i in range(N_BAGS):
    # XGBoost iz bag-a i
    px = xgb_models[i].predict_proba(X_test.values)[:, 1]
    votes += (px >= THRESHOLD).astype(int)
    # LR iz bag-a i
    pl = lr_models[i].predict_proba(X_test.values)[:, 1]
    votes += (pl >= THRESHOLD).astype(int)

# Evaluacija za svako K od 1 do 10
all_results = []
for k in range(1, 11):
    y_pred = (votes >= k).astype(int)  # DNF ako >= K modela glasa DNF
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = f1_score(y_test, y_pred, zero_division=0)
    all_results.append({
        "K": k, "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "Recall": rec, "Precision": prec, "F1": f1,
    })
    print(f"  {k:>4} {TN:>5} {FP:>5} {FN:>5} {TP:>5} {rec:>6.4f} {prec:>6.4f} {f1:>6.4f}")

# ============================================================
# 6. EVALUACIJA ZA DEFAULT K=2
# ============================================================
y_pred = (votes >= BEST_K).astype(int)
cm_best = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm_best.ravel()

print(f"\n--- Default K={BEST_K} ---")
print(f"Matrica konfuzije:")
print(f"                  Predvideno Zavrsio  Predvideno DNF")
print(f"  Stvarno Zavrsio       {TN:<5}            {FP:<5}")
print(f"  Stvarno DNF           {FN:<5}            {TP:<5}")

print(f"\nMetrike:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"\nDetaljan izvestaj:")
print(classification_report(y_test, y_pred, target_names=["Zavrsio", "DNF"], zero_division=0))

# ============================================================
# 7. CUVANJE MODELA
# ============================================================
# Spajamo svih 10 modela u jednu listu tuple-ova (naziv, model)
# Format: [("xgb_0", model), ("lr_0", model), ..., ("xgb_4", model), ("lr_4", model)]
all_models = (
    [(f"xgb_{i}", xgb_models[i]) for i in range(N_BAGS)]
    + [(f"lr_{i}", lr_models[i]) for i in range(N_BAGS)]
)

joblib.dump(all_models, os.path.join(MODEL_DIR, "bagged_models.pkl"))
joblib.dump(BEST_K, os.path.join(MODEL_DIR, "bagging_k.pkl"))
joblib.dump({"threshold": THRESHOLD, "K": BEST_K}, os.path.join(MODEL_DIR, "thresholds.pkl"))

# Kopiramo preprocessor iz models/ (generisan u koraku 3)
prep_src = os.path.join(BASE, "models", "preprocessor.pkl")
prep_dst = os.path.join(MODEL_DIR, "preprocessor.pkl")
if os.path.exists(prep_src):
    shutil.copy(prep_src, prep_dst)
    print(f"\nPreprocessor kopiran iz models/preprocessor.pkl")
else:
    print(f"\nUPOZORENJE: models/preprocessor.pkl ne postoji — pokreni src/preprocessing.py")

print(f"Modeli sacuvani u: {MODEL_DIR}/")

# ============================================================
# 8. CUVANJE METRIKA I PLOTOVA
# ============================================================

# 8a. Metrike za svih 10 K vrednosti (metrics_by_K.csv)
all_df = pd.DataFrame(all_results)
all_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics_by_K.csv"), index=False)

# 8b. Metrike za default K=2 (metrics.csv)
best_df = pd.DataFrame([{
    "Model": "Bagging Ensemble (K-voting)",
    "K": BEST_K, "Threshold": THRESHOLD,
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
}])
best_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics.csv"), index=False)

# 8c. Plot: K vs Recall/Precision/F1
fig, ax = plt.subplots(figsize=(8, 5))
df_plot = pd.DataFrame(all_results)
ax.plot(df_plot["K"], df_plot["Recall"], "b-o", label="Recall", linewidth=2)
ax.plot(df_plot["K"], df_plot["Precision"], "r-s", label="Precision", linewidth=2)
ax.plot(df_plot["K"], df_plot["F1"], "g-^", label="F1", linewidth=2)
ax.axvline(x=BEST_K, color="black", linestyle="--", label=f"Default K={BEST_K}")
ax.set_xlabel("K (minimalan broj glasova za DNF)")
ax.set_ylabel("Score")
ax.set_title(f"K-Voting performanse (threshold={THRESHOLD})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "k_voting_performance.png"), dpi=150)
plt.close()

# 8d. Plot: Matrica konfuzije za default K=2
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_best, display_labels=["Zavrsio", "DNF"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Model 3: Bagging (K={BEST_K}, thr={THRESHOLD})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "confusion_matrix.png"), dpi=150)
plt.close()

print(f"Metrike sacuvane u: {RESULTS_DIR}/")
print("Zavrseno.")
