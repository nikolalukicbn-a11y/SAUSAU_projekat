"""
Model 2: OR Ensemble Kalibrisan (XGBoost + LogisticRegression)
================================================================
Arhitektura: Dva nezavisna modela + OR logika.
  - XGBoost (gradient boosting) — primarni detektor DNF-a
  - LogisticRegression — sekundarni detektor, hvata DNF-ove koje XGBoost propusti
  - Oba modela su kalibrisana Platt scaling-om (CalibratedClassifierCV, method="sigmoid")
  - OR logika: DNF ako XGBoost ILI LR predvidi DNF

Kalibracija (Platt scaling):
  - XGBoost nekalibrisane verovatnoce su lose (Brier score 0.62, sve proba > 0.80)
  - Platt scaling popravlja Brier score na 0.33 i daje realne verovatnoce
  - Kalibracija koristi 5-fold CV unutar trening skupa

Hiperparametri:
  XGBoost:  lr=0.01, depth=6, n=200, scale_pos_weight=8
  LR:       C=10, class_weight="balanced"
  Kal.:     cv=5, method="sigmoid"

Thresholdi (optimizovani za minimalan FN):
  XGBoost: 0.35  (prag verovatnoce za XGBoost)
  LR:      0.30  (prag verovatnoce za LR)

NAPOMENA: Podaci (X_train.csv) su vec TargetEncoded + StandardScaled + SMOTE 60/40
iz koraka 3 (preprocessing.py). Nema potrebe za ponovnim preprocesiranjem.

Ulaz:
  - data/processed/X_train.csv, y_train.csv  (vec obradjeni)
  - data/processed/X_test.csv,  y_test.csv

Izlaz:
  - models/model2_or/xgboost_cal.pkl     (kalibrisani XGBoost)
  - models/model2_or/lr_cal.pkl          (kalibrisani LR)
  - models/model2_or/thresholds.pkl      (dict: {"xgb": 0.35, "lr": 0.30})
  - models/model2_or/preprocessor.pkl    (kopija iz models/ — za deployment)
  - results/model2_or/figures/           (confusion matrix + calibration curves)
  - results/model2_or/metrics/metrics.csv

Metrike na test skupu (2242 reda, 244 DNF-a):
  TN=398, FP=1600, FN=9, TP=235
  Accuracy=0.282 | Precision=0.128 | Recall=0.963 | F1=0.226
  Propuštenih DNF-ova: 9/244 (3.7%)
"""

import os
import shutil
import time

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
MODEL_DIR = os.path.join(BASE, "models", "model2_or")
RESULTS_DIR = os.path.join(BASE, "results", "model2_or")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# ============================================================
# 2. UCITAVANJE PODATAKA (vec enkodirani i skalirani iz koraka 3)
# ============================================================
# X_train.csv je vec TargetEncoded + StandardScaled + SMOTE 60/40
X_train = pd.read_csv(os.path.join(BASE, "data", "processed", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "processed", "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(BASE, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(BASE, "data", "processed", "y_test.csv")).values.ravel()

# ============================================================
# 3. KONFIGURACIJA
# ============================================================
TX, TL = 0.35, 0.30  # thresholdi: XGBoost, LogisticRegression

print("=" * 55)
print("MODEL 2: OR Ensemble Kalibrisan (Platt scaling)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Thresholds: XGB={TX}, LR={TL}")
print("=" * 55)

# ============================================================
# 4. TRENIRANJE + KALIBRACIJA XGBoost-a
# ============================================================
print("\nTreniranje XGBoost-a + Platt kalibracija...")
t0 = time.time()

# Bazni XGBoost model (nekalibrisan)
xgb = XGBClassifier(
    learning_rate=0.01,
    max_depth=6,
    n_estimators=200,
    scale_pos_weight=8,    # DNF klasa dobija 8x vecu tezinu
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

# Platt scaling: 5-fold CV kalibracija
# Za svaki fold: trenira XGBoost na 4 folda, fit-uje sigmoid na 1 foldu
# Rezultat: model.predict_proba() vraca kalibrisane verovatnoce
xgb_cal = CalibratedClassifierCV(xgb, cv=5, method="sigmoid", n_jobs=-1)
xgb_cal.fit(X_train, y_train)
print(f"  XGBoost kalibrisan: {time.time() - t0:.1f}s")

# ============================================================
# 5. TRENIRANJE + KALIBRACIJA LogisticRegression
# ============================================================
print("Treniranje LogisticRegression + Platt kalibracija...")
t0 = time.time()

lr = LogisticRegression(
    C=10,                    # manji C = jaca regularizacija
    class_weight="balanced",  # automatski balansira tezine klasa
    max_iter=2000,
    random_state=42,
)

lr_cal = CalibratedClassifierCV(lr, cv=5, method="sigmoid", n_jobs=-1)
lr_cal.fit(X_train, y_train)
print(f"  LR kalibrisan: {time.time() - t0:.1f}s")

# ============================================================
# 6. PREDIKCIJA SA OR LOGIKOM
# ============================================================
# Kalibrisane verovatnoce DNF-a od oba modela
px = xgb_cal.predict_proba(X_test)[:, 1]  # XGBoost
pl = lr_cal.predict_proba(X_test)[:, 1]   # LR

# OR logika: DNF ako BILO KOJI model predvidi DNF
y_pred = ((px >= TX) | (pl >= TL)).astype(int)

# ============================================================
# 7. EVALUACIJA
# ============================================================
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"\nMatrica konfuzije:")
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
# 8. CUVANJE MODELA
# ============================================================
joblib.dump(xgb_cal, os.path.join(MODEL_DIR, "xgboost_cal.pkl"))
joblib.dump(lr_cal, os.path.join(MODEL_DIR, "lr_cal.pkl"))
joblib.dump({"xgb": TX, "lr": TL}, os.path.join(MODEL_DIR, "thresholds.pkl"))

# Kopiramo preprocessor iz models/ (generisan u koraku 3)
# Potreban je za deployment — transformise nove ulazne podatke
prep_src = os.path.join(BASE, "models", "preprocessor.pkl")
prep_dst = os.path.join(MODEL_DIR, "preprocessor.pkl")
if os.path.exists(prep_src):
    shutil.copy(prep_src, prep_dst)
    print(f"\nPreprocessor kopiran iz models/preprocessor.pkl")
else:
    print(f"\nUPOZORENJE: models/preprocessor.pkl ne postoji — pokreni src/preprocessing.py")

print(f"Modeli sacuvani u: {MODEL_DIR}/")

# ============================================================
# 9. CUVANJE METRIKA I PLOTOVA
# ============================================================

# 9a. CSV sa metrikama
metrics_df = pd.DataFrame([{
    "Model": "OR Ensemble Kalibrisan (Platt scaling)",
    "XGB_Threshold": TX, "LR_Threshold": TL,
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
}])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics.csv"), index=False)

# 9b. Plot matrice konfuzije
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Zavrsio", "DNF"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Model 2: OR Kalibrisan (XGB={TX}, LR={TL})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "confusion_matrix.png"), dpi=150)
plt.close()

# 9c. Kalibracione krive — pokazuju pouzdanost verovatnoca
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, probs, label in [(axes[0], px, "XGBoost"), (axes[1], pl, "Logistic Regression")]:
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(prob_pred, prob_true, "s-", color="red", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Savrsena kalibracija")
    ax.set_xlabel("Predvidjena verovatnoca DNF")
    ax.set_ylabel("Stvarna ucestalost DNF")
    ax.set_title(f"Kalibraciona kriva — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "calibration_curves.png"), dpi=150)
plt.close()

print(f"Metrike sacuvane u: {RESULTS_DIR}/")
print("Zavrseno.")
