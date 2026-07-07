"""
Model 1: XGBoost Standalone (SMOTE 60/40)
==========================================
Algoritam: XGBoost (gradient boosting) — jedan model, direktna klasifikacija.
Treniran na SMOTE 60/40 podacima (40% DNF, 60% Zavrsio).

Hiperparametri:
  - learning_rate = 0.01
  - max_depth     = 6
  - n_estimators  = 200
  - scale_pos_weight = 8  (kompenzuje disbalans klasa)

Threshold: 0.80
  Verovatnoca DNF >= 0.80 => DNF, inace => Zavrsio.

Ulaz:
  - data/processed/X_train.csv, y_train.csv  (SMOTE 60/40 trening podaci, 15539 redova)
  - data/processed/X_test.csv,  y_test.csv   (test podaci, 2242 reda, 244 DNF)

Izlaz:
  - models/model1_xgboost/model.pkl           (istrenirani XGBoost model)
  - models/model1_xgboost/threshold.pkl       (threshold vrednost 0.80)
  - results/model1_xgboost/figures/confusion_matrix.png
  - results/model1_xgboost/metrics/metrics.csv

Metrike na test skupu (2242 reda, 244 DNF-a):
  TN=462, FP=1536, FN=20, TP=224
  Accuracy=0.306 | Precision=0.127 | Recall=0.918 | F1=0.224
  DNF predikcija: 78.5% svih uzoraka
  Propuštenih DNF-ova: 20/244 (8.2%)
"""

import os
import time

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# Tri nivoa iznad: production/model1_xgboost/ -> production/ -> root/
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, "models", "model1_xgboost")
RESULTS_DIR = os.path.join(BASE, "results", "model1_xgboost")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# ============================================================
# 2. UCITAVANJE PODATAKA
# ============================================================
# Trening podaci: SMOTE 60/40 (15539 redova, 16 feature-a, 40% DNF)
X_train = pd.read_csv(os.path.join(BASE, "data", "processed", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "processed", "y_train.csv")).values.ravel()
# Test podaci: originalna distribucija (2242 reda, 10.9% DNF, nije SMOTE-ovan)
X_test = pd.read_csv(os.path.join(BASE, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(BASE, "data", "processed", "y_test.csv")).values.ravel()

# ============================================================
# 3. KONFIGURACIJA
# ============================================================
THRESHOLD = 0.80  # prag verovatnoce DNF: >= 0.80 => DNF

print("=" * 55)
print("MODEL 1: XGBoost Standalone (SMOTE 60/40)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Threshold: {THRESHOLD}")
print("=" * 55)

# ============================================================
# 4. TRENIRANJE MODELA
# ============================================================
print("\nTreniranje XGBoost-a...")
t0 = time.time()

# Inicijalizacija XGBoost klasifikatora sa optimizovanim hiperparametrima
# scale_pos_weight=8 daje 8x vecu tezinu DNF klasi pri racunanju gradijenta
model = XGBClassifier(
    learning_rate=0.01,
    max_depth=6,
    n_estimators=200,
    scale_pos_weight=8,
    random_state=42,
    n_jobs=-1,               # koristi sve CPU jezgre
    eval_metric="logloss",    # metrika za early stopping (nije koriscen ovde)
)

# Fit na SMOTE 60/40 trening podacima
model.fit(X_train, y_train)
print(f"Vreme treniranja: {time.time() - t0:.1f}s")

# ============================================================
# 5. PREDIKCIJA NA TEST SKUPU
# ============================================================
# predict_proba vraca [verovatnoca_klase_0, verovatnoca_klase_1]
# uzimamo drugu kolonu [:, 1] = verovatnoca DNF-a
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= THRESHOLD).astype(int)

# ============================================================
# 6. EVALUACIJA
# ============================================================

# Matrica konfuzije: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"\nMatrica konfuzije:")
print(f"                  Predvideno Zavrsio  Predvideno DNF")
print(f"  Stvarno Zavrsio       {TN:<5}            {FP:<5}  (FP = lazni alarm)")
print(f"  Stvarno DNF           {FN:<5}            {TP:<5}  (FN = propusten DNF)")

# Metrike
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
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(THRESHOLD, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"\nModel sacuvan u: {MODEL_DIR}/")

# ============================================================
# 8. CUVANJE METRIKA I PLOTOVA
# ============================================================

# 8a. CSV sa metrikama
metrics_df = pd.DataFrame([{
    "Model": "XGBoost Standalone (SMOTE 60/40)",
    "Threshold": THRESHOLD,
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
}])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics.csv"), index=False)

# 8b. Plot matrice konfuzije
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Zavrsio", "DNF"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Model 1: XGBoost Standalone SMOTE (thr={THRESHOLD})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "confusion_matrix.png"), dpi=150)
plt.close()

print(f"Metrike sacuvane u: {RESULTS_DIR}/")
print("Zavrseno.")
