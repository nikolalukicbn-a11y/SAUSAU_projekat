"""
Joint threshold tuning za ensemble OR (XGBoost + LR).
Pronalazi optimalni par thresholda koji minimizuje FN * 5 + FP * 1.
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

COST_FN = 5
COST_FP = 1

X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

print("Ucitavanje modela...")
xgb = joblib.load(os.path.join(MODELS_DIR, "best_model_xgb.pkl"))
lr = joblib.load(os.path.join(MODELS_DIR, "best_model_lr.pkl"))

val_p_xgb = xgb.predict_proba(X_val)[:, 1]
val_p_lr = lr.predict_proba(X_val)[:, 1]
test_p_xgb = xgb.predict_proba(X_test)[:, 1]
test_p_lr = lr.predict_proba(X_test)[:, 1]

BASELINE_COST = y_test.sum() * COST_FN
print(f"Baseline cost (uvek Zavrsio): test={BASELINE_COST}")
print(f"Val: {len(y_val)} redova, DNF={y_val.sum()}")
print(f"Test: {len(y_test)} redova, DNF={y_test.sum()}")

# Individualni thresholdi (samostalno)
print("\n--- Individualni najbolji thresholdi (test) ---")
for name, proba, color in [
    ("XGBoost", test_p_xgb, "blue"),
    ("LR", test_p_lr, "green")
]:
    best_cost = float("inf")
    best_t = 0.5
    best_res = None
    for t in np.arange(0.01, 0.99, 0.01):
        yp = (proba >= t).astype(int)
        cm = confusion_matrix(y_test, yp)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            cost = FN * COST_FN + FP * COST_FP
            if cost < best_cost:
                best_cost = cost
                best_t = t
                best_res = (TN, FP, FN, TP)
    TN, FP, FN, TP = best_res
    print(f"  {name}: thr={best_t:.2f}, cost={best_cost}, "
          f"FN={FN}, FP={FP}, Rec={TP/(TP+FN):.4f}, Prec={TP/(TP+FP):.4f}")

# Joint threshold tuning (val skup)
print("\n--- Joint OR threshold tuning (val skup) ---")

# XGBoost radi dobro sam, LR mu pomaze na visem thresholdu
xgb_range = np.arange(0.78, 0.95, 0.01)
lr_range = np.arange(0.40, 0.65, 0.01)

all_pairs = []
for tx in xgb_range:
    for tl in lr_range:
        yp_val = ((val_p_xgb >= tx) | (val_p_lr >= tl)).astype(int)
        cm = confusion_matrix(y_val, yp_val)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            cost = FN * COST_FN + FP * COST_FP
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            all_pairs.append({
                "thr_xgb": tx, "thr_lr": tl,
                "cost": cost, "FN": FN, "FP": FP,
                "recall": rec,
                "f1": f1_score(y_val, yp_val, zero_division=0)
            })

df_val = pd.DataFrame(all_pairs).sort_values("cost")

print("\nTop 10 parova po costu (val):")
print(df_val.head(10).to_string(index=False))

# Testiramo top 5 na test skupu
print("\n--- Evaluacija top 5 parova na TEST skupu ---")
for i, row in df_val.head(5).iterrows():
    tx = row["thr_xgb"]
    tl = row["thr_lr"]
    yp_test = ((test_p_xgb >= tx) | (test_p_lr >= tl)).astype(int)
    cm = confusion_matrix(y_test, yp_test)
    TN, FP, FN, TP = cm.ravel()
    cost = FN * COST_FN + FP * COST_FP
    saving = (1 - cost / BASELINE_COST) * 100
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = f1_score(y_test, yp_test, zero_division=0)
    print(f"  #{i+1}: XGB={tx:.2f}, LR={tl:.2f} -> "
          f"FN={FN}, FP={FP}, Rec={rec:.4f}, Prec={prec:.4f}, "
          f"F1={f1:.4f}, Cost={cost} (saving={saving:.1f}%)")

# Poredjenje sa deployment XGBoost (FN=20, FP=1540)
print("\n--- Poredjenje sa XGBoost standalone (FN=20, FP=1540) ---")
deploy = {"FN": 20, "FP": 1540, "Recall": 0.918, "F1": 0.224}
for i, row in df_val.head(5).iterrows():
    tx = row["thr_xgb"]
    tl = row["thr_lr"]
    yp_test = ((test_p_xgb >= tx) | (test_p_lr >= tl)).astype(int)
    cm = confusion_matrix(y_test, yp_test)
    TN, FP, FN, TP = cm.ravel()
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    cost = FN * COST_FN + FP * COST_FP
    deploy_cost = deploy["FN"] * COST_FN + deploy["FP"] * COST_FP
    delta_cost = cost - deploy_cost
    print(f"  XGB={tx:.2f}+LR={tl:.2f}: FN={FN} (delta:{FN-deploy['FN']:+d}), "
          f"FP={FP} (delta:{FP-deploy['FP']:+d}), Rec={rec:.4f}, "
          f"Cost delta: {delta_cost:+d}")

# Snimi najbolji par
best = df_val.iloc[0]
joblib.dump(best["thr_xgb"], os.path.join(MODELS_DIR, "best_threshold_xgb.pkl"))
joblib.dump(best["thr_lr"], os.path.join(MODELS_DIR, "best_threshold_lr.pkl"))
print(f"\nSacuvani thresholdi: XGB={best['thr_xgb']:.2f}, LR={best['thr_lr']:.2f}")
print("Gotovo.")
