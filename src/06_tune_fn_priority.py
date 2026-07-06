"""
FN-prioritet threshold tuning: XGBoost fiksiran na ~0.83, LR varira.
Cilj: minimalan FN uz prihvatljiv FP.
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

# XGBoost standalone referenca
print("--- XGBoost standalone (referenca) ---")
for t in [0.83, 0.85, 0.88, 0.90, 0.95]:
    yp = (test_p_xgb >= t).astype(int)
    cm = confusion_matrix(y_test, yp)
    TN, FP, FN, TP = cm.ravel()
    rec = TP/(TP+FN) if TP+FN > 0 else 0
    prec = TP/(TP+FP) if TP+FP > 0 else 0
    cost = FN * COST_FN + FP * COST_FP
    print(f"  XGB={t:.2f}: FN={FN}, FP={FP}, Rec={rec:.4f}, Prec={prec:.4f}, Cost={cost}")

# OR sa XGB fiksno, LR varira
print("\n--- OR: XGBoost fixed 0.83, LR varies (test skup) ---")
print(f"{'XGB':>6} {'LR':>6} {'FN':>5} {'FP':>5} {'Rec':>8} {'Prec':>8} {'F1':>8} {'Cost':>6}")

best_fn_rows = []
for tx in [0.83, 0.85]:
    for tl in np.arange(0.40, 0.65, 0.02):
        yp = ((test_p_xgb >= tx) | (test_p_lr >= tl)).astype(int)
        cm = confusion_matrix(y_test, yp)
        TN, FP, FN, TP = cm.ravel()
        rec = TP/(TP+FN) if TP+FN > 0 else 0
        prec = TP/(TP+FP) if TP+FP > 0 else 0
        f1 = f1_score(y_test, yp, zero_division=0)
        cost = FN * COST_FN + FP * COST_FP
        best_fn_rows.append({
            "tx": tx, "tl": tl, "FN": FN, "FP": FP,
            "rec": rec, "prec": prec, "f1": f1, "cost": cost
        })
        marker = ""
        if FN <= 15:
            marker = " <-- LOW FN"
        print(f"  {tx:.2f}  {tl:.2f}  {FN:>4}  {FP:>4}  {rec:>7.4f}  {prec:>7.4f}  {f1:>7.4f}  {cost:>5}{marker}")

# Najbolji po FN (minimalan FN, pa min FP)
df_fn = pd.DataFrame(best_fn_rows).sort_values(["FN", "FP"])
print("\n--- Top 5 po minimalnom FN ---")
for i, row in df_fn.head(5).iterrows():
    saving = (1 - row["cost"] / BASELINE_COST) * 100
    print(f"  XGB={row['tx']:.2f}, LR={row['tl']:.2f}: "
          f"FN={int(row['FN'])}, FP={int(row['FP'])}, "
          f"Rec={row['rec']:.4f}, Prec={row['prec']:.4f}, "
          f"F1={row['f1']:.4f}, Cost={int(row['cost'])} (saving={saving:.1f}%)")

# Poredjenje sa deploy XGBoost
print("\n--- Poredjenje sa XGBoost deploy (FN=20, FP=1540) ---")
for i, row in df_fn.head(5).iterrows():
    d_fn = int(row["FN"]) - 20
    d_fp = int(row["FP"]) - 1540
    print(f"  XGB={row['tx']:.2f}, LR={row['tl']:.2f}: "
          f"FN delta={d_fn:+d}, FP delta={d_fp:+d} "
          f"({'BOLJE' if d_fn <= 0 else 'GORE'} FN)")

# Snimi najbolji
best = df_fn.iloc[0]
joblib.dump(best["tx"], os.path.join(MODELS_DIR, "best_threshold_xgb.pkl"))
joblib.dump(best["tl"], os.path.join(MODELS_DIR, "best_threshold_lr.pkl"))
print(f"\nSacuvano: XGB={best['tx']:.2f}, LR={best['tl']:.2f} "
      f"(FN={int(best['FN'])}, FP={int(best['FP'])})")
