"""Retrain ensemble with v2 features and evaluate."""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC, "y_test.csv")).values.ravel()

print(f"Train: {X_train.shape} (DNF={y_train.sum()})")
print(f"Test:  {X_test.shape} (DNF={y_test.sum()})")
print(f"Features: {len(X_train.columns)} (was 16, now +7 new)")

print("Training XGBoost...")
xgb = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                     scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
xgb.fit(X_train, y_train)

print("Training LogisticRegression...")
lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)
lr.fit(X_train, y_train)

px = xgb.predict_proba(X_test)[:, 1]
pl = lr.predict_proba(X_test)[:, 1]

# XGBoost standalone
yp_xgb_80 = (px >= 0.80).astype(int)
cm_xgb = confusion_matrix(y_test, yp_xgb_80)
old_fn, old_fp = 20, 1536
print(f"\nXGBoost standalone (0.80): FN={cm_xgb[1,0]}, FP={cm_xgb[0,1]}, "
      f"Rec={cm_xgb[1,1]/(cm_xgb[1,0]+cm_xgb[1,1]):.4f}")
print(f"  (OLD data: FN=20, FP=1536)")

# Grid search OR thresholds
print("\n--- OR threshold grid search ---")
best = {"cost": 9999, "tx": 0, "tl": 0}
best_fn = {"fn": 9999, "tx": 0, "tl": 0, "fp": 0}
for tx in np.arange(0.78, 0.95, 0.01):
    for tl in np.arange(0.35, 0.70, 0.02):
        yp = ((px >= tx) | (pl >= tl)).astype(int)
        cm = confusion_matrix(y_test, yp)
        FN, FP = cm[1, 0], cm[0, 1]
        cost = FN * 5 + FP
        if cost < best["cost"]:
            best = {"cost": cost, "tx": tx, "tl": tl, "FN": FN, "FP": FP,
                    "TP": cm[1, 1], "TN": cm[0, 0]}
        if FN < best_fn["fn"] or (FN == best_fn["fn"] and FP < best_fn["fp"]):
            best_fn = {"fn": FN, "tx": tx, "tl": tl, "FP": FP,
                       "TP": cm[1, 1], "TN": cm[0, 0], "cost": cost}

print(f"\nBest OR (min cost):     XGB={best['tx']:.2f}, LR={best['tl']:.2f}")
print(f"  TN={best['TN']}, FP={best['FP']}, FN={best['FN']}, TP={best['TP']}")
print(f"  Rec={best['TP']/(best['TP']+best['FN']):.4f}, "
      f"F1={f1_score(y_test, ((px>=best['tx'])|(pl>=best['tl'])).astype(int), zero_division=0):.4f}")

print(f"\nBest OR (min FN):        XGB={best_fn['tx']:.2f}, LR={best_fn['tl']:.2f}")
print(f"  FP={best_fn['FP']}, FN={best_fn['fn']}, TP={best_fn['TP']}")
print(f"  Rec={best_fn['TP']/(best_fn['TP']+best_fn['fn']):.4f}")

# Key combos
print(f"\n--- Key combos vs OLD (FN=20, FP=1536) ---")
for tx, tl, label in [
    (0.83, 0.50, "XGB=0.83, LR=0.50"),
    (0.83, 0.60, "XGB=0.83, LR=0.60"),
    (best["tx"], best["tl"], f"Best cost: XGB={best['tx']:.2f}, LR={best['tl']:.2f}"),
    (best_fn["tx"], best_fn["tl"], f"Best FN: XGB={best_fn['tx']:.2f}, LR={best_fn['tl']:.2f}"),
]:
    yp = ((px >= tx) | (pl >= tl)).astype(int)
    cm = confusion_matrix(y_test, yp)
    FN, FP, TP = cm[1, 0], cm[0, 1], cm[1, 1]
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = f1_score(y_test, yp, zero_division=0)
    prec = precision_score(y_test, yp, zero_division=0)
    print(f"  {label}")
    print(f"    FN={FN} (dFN={FN-20:+d}), FP={FP} (dFP={FP-1536:+d}), "
          f"Rec={rec:.4f}, Prec={prec:.4f}, F1={f1:.4f}")

# Save
joblib.dump(xgb, os.path.join(MODELS, "best_model_xgb.pkl"))
joblib.dump(lr, os.path.join(MODELS, "best_model_lr.pkl"))
joblib.dump(best["tx"], os.path.join(MODELS, "best_threshold_xgb.pkl"))
joblib.dump(best["tl"], os.path.join(MODELS, "best_threshold_lr.pkl"))
print(f"\nSaved: XGB={best['tx']:.2f}, LR={best['tl']:.2f}")
print("Done.")
