"""
Trenira SVM na SMOTE 60/40 podacima sa threshold tuningom.
Optimizuje za visok recall uz prihvatljiv FP.
"""
import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

print(f"Train: {X_train.shape}, DNF={y_train.sum()}")
print(f"Val:   {X_val.shape}, DNF={y_val.sum()}")
print(f"Test:  {X_test.shape}, DNF={y_test.sum()}")

# GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = SVC(probability=True, random_state=42, max_iter=5000)
params = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf"],
    "class_weight": ["balanced"],
}

print("\nGridSearchCV SVM (F1 scoring)...")
start = time.time()
grid = GridSearchCV(model, params, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Trajanje: {elapsed:.1f}s")
print(f"Najbolji params: {grid.best_params_}")
print(f"Najbolji CV F1: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# Threshold tuning — skeniramo 0.05 do 0.95 sa korakom 0.01
y_val_proba = best_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
best = {"threshold": 0.5, "f1": 0, "recall": 0, "precision": 0, "fp": 0, "fn": 0, "cost": 9999}

print(f"\n{'Threshold':>10} {'F1':>8} {'Recall':>8} {'Prec':>8} {'FP':>6} {'FN':>6} {'Cost':>6}")
print("-" * 60)

COST_FN = 5
COST_FP = 1
baseline_cost = y_test.sum() * COST_FN

all_rows = []
for t in thresholds:
    y_pred = (y_val_proba >= t).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        continue
    f1 = f1_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    cost = FN * COST_FN + FP * COST_FP
    all_rows.append({"threshold": t, "f1": f1, "recall": rec, "precision": prec,
                     "fp": FP, "fn": FN, "cost": cost})
    if cost < best["cost"] or (f1 > best["f1"] and rec >= 0.35):
        if cost < best["cost"]:
            best = {"threshold": t, "f1": f1, "recall": rec, "precision": prec,
                    "fp": FP, "fn": FN, "cost": cost}
    print(f"{t:>10.2f} {f1:>8.4f} {rec:>8.4f} {prec:>8.4f} {FP:>6} {FN:>6} {cost:>6}")

print(f"\nNajbolji threshold (min cost): {best['threshold']:.2f}")
print(f"  Val: F1={best['f1']:.4f}, Rec={best['recall']:.4f}, Prec={best['precision']:.4f}")
print(f"  Val: FP={best['fp']}, FN={best['fn']}, Cost={best['cost']}")

# Evaluacija na testu sa najboljim thresholdom
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best["threshold"]).astype(int)
cm = confusion_matrix(y_test, y_test_pred)
TN, FP, FN, TP = cm.ravel()
total_cost = FN * COST_FN + FP * COST_FP
saving = (1 - total_cost / baseline_cost) * 100

print(f"\n--- TEST SKUP (threshold={best['threshold']:.2f}) ---")
print(f"  CM: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
print(f"  Cost={total_cost} (baseline={baseline_cost}, saving={saving:.1f}%)")
print(f"  Acc={accuracy_score(y_test, y_test_pred):.4f}, "
      f"F1={f1_score(y_test, y_test_pred, zero_division=0):.4f}, "
      f"Rec={recall_score(y_test, y_test_pred, zero_division=0):.4f}, "
      f"Prec={precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Zavrsio", "DNF"], zero_division=0))

# Top 3 thresholda po costu na val skupu
df_val = pd.DataFrame(all_rows).sort_values("cost")
print(f"\n--- Top 5 thresholda po costu (val) ---")
print(df_val.head(5).to_string(index=False))

# Snimanje
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model_svm.pkl"))
joblib.dump(best["threshold"], os.path.join(MODELS_DIR, "best_threshold_svm.pkl"))
print(f"\nSacuvano: best_model_svm.pkl, best_threshold_svm.pkl (thresh={best['threshold']:.2f})")
