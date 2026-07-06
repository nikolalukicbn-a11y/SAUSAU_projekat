"""
KORAK 4c: STACKING ENSEMBLE
- Base modeli: XGBoost, Logistic Regression, Random Forest, MLP
- Meta-learner: Logistic Regression (sa calibration)
- Koristi out-of-fold predikcije za meta-features (StackingClassifier)
- Threshold tuning na val skupu
- Poredjenje sa XGBoost standalone
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

COST_FN = 5
COST_FP = 1

print("=" * 60)
print("KORAK 4c: STACKING ENSEMBLE")
print("  Base: XGBoost + LR + RF + MLP, Meta: LogisticRegression")
print("=" * 60)

# ============================================================
# 1. UCITAVANJE
# ============================================================
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

print(f"Train: {X_train.shape}, DNF={y_train.sum()}")
print(f"Val:   {X_val.shape}, DNF={y_val.sum()}")
print(f"Test:  {X_test.shape}, DNF={y_test.sum()}")

BASELINE_COST = y_test.sum() * COST_FN
print(f"Baseline cost: {BASELINE_COST}")

# ============================================================
# 2. BASE MODELI (sa najboljim parametrima iz prethodnih treninga)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = [
    ("xgb", XGBClassifier(
        learning_rate=0.01, max_depth=6, n_estimators=200,
        scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss"
    )),
    ("lr", LogisticRegression(
        C=10, class_weight="balanced", max_iter=2000, random_state=42
    )),
    ("rf", RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(50, 25), activation="relu", alpha=0.0001,
        learning_rate_init=0.001, early_stopping=True,
        max_iter=1000, random_state=42
    )),
]

# ============================================================
# 3. STACKING
# ============================================================
meta_learner = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=cv,  # out-of-fold predikcije
    n_jobs=-1,
    passthrough=False
)

print("\nTreniranje stacking ensemble-a...")
start = time.time()
stack.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Trajanje: {elapsed:.1f}s")

# ============================================================
# 4. TRENIRANJE INDIVIDUALNIH MODELA (za poredjenje)
# ============================================================
print("\nTreniranje individualnih modela (za poredjenje)...")
individual_models = {}

for name, model_cls in base_models:
    t0 = time.time()
    model_cls.fit(X_train, y_train)
    individual_models[name] = {
        "model": model_cls,
        "time": time.time() - t0
    }
    print(f"  {name}: {individual_models[name]['time']:.1f}s")

individual_models["stack"] = {"model": stack, "time": elapsed}

# ============================================================
# 5. THRESHOLD TUNING + EVALUACIJA
# ============================================================
results = {}

for name, info in individual_models.items():
    print(f"\n{'=' * 50}")
    print(f"EVALUACIJA: {name}")
    print(f"{'=' * 50}")

    model = info["model"]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Threshold tuning na val skupu
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thr = 0.5
    best_cost = float("inf")
    best_f1_on_val = 0
    thr_recall_target = None

    for t in thresholds:
        yp = (y_val_proba >= t).astype(int)
        cm = confusion_matrix(y_val, yp)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            cost = FN * COST_FN + FP * COST_FP
            f1v = f1_score(y_val, yp, zero_division=0)
            if cost < best_cost:
                best_cost = cost
                best_thr = t
                best_f1_on_val = f1v
            # Also track threshold that gives recall closest to XGBoost's (~92%)
            recv = recall_score(y_val, yp, zero_division=0)
            if recv >= 0.85 and (thr_recall_target is None or recv > recall_score(
                y_val, (y_val_proba >= thr_recall_target).astype(int), zero_division=0)):
                thr_recall_target = t

    # Evaluacija na testu sa najboljim thresholdom (min cost)
    yp_test = (y_test_proba >= best_thr).astype(int)
    cm_test = confusion_matrix(y_test, yp_test)
    TN, FP, FN, TP = cm_test.ravel()
    cost_test = FN * COST_FN + FP * COST_FP
    saving = (1 - cost_test / BASELINE_COST) * 100

    f1_t = f1_score(y_test, yp_test, zero_division=0)
    rec_t = recall_score(y_test, yp_test, zero_division=0)
    prec_t = precision_score(y_test, yp_test, zero_division=0)
    acc_t = accuracy_score(y_test, yp_test)

    print(f"  Best threshold (min cost): {best_thr:.2f}")
    print(f"    Val cost={best_cost:.0f}, Val F1={best_f1_on_val:.4f}")
    print(f"    Test: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    print(f"    Cost={cost_test} | Saving={saving:.1f}%")
    print(f"    F1={f1_t:.4f} | Rec={rec_t:.4f} | Prec={prec_t:.4f} | Acc={acc_t:.4f}")

    # Takođe test sa thresholdom koji daje recall ≥ 85% na val
    if thr_recall_target is not None:
        yp_rec = (y_test_proba >= thr_recall_target).astype(int)
        cm_rec = confusion_matrix(y_test, yp_rec)
        TNR, FPR, FNR, TPR = cm_rec.ravel()
        cost_rec = FNR * COST_FN + FPR * COST_FP
        saving_rec = (1 - cost_rec / BASELINE_COST) * 100
        rec_val_recall = recall_score(y_test, yp_rec, zero_division=0)
        print(f"\n  Threshold za recall ~92%: {thr_recall_target:.2f}")
        print(f"    Test: TN={TNR}, FP={FPR}, FN={FNR}, TP={TPR}")
        print(f"    Cost={cost_rec} | Rec={rec_val_recall:.4f} | "
              f"F1={f1_score(y_test, yp_rec, zero_division=0):.4f}")

    results[name] = {
        "model": model,
        "threshold": best_thr,
        "thr_recall": thr_recall_target,
        "cm": cm_test,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "f1": f1_t, "recall": rec_t, "precision": prec_t,
        "accuracy": acc_t, "cost": cost_test, "saving": saving,
        "val_cost": best_cost, "time": info["time"]
    }

# ============================================================
# 6. UPOREDNA TABELA
# ============================================================
print(f"\n{'=' * 60}")
print("UPOREDNA TABELA — STACKING vs INDIVIDUALNI")
print(f"{'=' * 60}")

rows = []
for name, r in results.items():
    rows.append({
        "Model": name.upper(),
        "Thresh": f"{r['threshold']:.2f}",
        "FN": r["FN"], "FP": r["FP"],
        "Recall": r["recall"], "Precision": r["precision"],
        "F1": r["f1"], "Cost": r["cost"],
        "Saving": f"{r['saving']:.1f}%",
        "Time": f"{r['time']:.1f}s"
    })
df = pd.DataFrame(rows).sort_values("F1", ascending=False)
print("\n" + df.to_string(index=False))
df.to_csv(os.path.join(METRICS_DIR, "stacking_comparison.csv"), index=False)

# ============================================================
# 7. POSEBNO POREDJENJE SA TRENUTNIM DEPLOYMENT MODELOM
# ============================================================
print(f"\n{'=' * 60}")
print("POREDJENJE SA TRENUTNIM XGBOOST DEPLOYMENTOM")
print(f"{'=' * 60}")

deploy = {"FN": 20, "FP": 1540, "Recall": 0.918, "F1": 0.224, "Cost": 1640}

for name, r in results.items():
    delta_fn = r["FN"] - deploy["FN"]
    delta_fp = r["FP"] - deploy["FP"]
    delta_f1 = r["f1"] - deploy["F1"]
    delta_rec = r["recall"] - deploy["Recall"]
    print(f"\n{name.upper()} vs XGBoost deploy:")
    print(f"  Delta FN:  {delta_fn:+d} ({'BOLJE' if delta_fn < 0 else 'GORE'})")
    print(f"  Delta FP:  {delta_fp:+d} ({'BOLJE' if delta_fp < 0 else 'GORE'})")
    print(f"  Delta F1:  {delta_f1:+.4f}")
    print(f"  Delta Rec: {delta_rec:+.4f}")

# ============================================================
# 8. SNIMANJE NAJBOLJEG
# ============================================================
best_name = df.iloc[0]["Model"]
best_res = results[best_name.lower()]

joblib.dump(best_res["model"], os.path.join(MODELS_DIR, "best_model_stack.pkl"))
joblib.dump(best_res["threshold"], os.path.join(MODELS_DIR, "best_threshold_stack.pkl"))
print(f"\nSacuvan najbolji stacking model: {best_name} (thresh={best_res['threshold']:.2f})")
print(f"  F1={best_res['f1']:.4f}, Rec={best_res['recall']:.4f}, "
      f"FN={best_res['FN']}, FP={best_res['FP']}")

print("\nKRAJ KORAKA 4c")
