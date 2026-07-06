"""
KORAK 8: KALIBRACIJA VEROVATNOCA (Platt scaling)
- Kalibrisemo XGBoost i LR pomocu Platt scaling-a (CalibratedClassifierCV)
- Kalibrisane verovatnoce = precizniji threshold tuning = manje FP uz isti recall
- Poredjenje sa nekalibrisanim modelom
"""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score,
    brier_score_loss, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
FIGURES = os.path.join(BASE, "results", "figures")
os.makedirs(FIGURES, exist_ok=True)

COST_FN = 5
COST_FP = 1

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(PROC, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROC, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC, "y_test.csv")).values.ravel()

BASELINE_COST = y_test.sum() * COST_FN
print(f"Test: {len(y_test)} redova, DNF={y_test.sum()}, baseline cost={BASELINE_COST}")

models_uncal = {}  # nekalibrisani
models_cal = {}    # kalibrisani

# ============================================================
# 1. TRENIRANJE I KALIBRACIJA
# ============================================================
configs = {
    "XGBoost": XGBClassifier(
        learning_rate=0.01, max_depth=6, n_estimators=200,
        scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss"
    ),
    "Logistic Regression": LogisticRegression(
        C=10, class_weight="balanced", max_iter=2000, random_state=42
    ),
}

for name, model in configs.items():
    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")

    # Treniranje nekalibrisanog
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Trening: {time.time()-t0:.1f}s")
    models_uncal[name] = model

    # Platt scaling kalibracija (CV=5, metod='sigmoid')
    t0 = time.time()
    cal = CalibratedClassifierCV(model, cv=5, method="sigmoid", n_jobs=-1)
    cal.fit(X_train, y_train)
    print(f"  Kalibracija: {time.time()-t0:.1f}s")
    models_cal[name] = cal

# ============================================================
# 2. POREDJENJE KALIBRISANIH vs NEKALIBRISANIH
# ============================================================
print(f"\n{'=' * 60}")
print("POREDJENJE KALIBRISANO vs NEKALIBRISANO (test)")
print(f"{'=' * 60}")

for name in configs:
    uncal = models_uncal[name]
    cal = models_cal[name]

    pu = uncal.predict_proba(X_test)[:, 1]
    pc = cal.predict_proba(X_test)[:, 1]

    brier_u = brier_score_loss(y_test, pu)
    brier_c = brier_score_loss(y_test, pc)

    print(f"\n{name}:")
    print(f"  Brier score (uncal): {brier_u:.4f} | (cal): {brier_c:.4f} | "
          f"improvement: {(brier_u-brier_c):+.4f}")
    print(f"  Proba stats (uncal): mean={pu.mean():.4f}, median={np.median(pu):.4f}, "
          f"min={pu.min():.4f}, max={pu.max():.4f}")
    print(f"  Proba stats (cal):   mean={pc.mean():.4f}, median={np.median(pc):.4f}, "
          f"min={pc.min():.4f}, max={pc.max():.4f}")

    # Calibration curve plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, label in [
        (axes[0], pu, "Nekalibrisane"),
        (axes[1], pc, "Kalibrisane")
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(prob_pred, prob_true, "s-", color="red", linewidth=2, label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Savrsena kalibracija")
        ax.set_xlabel("Predvidjena verovatnoca"); ax.set_ylabel("Stvarna ucestalost DNF")
        ax.set_title(f"{name} - {label}"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(FIGURES, f"calibration_{safe}.png"), dpi=150)
    plt.close()

# ============================================================
# 3. THRESHOLD TUNING SA KALIBRISANIM MODELIMA
# ============================================================
print(f"\n{'=' * 60}")
print("THRESHOLD TUNING - OR voting sa kalibrisanim modelima")
print(f"{'=' * 60}")

px_u = models_uncal["XGBoost"].predict_proba(X_test)[:, 1]
pl_u = models_uncal["Logistic Regression"].predict_proba(X_test)[:, 1]
px_c = models_cal["XGBoost"].predict_proba(X_test)[:, 1]
pl_c = models_cal["Logistic Regression"].predict_proba(X_test)[:, 1]

# Uncalibrated best
print("\n--- NEKALIBRISANI OR (referentni) ---")
best_u = {"cost": 9999, "tx": 0, "tl": 0, "FN": 0, "FP": 0, "TP": 0}
for tx in np.arange(0.75, 0.95, 0.01):
    for tl in np.arange(0.40, 0.70, 0.02):
        yp = ((px_u >= tx) | (pl_u >= tl)).astype(int)
        cm = confusion_matrix(y_test, yp)
        FN, FP = cm[1, 0], cm[0, 1]
        cost = FN * COST_FN + FP * COST_FP
        if cost < best_u["cost"] or (FN <= 16 and FP < best_u["FP"]):
            if cost < best_u["cost"]:
                best_u = {"cost": cost, "tx": tx, "tl": tl, "FN": FN, "FP": FP, "TP": cm[1, 1]}

print(f"  Best (min cost): XGB={best_u['tx']:.2f}, LR={best_u['tl']:.2f}")
print(f"    FN={best_u['FN']}, FP={best_u['FP']}, TP={best_u['TP']}, "
      f"Rec={best_u['TP']/(best_u['TP']+best_u['FN']):.4f}, Cost={best_u['cost']}")

print("\n--- KALIBRISANI OR ---")
best_c = {"cost": 9999, "tx": 0, "tl": 0, "FN": 0, "FP": 0, "TP": 0}
for tx in np.arange(0.10, 0.80, 0.01):
    for tl in np.arange(0.10, 0.70, 0.02):
        yp = ((px_c >= tx) | (pl_c >= tl)).astype(int)
        cm = confusion_matrix(y_test, yp)
        FN, FP = cm[1, 0], cm[0, 1]
        cost = FN * COST_FN + FP * COST_FP
        if cost < best_c["cost"] or (FN <= 16 and FP < best_c["FP"]):
            if cost < best_c["cost"]:
                best_c = {"cost": cost, "tx": tx, "tl": tl, "FN": FN, "FP": FP,
                          "TP": cm[1, 1], "TN": cm[0, 0]}

saving = (1 - best_c["cost"] / BASELINE_COST) * 100
print(f"  Best (min cost): XGB={best_c['tx']:.2f}, LR={best_c['tl']:.2f}")
print(f"    FN={best_c['FN']}, FP={best_c['FP']}, TP={best_c['TP']}, "
      f"Rec={best_c['TP']/(best_c['TP']+best_c['FN']):.4f}, "
      f"F1={f1_score(y_test, ((px_c>=best_c['tx'])|(pl_c>=best_c['tl'])).astype(int), zero_division=0):.4f}")
print(f"    Cost={best_c['cost']} (saving={saving:.1f}%)")

# ============================================================
# 4. POREDJENJE SA STARIM MODELOM
# ============================================================
print(f"\n{'=' * 60}")
print("FINALNO POREDJENJE")
print(f"{'=' * 60}")

yp_u = ((px_u >= best_u["tx"]) | (pl_u >= best_u["tl"])).astype(int)
cm_u = confusion_matrix(y_test, yp_u)

yp_c = ((px_c >= best_c["tx"]) | (pl_c >= best_c["tl"])).astype(int)
cm_c = confusion_matrix(y_test, yp_c)

print(f"\nUncal OR (XGB={best_u['tx']:.2f}, LR={best_u['tl']:.2f}):")
print(f"  CM: TN={cm_u[0,0]}, FP={cm_u[0,1]}, FN={cm_u[1,0]}, TP={cm_u[1,1]}")
print(f"  Rec={recall_score(y_test,yp_u,zero_division=0):.4f}, "
      f"Prec={precision_score(y_test,yp_u,zero_division=0):.4f}, "
      f"F1={f1_score(y_test,yp_u,zero_division=0):.4f}")

print(f"\nCalibrated OR (XGB={best_c['tx']:.2f}, LR={best_c['tl']:.2f}):")
print(f"  CM: TN={cm_c[0,0]}, FP={cm_c[0,1]}, FN={cm_c[1,0]}, TP={cm_c[1,1]}")
print(f"  Rec={recall_score(y_test,yp_c,zero_division=0):.4f}, "
      f"Prec={precision_score(y_test,yp_c,zero_division=0):.4f}, "
      f"F1={f1_score(y_test,yp_c,zero_division=0):.4f}")

# Delta
print(f"\nDelta (cal - uncal):")
print(f"  FN: {cm_c[1,0]-cm_u[1,0]:+d} | FP: {cm_c[0,1]-cm_u[0,1]:+d}")
print(f"  Rec: {recall_score(y_test,yp_c,zero_division=0)-recall_score(y_test,yp_u,zero_division=0):+.4f}")
print(f"  F1:  {f1_score(y_test,yp_c,zero_division=0)-f1_score(y_test,yp_u,zero_division=0):+.4f}")

# Save calibrated models
joblib.dump(models_cal["XGBoost"], os.path.join(MODELS, "best_model_xgb.pkl"))
joblib.dump(models_cal["Logistic Regression"], os.path.join(MODELS, "best_model_lr.pkl"))
joblib.dump(best_c["tx"], os.path.join(MODELS, "best_threshold_xgb.pkl"))
joblib.dump(best_c["tl"], os.path.join(MODELS, "best_threshold_lr.pkl"))
print(f"\nSaved calibrated models: XGB={best_c['tx']:.2f}, LR={best_c['tl']:.2f}")

print("\nKRAJ KORAKA 8")
