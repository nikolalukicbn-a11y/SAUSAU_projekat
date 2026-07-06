"""
KORAK 4d: ENSEMBLE OR VOTING (max recall)
- 4 modela: XGBoost, Logistic Regression, Random Forest, MLP
- Svaki model ima sopstveni threshold podešen za visok recall na val
- OR logika: ako BILO KOJI model predvidi DNF → DNF
- Cilj: minimalan FN po cenu višeg FP
"""
import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(METRICS_DIR, exist_ok=True)

COST_FN = 5
COST_FP = 1

print("=" * 60)
print("KORAK 4d: ENSEMBLE OR VOTING (max recall)")
print("  Ako bilo koji model kaze DNF => DNF")
print("=" * 60)

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

BASELINE_COST = y_test.sum() * COST_FN
print(f"Test: {len(y_test)} redova, DNF={y_test.sum()}, baseline cost={BASELINE_COST}")

# ============================================================
# 1. MODELI (najbolji parametri iz GridSearch)
# ============================================================
models_cfg = {
    "XGBoost": XGBClassifier(
        learning_rate=0.01, max_depth=6, n_estimators=200,
        scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss"
    ),
    "Logistic Regression": LogisticRegression(
        C=10, class_weight="balanced", max_iter=2000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(50, 25), activation="relu", alpha=0.0001,
        learning_rate_init=0.001, early_stopping=True,
        max_iter=1000, random_state=42
    ),
}

models = {}
val_probas = {}
test_probas = {}

# ============================================================
# 2. TRENIRANJE + THRESHOLD TUNING (po modelu)
# ============================================================
for name, model in models_cfg.items():
    print(f"\n{'=' * 50}")
    print(f"MODEL: {name}")
    print(f"{'=' * 50}")

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Trening: {elapsed:.1f}s")

    vp = model.predict_proba(X_val)[:, 1]
    tp = model.predict_proba(X_test)[:, 1]
    val_probas[name] = vp
    test_probas[name] = tp

    # Tražimo threshold koji daje RECALL >= 85% uz minimalan cost
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_min_cost = {"thr": None, "cost": float("inf"), "rec": 0, "prec": 0, "f1": 0, "fp": 0, "fn": 0}
    best_max_rec = {"thr": None, "cost": float("inf"), "rec": 0, "prec": 0, "f1": 0, "fp": 0, "fn": 0}

    for t in thresholds:
        yp = (vp >= t).astype(int)
        cm = confusion_matrix(y_val, yp)
        if cm.shape != (2, 2):
            continue
        TN, FP, FN, TP = cm.ravel()
        cost = FN * COST_FN + FP * COST_FP
        rec = recall_score(y_val, yp, zero_division=0)

        # Varijanta A: minimalni cost uz recall >= 85%
        if rec >= 0.85 and cost < best_min_cost["cost"]:
            best_min_cost = {"thr": t, "cost": cost, "rec": rec,
                             "prec": precision_score(y_val, yp, zero_division=0),
                             "f1": f1_score(y_val, yp, zero_division=0),
                             "fp": FP, "fn": FN}

        # Varijanta B: maksimalni recall (bez obzira na cost)
        if rec > best_max_rec["rec"] or (rec == best_max_rec["rec"] and cost < best_max_rec["cost"]):
            best_max_rec = {"thr": t, "cost": cost, "rec": rec,
                            "prec": precision_score(y_val, yp, zero_division=0),
                            "f1": f1_score(y_val, yp, zero_division=0),
                            "fp": FP, "fn": FN}

    # Varijanta C: threshold koji daje najbolji trade-off po F_beta (beta=2, recall 2x važniji)
    best_f2 = {"thr": None, "score": 0, "rec": 0, "prec": 0, "fp": 0, "fn": 0}
    for t in thresholds:
        yp = (vp >= t).astype(int)
        rec = recall_score(y_val, yp, zero_division=0)
        prec = precision_score(y_val, yp, zero_division=0)
        if rec + prec > 0:
            f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)
        else:
            f2 = 0
        if f2 > best_f2["score"]:
            cm = confusion_matrix(y_val, yp)
            TN, FP, FN, TP = cm.ravel()
            best_f2 = {"thr": t, "score": f2, "rec": rec, "prec": prec,
                       "fp": FP, "fn": FN, "cost": FN * COST_FN + FP * COST_FP}

    print(f"  A) Min cost uz Rec>=85%: thr={best_min_cost['thr']:.2f}, "
          f"val rec={best_min_cost['rec']:.3f}, val fp={best_min_cost['fp']}")
    print(f"  B) Max recall:            thr={best_max_rec['thr']:.2f}, "
          f"val rec={best_max_rec['rec']:.3f}, val fp={best_max_rec['fp']}")
    print(f"  C) Best F2 (recall×2):    thr={best_f2['thr']:.2f}, "
          f"val rec={best_f2['rec']:.3f}, val fp={best_f2['fp']}")

    models[name] = {
        "model": model,
        "min_cost": best_min_cost,
        "max_rec": best_max_rec,
        "best_f2": best_f2,
        "time": elapsed
    }

# ============================================================
# 3. ENSEMBLE OR VOTING — varijante
# ============================================================
print(f"\n{'=' * 60}")
print("ENSEMBLE OR VOTING — TEST SKUP")
print(f"{'=' * 60}")

ensemble_results = {}

# Varijanta A: svaki model koristi svoj threshold za recall >= 85%
for variant_name, thr_key in [
    ("A) Min cost, Rec>=85%", "min_cost"),
    ("B) Max recall", "max_rec"),
    ("C) Best F2", "best_f2"),
]:
    print(f"\n--- Varijanta {variant_name} ---")
    ensemble_preds = np.zeros(len(y_test), dtype=int)
    individual_preds = {}

    for name in models_cfg:
        thr = models[name][thr_key]["thr"]
        yp = (test_probas[name] >= thr).astype(int)
        individual_preds[name] = yp
        # OR: akumulacija — ako bilo koji model kaže 1 → 1
        ensemble_preds = ensemble_preds | yp

    cm = confusion_matrix(y_test, ensemble_preds)
    TN, FP, FN, TP = cm.ravel()
    cost = FN * COST_FN + FP * COST_FP
    saving = (1 - cost / BASELINE_COST) * 100
    rec = recall_score(y_test, ensemble_preds, zero_division=0)
    prec = precision_score(y_test, ensemble_preds, zero_division=0)
    f1 = f1_score(y_test, ensemble_preds, zero_division=0)

    # Koliko modela je glasalo za DNF?
    n_voters = np.zeros(len(y_test), dtype=int)
    for name in models_cfg:
        n_voters += individual_preds[name]
    avg_voters = n_voters[ensemble_preds == 1].mean() if (ensemble_preds == 1).sum() > 0 else 0

    print(f"  CM: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    print(f"  Cost={cost} (baseline={BASELINE_COST}, saving={saving:.1f}%)")
    print(f"  Rec={rec:.4f} | Prec={prec:.4f} | F1={f1:.4f} | Acc={accuracy_score(y_test, ensemble_preds):.4f}")
    print(f"  Prosecno glasova po DNF predikciji: {avg_voters:.1f}/4")

    # Koji modeli su najviše doprineli
    print(f"  Individualne metrike (test):")
    for name in models_cfg:
        cm_i = confusion_matrix(y_test, individual_preds[name])
        TNi, FPi, FNi, TPi = cm_i.ravel()
        print(f"    {name:22s}: thr={models[name][thr_key]['thr']:.2f}, "
              f"FN={FNi}, FP={FPi}, Rec={recall_score(y_test, individual_preds[name], zero_division=0):.3f}")

    # Varijanta D: samo 2 modela (XGBoost + LR) — manje agresivan OR
    if variant_name == "A) Min cost, Rec>=85%":
        xgb_preds = individual_preds["XGBoost"]
        lr_preds = individual_preds["Logistic Regression"]
        ensemble_2 = xgb_preds | lr_preds
        cm_2 = confusion_matrix(y_test, ensemble_2)
        TN2, FP2, FN2, TP2 = cm_2.ravel()
        cost_2 = FN2 * COST_FN + FP2 * COST_FP
        saving_2 = (1 - cost_2 / BASELINE_COST) * 100
        print(f"\n  Varijanta D: XGBoost OR LR (samo 2 modela):")
        print(f"    CM: TN={TN2}, FP={FP2}, FN={FN2}, TP={TP2}")
        print(f"    Cost={cost_2} (saving={saving_2:.1f}%)")
        print(f"    Rec={recall_score(y_test, ensemble_2, zero_division=0):.4f}, "
              f"F1={f1_score(y_test, ensemble_2, zero_division=0):.4f}")

        # Varijanta E: XGBoost + RF + MLP (bez LR)
        rf_preds = individual_preds["Random Forest"]
        mlp_preds = individual_preds["MLP"]
        ensemble_3 = xgb_preds | rf_preds | mlp_preds
        cm_3 = confusion_matrix(y_test, ensemble_3)
        TN3, FP3, FN3, TP3 = cm_3.ravel()
        cost_3 = FN3 * COST_FN + FP3 * COST_FP
        saving_3 = (1 - cost_3 / BASELINE_COST) * 100
        print(f"\n  Varijanta E: XGBoost OR RF OR MLP (bez LR):")
        print(f"    CM: TN={TN3}, FP={FP3}, FN={FN3}, TP={TP3}")
        print(f"    Cost={cost_3} (saving={saving_3:.1f}%)")
        print(f"    Rec={recall_score(y_test, ensemble_3, zero_division=0):.4f}, "
              f"F1={f1_score(y_test, ensemble_3, zero_division=0):.4f}")

    ensemble_results[variant_name] = {
        "cm": cm, "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "cost": cost, "saving": saving, "rec": rec, "prec": prec, "f1": f1,
        "avg_voters": avg_voters
    }

# ============================================================
# 4. POREDJENJE SA DEPLOYMENT XGBOOST
# ============================================================
print(f"\n{'=' * 60}")
print("POREDJENJE SA DEPLOYMENTOM (XGBoost standalone, FN=20, FP=1540)")
print(f"{'=' * 60}")

deploy = {"name": "XGBoost deploy", "FN": 20, "FP": 1540, "Recall": 0.918, "F1": 0.224, "Cost": 1640}

for name, r in ensemble_results.items():
    print(f"\n{name} vs Deploy XGBoost:")
    print(f"  FN:  {r['FN']:>4} vs {deploy['FN']:>4} | Delta: {r['FN'] - deploy['FN']:+d}")
    print(f"  FP:  {r['FP']:>4} vs {deploy['FP']:>4} | Delta: {r['FP'] - deploy['FP']:+d}")
    print(f"  Rec: {r['rec']:.4f} vs {deploy['Recall']:.4f}")
    print(f"  F1:  {r['f1']:.4f} vs {deploy['F1']:.4f}")

# ============================================================
# 5. SUMARNA TABELA
# ============================================================
print(f"\n{'=' * 60}")
print("SUMARNA TABELA")
print(f"{'=' * 60}")

all_rows = []
for name, r in ensemble_results.items():
    all_rows.append({"Model": name, "FN": r["FN"], "FP": r["FP"],
                     "Recall": r["rec"], "Precision": r["prec"],
                     "F1": r["f1"], "Cost": r["cost"],
                     "Saving": f"{r['saving']:.1f}%",
                     "Voters": f"{r['avg_voters']:.1f}/4"})
all_rows.append({"Model": deploy["name"], "FN": deploy["FN"], "FP": deploy["FP"],
                 "Recall": deploy["Recall"], "Precision": 0.127,
                 "F1": deploy["F1"], "Cost": deploy["Cost"],
                 "Saving": "-", "Voters": "1/1"})

df_all = pd.DataFrame(all_rows).sort_values("FN")
print("\n" + df_all.to_string(index=False))
df_all.to_csv(os.path.join(METRICS_DIR, "ensemble_or_voting.csv"), index=False)

print("\nKRAJ KORAKA 4d")
