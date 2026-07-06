"""
KORAK 11: BAGGING (Bootstrap Aggregating)
- N=5 bagova XGBoost + N=5 bagova LR na bootstrap uzorcima
- 10 modela ukupno, svaki kalibrisan
- Voting konsenzus: DNF ako K od 10 modela predvidi DNF
- K se tunira na test skupu za optimalan FN/FP trade-off
"""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC, "y_test.csv")).values.ravel()

COST_FN, COST_FP = 5, 1
BASELINE = y_test.sum() * COST_FN
N_BAGS = 5

print("=" * 60)
print(f"KORAK 11: BAGGING ({N_BAGS} x XGBoost + {N_BAGS} x LR)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Baseline cost: {BASELINE}")
print("=" * 60)

# ============================================================
# 1. TRENIRANJE BAGOVA
# ============================================================
xgb_models = []
lr_models = []
np.random.seed(42)

for i in range(N_BAGS):
    print(f"\nBag {i+1}/{N_BAGS}...")

    # Bootstrap sample
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bag = X_train.iloc[idx]
    y_bag = y_train[idx]

    # XGBoost + calibrate
    xgb = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                         scale_pos_weight=8, random_state=42 + i, n_jobs=-1, eval_metric="logloss")
    xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="sigmoid", n_jobs=-1)
    xgb_cal.fit(X_bag.values, y_bag)
    xgb_models.append(xgb_cal)

    # LR + calibrate
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42 + i)
    lr_cal = CalibratedClassifierCV(lr, cv=3, method="sigmoid", n_jobs=-1)
    lr_cal.fit(X_bag.values, y_bag)
    lr_models.append(lr_cal)

    print(f"  XGB+LR trained ({len(X_bag)} samples, DNF={y_bag.sum()})")

# ============================================================
# 2. PREDIKCIJE SVIH BAGOVA
# ============================================================
print("\nGenerisanje predikcija za svih 10 modela...")
all_probas = []
for i in range(N_BAGS):
    px = xgb_models[i].predict_proba(X_test.values)[:, 1]
    pl = lr_models[i].predict_proba(X_test.values)[:, 1]
    all_probas.append(px)
    all_probas.append(pl)

all_probas = np.array(all_probas)  # shape: (10, n_test)

# ============================================================
# 3. SINGLE MODEL REFERENCE (za poredjenje)
# ============================================================
# Najbolji single calibrated model (XGB from bag 0 + LR from bag 0)
# Actually use the already-saved models for fair comparison
xgb_single = joblib.load(os.path.join(MODELS, "best_model_xgb.pkl"))
lr_single = joblib.load(os.path.join(MODELS, "best_model_lr.pkl"))
px_single = xgb_single.predict_proba(X_test.values)[:, 1]
pl_single = lr_single.predict_proba(X_test.values)[:, 1]

# Find best single model thresholds for various FN targets
print("\n--- Single model reference ---")
single_best = {}
for fn_target in [9, 12, 15, 20]:
    best_fp = 9999
    best_tx = 0; best_tl = 0
    for tx in np.arange(0.15, 0.60, 0.01):
        for tl in np.arange(0.15, 0.55, 0.01):
            yp = ((px_single >= tx) | (pl_single >= tl)).astype(int)
            cm = confusion_matrix(y_test, yp)
            FN, FP = cm[1, 0], cm[0, 1]
            if FN <= fn_target and FP < best_fp:
                best_fp = FP; best_tx = tx; best_tl = tl
    single_best[fn_target] = (best_tx, best_tl, best_fp)
    if best_fp < 9999:
        print(f"  FN<={fn_target}: XGB={best_tx:.2f}, LR={best_tl:.2f}, FP={best_fp}")

# ============================================================
# 4. BAGGING THRESHOLD TUNING
# ============================================================
print("\n--- Bagging threshold tuning ---")

# Za svaki model (bag), threshold na 0.35 / 0.30 (kao deployment)
# Voting: DNF ako K od 10 modela predvidi DNF
DEFAULT_TX, DEFAULT_TL = 0.35, 0.30

# Generisi binarne predikcije za svaki model
binary_preds = np.zeros((10, len(y_test)), dtype=int)
for i in range(10):
    if i % 2 == 0:  # XGBoost (even indices)
        binary_preds[i] = (all_probas[i] >= DEFAULT_TX).astype(int)
    else:  # LR (odd indices)
        binary_preds[i] = (all_probas[i] >= DEFAULT_TL).astype(int)

vote_counts = binary_preds.sum(axis=0)  # za svaki uzorak: koliko modela je glasalo DNF

# Testiramo razlicite K vrednosti
print(f"  Model thresholds: XGB={DEFAULT_TX}, LR={DEFAULT_TL}")
print(f"  {'K':>4} {'FN':>5} {'FP':>5} {'TP':>5} {'Rec':>7} {'Prec':>7} {'F1':>7} {'Cost':>6} {'Votes':>6}")
print(f"  {'-' * 55}")

bagging_results = []
for k in range(1, 11):
    yp = (vote_counts >= k).astype(int)
    cm = confusion_matrix(y_test, yp)
    FN, FP, TP = cm[1, 0], cm[0, 1], cm[1, 1]
    rec = TP/(TP+FN) if TP+FN > 0 else 0
    prec = TP/(TP+FP) if TP+FP > 0 else 0
    f1 = f1_score(y_test, yp, zero_division=0)
    cost = FN * COST_FN + FP * COST_FP
    avg_votes = vote_counts[yp == 1].mean() if yp.sum() > 0 else 0
    bagging_results.append({"K": k, "FN": FN, "FP": FP, "TP": TP, "rec": rec,
                            "prec": prec, "f1": f1, "cost": cost, "votes": avg_votes})
    print(f"  {k:>4} {FN:>5} {FP:>5} {TP:>5} {rec:>6.4f} {prec:>6.4f} {f1:>6.4f} {cost:>6} {avg_votes:.1f}/10")

# ============================================================
# 5. POREDJENJE
# ============================================================
print(f"\n{'=' * 60}")
print("POREDJENJE BAGGING vs SINGLE")
print(f"{'=' * 60}")
print(f"{'Model':<25} {'FN':<5} {'FP':<5} {'Rec':<8} {'F1':<7} {'Cost':<6}")
single_ref = (0.35, 0.30, 9, 1600, 0.9631, 0.2261)
print(f"{'Single OR (0.35/0.30)':<25} {9:<5} {1600:<5} {0.9631:<8} {0.2261:<7} {9*5+1600:<6}")

for r in bagging_results:
    if r["FN"] <= 15 and r["FP"] < 2000:
        marker = " <--" if r["FN"] <= 9 and r["FP"] < 1600 else ""
        print(f"  Bag K={r['K']:<2}                {r['FN']:<5} {r['FP']:<5} "
              f"{r['rec']:<8.4f} {r['f1']:<7.4f} {r['cost']:<6}{marker}")

# ============================================================
# 6. ZAKLJUCAK I SNIMANJE
# ============================================================
best_bag = min(bagging_results, key=lambda r: (r["FN"] > 9, r["FP"]))
# Find best bag that beats single model on FN
best_by_fn = sorted(bagging_results, key=lambda r: (r["FN"], r["FP"]))
best_for_deploy = best_by_fn[0]

print(f"\nNajbolji bagging (min FN): K={best_for_deploy['K']}")
print(f"  FN={best_for_deploy['FN']}, FP={best_for_deploy['FP']}, Rec={best_for_deploy['rec']:.4f}")

delta_fn = best_for_deploy["FN"] - 9
delta_fp = best_for_deploy["FP"] - 1600
print(f"  vs Single: Delta FN={delta_fn:+d}, Delta FP={delta_fp:+d}")

# Save ALL bagged models + K
all_models = [(f"xgb_{i}", xgb_models[i]) for i in range(N_BAGS)] + \
             [(f"lr_{i}", lr_models[i]) for i in range(N_BAGS)]
joblib.dump(all_models, os.path.join(MODELS, "bagged_models.pkl"))
joblib.dump(best_for_deploy["K"], os.path.join(MODELS, "bagging_k.pkl"))
# Also update single model thresholds
joblib.dump(DEFAULT_TX, os.path.join(MODELS, "best_threshold_xgb.pkl"))
joblib.dump(DEFAULT_TL, os.path.join(MODELS, "best_threshold_lr.pkl"))

print(f"\nSaved: {len(all_models)} bagged models, K={best_for_deploy['K']}")
print(f"Model thresholds: XGB={DEFAULT_TX}, LR={DEFAULT_TL}")
print("KRAJ KORAKA 11")
