"""
Model 3: Bagging Ensemble (5x XGBoost + 5x LR, K-voting)
- Bootstrap aggregating na SMOTE podacima
- 10 kalibrisanih modela (5+5)
- Voting: DNF ako K od 10 modela predvidi DNF
- Evaluacija za razlicite K vrednosti
"""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, "models", "model3_bagging")
RESULTS_DIR = os.path.join(BASE, "results", "model3_bagging")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

X_train = pd.read_csv(os.path.join(BASE, "data", "processed", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "processed", "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(BASE, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(BASE, "data", "processed", "y_test.csv")).values.ravel()

N_BAGS, THRESHOLD = 5, 0.35
BEST_K = 2

print("=" * 55)
print(f"MODEL 3: Bagging Ensemble ({N_BAGS}x XGBoost + {N_BAGS}x LR)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Threshold per model: {THRESHOLD}, Best K: {BEST_K}")
print("=" * 55)

# Train bagged models
xgb_models, lr_models = [], []
np.random.seed(42)
for i in range(N_BAGS):
    print(f"\nBag {i+1}/{N_BAGS}...")
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bag, y_bag = X_train.iloc[idx].values, y_train[idx]

    xgb = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                         scale_pos_weight=8, random_state=42 + i, n_jobs=-1, eval_metric="logloss")
    xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="sigmoid", n_jobs=-1)
    xgb_cal.fit(X_bag, y_bag)
    xgb_models.append(xgb_cal)

    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42 + i)
    lr_cal = CalibratedClassifierCV(lr, cv=3, method="sigmoid", n_jobs=-1)
    lr_cal.fit(X_bag, y_bag)
    lr_models.append(lr_cal)
    print(f"  Done ({len(X_bag)} samples, DNF={y_bag.sum()})")

# Evaluate for different K
print("\n--- Evaluacija K-voting (thr=" + str(THRESHOLD) + ") ---")
print(f"  {'K':>4} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5} {'Rec':>7} {'Prec':>7} {'F1':>7}")

all_probas, all_binary = [], []
for i in range(N_BAGS):
    px = xgb_models[i].predict_proba(X_test.values)[:, 1]
    pl = lr_models[i].predict_proba(X_test.values)[:, 1]
    all_probas.extend([px, pl])

votes = np.sum([(p >= THRESHOLD).astype(int) for p in all_probas], axis=0)
all_results = []

for k in range(1, 11):
    y_pred = (votes >= k).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    rec = TP/(TP+FN) if TP+FN > 0 else 0
    prec = TP/(TP+FP) if TP+FP > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)
    all_results.append({"K": k, "TN": TN, "FP": FP, "FN": FN, "TP": TP,
                        "Recall": rec, "Precision": prec, "F1": f1})
    print(f"  {k:>4} {TN:>5} {FP:>5} {FN:>5} {TP:>5} {rec:>6.4f} {prec:>6.4f} {f1:>6.4f}")

# Best K (by default)
y_pred = (votes >= BEST_K).astype(int)
cm_best = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm_best.ravel()

print(f"\n--- Best K={BEST_K} (default) ---")
print(f"Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Zavrsio', 'DNF'], zero_division=0)}")

# Save models
all_models = ([(f"xgb_{i}", xgb_models[i]) for i in range(N_BAGS)] +
              [(f"lr_{i}", lr_models[i]) for i in range(N_BAGS)])
joblib.dump(all_models, os.path.join(MODEL_DIR, "bagged_models.pkl"))
joblib.dump(BEST_K, os.path.join(MODEL_DIR, "bagging_k.pkl"))
joblib.dump({"threshold": THRESHOLD, "K": BEST_K}, os.path.join(MODEL_DIR, "thresholds.pkl"))

# Save metrics
all_df = pd.DataFrame(all_results)
all_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics_by_K.csv"), index=False)

best_df = pd.DataFrame([{"Model": "Bagging Ensemble", "K": BEST_K, "Threshold": THRESHOLD,
                          "TN": TN, "FP": FP, "FN": FN, "TP": TP,
                          "Accuracy": accuracy_score(y_test, y_pred),
                          "Precision": precision_score(y_test, y_pred, zero_division=0),
                          "Recall": recall_score(y_test, y_pred, zero_division=0),
                          "F1-Score": f1_score(y_test, y_pred, zero_division=0)}])
best_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics.csv"), index=False)

# K vs metrics plot
fig, ax = plt.subplots(figsize=(8, 5))
df_plot = pd.DataFrame(all_results)
ax.plot(df_plot["K"], df_plot["Recall"], "b-o", label="Recall", linewidth=2)
ax.plot(df_plot["K"], df_plot["Precision"], "r-s", label="Precision", linewidth=2)
ax.plot(df_plot["K"], df_plot["F1"], "g-^", label="F1", linewidth=2)
ax.axvline(x=BEST_K, color="black", linestyle="--", label=f"Best K={BEST_K}")
ax.set_xlabel("K (min votes for DNF)"); ax.set_ylabel("Score")
ax.set_title(f"K-Voting Performance (threshold={THRESHOLD})"); ax.legend()
ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "k_voting_performance.png"), dpi=150)
plt.close()

# Confusion matrix for best K
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_best, display_labels=["Zavrsio", "DNF"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Model 3: Bagging (K={BEST_K}, thr={THRESHOLD})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "confusion_matrix.png"), dpi=150)
plt.close()

print(f"\nModels saved to models/model3_bagging/")
print(f"Results saved to results/model3_bagging/")
print("Done.")
