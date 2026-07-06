"""
Model 2: OR Ensemble Calibrated (XGBoost + LogisticRegression)
- Trenira kalibrisani XGBoost i LR na SMOTE podacima
- OR logika: DNF ako XGBoost ILI LR predvidi DNF
- Evaluacija na test skupu, cuvanje modela i metrika
"""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, "models", "model2_or")
RESULTS_DIR = os.path.join(BASE, "results", "model2_or")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

X_train = pd.read_csv(os.path.join(BASE, "data", "processed", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "processed", "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(BASE, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(BASE, "data", "processed", "y_test.csv")).values.ravel()

TX, TL = 0.35, 0.30

print("=" * 55)
print("MODEL 2: OR Ensemble Calibrated (Platt scaling)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Thresholds: XGB={TX}, LR={TL}")
print("=" * 55)

# Train + calibrate XGBoost
print("\nTraining XGBoost + calibration...")
t0 = time.time()
xgb = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                     scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
xgb_cal = CalibratedClassifierCV(xgb, cv=5, method="sigmoid", n_jobs=-1)
xgb_cal.fit(X_train, y_train)
print(f"  XGBoost: {time.time()-t0:.1f}s")

# Train + calibrate LR
print("Training LR + calibration...")
t0 = time.time()
lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)
lr_cal = CalibratedClassifierCV(lr, cv=5, method="sigmoid", n_jobs=-1)
lr_cal.fit(X_train, y_train)
print(f"  LR: {time.time()-t0:.1f}s")

# Predict
px = xgb_cal.predict_proba(X_test)[:, 1]
pl = lr_cal.predict_proba(X_test)[:, 1]
y_pred = ((px >= TX) | (pl >= TL)).astype(int)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"\nConfusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Zavrsio', 'DNF'], zero_division=0)}")

# Save models
joblib.dump(xgb_cal, os.path.join(MODEL_DIR, "xgboost_cal.pkl"))
joblib.dump(lr_cal, os.path.join(MODEL_DIR, "lr_cal.pkl"))
joblib.dump({"xgb": TX, "lr": TL}, os.path.join(MODEL_DIR, "thresholds.pkl"))

# Copy preprocessor from main models/ if not exists
prep_src = os.path.join(BASE, "models", "model3_bagging", "preprocessor.pkl")
if not os.path.exists(os.path.join(MODEL_DIR, "preprocessor.pkl")) and os.path.exists(prep_src):
    joblib.dump(joblib.load(prep_src), os.path.join(MODEL_DIR, "preprocessor.pkl"))

# Save metrics
metrics_df = pd.DataFrame([{
    "Model": "OR Ensemble Calibrated (Platt scaling)",
    "XGB_Threshold": TX, "LR_Threshold": TL,
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
}])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics.csv"), index=False)

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Zavrsio", "DNF"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Model 2: OR Calibrated (XGB={TX}, LR={TL})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "confusion_matrix.png"), dpi=150)
plt.close()

# Calibration curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, probs, label in [(axes[0], px, "XGBoost"), (axes[1], pl, "Logistic Regression")]:
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(prob_pred, prob_true, "s-", color="red", linewidth=2, label="Calibrated")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("True DNF frequency")
    ax.set_title(f"Calibration Curve - {label}"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "calibration_curves.png"), dpi=150)
plt.close()

print("\nModels saved to models/model2_or/")
print("Results saved to results/model2_or/")
print("Done.")
