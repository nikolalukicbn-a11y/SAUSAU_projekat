"""
Model 1: XGBoost Standalone (SMOTE 60/40)
- Trenira XGBoost na SMOTE podacima
- Evaluacija na test skupu (confusion matrix, metrics)
- Snima model i metrike
"""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, "models", "model1_xgboost")
RESULTS_DIR = os.path.join(BASE, "results", "model1_xgboost")
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

X_train = pd.read_csv(os.path.join(BASE, "data", "processed", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "processed", "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(BASE, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(BASE, "data", "processed", "y_test.csv")).values.ravel()

THRESHOLD = 0.80

print("=" * 55)
print("MODEL 1: XGBoost Standalone (SMOTE 60/40)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Threshold: {THRESHOLD}")
print("=" * 55)

print("\nTraining XGBoost...")
t0 = time.time()
model = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                       scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
model.fit(X_train, y_train)
print(f"Training time: {time.time()-t0:.1f}s")

# Predict
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= THRESHOLD).astype(int)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"\nConfusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Zavrsio', 'DNF'], zero_division=0)}")

# Save model
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(THRESHOLD, os.path.join(MODEL_DIR, "threshold.pkl"))

# Save metrics
metrics_df = pd.DataFrame([{
    "Model": "XGBoost Standalone (SMOTE 60/40)",
    "Threshold": THRESHOLD,
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
}])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "metrics.csv"), index=False)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Zavrsio", "DNF"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Model 1: XGBoost SMOTE (thr={THRESHOLD})")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures", "confusion_matrix.png"), dpi=150)
plt.close()

print("\nModel saved to models/model1_xgboost/")
print("Results saved to results/model1_xgboost/")
print("Done.")
