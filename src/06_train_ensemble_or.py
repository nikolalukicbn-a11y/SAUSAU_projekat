"""
Trenira i snima XGBoost + Logistic Regression za ensemble OR voting.
Threshold-i optimizovani za recall >= 85% na val skupu.
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

print("Treniranje XGBoost (SMOTE 60/40)...")
xgb = XGBClassifier(
    learning_rate=0.01, max_depth=6, n_estimators=200,
    scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss"
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, os.path.join(MODELS_DIR, "best_model_xgb.pkl"))

XGB_THRESHOLD = 0.83
joblib.dump(XGB_THRESHOLD, os.path.join(MODELS_DIR, "best_threshold_xgb.pkl"))
print(f"  XGBoost snimljen, threshold={XGB_THRESHOLD}")

print("Treniranje Logistic Regression (SMOTE 60/40)...")
lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
joblib.dump(lr, os.path.join(MODELS_DIR, "best_model_lr.pkl"))

LR_THRESHOLD = 0.36
joblib.dump(LR_THRESHOLD, os.path.join(MODELS_DIR, "best_threshold_lr.pkl"))
print(f"  LR snimljen, threshold={LR_THRESHOLD}")

print("Gotovo. Oba modela spremna za OR voting deployment.")
