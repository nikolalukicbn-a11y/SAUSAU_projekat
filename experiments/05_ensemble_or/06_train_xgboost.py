"""Trenira XGBoost SMOTE 60/40 sa F1 scoring-om (bez cost-sensitive)."""
import pandas as pd
import numpy as np
import os
import joblib
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

print(f"Train: {X_train.shape}, DNF={y_train.sum()}, Non-DNF={len(y_train)-y_train.sum()}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1, 0.2],
    "scale_pos_weight": [1, 3, 5, 8]
}

print("GridSearchCV XGBoost (F1 scoring)...")
start = time.time()
grid = GridSearchCV(model, params, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Trajanje: {elapsed:.1f}s")
print(f"Najbolji params: {grid.best_params_}")
print(f"Najbolji CV F1: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model_xgb.pkl"))
joblib.dump(0.80, os.path.join(MODELS_DIR, "best_threshold_xgb.pkl"))
print("Sacuvano: best_model_xgb.pkl, best_threshold_xgb.pkl (thresh=0.80)")
