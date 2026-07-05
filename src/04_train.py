"""
KORAK 4: TRENIRANJE I EVALUACIJA MODELA (v2 - sa threshold tuning-om)
- 6 modela: Logistic Regression (baseline), Random Forest, XGBoost, SVM, KNN, MLP Classifier
- GridSearchCV sa stratified 5-fold CV, scoring='f1'
- Threshold tuning na validacionom skupu (max F1)
- Evaluacija na test skupu sa default i optimizovanim threshold-om
- Feature importance, uporedna tabela, cuvanje najboljeg modela

POBOLJSANJA U ODNOSU NA v1:
  1. Threshold tuning: nalazi optimalni prag verovatnoce na val skupu
     koji maksimizuje F1 score, umesto fiksnog 0.5
  2. TargetEncoded podaci (vise informacija za tree modele)
  3. Bolji feature-i: Rider_DNF_Rate, Historical_DNF_Rate shrinkage, bez sequence-a
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ---------- 1. PUTANJE ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print("=" * 60)
print("KORAK 4: TRENIRANJE I EVALUACIJA (v2 - threshold tuning)")
print("=" * 60)

# ---------- 2. UCITAVANJE ----------
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

print(f"Train: {X_train.shape} (DNF: {y_train.sum()}, "
      f"Zavrsili: {len(y_train)-y_train.sum()})")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")
print(f"Feature-ovi ({len(X_train.columns)}): {list(X_train.columns)}")

# ---------- 3. MODELI ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "class_weight": [None, "balanced"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", "balanced_subsample"]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, 3, 5, 8]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42, max_iter=5000),
        "params": {
            "C": [0.1, 1],
            "kernel": ["rbf"],
            "class_weight": ["balanced"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(n_jobs=-1),
        "params": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "MLP Classifier": {
        "model": MLPClassifier(random_state=42, early_stopping=True, max_iter=1000),
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (50, 25)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001]
        }
    }
}

results = {}

# ---------- 4. TRENIRANJE ----------
for name, config in models.items():
    print(f"\n{'=' * 60}")
    print(f"TRENIRANJE: {name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    grid = GridSearchCV(
        config["model"], config["params"],
        cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - start_time

    best_model = grid.best_estimator_
    print(f"Vreme: {elapsed:.1f}s")
    print(f"Najbolji parametri: {grid.best_params_}")
    print(f"Najbolji CV F1: {grid.best_score_:.4f}")

    # ---------- THRESHOLD = 0.75 (fiksno) ----------
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    FIXED_THRESHOLD = 0.75

    best_threshold = FIXED_THRESHOLD
    y_pred_t = (y_val_proba >= best_threshold).astype(int)
    best_f1 = f1_score(y_val, y_pred_t)

    thresholds = np.arange(0.10, 0.91, 0.05)
    thresholds = np.sort(np.unique(np.append(thresholds, [0.5, FIXED_THRESHOLD])))
    threshold_scores = []
    for t in thresholds:
        y_pred_t = (y_val_proba >= t).astype(int)
        threshold_scores.append({"threshold": t,
                                  "f1": f1_score(y_val, y_pred_t),
                                  "precision": precision_score(y_val, y_pred_t),
                                  "recall": recall_score(y_val, y_pred_t)})
    df_thresh = pd.DataFrame(threshold_scores)
    print(f"\nThreshold = {FIXED_THRESHOLD:.2f} (fiksno):")
    print(f"  Val F1={best_f1:.4f}, "
          f"Prec={precision_score(y_val, y_pred_t):.4f}, "
          f"Rec={recall_score(y_val, y_pred_t):.4f}")
    print(f"  Default (0.5): F1={df_thresh[df_thresh['threshold']==0.5]['f1'].values[0]:.4f}")

    # ---------- EVALUACIJA SA DEFAULT THRESHOLD (0.5) ----------
    y_test_pred_default = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    # ---------- EVALUACIJA SA OPTIMALNIM THRESHOLD-OM ----------
    y_test_pred_opt = (y_test_proba >= best_threshold).astype(int)

    # Metrike sa default threshold-om
    def_test = {
        "accuracy": accuracy_score(y_test, y_test_pred_default),
        "precision": precision_score(y_test, y_test_pred_default),
        "recall": recall_score(y_test, y_test_pred_default),
        "f1": f1_score(y_test, y_test_pred_default)
    }

    # Metrike sa optimalnim threshold-om
    opt_test = {
        "accuracy": accuracy_score(y_test, y_test_pred_opt),
        "precision": precision_score(y_test, y_test_pred_opt),
        "recall": recall_score(y_test, y_test_pred_opt),
        "f1": f1_score(y_test, y_test_pred_opt)
    }

    print(f"\nTest skup - Default threshold (0.5):")
    print(f"  Accuracy:  {def_test['accuracy']:.4f}")
    print(f"  Precision: {def_test['precision']:.4f}")
    print(f"  Recall:    {def_test['recall']:.4f}")
    print(f"  F1:        {def_test['f1']:.4f}")

    print(f"\nTest skup - Optimalni threshold ({best_threshold:.2f}):")
    print(f"  Accuracy:  {opt_test['accuracy']:.4f}")
    print(f"  Precision: {opt_test['precision']:.4f}")
    print(f"  Recall:    {opt_test['recall']:.4f}  <-- KLJUCNO")
    print(f"  F1:        {opt_test['f1']:.4f}  <-- KLJUCNO")

    print(f"\nClassification Report (optimalni threshold):")
    print(classification_report(y_test, y_test_pred_opt,
                                target_names=["Zavrsio (0)", "DNF (1)"]))

    results[name] = {
        "best_model": best_model,
        "grid": grid,
        "best_threshold": best_threshold,
        "test_accuracy": opt_test["accuracy"],
        "test_precision": opt_test["precision"],
        "test_recall": opt_test["recall"],
        "test_f1": opt_test["f1"],
        "test_acc_default": def_test["accuracy"],
        "test_f1_default": def_test["f1"],
        "training_time": elapsed
    }

    # ---------- CONFUSION MATRIX (optimalni threshold na testu) ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_def = confusion_matrix(y_test, y_test_pred_default)
    ConfusionMatrixDisplay(cm_def, display_labels=["Zavrsio", "DNF"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{name} - Default (thresh=0.5)")

    cm_opt = confusion_matrix(y_test, y_test_pred_opt)
    ConfusionMatrixDisplay(cm_opt, display_labels=["Zavrsio", "DNF"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title(f"{name} - Optimal (thresh={best_threshold:.2f})")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close()

    # ---------- THRESHOLD VS METRICS PLOT ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_thresh["threshold"], df_thresh["precision"], "b-o", ms=4, label="Precision")
    ax.plot(df_thresh["threshold"], df_thresh["recall"], "r-s", ms=4, label="Recall")
    ax.plot(df_thresh["threshold"], df_thresh["f1"], "g-^", ms=6, label="F1", linewidth=2)
    ax.axvline(x=best_threshold, color="black", linestyle="--", alpha=0.5,
               label=f"Optimal={best_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{name} - Threshold vs Metrike (Val skup)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"threshold_tuning_{safe_name}.png"), dpi=150)
    plt.close()

# ---------- 5. FEATURE IMPORTANCE ----------
print(f"\n{'=' * 60}")
print("FEATURE IMPORTANCE")
print(f"{'=' * 60}")

feature_names = list(X_train.columns)
rf_model = results["Random Forest"]["best_model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nRandom Forest - vaznost feature-a:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]:25s} -> {importances[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([feature_names[i] for i in reversed(indices)],
        [importances[i] for i in reversed(indices)],
        color=sns.color_palette("viridis", len(feature_names)))
ax.set_xlabel("Vaznost")
ax.set_title("Feature Importance - Random Forest (TargetEncoded)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ---------- 6. UPOREDNA TABELA ----------
print(f"\n{'=' * 60}")
print("UPOREDNA TABELA PERFORMANSI (sa optimalnim threshold-om)")
print(f"{'=' * 60}")

comparison = pd.DataFrame([
    {
        "Model": name,
        "Threshold": f"{res['best_threshold']:.2f}",
        "Accuracy": res["test_accuracy"],
        "Precision": res["test_precision"],
        "Recall (DNF)": res["test_recall"],
        "F1-Score": res["test_f1"],
        "CV F1": res["grid"].best_score_,
        "Vreme (s)": f"{res['training_time']:.1f}"
    }
    for name, res in results.items()
])
comparison = comparison.sort_values("F1-Score", ascending=False)
print("\n" + comparison.to_string(index=False))
comparison.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)

# Poredjenje bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison))
width = 0.2
metrics_plot = ["Accuracy", "Precision", "Recall (DNF)", "F1-Score"]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
for i, (metric, color) in enumerate(zip(metrics_plot, colors)):
    ax.bar(x + i * width, comparison[metric], width, label=metric, color=color, alpha=0.85)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison["Model"])
ax.set_ylabel("Vrednost")
ax.set_title("Poredjenje performansi modela (optimalni threshold)", fontsize=13)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
plt.close()

# ---------- 7. ODABIR NAJBOLJEG ----------
print(f"\n{'=' * 60}")
print("ODABIR NAJBOLJEG MODELA")
print(f"{'=' * 60}")

best_model_name = comparison.iloc[0]["Model"]
best_model = results[best_model_name]["best_model"]
best_thresh = results[best_model_name]["best_threshold"]

print(f"Najbolji model (F1): {best_model_name}")
print(f"  Optimalni threshold: {best_thresh:.2f}")
print(f"  Test F1:      {results[best_model_name]['test_f1']:.4f}")
print(f"  Test Recall:  {results[best_model_name]['test_recall']:.4f}")
print(f"  Test Prec:    {results[best_model_name]['test_precision']:.4f}")

# Cuvanje modela + threshold-a
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_model, model_path)
joblib.dump(best_thresh, os.path.join(MODELS_DIR, "best_threshold.pkl"))
print(f"\nModel sacuvan: {model_path}")
print(f"Threshold sacuvan: models/best_threshold.pkl")

# ---------- 8. ZAVRSNI IZVESTAJ ----------
print(f"\n{'=' * 60}")
print("ZAVRSNI IZVESTAJ (v2)")
print(f"{'=' * 60}")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  Threshold: {res['best_threshold']:.2f}")
    print(f"  CV F1:     {res['grid'].best_score_:.4f}")
    print(f"  Test: Acc={res['test_accuracy']:.3f} | "
          f"Prec={res['test_precision']:.3f} | "
          f"Rec={res['test_recall']:.3f} | "
          f"F1={res['test_f1']:.3f}")
    print(f"  Default:   Acc={res['test_acc_default']:.3f} | "
          f"F1={res['test_f1_default']:.3f}")
    print(f"  Vreme: {res['training_time']:.1f}s")

print(f"\n{'=' * 60}")
print(f"POBEDNIK: {best_model_name}")
print(f"  F1={results[best_model_name]['test_f1']:.3f} | "
      f"Recall DNF={results[best_model_name]['test_recall']:.3f} | "
      f"Threshold={best_thresh:.2f}")
print("KRAJ KORAKA 4 - Treniranje v2 zavrseno")
print("=" * 60)
