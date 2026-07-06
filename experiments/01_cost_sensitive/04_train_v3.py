"""
KORAK 4 v3: COST-SENSITIVE TRENIRANJE I EVALUACIJA
- False Negative (DNF propušten) je SKUPLJI od False Positive (lažni alarm)
- COST_FN / COST_FP ratio = 5:1 (podesivo)
- sample_weight za DNF redove = COST_FN, za ostale = COST_FP
- Custom cost scorer za GridSearchCV (umesto F1)
- Threshold tuning minimizuje total cost na val skupu
- Evaluacija sa cost matricom
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
    make_scorer, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ============================================================
# 1. PUTANJE
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ============================================================
# 2. KONFIGURACIJA COST-SENSITIVE
# ============================================================
COST_FN = 5   # kazna za propušten DNF (false negative)
COST_FP = 1   # kazna za lažni alarm (false positive)
# Ratio 5:1 znači: propustiti DNF je 5x gore nego lažno predvideti DNF

print("=" * 60)
print("KORAK 4 v3: COST-SENSITIVE TRENIRANJE")
print(f"  COST_FN (propušten DNF) = {COST_FN}")
print(f"  COST_FP (lažni alarm)   = {COST_FP}")
print(f"  Ratio FN:FP = {COST_FN}:{COST_FP}")
print("=" * 60)

# ============================================================
# 3. CUSTOM COST SCORER
# ============================================================

def cost_loss(y_true, y_pred):
    """
    Računa ukupni trošak: FN * COST_FN + FP * COST_FP
    Vraća NEGATIVNU vrednost jer GridSearchCV maksimizuje scorer.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        return -(FN * COST_FN + FP * COST_FP)
    return -float("inf")


cost_scorer = make_scorer(cost_loss, greater_is_better=True)


def decompose_cost(cm):
    """Razlaže matricu konfuzije na komponente troška."""
    TN, FP, FN, TP = cm.ravel()
    return {
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "cost_FN": FN * COST_FN,
        "cost_FP": FP * COST_FP,
        "total_cost": FN * COST_FN + FP * COST_FP,
    }


# ============================================================
# 4. UCITAVANJE
# ============================================================
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

print(f"\nTrain: {X_train.shape} (DNF: {y_train.sum()}, "
      f"Zavrsili: {len(y_train)-y_train.sum()})")
print(f"Val:   {X_val.shape} (DNF: {y_val.sum()})")
print(f"Test:  {X_test.shape} (DNF: {y_test.sum()})")

# -----------------------------------------------------------
# Sample weight: viši za DNF redove
# -----------------------------------------------------------
sample_weight_train = np.where(y_train == 1, COST_FN, COST_FP).astype(float)
sample_weight_train = sample_weight_train / sample_weight_train.mean()

print(f"\nSample weight - DNF prosek: {sample_weight_train[y_train==1].mean():.2f}, "
      f"Non-DNF prosek: {sample_weight_train[y_train==0].mean():.2f}")

# Baseline cost: kad model uvek predvidi "Završio" (svi 0)
baseline_cost = (y_test == 1).sum() * COST_FN
print(f"\nBaseline cost (always predict 'Završio'): {baseline_cost}")
print(f"  ({ (y_test==1).sum() } propuštenih DNF-ova × COST_FN={COST_FN})")

# ============================================================
# 5. MODELI
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "class_weight": [None, "balanced"]
        },
        "supports_sw": True
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", "balanced_subsample"]
        },
        "supports_sw": True
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, 3, 5, 8]
        },
        "supports_sw": True
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42, max_iter=5000),
        "params": {
            "C": [0.1, 1],
            "kernel": ["rbf"],
            "class_weight": ["balanced"]
        },
        "supports_sw": True
    },
    "KNN": {
        "model": KNeighborsClassifier(n_jobs=-1),
        "params": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        },
        "supports_sw": False
    },
    "MLP Classifier": {
        "model": MLPClassifier(random_state=42, early_stopping=True, max_iter=1000),
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (50, 25)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001]
        },
        "supports_sw": True
    }
}

results = {}

# ============================================================
# 6. TRENIRANJE
# ============================================================
for name, config in models.items():
    print(f"\n{'=' * 60}")
    print(f"TRENIRANJE: {name} (cost-sensitive)")
    print(f"{'=' * 60}")

    start_time = time.time()

    grid = GridSearchCV(
        config["model"], config["params"],
        cv=cv, scoring=cost_scorer, n_jobs=-1, verbose=0
    )

    if config["supports_sw"]:
        grid.fit(X_train, y_train, sample_weight=sample_weight_train)
        print(f"  -&gt; sample_weight primenjen (DNF težina = {COST_FN}x)")
    else:
        grid.fit(X_train, y_train)
        print(f"  -&gt; sample_weight NIJE podržan, koristi se samo cost scorer")

    elapsed = time.time() - start_time

    best_model = grid.best_estimator_
    print(f"  Vreme: {elapsed:.1f}s")
    print(f"  Najbolji parametri: {grid.best_params_}")
    print(f"  Najbolji CV Cost: {-grid.best_score_:.1f} (negativan = bolji)")

    # ---------- THRESHOLD TUNING (minimizuje cost) ----------
    y_val_proba = best_model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.05)
    best_threshold = 0.5
    best_cost_val = float("inf")

    threshold_costs = []
    for t in thresholds:
        y_pred_t = (y_val_proba >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred_t)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            c = FN * COST_FN + FP * COST_FP
        else:
            c = float("inf")
        threshold_costs.append({
            "threshold": t,
            "cost": c,
            "f1": f1_score(y_val, y_pred_t),
            "precision": precision_score(y_val, y_pred_t, zero_division=0),
            "recall": recall_score(y_val, y_pred_t, zero_division=0),
        })
        if c < best_cost_val:
            best_cost_val = c
            best_threshold = t

    df_thresh = pd.DataFrame(threshold_costs)
    best_row = df_thresh[df_thresh["threshold"] == best_threshold].iloc[0]

    print(f"\n  Optimalni threshold (min cost): {best_threshold:.2f}")
    print(f"    Val Cost = {best_cost_val:.0f} | "
          f"F1 = {best_row['f1']:.4f} | "
          f"Prec = {best_row['precision']:.4f} | "
          f"Rec = {best_row['recall']:.4f}")

    # ---------- EVALUACIJA NA TESTU ----------
    y_test_pred_default = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred_opt = (y_test_proba >= best_threshold).astype(int)

    cm_def = confusion_matrix(y_test, y_test_pred_default)
    cm_opt = confusion_matrix(y_test, y_test_pred_opt)

    cost_def = decompose_cost(cm_def)
    cost_opt = decompose_cost(cm_opt)

    # Cost ratio: koliko smo uštedeli u odnosu na baseline
    cost_saving_pct = (1 - cost_opt["total_cost"] / baseline_cost) * 100 if baseline_cost > 0 else 0

    print(f"\n  --- Test skup ---")
    print(f"  Default threshold (0.5):")
    print(f"    CM: TN={cost_def['TN']}, FP={cost_def['FP']}, "
          f"FN={cost_def['FN']}, TP={cost_def['TP']}")
    print(f"    Cost: FP={cost_def['cost_FP']}, FN={cost_def['cost_FN']}, "
          f"TOTAL={cost_def['total_cost']}")
    print(f"    Acc={accuracy_score(y_test, y_test_pred_default):.3f}, "
          f"F1={f1_score(y_test, y_test_pred_default):.3f}")

    print(f"\n  Optimalni threshold ({best_threshold:.2f}):")
    print(f"    CM: TN={cost_opt['TN']}, FP={cost_opt['FP']}, "
          f"FN={cost_opt['FN']}, TP={cost_opt['TP']}")
    print(f"    Cost: FP={cost_opt['cost_FP']}, FN={cost_opt['cost_FN']}, "
          f"TOTAL={cost_opt['total_cost']} (baseline={baseline_cost}, "
          f"ušteda={cost_saving_pct:.1f}%)")
    print(f"    Acc={accuracy_score(y_test, y_test_pred_opt):.3f}, "
          f"Prec={precision_score(y_test, y_test_pred_opt, zero_division=0):.3f}, "
          f"Rec={recall_score(y_test, y_test_pred_opt, zero_division=0):.3f}, "
          f"F1={f1_score(y_test, y_test_pred_opt, zero_division=0):.3f}")

    print(f"\n  Classification Report (optimalni threshold):")
    print(classification_report(y_test, y_test_pred_opt,
                                target_names=["Zavrsio (0)", "DNF (1)"]))
    print(f"  Ušteda u odnosu na 'uvek Završio': {cost_saving_pct:.1f}%")

    results[name] = {
        "best_model": best_model,
        "grid": grid,
        "best_threshold": best_threshold,
        "test_acc_def": accuracy_score(y_test, y_test_pred_default),
        "test_f1_def": f1_score(y_test, y_test_pred_default),
        "test_acc_opt": accuracy_score(y_test, y_test_pred_opt),
        "test_prec_opt": precision_score(y_test, y_test_pred_opt, zero_division=0),
        "test_rec_opt": recall_score(y_test, y_test_pred_opt, zero_division=0),
        "test_f1_opt": f1_score(y_test, y_test_pred_opt, zero_division=0),
        "cost_opt": cost_opt,
        "cost_saving_pct": cost_saving_pct,
        "training_time": elapsed,
        "cm_opt": cm_opt,
        "cm_def": cm_def,
        "supports_sw": config["supports_sw"],
    }

    # ---------- CONFUSION MATRIX PLOT ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_def, display_labels=["Zavrsio", "DNF"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{name} - Default (0.5)\nCost={cost_def['total_cost']}")

    ConfusionMatrixDisplay(cm_opt, display_labels=["Zavrsio", "DNF"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title(f"{name} - Optimal ({best_threshold:.2f})\n"
                      f"Cost={cost_opt['total_cost']} (ušteda {cost_saving_pct:.1f}%)")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(FIGURES_DIR,
                             f"confusion_matrix_cost_{safe_name}.png"), dpi=150)
    plt.close()

    # ---------- THRESHOLD VS COST PLOT ----------
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df_thresh["threshold"], df_thresh["cost"],
             "r-o", ms=5, label="Total Cost", linewidth=2.5)
    ax1.axvline(x=best_threshold, color="black", linestyle="--", alpha=0.7,
                label=f"Optimal={best_threshold:.2f}")
    ax1.axhline(y=baseline_cost, color="gray", linestyle=":", alpha=0.6,
                label=f"Baseline cost={baseline_cost}")
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Total Cost (FN×COST_FN + FP×COST_FP)", fontsize=12, color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(df_thresh["threshold"], df_thresh["f1"],
             "g--^", ms=5, label="F1 (ref)", alpha=0.5)
    ax2.plot(df_thresh["threshold"], df_thresh["recall"],
             "b--s", ms=4, label="Recall (ref)", alpha=0.4)
    ax2.set_ylabel("F1 / Recall", fontsize=12, color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{name} - Threshold vs Cost (FN/FP={COST_FN}/{COST_FP})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,
                             f"threshold_cost_{safe_name}.png"), dpi=150)
    plt.close()

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
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
ax.set_title("Feature Importance - Random Forest (Cost-Sensitive)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance_cost.png"), dpi=150)
plt.close()

# ============================================================
# 8. COST ANALYSIS — MULTIPLE RATIOS
# ============================================================
print(f"\n{'=' * 60}")
print("COST ANALIZA ZA VIŠE FN/FP RATIOSA")
print(f"{'=' * 60}")

cost_ratios = [1, 2, 3, 5, 10]
ratio_analysis = []

for ratio in cost_ratios:
    for name, res in results.items():
        cm = res["cm_opt"]
        TN, FP, FN, TP = cm.ravel()
        total_cost = FN * ratio + FP * 1
        ratio_analysis.append({
            "Model": name,
            "FN/FP Ratio": f"{ratio}:1",
            "FN": FN, "FP": FP,
            f"Cost (ratio={ratio}:1)": total_cost,
            "F1": res["test_f1_opt"],
            "Recall": res["test_rec_opt"],
            "Precision": res["test_prec_opt"],
        })

df_ratio = pd.DataFrame(ratio_analysis)
print("\nPivot: ukupni cost po modelu i ratio-u (manji = bolji):")
pivot = df_ratio.pivot_table(
    index="Model",
    columns="FN/FP Ratio",
    values=f"Cost (ratio=5:1)",
    aggfunc="first"
)
# Print full table
print(df_ratio.to_string(index=False))
df_ratio.to_csv(os.path.join(METRICS_DIR,
                              "cost_ratio_analysis.csv"), index=False)

# ============================================================
# 9. UPOREDNA TABELA
# ============================================================
print(f"\n{'=' * 60}")
print("UPOREDNA TABELA PERFORMANSI (cost-sensitive)")
print(f"{'=' * 60}")

comparison = pd.DataFrame([
    {
        "Model": name,
        "SW": "Da" if res["supports_sw"] else "Ne",
        "Thresh": f"{res['best_threshold']:.2f}",
        "Accuracy": res["test_acc_opt"],
        "Precision": res["test_prec_opt"],
        "Recall (DNF)": res["test_rec_opt"],
        "F1-Score": res["test_f1_opt"],
        "Cost (FN+FP)": res["cost_opt"]["total_cost"],
        "Ušteda vs baseline": f"{res['cost_saving_pct']:.1f}%",
        "CV Cost": -res["grid"].best_score_,
        "Vreme (s)": f"{res['training_time']:.1f}"
    }
    for name, res in results.items()
])
# Sort by cost (manji = bolji)
comparison = comparison.sort_values("Cost (FN+FP)", ascending=True)
print("\n" + comparison.to_string(index=False))
comparison.to_csv(os.path.join(METRICS_DIR,
                                "model_comparison_cost.csv"), index=False)

# Poredjenje bar chart (cost-focused)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(comparison))
width = 0.2
metrics_plot = ["Accuracy", "Precision", "Recall (DNF)", "F1-Score"]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
for i, (metric, color) in enumerate(zip(metrics_plot, colors)):
    ax.bar(x + i * width, comparison[metric], width,
           label=metric, color=color, alpha=0.85)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison["Model"], rotation=20, ha="right")
ax.set_ylabel("Vrednost")
ax.set_title(f"Poredjenje performansi (cost-sensitive, FN/FP={COST_FN}:{COST_FP})",
             fontsize=13)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_cost.png"), dpi=150)
plt.close()

# Cost bar chart
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(comparison["Model"], comparison["Cost (FN+FP)"],
              color=["#2ecc71" if c < baseline_cost else "#e74c3c"
                     for c in comparison["Cost (FN+FP)"]])
ax.axhline(y=baseline_cost, color="red", linestyle="--",
           linewidth=2, label=f"Baseline (uvek 'Završio') = {baseline_cost}")
ax.set_ylabel("Total Cost")
ax.set_title(f"Ukupni trošak po modelu (FN×{COST_FN} + FP×{COST_FP})",
             fontsize=13)
for bar, cost in zip(bars, comparison["Cost (FN+FP)"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(cost), ha="center", fontsize=9, fontweight="bold")
ax.legend()
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "cost_comparison.png"), dpi=150)
plt.close()

# ============================================================
# 10. ODABIR NAJBOLJEG (po cost-u)
# ============================================================
print(f"\n{'=' * 60}")
print("ODABIR NAJBOLJEG MODELA (po minimalnom cost-u)")
print(f"{'=' * 60}")

best_model_name = comparison.iloc[0]["Model"]
best_model = results[best_model_name]["best_model"]
best_thresh = results[best_model_name]["best_threshold"]
best_cost = results[best_model_name]["cost_opt"]["total_cost"]
best_saving = results[best_model_name]["cost_saving_pct"]

print(f"Najbolji model (min cost): {best_model_name}")
print(f"  Threshold:   {best_thresh:.2f}")
print(f"  Total cost:  {best_cost} (baseline={baseline_cost}, "
      f"ušteda={best_saving:.1f}%)")
print(f"  Test F1:     {results[best_model_name]['test_f1_opt']:.4f}")
print(f"  Test Recall: {results[best_model_name]['test_rec_opt']:.4f}")
print(f"  Test Prec:   {results[best_model_name]['test_prec_opt']:.4f}")

# Cuvanje
model_path = os.path.join(MODELS_DIR, "best_model_cost.pkl")
thresh_path = os.path.join(MODELS_DIR, "best_threshold_cost.pkl")
joblib.dump(best_model, model_path)
joblib.dump(best_thresh, thresh_path)
print(f"\nModel: {model_path}")
print(f"Threshold: {thresh_path}")

# ============================================================
# 11. ZAVRSNI IZVESTAJ
# ============================================================
print(f"\n{'=' * 60}")
print(f"ZAVRSNI IZVESTAJ v3 (cost-sensitive, FN/FP={COST_FN}:{COST_FP})")
print(f"{'=' * 60}")

for name, res in results.items():
    c = res["cost_opt"]
    print(f"\n{name}:")
    print(f"  SampleWeight: {'Da' if res['supports_sw'] else 'Ne'}")
    print(f"  Threshold:    {res['best_threshold']:.2f}")
    print(f"  Cost:         {c['total_cost']} | "
          f"Usteda: {res['cost_saving_pct']:.1f}%")
    print(f"  CM:           TN={c['TN']}, FP={c['FP']}, "
          f"FN={c['FN']}, TP={c['TP']}")
    print(f"  Test:         Acc={res['test_acc_opt']:.3f} | "
          f"Prec={res['test_prec_opt']:.3f} | "
          f"Rec={res['test_rec_opt']:.3f} | "
          f"F1={res['test_f1_opt']:.3f}")
    print(f"  Vreme:        {res['training_time']:.1f}s")

print(f"\n{'=' * 60}")
print(f"POBEDNIK: {best_model_name}")
print(f"  Cost = {best_cost} | Usteda = {best_saving:.1f}%")
print(f"  F1 = {results[best_model_name]['test_f1_opt']:.3f} | "
      f"Recall = {results[best_model_name]['test_rec_opt']:.3f} | "
      f"Threshold = {best_thresh:.2f}")
print(f"\n  -&gt; {results[best_model_name]['cm_opt'].ravel()}")
print("KRAJ KORAKA 4 v3 - Cost-sensitive treniranje zavrseno")
print("=" * 60)
