"""
KORAK 4b: ANOMALY DETECTION (Isolation Forest + One-Class SVM)
- DNF se tretira kao outlier/anomalija
- Modeli se treniraju SAMO na klasi "Zavrsio" (0) — uce sta je "normalno"
- Anomaly score = verovatnoca DNF-a
- Threshold tuning na val skupu, evaluacija na testu
- BEZ SMOTE-a — anomaly detection radi sa prirodnom distribucijom
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. PUTANJE
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

COST_FN = 5
COST_FP = 1
BASELINE_COST = None

print("=" * 60)
print("KORAK 4b: ANOMALY DETECTION")
print("  DNF se tretira kao outlier — modeli uce sta je 'normalna' trka")
print("=" * 60)

import itertools

def _expand_grid(param_grid):
    for combo in itertools.product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), combo))

# ============================================================
# 2. UCITAVANJE + FEATURE ENGINEERING
# ============================================================
df = pd.read_csv(CLEANED_PATH)
print(f"\nUcitano: {df.shape[0]} redova")

df = df.sort_values(["year", "sequence"]).reset_index(drop=True)

# --- Feature engineering (isti kao 03_preprocessing.py) ---
print("Feature engineering...")

df["Historical_DNF_Rate"] = (
    df.groupby(["rider_name", "circuit_name"])["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
)
rider_global = df.groupby("rider_name")["Status_Zavrsetka"].transform(
    lambda x: x.shift().expanding().mean()
)
df["Historical_DNF_Rate"] = df["Historical_DNF_Rate"].fillna(rider_global).fillna(0.0)

df["Rider_Experience"] = df.groupby("rider_name").cumcount()

df["Team_DNF_Rate"] = (
    df.groupby("team_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

df["Bike_DNF_Rate"] = (
    df.groupby("bike_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

df["Track_DNF_Rate"] = (
    df.groupby("circuit_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

df["Season_Phase"] = df.groupby("year")["sequence"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)

# --- Qualifying circuit features ---
qual_path = os.path.join(RAW_DIR, "Qualifying.csv")
df_qual = pd.read_csv(qual_path)
df_qual = df_qual[df_qual["class"] == "MotoGP"].copy()
df_qual = df_qual.drop_duplicates()

def _parse_lap_time(t):
    t = str(t)
    if ":" not in t:
        return None
    parts = t.split(":")
    return float(parts[0]) * 60 + float(parts[1])

df_qual["time_sec"] = df_qual["time"].apply(_parse_lap_time)
q2 = df_qual[df_qual["session"] == "Q2"].dropna(subset=["time_sec"])
circuit_year = q2.groupby(["event", "year"]).apply(
    lambda g: pd.Series({
        "Quali_Spread":    g["time_sec"].max() - g["time_sec"].min(),
        "Quali_Top6_Gap":  g.nsmallest(6, "time_sec")["time_sec"].max()
                           - g["time_sec"].min(),
        "Quali_Time_Std":  g["time_sec"].std()
    })
).reset_index()
circuit_features = (circuit_year
    .groupby("event")[["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]]
    .median().reset_index()
    .rename(columns={"event": "shortname"})
)
df = df.merge(circuit_features, on="shortname", how="left")
for col in ["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]:
    df[col] = df[col].fillna(df[col].median())

print(f"  Ukupno feature-a: {len(df.columns)}")
print(f"  DNF distribution: {df['Status_Zavrsetka'].sum()} ({df['Status_Zavrsetka'].mean()*100:.1f}%)")

# ============================================================
# 3. SPLIT (bez SMOTE-a!)
# ============================================================
categorical_cols = ["shortname", "circuit_name", "rider_name", "team_name", "bike_name"]
numerical_cols = ["year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
                  "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase",
                  "Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]
feature_cols = categorical_cols + numerical_cols

y = df["Status_Zavrsetka"]
X = df[feature_cols]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"  Train DNF: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Val DNF:   {y_val.sum()} ({y_val.mean()*100:.1f}%)")
print(f"  Test DNF:  {y_test.sum()} ({y_test.mean()*100:.1f}%)")

BASELINE_COST = y_test.sum() * COST_FN
print(f"\nBaseline cost (uvek 'Zavrsio'): {BASELINE_COST}")

# ============================================================
# 4. PREPROCESSING (bez SMOTE)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(target_type="binary", random_state=42), categorical_cols),
        ("scaler", StandardScaler(), numerical_cols)
    ],
    remainder="passthrough"
)

X_train_enc = preprocessor.fit_transform(X_train, y_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)

# Izdvajanje SAMO normalne klase (Zavrsio=0) za anomaly trening
X_normal_train = X_train_enc[y_train.values == 0]
X_normal_val = X_val_enc[y_val.values == 0]

print(f"\nAnomaly trening — samo 'Zavrsio' klasa: {len(X_normal_train)} uzoraka")

# ============================================================
# 5. ANOMALY DETECTION MODELI
# ============================================================
anomaly_models = {
    "Isolation Forest": {
        "model": IsolationForest(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_samples": [0.5, 0.75, 1.0],
            "contamination": [0.05, 0.08, 0.12, 0.15],
        }
    },
    "One-Class SVM": {
        "model": OneClassSVM(),
        "params": {
            "kernel": ["rbf"],
            "nu": [0.05, 0.08, 0.12, 0.15],
            "gamma": ["scale", "auto", 0.01, 0.1],
        }
    }
}

results = {}

for name, config in anomaly_models.items():
    print(f"\n{'=' * 60}")
    print(f"ANOMALY DETECTION: {name}")
    print(f"{'=' * 60}")

    best_model = None
    best_params = None
    best_cost_val = float("inf")
    best_threshold = 0.5
    overall_start = time.time()

    for params_combo in _expand_grid(config["params"]):
        model = type(config["model"])(**{**params_combo, "random_state": 42}
                                       if name == "Isolation Forest" else params_combo)

        start_time = time.time()

        if name == "Isolation Forest":
            model.fit(X_normal_train)
            anomaly_scores = -model.score_samples(X_val_enc)
        else:
            model.fit(X_normal_train)
            anomaly_scores = -model.decision_function(X_val_enc)

        for t in np.arange(
            np.percentile(anomaly_scores, 80),
            np.percentile(anomaly_scores, 99),
            (np.percentile(anomaly_scores, 99) - np.percentile(anomaly_scores, 80)) / 30
        ):
            y_pred = (anomaly_scores >= t).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
                cost = FN * COST_FN + FP * COST_FP
                if cost < best_cost_val:
                    best_cost_val = cost
                    best_threshold = t
                    best_model = model
                    best_params = params_combo

    elapsed = time.time() - overall_start

    print(f"  Najbolji params: {best_params}")
    print(f"  Best val cost: {best_cost_val:.0f}, threshold={best_threshold:.4f}")

    # --- Evaluacija na testu ---
    if name == "Isolation Forest":
        test_scores = -best_model.score_samples(X_test_enc)
    else:
        test_scores = -best_model.decision_function(X_test_enc)

    y_pred_test = (test_scores >= best_threshold).astype(int)
    cm_test = confusion_matrix(y_test, y_pred_test)
    TN, FP, FN, TP = cm_test.ravel() if cm_test.shape == (2, 2) else (0, 0, 0, 0)
    total_cost = FN * COST_FN + FP * COST_FP
    saving_pct = (1 - total_cost / BASELINE_COST) * 100 if BASELINE_COST > 0 else 0

    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    prec = precision_score(y_test, y_pred_test, zero_division=0)

    print(f"\n  Test skup:")
    print(f"    CM: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    print(f"    Cost={total_cost} | Saving={saving_pct:.1f}%")
    print(f"    F1={f1:.4f} | Rec={rec:.4f} | Prec={prec:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["Zavrsio", "DNF"], zero_division=0))

    results[name] = {
        "model": best_model,
        "params": best_params,
        "threshold": best_threshold,
        "test_scores": test_scores,
        "y_pred": y_pred_test,
        "cm": cm_test,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "cost": total_cost,
        "saving_pct": saving_pct,
        "f1": f1, "recall": rec, "precision": prec,
    }

    # --- Score distribucija ---
    # Recalculate val scores with best model for plotting
    if name == "Isolation Forest":
        val_scores = -best_model.score_samples(X_val_enc)
    else:
        val_scores = -best_model.decision_function(X_val_enc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label, scores_subset, y_subset in [
        (axes[0], "Val", val_scores, y_val),
        (axes[1], "Test", test_scores, y_test)
    ]:
        for cls, color, lbl in [(0, "#2ecc71", "Zavrsio"), (1, "#e74c3c", "DNF")]:
            mask = y_subset.values == cls
            if mask.sum() > 0:
                ax.hist(scores_subset[mask], bins=40, alpha=0.6, color=color, label=lbl)
        ax.axvline(x=best_threshold, color="black", linestyle="--", linewidth=2,
                   label=f"Thresh={best_threshold:.4f}")
        ax.set_title(f"{label} — {name}")
        ax.set_xlabel("Anomaly Score (vise = DNF)")
        ax.legend()
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower().replace("-", "_")
    plt.savefig(os.path.join(FIGURES_DIR, f"anomaly_scores_{safe_name}.png"), dpi=150)
    plt.close()

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_test, display_labels=["Zavrsio", "DNF"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} — Cost={total_cost} (saving {saving_pct:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"anomaly_cm_{safe_name}.png"), dpi=150)
    plt.close()

# ============================================================
# 6. UPOREDNA TABELA
# ============================================================
print(f"\n{'=' * 60}")
print("UPOREDNA TABELA — ANOMALY DETECTION")
print(f"{'=' * 60}")

comparison = []
for name, res in results.items():
    comparison.append({
        "Model": name,
        "Threshold": f"{res['threshold']:.4f}",
        "TN": res["TN"], "FP": res["FP"],
        "FN": res["FN"], "TP": res["TP"],
        "Accuracy": accuracy_score(y_test, res["y_pred"]),
        "Precision": res["precision"],
        "Recall (DNF)": res["recall"],
        "F1-Score": res["f1"],
        "Cost": res["cost"],
        "Saving": f"{res['saving_pct']:.1f}%",
    })

comparison_df = pd.DataFrame(comparison)
comparison_df = comparison_df.sort_values("F1-Score", ascending=False)
print("\n" + comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(METRICS_DIR, "anomaly_comparison.csv"), index=False)

# ============================================================
# 7. SNIMANJE NAJBOLJEG
# ============================================================
best_name = comparison_df.iloc[0]["Model"]
best_res = results[best_name]

model_path = os.path.join(MODELS_DIR, "best_model_anomaly.pkl")
thresh_path = os.path.join(MODELS_DIR, "best_threshold_anomaly.pkl")
prep_path = os.path.join(MODELS_DIR, "preprocessor_anomaly.pkl")

joblib.dump(best_res["model"], model_path)
joblib.dump(best_res["threshold"], thresh_path)
joblib.dump(preprocessor, prep_path)

print(f"\n{'=' * 60}")
print(f"POBEDNIK: {best_name}")
print(f"  F1={best_res['f1']:.4f} | Rec={best_res['recall']:.4f} | "
      f"Prec={best_res['precision']:.4f} | Cost={best_res['cost']}")
print(f"  Model: {model_path}")
print(f"  Threshold: {thresh_path}")
print("KRAJ KORAKA 4b")
