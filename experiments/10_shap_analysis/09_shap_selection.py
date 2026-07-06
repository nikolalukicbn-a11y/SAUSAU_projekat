"""
KORAK 9: SHAP ANALIZA + FEATURE SELECTION (v2)
- Koristi nekalibrisani XGBoost za SHAP (CalibratedClassifierCV nije podrzan)
- Permutation importance + SHAP + RFE
- Testira podskupove feature-a
"""
import pandas as pd, numpy as np, joblib, os, time, shap
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
FIGURES = os.path.join(BASE, "results", "figures")
METRICS = os.path.join(BASE, "results", "metrics")
os.makedirs(FIGURES, exist_ok=True)

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC, "y_test.csv")).values.ravel()

feature_names = list(X_train.columns)
COST_FN, COST_FP = 5, 1

print("=" * 60)
print("KORAK 9: SHAP + FEATURE SELECTION")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Treniramo nekalibrisani XGBoost za SHAP
print("\nTreniranje XGBoost (nekalibrisani, za SHAP)...")
xgb = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                     scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
xgb.fit(X_train, y_train)

# ============================================================
# 1. SHAP
# ============================================================
print("\n--- 1. SHAP analiza ---")
explainer = shap.TreeExplainer(xgb)
sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].values

t0 = time.time()
shap_values = explainer.shap_values(X_sample)
print(f"  Vreme: {time.time()-t0:.1f}s")

shap_imp = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("  SHAP vaznost:")
for _, row in shap_imp.iterrows():
    print(f"    {row['feature']:30s}: {row['mean_abs_shap']:.6f}")

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP plotovi sacuvani")

# ============================================================
# 2. PERMUTATION IMPORTANCE
# ============================================================
print("\n--- 2. Permutation importance (F1) ---")
perm = permutation_importance(xgb, X_test, y_test, n_repeats=5, random_state=42,
                               scoring="f1", n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

print("  Permutation importance:")
for _, row in perm_df.iterrows():
    print(f"    {row['feature']:30s}: {row['importance_mean']:.6f} +/- {row['importance_std']:.6f}")

# ============================================================
# 3. KOMBINOVANA VAZNOST
# ============================================================
print("\n--- 3. Kombinovana vaznost ---")
comb = shap_imp.copy().rename(columns={"mean_abs_shap": "shap"})
comb = comb.merge(perm_df[["feature", "importance_mean"]], on="feature")
comb = comb.rename(columns={"importance_mean": "permutation"})
comb["shap_norm"] = comb["shap"] / comb["shap"].max()
comb["perm_norm"] = comb["permutation"] / (comb["permutation"].max() + 1e-9)
comb["combined"] = (comb["shap_norm"] + comb["perm_norm"]) / 2
comb = comb.sort_values("combined", ascending=False)

print("  Rankirani feature-i:")
for i, (_, row) in enumerate(comb.iterrows()):
    print(f"    {i+1:2}. {row['feature']:30s}: SHAP={row['shap_norm']:.4f}, "
          f"Perm={row['perm_norm']:.4f}, Comb={row['combined']:.4f}")

# ============================================================
# 4. TESTIRANJE PODSKUPOVA (samo XGBoost, brza evaluacija)
# ============================================================
print("\n--- 4. Podskupovi feature-a (XGBoost standalone, threshold tuning) ---")
print(f"    Baseline (16 feat): ref FN=20, FP=1536")

for k in [4, 6, 8, 10, 12, 14, 16]:
    top_feat = comb["feature"].head(k).tolist()
    Xt_train = X_train[top_feat]
    Xt_test = X_test[top_feat]

    mx = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                        scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
    mx.fit(Xt_train, y_train)
    px = mx.predict_proba(Xt_test)[:, 1]

    best = {"f1": 0, "FN": 999, "FP": 999, "thr": 0, "rec": 0}
    for t in np.arange(0.10, 0.90, 0.01):
        yp = (px >= t).astype(int)
        cm = confusion_matrix(y_test, yp)
        FN, FP = cm[1, 0], cm[0, 1]
        rec = recall_score(y_test, yp, zero_division=0)
        # Try to find FN <= 25, if not found relax
        target_fn = 25 if k <= 8 else 20
        if FN <= target_fn and FP < best["FP"]:
            best = {"f1": f1_score(y_test, yp, zero_division=0), "FN": FN, "FP": FP,
                    "thr": t, "rec": rec}
    if best["FP"] == 999:
        for t in np.arange(0.10, 0.90, 0.01):
            yp = (px >= t).astype(int)
            cm = confusion_matrix(y_test, yp)
            FN, FP = cm[1, 0], cm[0, 1]
            rec = recall_score(y_test, yp, zero_division=0)
            if FN < best["FN"]:
                best = {"f1": f1_score(y_test, yp, zero_division=0), "FN": FN, "FP": FP,
                        "thr": t, "rec": rec}

    print(f"    Top {k:2}: FN={best['FN']}, FP={best['FP']}, Rec={best['rec']:.4f}, "
          f"F1={best['f1']:.4f}  (thr={best['thr']:.2f})")

# ============================================================
# 5. OR TEST SA NAJBOLJIM PODSKUPOM
# ============================================================
print("\n--- 5. OR ensemble sa top podskupovima ---")
for k in [6, 8, 10, 12, 16]:
    top_feat = comb["feature"].head(k).tolist()
    Xt_train = X_train[top_feat]
    Xt_test = X_test[top_feat]

    mx = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                        scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
    mx.fit(Xt_train, y_train)
    ml = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)
    ml.fit(Xt_train, y_train)

    px = mx.predict_proba(Xt_test)[:, 1]
    pl = ml.predict_proba(Xt_test)[:, 1]

    best = {"f1": 0, "FN": 99, "FP": 9999, "rec": 0, "tx": 0, "tl": 0}
    for tx in np.arange(0.10, 0.60, 0.02):
        for tl in np.arange(0.10, 0.50, 0.02):
            yp = ((px >= tx) | (pl >= tl)).astype(int)
            cm = confusion_matrix(y_test, yp)
            FN, FP = cm[1, 0], cm[0, 1]
            rec = recall_score(y_test, yp, zero_division=0)
            # Prioritize FN <= 15, then minimize FP
            if FN <= 15 and FP < best["FP"]:
                best = {"f1": f1_score(y_test, yp, zero_division=0), "FN": FN, "FP": FP,
                        "rec": rec, "tx": tx, "tl": tl}
    if best["FP"] >= 9999:
        # Relax: find best FN
        for tx in np.arange(0.10, 0.60, 0.02):
            for tl in np.arange(0.10, 0.50, 0.02):
                yp = ((px >= tx) | (pl >= tl)).astype(int)
                cm = confusion_matrix(y_test, yp)
                FN, FP = cm[1, 0], cm[0, 1]
                rec = recall_score(y_test, yp, zero_division=0)
                if FN < best["FN"]:
                    best = {"f1": f1_score(y_test, yp, zero_division=0), "FN": FN, "FP": FP,
                            "rec": rec, "tx": tx, "tl": tl}

    print(f"    Top {k:2} OR: FN={best['FN']}, FP={best['FP']}, Rec={best['rec']:.4f}, "
          f"F1={best['f1']:.4f}  (XGB={best['tx']:.2f}, LR={best['tl']:.2f})")

# ============================================================
# 6. ZAKLJUCAK
# ============================================================
comb.to_csv(os.path.join(METRICS, "feature_importance_combined.csv"), index=False)
print(f"\nFeature importance -> results/metrics/feature_importance_combined.csv")
print("KRAJ KORAKA 9")
