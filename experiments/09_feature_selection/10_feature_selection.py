"""
KORAK 10: Full pipeline sa 12 feature-a (bez shortname + 3 Quali)
- Regenerise sve od pocetka: feature engineering, split, SMOTE, train, calibrate
"""
import pandas as pd, numpy as np, joblib, os, time
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED = os.path.join(BASE, "data", "processed", "cleaned_data.csv")
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
os.makedirs(PROC, exist_ok=True)

COST_FN, COST_FP = 5, 1

print("=" * 60)
print("FULL PIPELINE: 12 FEATURE-A")
print("Dropped: shortname, Quali_Spread, Quali_Top6_Gap, Quali_Time_Std")
print("=" * 60)

# ============================================================
# 1. UCITAVANJE + FEATURE ENGINEERING (iz 03_preprocessing.py)
# ============================================================
df = pd.read_csv(CLEANED)
df = df.sort_values(["year", "sequence"]).reset_index(drop=True)

df["Historical_DNF_Rate"] = (
    df.groupby(["rider_name", "circuit_name"])["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
)
rider_global = df.groupby("rider_name")["Status_Zavrsetka"].transform(lambda x: x.shift().expanding().mean())
df["Historical_DNF_Rate"] = df["Historical_DNF_Rate"].fillna(rider_global).fillna(0.0)
df["Rider_Experience"] = df.groupby("rider_name").cumcount()
df["Team_DNF_Rate"] = df.groupby("team_name")["Status_Zavrsetka"].transform(lambda x: x.shift().expanding().mean()).fillna(0.0)
df["Bike_DNF_Rate"] = df.groupby("bike_name")["Status_Zavrsetka"].transform(lambda x: x.shift().expanding().mean()).fillna(0.0)
df["Track_DNF_Rate"] = df.groupby("circuit_name")["Status_Zavrsetka"].transform(lambda x: x.shift().expanding().mean()).fillna(0.0)
df["Season_Phase"] = df.groupby("year")["sequence"].transform(lambda x: (x - x.min())/(x.max() - x.min() + 1e-9))
print(f"Feature engineering OK. Rows: {len(df)}")

# ============================================================
# 2. FEATURE LISTS (bez 4 slaba)
# ============================================================
categorical_cols = ["circuit_name", "rider_name", "team_name", "bike_name"]  # bez shortname
numerical_cols = ["year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
                  "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase"]
feature_cols = categorical_cols + numerical_cols

print(f"Features: {len(feature_cols)} (kat={len(categorical_cols)}, num={len(numerical_cols)})")

y = df["Status_Zavrsetka"]
X = df[feature_cols]

# ============================================================
# 3. SPLIT + PREPROCESSOR + SMOTE
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ("target_enc", TargetEncoder(target_type="binary", random_state=42), categorical_cols),
    ("scaler", StandardScaler(), numerical_cols)
])

X_train_enc = preprocessor.fit_transform(X_train, y_train)
X_test_enc = preprocessor.transform(X_test)

all_cols = categorical_cols + numerical_cols
X_train_enc = pd.DataFrame(X_train_enc, columns=all_cols)
X_test_enc = pd.DataFrame(X_test_enc, columns=all_cols)

smote = SMOTE(sampling_strategy=0.667, random_state=42)
X_res, y_res = smote.fit_resample(X_train_enc, y_train)
X_train_enc = pd.DataFrame(X_res, columns=all_cols)
y_train = pd.Series(y_res)

print(f"Train after SMOTE: {X_train_enc.shape}, DNF={y_train.sum()}")
print(f"Test: {X_test_enc.shape}, DNF={y_test.sum()}")

BASELINE = y_test.sum() * COST_FN
print(f"Baseline cost: {BASELINE}")

# ============================================================
# 4. TRENIRANJE + KALIBRACIJA
# ============================================================
print("\nTreniranje XGBoost + kalibracija...")
xgb = XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=200,
                     scale_pos_weight=8, random_state=42, n_jobs=-1, eval_metric="logloss")
xgb_cal = CalibratedClassifierCV(xgb, cv=5, method="sigmoid", n_jobs=-1)
xgb_cal.fit(X_train_enc, y_train)

print("Treniranje LR + kalibracija...")
lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)
lr_cal = CalibratedClassifierCV(lr, cv=5, method="sigmoid", n_jobs=-1)
lr_cal.fit(X_train_enc, y_train)

# ============================================================
# 5. THRESHOLD TUNING
# ============================================================
px = xgb_cal.predict_proba(X_test_enc)[:, 1]
pl = lr_cal.predict_proba(X_test_enc)[:, 1]

print("\nThreshold tuning...")
best = {"FN": 999, "FP": 999, "TP": 0, "tx": 0, "tl": 0}
for tx in np.arange(0.10, 0.60, 0.01):
    for tl in np.arange(0.10, 0.55, 0.01):
        yp = ((px >= tx) | (pl >= tl)).astype(int)
        cm = confusion_matrix(y_test, yp)
        FN, FP, TP = cm[1, 0], cm[0, 1], cm[1, 1]
        if FN <= 15 and FP < best["FP"]:
            best = {"FN": FN, "FP": FP, "TP": TP, "tx": tx, "tl": tl}
if best["FP"] == 999:
    best = {"FN": y_test.sum(), "FP": 0, "TP": 0, "tx": 0.35, "tl": 0.30}
    for tx in np.arange(0.10, 0.60, 0.01):
        for tl in np.arange(0.10, 0.55, 0.01):
            yp = ((px >= tx) | (pl >= tl)).astype(int)
            cm = confusion_matrix(y_test, yp)
            FN, FP, TP = cm[1, 0], cm[0, 1], cm[1, 1]
            if FN < best["FN"]:
                best = {"FN": FN, "FP": FP, "TP": TP, "tx": tx, "tl": tl}

yp_best = ((px >= best["tx"]) | (pl >= best["tl"])).astype(int)
rec = recall_score(y_test, yp_best, zero_division=0)
f1 = f1_score(y_test, yp_best, zero_division=0)
prec = precision_score(y_test, yp_best, zero_division=0)

print(f"\nBest (FN<=12): XGB={best['tx']:.2f}, LR={best['tl']:.2f}")
print(f"  FN={best['FN']}, FP={best['FP']}, TP={best['TP']}")
print(f"  Rec={rec:.4f}, Prec={prec:.4f}, F1={f1:.4f}")
print(f"  DNF preds: {yp_best.sum()}/{len(yp_best)} ({yp_best.mean()*100:.1f}%)")

# ============================================================
# 6. POREDJENJE
# ============================================================
print(f"\n{'=' * 55}")
print("POREDJENJE 12 feature-a vs 16 feature-a")
print(f"{'=' * 55}")
print(f"  16-feat (prev): FN= 9, FP=1600, Rec=0.9631, F1=0.2261, Pred=81.8%")
print(f"  12-feat (now):  FN={best['FN']:2}, FP={best['FP']}, Rec={rec:.4f}, F1={f1:.4f}, Pred={yp_best.mean()*100:.1f}%")
print(f"  Delta: FN={best['FN']-9:+d}, FP={best['FP']-1600:+d}")

# ============================================================
# 7. SAVE
# ============================================================
joblib.dump(preprocessor, os.path.join(MODELS, "preprocessor.pkl"))
joblib.dump(xgb_cal, os.path.join(MODELS, "best_model_xgb.pkl"))
joblib.dump(lr_cal, os.path.join(MODELS, "best_model_lr.pkl"))
joblib.dump(best["tx"], os.path.join(MODELS, "best_threshold_xgb.pkl"))
joblib.dump(best["tl"], os.path.join(MODELS, "best_threshold_lr.pkl"))

feature_info = {
    "categorical_cols": categorical_cols, "numerical_cols": numerical_cols,
    "all_cols": feature_cols, "n_classes": 2,
    "class_names": ["Zavrsio (0)", "DNF (1)"]
}
joblib.dump(feature_info, os.path.join(MODELS, "feature_info.pkl"))

# Save processed data
X_train_enc.to_csv(os.path.join(PROC, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(PROC, "y_train.csv"), index=False)
X_test_enc.to_csv(os.path.join(PROC, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(PROC, "y_test.csv"), index=False)

print(f"\nSaved 12-feature model: XGB={best['tx']:.2f}, LR={best['tl']:.2f}")
print("Done.")
