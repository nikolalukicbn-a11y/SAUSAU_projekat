"""
KORAK 3b: PROSIRENI FEATURE ENGINEERING
- Svi postojeci feature-i + NOVI:
  1. Rider_Global_DNF_Rate   (expanding mean DNF per vozaca, sve staze)
  2. Rider_Recent_DNF        (DNF rate u poslednjih 5 trka)
  3. Bike_Recent_DNF         (DNF rate konstruktora u poslednjih 5 trka)
  4. Team_Recent_DNF         (DNF rate tima u poslednjih 5 trka)
  5. Rider_Season_DNF        (DNF rate ove sezone do sada)
  6. Circuit_Global_DNF_Rate (ukupna DNF stopa staze, nije expanding)
  7. Race_Count_in_Season    (koliko trka je vec odvezeno ove sezone)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("KORAK 3b: PROSIRENI FEATURE ENGINEERING (v2)")
print("=" * 60)

# ============================================================
# 1. UCITAVANJE
# ============================================================
df = pd.read_csv(CLEANED_PATH)
print(f"\nUcitano: {df.shape[0]} redova x {df.shape[1]} kolona")
df = df.sort_values(["year", "sequence"]).reset_index(drop=True)

# ============================================================
# 2. STARI FEATURE-I
# ============================================================
print("\n--- Stari feature-i ---")

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

# Qualifying features
qual_path = os.path.join(RAW_DIR, "Qualifying.csv")
df_qual = pd.read_csv(qual_path)
df_qual = df_qual[df_qual["class"] == "MotoGP"].copy()
df_qual = df_qual.drop_duplicates()

def _parse_lap_time(t):
    t = str(t)
    if ":" not in t: return None
    parts = t.split(":")
    return float(parts[0]) * 60 + float(parts[1])

df_qual["time_sec"] = df_qual["time"].apply(_parse_lap_time)
q2 = df_qual[df_qual["session"] == "Q2"].dropna(subset=["time_sec"])
circuit_year = q2.groupby(["event", "year"]).apply(
    lambda g: pd.Series({
        "Quali_Spread":    g["time_sec"].max() - g["time_sec"].min(),
        "Quali_Top6_Gap":  g.nsmallest(6, "time_sec")["time_sec"].max() - g["time_sec"].min(),
        "Quali_Time_Std":  g["time_sec"].std()
    })
).reset_index()
circuit_features = (circuit_year
    .groupby("event")[["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]]
    .median().reset_index().rename(columns={"event": "shortname"}))
df = df.merge(circuit_features, on="shortname", how="left")
for col in ["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]:
    df[col] = df[col].fillna(df[col].median())

print("  Stari feature-i dodati")

# ============================================================
# 3. NOVI FEATURE-I
# ============================================================
print("\n--- Novi feature-i ---")

# 3a. Rider_Global_DNF_Rate — expanding mean, sve staze
df["Rider_Global_DNF_Rate"] = rider_global.fillna(0.0)
print("  3a. Rider_Global_DNF_Rate: OK")

# 3b. Rider_Recent_DNF — poslednjih 5 trka
def rolling_dnf_last_n(group, n=5):
    series = group.shift(1).rolling(window=n, min_periods=1).mean()
    return series

df["Rider_Recent_DNF"] = (
    df.groupby("rider_name")["Status_Zavrsetka"]
    .transform(lambda x: rolling_dnf_last_n(x, 5))
).fillna(0.0)
print("  3b. Rider_Recent_DNF (last 5): OK")

# 3c. Bike_Recent_DNF
df["Bike_Recent_DNF"] = (
    df.groupby("bike_name")["Status_Zavrsetka"]
    .transform(lambda x: rolling_dnf_last_n(x, 5))
).fillna(0.0)
print("  3c. Bike_Recent_DNF (last 5): OK")

# 3d. Team_Recent_DNF
df["Team_Recent_DNF"] = (
    df.groupby("team_name")["Status_Zavrsetka"]
    .transform(lambda x: rolling_dnf_last_n(x, 5))
).fillna(0.0)
print("  3d. Team_Recent_DNF (last 5): OK")

# 3e. Rider_Season_DNF — DNF rate u tekucoj sezoni
df["Rider_Season_DNF"] = (
    df.groupby(["rider_name", "year"])["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)
print("  3e. Rider_Season_DNF: OK")

# 3f. Circuit_Global_DNF_Rate — ukupna DNF stopa staze (svi vozaci, sve godine)
# Racunamo expanding mean po stazi (bez shifta — koristi trenutnu trku u racunu za global)
df["Circuit_Global_DNF_Rate"] = (
    df.groupby("circuit_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)
print("  3f. Circuit_Global_DNF_Rate: OK")

# 3g. Race_Count_in_Season — koliko trka je vec odvezeno
df["Race_Count_in_Season"] = df.groupby("year").cumcount()
print("  3g. Race_Count_in_Season: OK")

# Provera novih feature-a
new_features = [
    "Rider_Global_DNF_Rate", "Rider_Recent_DNF",
    "Bike_Recent_DNF", "Team_Recent_DNF",
    "Rider_Season_DNF", "Circuit_Global_DNF_Rate",
    "Race_Count_in_Season"
]
print("\n  Provera novih feature-a:")
for col in new_features:
    print(f"    {col:30s}: mean={df[col].mean():.4f}, NaN={df[col].isnull().sum()}")

# ============================================================
# 4. EDA VIZUALIZACIJE
# ============================================================
print("\n--- EDA Vizualizacije ---")

# Korelaciona matrica sa novim feature-ima
num_cols = ["year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
            "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate",
            "Season_Phase", "Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std",
            "Rider_Global_DNF_Rate", "Rider_Recent_DNF",
            "Bike_Recent_DNF", "Team_Recent_DNF",
            "Rider_Season_DNF", "Circuit_Global_DNF_Rate",
            "Race_Count_in_Season", "Status_Zavrsetka"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.6})
ax.set_title("Korelaciona matrica (prosireni feature-i v2)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "correlation_matrix_v2.png"), dpi=150)
plt.close()

print("  Korelacije sa targetom (Status_Zavrsetka):")
for col in [c for c in num_cols if c != "Status_Zavrsetka"]:
    corr_val = df[col].corr(df["Status_Zavrsetka"])
    marker = " *** NOVO" if col in new_features else ""
    print(f"    {col:30s}: {corr_val:+.4f}{marker}")

# ============================================================
# 5. PRIPREMA ZA ENKODIRANJE
# ============================================================
print("\n--- Priprema za enkodiranje ---")

categorical_cols = ["shortname", "circuit_name", "rider_name", "team_name", "bike_name"]
numerical_cols = ["year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
                  "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase",
                  "Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std",
                  "Rider_Global_DNF_Rate", "Rider_Recent_DNF",
                  "Bike_Recent_DNF", "Team_Recent_DNF",
                  "Rider_Season_DNF", "Circuit_Global_DNF_Rate",
                  "Race_Count_in_Season"]

y = df["Status_Zavrsetka"]
feature_cols = categorical_cols + numerical_cols
X = df[feature_cols]

print(f"  Kategoricki: {len(categorical_cols)}, Numericki: {len(numerical_cols)}, Ukupno: {len(feature_cols)}")
print(f"  NOVI numericki: {[c for c in numerical_cols if c in new_features]}")

# ============================================================
# 6. SPLIT (70/15/15)
# ============================================================
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

# ============================================================
# 7. TARGET ENCODING + SCALER + SMOTE
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

all_cols = categorical_cols + numerical_cols
X_train_enc = pd.DataFrame(X_train_enc, columns=all_cols)
X_val_enc = pd.DataFrame(X_val_enc, columns=all_cols)
X_test_enc = pd.DataFrame(X_test_enc, columns=all_cols)

# SMOTE 60/40
smote = SMOTE(sampling_strategy=0.667, random_state=42)
X_res, y_res = smote.fit_resample(X_train_enc, y_train)
X_train_enc = pd.DataFrame(X_res, columns=all_cols)
y_train = pd.Series(y_res, name="Status_Zavrsetka")

print(f"\nPosle SMOTE: Train={X_train_enc.shape}, DNF={y_train.sum()}, "
      f"Non-DNF={len(y_train)-y_train.sum()}")

# ============================================================
# 8. CUVANJE
# ============================================================
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

feature_info = {
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "all_cols": feature_cols,
    "new_features": [c for c in new_features if c in numerical_cols],
    "n_classes": 2,
    "class_names": ["Zavrsio (0)", "DNF (1)"]
}
joblib.dump(feature_info, os.path.join(MODELS_DIR, "feature_info.pkl"))

# Cuvanje procesiranih skupova
X_train_enc.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
X_val_enc.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)
X_test_enc.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

print(f"\nProcesirani podaci sacuvani (v2 sa {len(numerical_cols)} numerickih feature-a)")

# ============================================================
# 9. SUMARNI IZVESTAJ
# ============================================================
print(f"\n{'=' * 60}")
print("SUMMARNI IZVESTAJ v2")
print(f"{'=' * 60}")
print(f"  Ukupno feature-a: {len(feature_cols)} (kategoricki: {len(categorical_cols)}, numericki: {len(numerical_cols)})")
print(f"  Stari numericki:  {len([c for c in numerical_cols if c not in new_features])}")
print(f"  NOVI numericki:   {len([c for c in numerical_cols if c in new_features])}")
print(f"  Novi feature-i:   {', '.join(new_features)}")
print(f"  Target disbalans: DNF={y.sum()/len(y)*100:.1f}%")
print("KRAJ KORAKA 3b")
print("=" * 60)
