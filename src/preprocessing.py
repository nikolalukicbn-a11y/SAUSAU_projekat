"""
KORAK 3: PREPROCESIRANJE + EDA + FEATURE ENGINEERING (v2)
- Prosireni feature engineering (vise izvedenih karakteristika)
- EDA vizualizacije
- TargetEncoder za kategoricke varijable (umesto LabelEncoder-a)
- StandardScaler za numericke varijable
- Train/val/test split (70/15/15) PRE enkodiranja (sprecava data leakage)
- Cuvanje ColumnTransformer-a, enkodera i procesiranih podataka

POBOLJSANJA U ODNOSU NA v1:
  1. TargetEncoding: kategorije se enkodiraju prosecnom vrednosti targeta
     (sa cross-fitting-om), sto daje mnogo vise informacija tree modelima
     nego proizvoljni integer LabelEncoder-a
  2. Vise feature-a: RiderExperience, Team_DNF_Rate, Bike_DNF_Rate,
     Track_DNF_Rate, SeasonPhase
  3. Split PRE enkodiranja - sprecava data leakage kroz TargetEncoder
"""

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

# ---------- 1. PUTANJE ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("KORAK 3: PREPROCESIRANJE I EDA (v2 - poboljsano)")
print("=" * 60)

# ---------- 2. UCITAVANJE ----------
df = pd.read_csv(CLEANED_PATH)
print(f"\nUcitano: {df.shape[0]} redova x {df.shape[1]} kolona")
print(f"Kolone: {list(df.columns)}")

# ---------- 2b. QUALIFYING CIRCUIT FEATURES ----------
print("\n--- 2b. Qualifying circuit features (MotoGP Q2, 2022-2025) ---")

qual_path = os.path.join(RAW_DIR, "Qualifying.csv")
df_qual = pd.read_csv(qual_path)
df_qual = df_qual[df_qual["class"] == "MotoGP"].copy()
df_qual = df_qual.drop_duplicates()
print(f"  MotoGP qualifying redova: {len(df_qual)} (2022-2025)")

# Parse lap time "MM:SS.sss" → seconds


def _parse_lap_time(t):
    t = str(t)
    if ":" not in t:
        return None
    parts = t.split(":")
    return float(parts[0]) * 60 + float(parts[1])


df_qual["time_sec"] = df_qual["time"].apply(_parse_lap_time)

# Samo Q2 — najbrzi vozaci, konzistentniji uslovi
q2 = df_qual[df_qual["session"] == "Q2"].dropna(subset=["time_sec"])

# Po stazi i godini
circuit_year = q2.groupby(["event", "year"]).apply(
    lambda g: pd.Series({
        "Quali_Spread":    g["time_sec"].max() - g["time_sec"].min(),
        "Quali_Top6_Gap":  g.nsmallest(6, "time_sec")["time_sec"].max()
        - g["time_sec"].min(),
        "Quali_Time_Std":  g["time_sec"].std()
    })
).reset_index()

# Medijan kroz sve dostupne godine — robustan na wet session outlier-e
circuit_features = (circuit_year
                    .groupby("event")[["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]]
                    .median().reset_index()
                    .rename(columns={"event": "shortname"})
                    )

# Merge na glavni dataset (left join — cuva sve redove)
df = df.merge(circuit_features, on="shortname", how="left")

# Staze bez qualifying podataka → popuni medijanom svih staza
for col in ["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]:
    median_val = df[col].median()
    missing = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: mean={df[col].mean():.3f}s, "
          f"median={df[col].median():.3f}s, "
          f"filled NaN: {missing}")

print(f"  Ukupno staza u qualifying: {len(circuit_features)}, "
      f"novih feature-a: 3")

# DATA LEAKAGE CHECK: ovi feature-i su statični po stazi (medijan kroz 2022-2025).
# Ne zavise od pojedinačne trke, vozača ili godine — ne unose leakage.
# Primenjeni PRE train/val/test split-a (samo mapiranje shortname → broj).

# ---------- 3. FEATURE ENGINEERING ----------
print("\n--- 3. Feature Engineering ---")

df = df.sort_values(["year", "sequence"]).reset_index(drop=True)

# 3a. Historical DNF Rate (vozac + staza, iz prethodnih sezona)
print("  3a. Historical_DNF_Rate (vozac na stazi)")
df["Historical_DNF_Rate"] = (
    df.groupby(["rider_name", "circuit_name"])["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
)

# Popuni NaN: prvo globalni DNF rate za vozaca, pa 0
rider_global = df.groupby("rider_name")["Status_Zavrsetka"].transform(
    lambda x: x.shift().expanding().mean()
)
df["Historical_DNF_Rate"] = df["Historical_DNF_Rate"].fillna(
    rider_global).fillna(0.0)

# 3b. Rider Experience (broj prethodnih trka vozaca)
print("  3b. Rider_Experience (broj prethodnih nastupa)")
df["Rider_Experience"] = df.groupby("rider_name").cumcount()

# 3c. Team DNF Rate (procenat DNF-ova tima na svim stazama, prethodne sezone)
print("  3c. Team_DNF_Rate")
df["Team_DNF_Rate"] = (
    df.groupby("team_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

# 3d. Bike DNF Rate (procenat DNF-ova konstruktora, prethodne sezone)
print("  3d. Bike_DNF_Rate")
df["Bike_DNF_Rate"] = (
    df.groupby("bike_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

# 3e. Track DNF Rate (procenat DNF-ova na stazi, svi vozaci, prethodne sezone)
print("  3e. Track_DNF_Rate")
df["Track_DNF_Rate"] = (
    df.groupby("circuit_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

# 3f. Season Phase (koji deo sezone: sequence / max_sequence po godini)
print("  3f. Season_Phase (faza sezone 0-1)")
df["Season_Phase"] = df.groupby("year")["sequence"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)

print("\nFeature engineering zavrsen. Nove izvedene kolone:")
derived = ["Historical_DNF_Rate", "Rider_Experience", "Team_DNF_Rate",
           "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase"]
for col in derived:
    print(f"  {col}: mean={df[col].mean():.4f}, median={df[col].median():.4f}, "
          f"NaN={df[col].isnull().sum()}")

# ---------- 4. EDA VIZUALIZACIJE ----------
print("\n--- 4. EDA Vizualizacije ---")

# 4a. Distribucija targeta
print("  4a. Distribucija targeta")
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Status_Zavrsetka"].value_counts()
bars = ax.bar(["Zavrsio (0)", "DNF (1)"], [counts.get(0, 0), counts.get(1, 0)],
              color=["#2ecc71", "#e74c3c"])
ax.set_title("Distribucija target varijable - Status_Zavrsetka")
for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{count}\n({count/len(df)*100:.1f}%)", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"), dpi=150)
plt.close()

# 4b. Korelaciona matrica (svi numericki + izvedeni feature-i)
print("  4b. Korelaciona matrica")
num_cols = ["year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
            "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate",
            "Season_Phase", "Status_Zavrsetka"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Korelaciona matrica (prosireni feature-i)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "correlation_matrix.png"), dpi=150)
plt.close()
print("    Korelacije sa targetom:")
for col in [c for c in num_cols if c != "Status_Zavrsetka"]:
    corr_val = df[col].corr(df["Status_Zavrsetka"])
    print(f"      {col:25s}: {corr_val:.4f}")

# 4c. Boxplot novih feature-a po klasama
print("  4c. Boxplot novih feature-a po klasama")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, derived):
    df.boxplot(column=col, by="Status_Zavrsetka", ax=ax)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "boxplots_by_class.png"), dpi=150)
plt.close()

# ---------- 5. PRIPREMA ZA ENKODIRANJE ----------
print("\n--- 5. Priprema za enkodiranje ---")

categorical_cols = ["shortname", "circuit_name",
                    "rider_name", "team_name", "bike_name"]
numerical_cols = ["year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
                  "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase",
                  "Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]

# Target
y = df["Status_Zavrsetka"]
feature_cols = categorical_cols + numerical_cols
X = df[feature_cols]

print(f"Kategoricki feature-i ({len(categorical_cols)}): {categorical_cols}")
print(f"Numericki feature-i ({len(numerical_cols)}): {numerical_cols}")
print(f"Ukupno feature-a: {len(feature_cols)}")

# ---------- 6. TRAIN/VAL/TEST SPLIT (PRE enkodiranja!) ----------
print("\n--- 6. Train/Val/Test split (70/15/15) PRE enkodiranja ---")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%) - "
      f"DNF: {y_train.sum()}, Zavrsili: {len(y_train)-y_train.sum()}")
print(f"Val:   {len(X_val)} ({len(X_val)/len(df)*100:.1f}%) - "
      f"DNF: {y_val.sum()}, Zavrsili: {len(y_val)-y_val.sum()}")
print(f"Test:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%) - "
      f"DNF: {y_test.sum()}, Zavrsili: {len(y_test)-y_test.sum()}")

# ---------- 7. TARGET ENCODING + STANDARD SCALING ----------
print("\n--- 7. TargetEncoder + StandardScaler ---")

# ColumnTransformer: TargetEncoder za kategoricke, StandardScaler za numericke
preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(
            target_type="binary", random_state=42), categorical_cols),
        ("scaler", StandardScaler(), numerical_cols)
    ],
    remainder="passthrough"
)

# Fit na train, transform na sve
X_train_enc = preprocessor.fit_transform(X_train, y_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)

# Konvertuj u DataFrame za cuvanje (sve kolone su numericke)
all_cols = categorical_cols + numerical_cols
X_train_enc = pd.DataFrame(X_train_enc, columns=all_cols)
X_val_enc = pd.DataFrame(X_val_enc, columns=all_cols)
X_test_enc = pd.DataFrame(X_test_enc, columns=all_cols)

print(f"Target encoding + scaling zavrsen")
print(f"Train shape: {X_train_enc.shape}")
print(f"Val shape:   {X_val_enc.shape}")
print(f"Test shape:  {X_test_enc.shape}")

# ---------- 7b. SMOTE OVERSAMPLING 60/40 ----------
print("\n--- 7b. SMOTE oversampling 60/40 ---")
print(f"Pre SMOTE - Train skup:")
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()
print(f"  Klasa 0 (Zavrsio): {n0}")
print(f"  Klasa 1 (DNF):     {n1}")
print(f"  Ratio DNF/Ukupno:  {n1/(n0+n1)*100:.1f}%")

smote = SMOTE(sampling_strategy=0.667, random_state=42)
X_res, y_res = smote.fit_resample(X_train_enc, y_train)

X_train_enc = pd.DataFrame(X_res, columns=all_cols)
y_train = pd.Series(y_res, name="Status_Zavrsetka")

n0_new = (y_train == 0).sum()
n1_new = (y_train == 1).sum()
print(f"Posle SMOTE 60/40 - Train skup:")
print(f"  Klasa 0 (Zavrsio): {n0_new}")
print(f"  Klasa 1 (DNF):     {n1_new}")
print(f"  Ratio DNF/Ukupno:  {n1_new/(n0_new+n1_new)*100:.1f}%")
print(f"  Ukupno redova:     {len(X_train_enc)}")

# ---------- 8. CUVANJE ----------
print("\n--- 8. Cuvanje procesiranih podataka ---")

# Cuvanje preprocessora (TargetEncoder + StandardScaler)
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
print("preprocessor.pkl sacuvan (TargetEncoder + StandardScaler)")

# Cuvanje feature info (bitno za deployment)
feature_info = {
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "all_cols": feature_cols,
    "n_classes": 2,
    "class_names": ["Zavrsio (0)", "DNF (1)"]
}
joblib.dump(feature_info, os.path.join(MODELS_DIR, "feature_info.pkl"))
print("feature_info.pkl sacuvan")

# Cuvanje procesiranih skupova
X_train_enc.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
pd.Series(y_train.values, name="Status_Zavrsetka").to_csv(
    os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
X_val_enc.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
pd.Series(y_val.values, name="Status_Zavrsetka").to_csv(
    os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)
X_test_enc.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
pd.Series(y_test.values, name="Status_Zavrsetka").to_csv(
    os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
print("Procesirani skupovi sacuvani")

# ---------- 9. SUMIRANI IZVESTAJ ----------
print("\n" + "=" * 60)
print("SUMIRANI IZVESTAJ PREPROCESIRANJA (v2)")
print("=" * 60)
print(f"""
Novi feature-i:       Historical_DNF_Rate, Rider_Experience, Team_DNF_Rate,
                      Bike_DNF_Rate, Track_DNF_Rate, Season_Phase,
                      Quali_Spread, Quali_Top6_Gap, Quali_Time_Std (2022-2025)
Enkodiranje:          TargetEncoder (umesto LabelEncoder-a) - daje vise info
Skaliranje:           StandardScaler
Split:                70/15/15 PRE enkodiranja (bez leakage-a)
Balansiranje:         SMOTE (60/40) + class_weight/scale_pos_weight u modelima

UKUPNO FEATURE-A:     {len(feature_cols)}
  Kategoricki:        {len(categorical_cols)} (TargetEncoded)
  Numericki:          {len(numerical_cols)} (StandardScaled)

TARGET:               Status_Zavrsetka (0=Zavrsio, 1=DNF)
  Disbalans:          {y.sum()/len(y)*100:.1f}% DNF
""")
print("=" * 60)
print("KRAJ KORAKA 3 - Preprocesiranje v2 zavrseno")
print("=" * 60)
