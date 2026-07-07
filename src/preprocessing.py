"""
KORAK 3: PREPROCESIRANJE, FEATURE ENGINEERING I EDA
=====================================================
Generise 16 feature-a (5 kategorickih + 11 numerickih) iz ociscenog dataseta.

OPERACIJE:
  1. Feature engineering — 6 izvedenih numerickih feature-a
  2. Qualifying circuit features (3) — podaci o stazi iz Q2 sesija
  3. EDA vizualizacije — distribucija targeta, korelacije, boxplotovi
  4. TargetEncoder za kategoricke feature-e (prosecna vrednost targeta)
  5. StandardScaler za numericke feature-e (mean=0, std=1)
  6. Train/Val/Test split 70/15/15 PRE enkodiranja (sprecava data leakage)
  7. SMOTE oversampling 60/40 (samo na trening skupu!)

FEATURE-I (16 ukupno):
  Kategoricki (5) — enkodirani TargetEncoder-om:
    shortname, circuit_name, rider_name, team_name, bike_name

  Numericki (11) — skalirani StandardScaler-om:
    Osnovni: year, sequence
    Izvedeni: Historical_DNF_Rate, Rider_Experience, Team_DNF_Rate,
              Bike_DNF_Rate, Track_DNF_Rate, Season_Phase
    Qualifying: Quali_Spread, Quali_Top6_Gap, Quali_Time_Std

VAZNO — Zasto split PRE enkodiranja?
  TargetEncoder racuna prosecnu vrednost targeta za svaku kategoriju.
  Ako bismo prvo enkodirali pa onda delili, TargetEncoder bi "video"
  test podatke kroz target vrednosti — to je DATA LEAKAGE.
  Zato se split radi PRE enkodiranja, a TargetEncoder.fit() samo na train.

Ulaz:
  - data/processed/cleaned_data.csv  (iz koraka 2)
  - data/raw/Qualifying.csv          (za qualifying feature-e)

Izlaz:
  - data/processed/X_train.csv, y_train.csv  (SMOTE 60/40)
  - data/processed/X_val.csv,   y_val.csv    (validacija)
  - data/processed/X_test.csv,  y_test.csv   (test)
  - models/preprocessor.pkl                 (ColumnTransformer)
  - models/feature_info.pkl                 (metapodaci o feature-ima)
  - results/figures/                        (EDA plotovi)
"""

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder

matplotlib.use("Agg")

# ============================================================
# 1. DEFINISANJE PUTANJA
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "experiments", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("KORAK 3: PREPROCESIRANJE I FEATURE ENGINEERING")
print("=" * 60)

# ============================================================
# 2. UCITAVANJE OCISCENOG DATASETA
# ============================================================
df = pd.read_csv(CLEANED_PATH)
print(f"\nUcitano: {df.shape[0]} redova x {df.shape[1]} kolona")
print(f"Kolone: {list(df.columns)}")

# ============================================================
# 3. QUALIFYING CIRCUIT FEATURES (iz Qualifying.csv)
# ============================================================
# Racunamo prosecne karakteristike staza iz Q2 kvalifikacionih sesija.
# Q2 = najbrzi vozaci, najkonzistentniji uslovi.
# Ovi feature-i su STATICNI po stazi (medijan kroz 2022-2025),
# ne unose data leakage jer ne zavise od pojedinacne trke ili vozaca.
print("\n--- 3. Qualifying circuit features (MotoGP Q2, 2022-2025) ---")

qual_path = os.path.join(RAW_DIR, "Qualifying.csv")
df_qual = pd.read_csv(qual_path)

# Filtriramo samo MotoGP klasu i uklanjamo duplikate
df_qual = df_qual[df_qual["class"] == "MotoGP"].copy()
df_qual = df_qual.drop_duplicates()
print(f"  MotoGP qualifying redova: {len(df_qual)} (2022-2025)")


def _parse_lap_time(t):
    """Konvertuje vreme kruga iz formata "MM:SS.sss" u sekunde."""
    t = str(t)
    if ":" not in t:
        return None
    parts = t.split(":")
    return float(parts[0]) * 60 + float(parts[1])


df_qual["time_sec"] = df_qual["time"].apply(_parse_lap_time)

# Samo Q2 sesije — najbrzi vozaci, konzistentniji uslovi od Q1
q2 = df_qual[df_qual["session"] == "Q2"].dropna(subset=["time_sec"])

# Grupisi po stazi (event) i godini (year), izracunaj metrike
circuit_year = (
    q2.groupby(["event", "year"])
    .apply(
        lambda g: pd.Series({
            "Quali_Spread":   g["time_sec"].max() - g["time_sec"].min(),
            "Quali_Top6_Gap": g.nsmallest(6, "time_sec")["time_sec"].max()
                               - g["time_sec"].min(),
            "Quali_Time_Std": g["time_sec"].std(),
        })
    )
    .reset_index()
)

# Medijan kroz sve dostupne godine — robustan na outlier-e (npr. wet sessions)
circuit_features = (
    circuit_year
    .groupby("event")[["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]]
    .median()
    .reset_index()
    .rename(columns={"event": "shortname"})
)

# Spajanje (merge) sa glavnim datasetom — left join cuva sve redove
df = df.merge(circuit_features, on="shortname", how="left")

# Staze bez qualifying podataka (istorijske staze koje vise nisu u kalendaru)
# popunjavamo medijanom svih staza
for col in ["Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std"]:
    median_val = df[col].median()
    missing = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: mean={df[col].mean():.3f}s, "
          f"median={df[col].median():.3f}s, popunjeno NaN: {missing}")

print(f"  Ukupno staza u qualifying bazi: {len(circuit_features)}, "
      f"dodato 3 nova feature-a")

# ============================================================
# 4. FEATURE ENGINEERING (izvedeni numericki feature-i)
# ============================================================
# Sortiramo hronoloski pre racunanja vremenskih feature-a
df = df.sort_values(["year", "sequence"]).reset_index(drop=True)

print("\n--- 4. Feature Engineering ---")

# 4a. Historical_DNF_Rate — DNF stopa vozaca na konkretnoj stazi
# Racuna se EXPANDING MEAN (samo prethodne trke, ne ukljucuje trenutnu)
# Ako vozac nema istoriju na toj stazi, koristi globalni DNF rate vozaca
print("  4a. Historical_DNF_Rate (vozac na konkretnoj stazi)")
df["Historical_DNF_Rate"] = (
    df.groupby(["rider_name", "circuit_name"])["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
)
# Fallback: globalni DNF rate vozaca (sve staze)
rider_global = df.groupby("rider_name")["Status_Zavrsetka"].transform(
    lambda x: x.shift().expanding().mean()
)
df["Historical_DNF_Rate"] = (
    df["Historical_DNF_Rate"].fillna(rider_global).fillna(0.0)
)

# 4b. Rider_Experience — broj prethodnih MotoGP/500cc nastupa
print("  4b. Rider_Experience (broj prethodnih nastupa)")
df["Rider_Experience"] = df.groupby("rider_name").cumcount()

# 4c. Team_DNF_Rate — DNF stopa tima (svi vozaci tog tima, sve staze)
print("  4c. Team_DNF_Rate (prosecna DNF stopa tima)")
df["Team_DNF_Rate"] = (
    df.groupby("team_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

# 4d. Bike_DNF_Rate — DNF stopa konstruktora (svi vozaci na toj masini)
print("  4d. Bike_DNF_Rate (prosecna DNF stopa konstruktora)")
df["Bike_DNF_Rate"] = (
    df.groupby("bike_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

# 4e. Track_DNF_Rate — DNF stopa same staze (svi vozaci, sve godine)
print("  4e. Track_DNF_Rate (prosecna DNF stopa staze)")
df["Track_DNF_Rate"] = (
    df.groupby("circuit_name")["Status_Zavrsetka"]
    .transform(lambda x: x.shift().expanding().mean())
).fillna(0.0)

# 4f. Season_Phase — faza sezone (0 = pocetak, 1 = kraj)
# Normalizovano po godini: sequence / max_sequence za tu godinu
print("  4f. Season_Phase (faza sezone 0-1)")
df["Season_Phase"] = df.groupby("year")["sequence"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)

# Provera novih feature-a
print("\n  Pregled izvedenih feature-a:")
derived = [
    "Historical_DNF_Rate", "Rider_Experience", "Team_DNF_Rate",
    "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase",
]
for col in derived:
    print(f"    {col:30s}: mean={df[col].mean():.4f}, "
          f"median={df[col].median():.4f}, NaN={df[col].isnull().sum()}")

# ============================================================
# 5. EDA VIZUALIZACIJE
# ============================================================

# 5a. Distribucija target varijable (bar chart)
print("\n--- 5. EDA Vizualizacije ---")
print("  5a. Distribucija targeta")
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Status_Zavrsetka"].value_counts()
bars = ax.bar(
    ["Zavrsio (0)", "DNF (1)"],
    [counts.get(0, 0), counts.get(1, 0)],
    color=["#2ecc71", "#e74c3c"],
)
ax.set_title("Distribucija target varijable — Status_Zavrsetka")
for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{count}\n({count / len(df) * 100:.1f}%)",
        ha="center", fontsize=11,
    )
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"), dpi=150)
plt.close()

# 5b. Korelaciona matrica numerickih feature-a
print("  5b. Korelaciona matrica")
num_cols = [
    "year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
    "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate",
    "Season_Phase", "Status_Zavrsetka",
]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Korelaciona matrica numerickih feature-a")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "correlation_matrix.png"), dpi=150)
plt.close()

print("    Korelacije numerickih feature-a sa targetom (Status_Zavrsetka):")
for col in [c for c in num_cols if c != "Status_Zavrsetka"]:
    corr_val = df[col].corr(df["Status_Zavrsetka"])
    print(f"      {col:25s}: {corr_val:+.4f}")

# 5c. Boxplotovi izvedenih feature-a po klasama
print("  5c. Boxplotovi izvedenih feature-a po klasama")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, derived):
    df.boxplot(column=col, by="Status_Zavrsetka", ax=ax)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "boxplots_by_class.png"), dpi=150)
plt.close()

# ============================================================
# 6. DEFINISANJE FEATURE KOLONA
# ============================================================
# Kategoricki: TargetEncoder (prosecna vrednost targeta po kategoriji)
categorical_cols = [
    "shortname", "circuit_name", "rider_name", "team_name", "bike_name",
]
# Numericki: StandardScaler (mean=0, std=1)
numerical_cols = [
    "year", "sequence", "Historical_DNF_Rate", "Rider_Experience",
    "Team_DNF_Rate", "Bike_DNF_Rate", "Track_DNF_Rate", "Season_Phase",
    "Quali_Spread", "Quali_Top6_Gap", "Quali_Time_Std",
]

y = df["Status_Zavrsetka"]
feature_cols = categorical_cols + numerical_cols
X = df[feature_cols]

print(f"\n--- 6. Feature kolone ---")
print(f"  Kategoricki {len(categorical_cols)}: {categorical_cols}")
print(f"  Numericki   {len(numerical_cols)}: {numerical_cols}")
print(f"  Ukupno:     {len(feature_cols)}")

# ============================================================
# 7. TRAIN/VAL/TEST SPLIT (PRE ENKODIRANJA!)
# ============================================================
# Vazno: Split PRE TargetEncoder-a da bi se izbegao data leakage.
# TargetEncoder.fit() se poziva samo na train skupu.
print("\n--- 7. Train/Val/Test split (70/15/15) ---")

# Prvo: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
# Zatim: 30% temp podelimo na pola => 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"  Train: {len(X_train)} ({len(X_train) / len(df) * 100:.1f}%) — "
      f"DNF: {y_train.sum()}")
print(f"  Val:   {len(X_val)} ({len(X_val) / len(df) * 100:.1f}%) — "
      f"DNF: {y_val.sum()}")
print(f"  Test:  {len(X_test)} ({len(X_test) / len(df) * 100:.1f}%) — "
      f"DNF: {y_test.sum()}")

# ============================================================
# 8. TARGET ENCODING + STANDARD SCALING
# ============================================================
# ColumnTransformer primenjuje razlicite transformacije na razlicite kolone:
#   - Kategoricke kolone -> TargetEncoder (prosecna vrednost targeta)
#   - Numericke kolone   -> StandardScaler (mean=0, std=1)
print("\n--- 8. TargetEncoder + StandardScaler ---")

preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(target_type="binary", random_state=42),
         categorical_cols),
        ("scaler", StandardScaler(), numerical_cols),
    ],
    remainder="passthrough",
)

# Fit SAMO na train skupu, pa transform na sve
X_train_enc = preprocessor.fit_transform(X_train, y_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)

# Konverzija nazad u DataFrame (kolone postaju numericke nakon enkodiranja)
all_cols = categorical_cols + numerical_cols
X_train_enc = pd.DataFrame(X_train_enc, columns=all_cols)
X_val_enc = pd.DataFrame(X_val_enc, columns=all_cols)
X_test_enc = pd.DataFrame(X_test_enc, columns=all_cols)

print(f"  Enkodiranje i skaliranje zavrseno")
print(f"  Train shape: {X_train_enc.shape}")
print(f"  Val shape:   {X_val_enc.shape}")
print(f"  Test shape:  {X_test_enc.shape}")

# ============================================================
# 9. SMOTE OVERSAMPLING (60% Zavrsio / 40% DNF)
# ============================================================
# SMOTE (Synthetic Minority Oversampling TEchnique):
#   - Kreira sinteticke uzorke DNF klase interpolacijom izmedju postojecih
#   - sampling_strategy=0.667 znaci: DNF / Zavrsio = 0.667 => 40% DNF
#   - Primenjuje se SAMO na trening skup (nikad na val/test!)
print("\n--- 9. SMOTE oversampling 60/40 ---")
print(f"  Pre SMOTE — Train skup:")
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()
print(f"    Klasa 0 (Zavrsio): {n0}")
print(f"    Klasa 1 (DNF):     {n1}")
print(f"    Ratio DNF/Ukupno:  {n1 / (n0 + n1) * 100:.1f}%")

smote = SMOTE(sampling_strategy=0.667, random_state=42)
X_res, y_res = smote.fit_resample(X_train_enc, y_train)

X_train_enc = pd.DataFrame(X_res, columns=all_cols)
y_train = pd.Series(y_res, name="Status_Zavrsetka")

n0_new = (y_train == 0).sum()
n1_new = (y_train == 1).sum()
print(f"  Posle SMOTE — Train skup:")
print(f"    Klasa 0 (Zavrsio): {n0_new}")
print(f"    Klasa 1 (DNF):     {n1_new}")
print(f"    Ratio DNF/Ukupno:  {n1_new / (n0_new + n1_new) * 100:.1f}%")
print(f"    Ukupno redova:     {len(X_train_enc)}")

# ============================================================
# 10. CUVANJE SVIH REZULTATA
# ============================================================
print("\n--- 10. Cuvanje procesiranih podataka ---")

# Preprocessor — potreban za deployment (transformacija novih uzoraka)
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
print("  preprocessor.pkl (TargetEncoder + StandardScaler)")

# Feature info — metapodaci o feature-ima
feature_info = {
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "all_cols": feature_cols,
    "n_classes": 2,
    "class_names": ["Zavrsio (0)", "DNF (1)"],
}
joblib.dump(feature_info, os.path.join(MODELS_DIR, "feature_info.pkl"))
print("  feature_info.pkl")

# Procesirani skupovi podataka (spremni za treniranje modela)
X_train_enc.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
X_val_enc.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)
X_test_enc.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
print("  X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv")

# ============================================================
# 11. SUMARNI IZVESTAJ
# ============================================================
print("\n" + "=" * 60)
print("SUMARNI IZVESTAJ PREPROCESIRANJA")
print("=" * 60)
print(f"""
  Izvedeni feature-i:   Historical_DNF_Rate, Rider_Experience, Team_DNF_Rate,
                         Bike_DNF_Rate, Track_DNF_Rate, Season_Phase
  Qualifying feature-i:  Quali_Spread, Quali_Top6_Gap, Quali_Time_Std
  Enkodiranje:           TargetEncoder (kategoricki)
  Skaliranje:            StandardScaler (numericki)
  Split:                 70/15/15 PRE enkodiranja (bez leakage-a)
  Balansiranje:          SMOTE 60/40 + class_weight/scale_pos_weight

  UKUPNO FEATURE-A:      {len(feature_cols)}
    Kategoricki:         {len(categorical_cols)} (TargetEncoded)
    Numericki:           {len(numerical_cols)} (StandardScaled)

  TARGET:                Status_Zavrsetka (0 = Zavrsio, 1 = DNF)
  Disbalans:             {y.sum() / len(y) * 100:.1f}% DNF u originalnim podacima
""")
print("=" * 60)
print("KORAK 3 ZAVRSEN — Podaci spremni za treniranje modela")
print("=" * 60)
