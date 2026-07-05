"""
KORAK 1: UCITAVANJE I PREGLED PODATAKA
- Ucitava glavni dataset (FILTERED_ROWS.csv) i dopunske (Qualifying.csv, Race.csv)
- Prikazuje osnovne informacije: dimenzije, tipove podataka, missing values
- Snima spojeni sirovi dataset u data/processed/

Podaci:
  Glavni (1949-2022):  56,396 redova, 15 kolona (MotoGP istorijski rezultati)
  Dopunski (2022):     Qualifying.csv + Race.csv (noviji format, razlicite kolone)
"""

import pandas as pd
import os

# ---------- 1. PUTANJE ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------- 2. UCITAVANJE GLAVNOG DATASETA ----------
print("=" * 60)
print("UCITAVANJE GLAVNOG DATASETA: FILTERED_ROWS.csv")
print("=" * 60)

main_path = os.path.join(RAW_DIR, "FILTERED_ROWS.csv")
df_main = pd.read_csv(main_path)

print(f"\nDimenzije: {df_main.shape[0]} redova x {df_main.shape[1]} kolona")
print(f"\nKolone: {list(df_main.columns)}")

print("\n--- Tipovi podataka (dtypes) ---")
print(df_main.dtypes)

print("\n--- Prvih 5 redova ---")
print(df_main.head())

print("\n--- Poslednjih 5 redova ---")
print(df_main.tail())

print("\n--- Statisticki pregled (describe) ---")
print(df_main.describe(include="all"))

print("\n--- Missing values ---")
missing = df_main.isnull().sum()
missing_pct = (df_main.isnull().sum() / len(df_main)) * 100
missing_df = pd.DataFrame({"Nedostaje": missing, "Procenat (%)": missing_pct.round(2)})
print(missing_df[missing_df["Nedostaje"] > 0] if missing_df["Nedostaje"].sum() > 0 else "NEMA missing vrednosti")

# ---------- 3. UCITAVANJE DOPUNSKIH DATASETA (2022) ----------
print("\n" + "=" * 60)
print("UCITAVANJE DOPUNSKIH DATASETA (2022)")
print("=" * 60)

qual_path = os.path.join(RAW_DIR, "Qualifying.csv")
if os.path.exists(qual_path):
    df_qual = pd.read_csv(qual_path)
    print(f"\nQualifying.csv: {df_qual.shape[0]} redova x {df_qual.shape[1]} kolona")
    print(f"Kolone: {list(df_qual.columns)}")
    print(f"\nPrvih 3 reda:\n{df_qual.head(3)}")
    print(f"\nMissing values:\n{df_qual.isnull().sum()[df_qual.isnull().sum() > 0]}")
else:
    print("\nQualifying.csv nije pronadjen")

race_path = os.path.join(RAW_DIR, "Race.csv")
if os.path.exists(race_path):
    df_race = pd.read_csv(race_path)
    print(f"\nRace.csv: {df_race.shape[0]} redova x {df_race.shape[1]} kolona")
    print(f"Kolone: {list(df_race.columns)}")
    print(f"\nPrvih 3 reda:\n{df_race.head(3)}")
    print(f"\nMissing values:\n{df_race.isnull().sum()[df_race.isnull().sum() > 0]}")
else:
    print("\nRace.csv nije pronadjen")

# ---------- 4. INICIJALNA ANALIZA GLAVNOG DATASETA ----------
print("\n" + "=" * 60)
print("INICIJALNA ANALIZA GLAVNOG DATASETA")
print("=" * 60)

print(f"\ncategory jedinstvene vrednosti: {df_main['category'].unique()}")
print(f"\ncategory distribucija:")
print(df_main["category"].value_counts())

print(f"\nposition jedinstvene vrednosti ({df_main['position'].nunique()}):")
print(df_main["position"].value_counts().sort_index().head(20))

print(f"\nposition <= 0 (potencijalni DNF): {(df_main['position'] <= 0).sum()}")
print(f"position > 0 (zavrsili): {(df_main['position'] > 0).sum()}")

# ---------- 5. CUVANJE SIROVIH PODATAKA ----------
print("\n" + "=" * 60)
print("CUVANJE SIROVIH PODATAKA")
print("=" * 60)

main_output = os.path.join(PROCESSED_DIR, "raw_main.csv")
df_main.to_csv(main_output, index=False)
print(f"Glavni dataset sacuvan: {main_output} ({df_main.shape[0]} redova)")

if os.path.exists(qual_path):
    df_qual.to_csv(os.path.join(PROCESSED_DIR, "raw_qualifying.csv"), index=False)
    print(f"Qualifying sacuvan ({df_qual.shape[0]} redova)")

if os.path.exists(race_path):
    df_race.to_csv(os.path.join(PROCESSED_DIR, "raw_race.csv"), index=False)
    print(f"Race sacuvan ({df_race.shape[0]} redova)")

print("\n" + "=" * 60)
print("KRAJ KORAKA 1 - Podaci uspesno ucitani i pregledani")
print("=" * 60)
