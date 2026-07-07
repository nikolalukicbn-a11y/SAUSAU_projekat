"""
KORAK 1: UCITAVANJE I PREGLED PODATAKA
=======================================
Ucitava glavni dataset FILTERED_ROWS.csv (1949-2022, 56.396 redova)
i dopunske datasetove Qualifying.csv i Race.csv (2022, noviji format).

Prikazuje:
  - Dimenzije datasetova
  - Tipove podataka (dtypes)
  - Prvih i poslednjih 5 redova
  - Statisticki pregled (describe)
  - Missing values po kolonama
  - Distribuciju kategorija (MotoGP, 500cc, Moto2, itd.)
  - Distribuciju position vrednosti (koliko DNF-ova)

Snima sirove ucitane podatke u data/processed/:
  - raw_main.csv       (glavni dataset)
  - raw_qualifying.csv  (kvalifikacije 2022-2025)
  - raw_race.csv        (rezultati trka 2022)

Ulaz:
  - data/raw/FILTERED_ROWS.csv  (glavni dataset, 56.396 redova x 15 kolona)
  - data/raw/Qualifying.csv     (kvalifikacije, 8.019 redova x 12 kolona)
  - data/raw/Race.csv           (rezultati trka, varijabilna velicina)

Izlaz:
  - data/processed/raw_main.csv
  - data/processed/raw_qualifying.csv
  - data/processed/raw_race.csv
"""

import os

import pandas as pd

# ============================================================
# 1. DEFINISANJE PUTANJA
# ============================================================
# Skripta je u src/, pa idemo jedan nivo gore do root-a projekta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)  # kreira data/processed/ ako ne postoji

# ============================================================
# 2. UCITAVANJE GLAVNOG DATASETA (FILTERED_ROWS.csv)
# ============================================================
print("=" * 60)
print("UCITAVANJE GLAVNOG DATASETA: FILTERED_ROWS.csv")
print("=" * 60)

main_path = os.path.join(RAW_DIR, "FILTERED_ROWS.csv")
df_main = pd.read_csv(main_path)

# Osnovne informacije o datasetu
print(f"\nDimenzije: {df_main.shape[0]} redova x {df_main.shape[1]} kolona")
print(f"\nKolone: {list(df_main.columns)}")

# Tipovi podataka — bitno za dalju obradu
print("\n--- Tipovi podataka (dtypes) ---")
print(df_main.dtypes)

# Pregled pocetka i kraja dataseta
print("\n--- Prvih 5 redova ---")
print(df_main.head())
print("\n--- Poslednjih 5 redova ---")
print(df_main.tail())

# Statisticki pregled svih kolona (numerickih i kategorickih)
print("\n--- Statisticki pregled (describe) ---")
print(df_main.describe(include="all"))

# Provera nedostajucih vrednosti — koje kolone imaju rupe i koliko
print("\n--- Missing values ---")
missing = df_main.isnull().sum()
missing_pct = (df_main.isnull().sum() / len(df_main)) * 100
missing_df = pd.DataFrame({
    "Nedostaje": missing,
    "Procenat (%)": missing_pct.round(2)
})
if missing_df["Nedostaje"].sum() > 0:
    print(missing_df[missing_df["Nedostaje"] > 0])
else:
    print("NEMA missing vrednosti u glavnom datasetu")

# ============================================================
# 3. UCITAVANJE DOPUNSKIH DATASETA (2022-2025)
# ============================================================
# Ovi datasetovi imaju drugaciji format kolona u odnosu na glavni.
# Koriste se za izvlacenje dodatnih feature-a u koraku 3 (preprocessing).
print("\n" + "=" * 60)
print("UCITAVANJE DOPUNSKIH DATASETA")
print("=" * 60)

# 3a. Qualifying.csv — kvalifikacioni rezultati (2022-2025, sve klase)
qual_path = os.path.join(RAW_DIR, "Qualifying.csv")
if os.path.exists(qual_path):
    df_qual = pd.read_csv(qual_path)
    print(f"\nQualifying.csv: {df_qual.shape[0]} redova x {df_qual.shape[1]} kolona")
    print(f"Kolone: {list(df_qual.columns)}")
    print(f"\nPrvih 3 reda:\n{df_qual.head(3)}")
    missing_q = df_qual.isnull().sum()
    missing_q = missing_q[missing_q > 0]
    print(f"\nMissing values:\n{missing_q if len(missing_q) > 0 else 'NEMA'}")
else:
    print("\nQualifying.csv nije pronadjen — preskace se")
    df_qual = None

# 3b. Race.csv — rezultati trka (2022, sve klase)
race_path = os.path.join(RAW_DIR, "Race.csv")
if os.path.exists(race_path):
    df_race = pd.read_csv(race_path)
    print(f"\nRace.csv: {df_race.shape[0]} redova x {df_race.shape[1]} kolona")
    print(f"Kolone: {list(df_race.columns)}")
    print(f"\nPrvih 3 reda:\n{df_race.head(3)}")
    missing_r = df_race.isnull().sum()
    missing_r = missing_r[missing_r > 0]
    print(f"\nMissing values:\n{missing_r if len(missing_r) > 0 else 'NEMA'}")
else:
    print("\nRace.csv nije pronadjen — preskace se")
    df_race = None

# ============================================================
# 4. INICIJALNA ANALIZA GLAVNOG DATASETA
# ============================================================
# Pre same obrade, pregledamo kljucne kolone za razumevanje podataka
print("\n" + "=" * 60)
print("INICIJALNA ANALIZA GLAVNOG DATASETA")
print("=" * 60)

# Kategorije trka (MotoGP, 500cc, Moto2, Moto3, ...)
print(f"\ncategory jedinstvene vrednosti: {df_main['category'].unique()}")
print(f"\ncategory distribucija:")
print(df_main["category"].value_counts())

# Position — kljucna kolona za target varijablu
# position > 0 = zavrsio trku, position <= 0 = DNF (odustajanje)
print(f"\nposition jedinstvene vrednosti ({df_main['position'].nunique()}):")
print(df_main["position"].value_counts().sort_index().head(20))

print(f"\nposition <= 0 (DNF — odustajanje): {(df_main['position'] <= 0).sum()}")
print(f"position > 0 (Zavrsio trku): {(df_main['position'] > 0).sum()}")

# ============================================================
# 5. CUVANJE SIROVIH UCITANIH PODATAKA
# ============================================================
# Snimamo u data/processed/ za dalju obradu u narednim koracima
print("\n" + "=" * 60)
print("CUVANJE SIROVIH PODATAKA")
print("=" * 60)

main_output = os.path.join(PROCESSED_DIR, "raw_main.csv")
df_main.to_csv(main_output, index=False)
print(f"Glavni dataset: {main_output} ({df_main.shape[0]} redova)")

if df_qual is not None:
    qual_out = os.path.join(PROCESSED_DIR, "raw_qualifying.csv")
    df_qual.to_csv(qual_out, index=False)
    print(f"Qualifying: {qual_out} ({df_qual.shape[0]} redova)")

if df_race is not None:
    race_out = os.path.join(PROCESSED_DIR, "raw_race.csv")
    df_race.to_csv(race_out, index=False)
    print(f"Race: {race_out} ({df_race.shape[0]} redova)")

print("\n" + "=" * 60)
print("KORAK 1 ZAVRSEN — Podaci ucitani i snimljeni u data/processed/")
print("=" * 60)
