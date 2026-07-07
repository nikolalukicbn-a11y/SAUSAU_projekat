"""
KORAK 2: CISCENJE PODATAKA
===========================
Obrada sirovog dataseta — filtrira, kreira target varijablu,
uklanja data leakage i redundantne kolone.

Operacije:
  1. Filtrira samo MotoGP i 500cc kategorije (najvisa klasa)
  2. Kreira target varijablu Status_Zavrsetka iz kolone position
     Pravilo: position > 0 => Zavrsio (0), position <= 0 => DNF (1)
  3. Proverava i izvestava o missing vrednostima
  4. Uklanja duplikate
  5. Uklanja DATA LEAKAGE kolone: position, points, speed, time
     Ovo su podaci poznati tek NAKON trke — model sme da koristi
     SAMO informacije dostupne PRE starta trke
  6. Uklanja redundantne kolone: rider (ID), number, country, category

VAZNO — Data Leakage:
  Kolone "points", "speed" i "time" sadrze informacije koje su poznate
  tek nakon zavrsetka trke (koliko je voza bio brz, koliko je poena
  osvojio, koje je vreme ostvario). Model koji bi koristio ove kolone
  za predikciju DNF-a bi varao — znao bi rezultat trke unapred.
  Zato se ove kolone MORAJU ukloniti pre treniranja.

Ulaz:
  - data/processed/raw_main.csv  (ucitan u koraku 1)

Izlaz:
  - data/processed/cleaned_data.csv  (14.944 reda x 8 kolona)
"""

import os

import pandas as pd

# ============================================================
# 1. DEFINISANJE PUTANJA
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_MAIN = os.path.join(BASE_DIR, "data", "processed", "raw_main.csv")
CLEANED_OUTPUT = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ============================================================
# 2. UCITAVANJE SIROVIH PODATAKA
# ============================================================
print("=" * 60)
print("KORAK 2: CISCENJE PODATAKA")
print("=" * 60)

df = pd.read_csv(RAW_MAIN)
print(f"\nUcitano: {df.shape[0]} redova x {df.shape[1]} kolona")

# ============================================================
# 3. FILTRIRANJE KATEGORIJA (samo najvisa klasa)
# ============================================================
# Zadrzavamo samo MotoGP (od 2002. godine) i 500cc (prethodnik, 1949-2001)
# Ovo su ekvivalentne klase — najvisi nivo takmicenja
# Moto2, Moto3, MotoE su nize klase sa drugacijim karakteristikama
print("\n--- 3. Filtriranje kategorija ---")
print(f"Pre filtriranja: {len(df)} redova")

df = df[df["category"].isin(["MotoGP", "500cc"])].copy()
print(f"Posle filtriranja (samo MotoGP + 500cc): {len(df)} redova")
print(f"Raspodela:\n{df['category'].value_counts()}")

# ============================================================
# 4. KREIRANJE TARGET VARIJABLE: Status_Zavrsetka
# ============================================================
# Pravilo:
#   position > 0  => vozac je zavrsio trku (nebitno na kojoj poziciji)
#   position <= 0 => vozac NIJE zavrsio (DNF = Did Not Finish)
#     0 = nije startovao, -1 = odustao u prvom krugu, itd.
print("\n--- 4. Kreiranje target varijable: Status_Zavrsetka ---")
print("Pravilo: position > 0 => Zavrsio (0), position <= 0 => DNF (1)")

df["Status_Zavrsetka"] = (df["position"] <= 0).astype(int)

# Prikaz distribucije klasa — ocekuje se jak disbalans (DNF je redak)
print(f"\nRaspodela targeta:")
n_finished = (df["Status_Zavrsetka"] == 0).sum()
n_dnf = (df["Status_Zavrsetka"] == 1).sum()
print(f"  Klasa 0 (Zavrsio):  {n_finished} ({n_finished / len(df) * 100:.1f}%)")
print(f"  Klasa 1 (DNF):      {n_dnf} ({n_dnf / len(df) * 100:.1f}%)")
print(f"  Ovo je NEURAVNOTEZEN dataset — bice potreban SMOTE u preprocesiranju")

# ============================================================
# 5. PROVERA NEDOSTAJUCIH VREDNOSTI
# ============================================================
print("\n--- 5. Provera missing values ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_report = pd.DataFrame({
    "Nedostaje": missing,
    "Procenat": missing_pct.round(2),
})
missing_report = missing_report[missing_report["Nedostaje"] > 0].sort_values(
    "Nedostaje", ascending=False
)
print(missing_report if len(missing_report) > 0 else "NEMA missing vrednosti")

# Kolone sa mnogo missing vrednosti:
#   number: 56% missing — broj motocikla, nebitno za predikciju
#   speed:  41% missing — brzina tokom trke (leak, bice uklonjena)
#   time:   0.8% missing — vreme trke (leak, bice uklonjen)

# ============================================================
# 6. UKLANJANJE DUPLIKATA
# ============================================================
print("\n--- 6. Uklanjanje duplikata ---")
dup_count = df.duplicated().sum()
print(f"Pronadjeno duplikata: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"Uklonjeni. Novo stanje: {len(df)} redova")
else:
    print("Nema duplikata — dataset je cist")

# ============================================================
# 7. UKLANJANJE DATA LEAKAGE KOLONA
# ============================================================
# Data leakage = informacije koje su poznate tek NAKON trke.
# Model sme da koristi SAMO podatke poznate PRE starta trke:
#   - staza (circuit_name, shortname)
#   - vozac (rider_name)
#   - tim (team_name)
#   - konstruktor (bike_name)
#   - godina (year)
#   - redni broj trke u sezoni (sequence)
print("\n--- 7. Uklanjanje DATA LEAKAGE kolona ---")
leak_cols = ["position", "points", "speed", "time"]
print(f"Leak kolone (informacije poznate tek NAKON trke): {leak_cols}")

existing_leak = [c for c in leak_cols if c in df.columns]
df = df.drop(columns=existing_leak)
print(f"Uklonjene: {existing_leak}")

# ============================================================
# 8. UKLANJANJE REDUNDANTNIH KOLONA
# ============================================================
# Redundantne kolone ne dodaju nove informacije:
#   rider:   numericki ID vozaca — redundantno sa rider_name (ime i prezime)
#   number:  broj motocikla — 56% missing, ne utice na ishod trke
#   country: zemlja vozaca — redundantno (vozac vec identifikuje sebe kroz rider_name)
#   category: sve su MotoGP/500cc nakon filtriranja — konstantna vrednost
print("\n--- 8. Uklanjanje redundantnih kolona ---")
redundant_cols = ["rider", "number", "country", "category"]
print(f"Redundantne kolone: {redundant_cols}")

existing_red = [c for c in redundant_cols if c in df.columns]
df = df.drop(columns=existing_red)
print(f"Uklonjene: {existing_red}")

# ============================================================
# 9. FINALNI PREGLED OCISCENOG DATASETA
# ============================================================
print("\n--- 9. Finalni pregled ---")
print(f"Dimenzije: {df.shape[0]} redova x {df.shape[1]} kolona")
print(f"Kolone: {list(df.columns)}")
print(f"\nTipovi podataka:")
print(df.dtypes)
print(f"\nPrvih 5 redova:")
print(df.head())
print(f"\nPreostale missing vrednosti:")
remaining_missing = df.isnull().sum()
print(remaining_missing[remaining_missing > 0] if remaining_missing.sum() > 0 else "NEMA")

# ============================================================
# 10. CUVANJE OCISCENOG DATASETA
# ============================================================
print("\n--- 10. Cuvanje ---")
df.to_csv(CLEANED_OUTPUT, index=False)
print(f"Ociscen dataset: {CLEANED_OUTPUT}")
print(f"  {len(df)} redova x {len(df.columns)} kolona")

# ============================================================
# 11. SUMARNI IZVESTAJ
# ============================================================
print("\n" + "=" * 60)
print("SUMARNI IZVESTAJ CISCENJA")
print("=" * 60)
print(f"""
  POCETNO STANJE:   56.396 redova x 15 kolona
  FILTRIRANJE:      Samo MotoGP + 500cc (najvisa klasa)
  KREIRAN TARGET:   Status_Zavrsetka (0 = Zavrsio, 1 = DNF)
  UKLONJENI LEAK:   position, points, speed, time
                     (informacije poznate tek NAKON trke)
  UKLONJENI REDUN:  rider, number, country, category
                     (redundantne i nepotrebne kolone)

  KONACNO STANJE:   {len(df)} redova x {len(df.columns)} kolona
  KOLONE:           {list(df.columns)}
  TARGET RASPODELA: {n_finished} Zavrsili / {n_dnf} DNF
                     ({n_dnf / len(df) * 100:.1f}% DNF — neuravnotezen dataset)
""")
print("=" * 60)
print("KORAK 2 ZAVRSEN — Podaci ocisceni i snimljeni")
print("=" * 60)
