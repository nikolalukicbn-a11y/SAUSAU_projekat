"""
KORAK 2: CISCENJE PODATAKA
- Filtriranje: samo MotoGP i 500cc kategorije (najvisa klasa)
- Kreiranje target varijable Status_Zavrsetka iz kolone position
- Provera i obrada missing values
- Uklanjanje duplikata
- Uklanjanje kolona koje izazivaju data leakage (pozicija, poeni, brzina, vreme)
- Uklanjanje redundantnih i nepotrebnih kolona

DATA LEAKAGE OBJASNJENJE:
  Kolone 'points', 'speed' i 'time' su podaci koji su poznati tek NAKON
  zavrsetka trke. Model sme da koristi SAMO informacije poznate PRE trke
  (staza, vozac, tim, godina, redni broj trke). Koriscenje post-race
  podataka bi dalo lazno dobre rezultate.
"""

import pandas as pd
import os

# ---------- 1. PUTANJE ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_MAIN = os.path.join(BASE_DIR, "data", "processed", "raw_main.csv")
CLEANED_OUTPUT = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ---------- 2. UCITAVANJE ----------
print("=" * 60)
print("KORAK 2: CISCENJE PODATAKA")
print("=" * 60)

df = pd.read_csv(RAW_MAIN)
print(f"\nUcitano: {df.shape[0]} redova x {df.shape[1]} kolona")

# ---------- 3. FILTRIRANJE KATEGORIJA ----------
print("\n--- 3. Filtriranje kategorija ---")
print(f"Pre filtriranja: {len(df)} redova")

# Zadrzavamo samo MotoGP (od 2002) i 500cc (prethodnik, 1949-2001)
df = df[df["category"].isin(["MotoGP", "500cc"])].copy()
print(f"Posle filtriranja (MotoGP + 500cc): {len(df)} redova")
print(f"Raspodela po kategorijama:\n{df['category'].value_counts()}")

# ---------- 4. KREIRANJE TARGET VARIJABLE ----------
print("\n--- 4. Kreiranje target varijable: Status_Zavrsetka ---")
print("Pravilo: position > 0 => 0 (Zavrsio trku), position <= 0 => 1 (DNF)")

df["Status_Zavrsetka"] = (df["position"] <= 0).astype(int)

print(f"\nRaspodela targeta:")
print(f"  Klasa 0 (Zavrsio):  {(df['Status_Zavrsetka'] == 0).sum()} ({(df['Status_Zavrsetka'] == 0).mean()*100:.1f}%)")
print(f"  Klasa 1 (DNF):      {(df['Status_Zavrsetka'] == 1).sum()} ({(df['Status_Zavrsetka'] == 1).mean()*100:.1f}%)")
print(f"  Ovo je NEURAVNOTEZEN dataset - bicemo potreban SMOTE u preprocesiranju")

# ---------- 5. PROVERA MISSING VALUES ----------
print("\n--- 5. Provera missing values ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_report = pd.DataFrame({
    "Nedostaje": missing,
    "Procenat": missing_pct.round(2)
})
missing_report = missing_report[missing_report["Nedostaje"] > 0].sort_values("Nedostaje", ascending=False)
print(missing_report)

# Kolone sa mnogo missing vrednosti koje cemo svakako ukloniti:
# number (56%), speed (41%) - bice uklonjene u koraku 7
# time - 0.79% missing, ali je post-race podatak, bice uklonjen

# ---------- 6. UKLANJANJE DUPLIKATA ----------
print("\n--- 6. Uklanjanje duplikata ---")
dup_count = df.duplicated().sum()
print(f"Pronadjeno duplikata: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"Uklonjeni duplikati. Novo stanje: {len(df)} redova")

# ---------- 7. UKLANJANJE LEAK KOLONA ----------
print("\n--- 7. Uklanjanje data leakage kolona ---")
leak_cols = ["position", "points", "speed", "time"]
print(f"Leak kolone (poznate tek nakon trke): {leak_cols}")

# Proveravamo da li postoje pre uklanjanja
existing_leak = [c for c in leak_cols if c in df.columns]
df = df.drop(columns=existing_leak)
print(f"Uklonjene: {existing_leak}")

# ---------- 8. UKLANJANJE REDUNDANTNIH I NEPOTREBNIH KOLONA ----------
print("\n--- 8. Uklanjanje redundantnih kolona ---")

# rider: numericki ID vozaca - redudantno sa rider_name
# number: broj motocikla - 56% missing, nebitno za predikciju
# country: zemlja vozaca - redundantno (vozac vec identifikuje sebe)
# category: sve su MotoGP/500cc nakon filtriranja
redundant_cols = ["rider", "number", "country", "category"]
existing_red = [c for c in redundant_cols if c in df.columns]
df = df.drop(columns=existing_red)
print(f"Uklonjene redundantne kolone: {existing_red}")

# ---------- 9. FINALNI PREGLED ----------
print("\n--- 9. Finalni pregled ociscenog dataseta ---")
print(f"Dimenzije: {df.shape[0]} redova x {df.shape[1]} kolona")
print(f"Kolone: {list(df.columns)}")
print(f"\nTipovi podataka:")
print(df.dtypes)
print(f"\nPrvih 5 redova:")
print(df.head())
print(f"\nMissing values (sve kolone):")
print(df.isnull().sum())

print(f"\nOpis dataseta:")
print(df.describe(include="all"))

# ---------- 10. CUVANJE ----------
print("\n--- 10. Cuvanje ociscenog dataseta ---")
df.to_csv(CLEANED_OUTPUT, index=False)
print(f"Ociscen dataset sacuvan: {CLEANED_OUTPUT}")
print(f"Broj redova: {len(df)}, Broj kolona: {len(df.columns)}")

# ---------- 11. SUMARNI IZVESTAJ ----------
print("\n" + "=" * 60)
print("SUMMARNI IZVESTAJ CISCENJA")
print("=" * 60)
print(f"""
POCETNO STANJE:  56,396 redova x 15 kolona
FILTRIRANJE:     Samo MotoGP + 500cc (najvisa klasa)
KREIRAN TARGET:  Status_Zavrsetka (0 = Zavrsio, 1 = DNF)
UKLONJENI LEAK:  position, points, speed, time
UKLONJENI REDUN: rider, number, country, category

KONACNO STANJE:  {len(df)} redova x {len(df.columns)} kolona
KOLONE:          {list(df.columns)}
TARGET RASPODELA: {(df['Status_Zavrsetka'] == 0).sum()} Zavrsili / {(df['Status_Zavrsetka'] == 1).sum()} DNF
""")
print("=" * 60)
print("KRAJ KORAKA 2 - Podaci ocisceni")
print("=" * 60)
