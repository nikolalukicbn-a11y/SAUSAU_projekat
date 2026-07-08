"""
Generise PDF dokumentaciju za MotoGP DNF Prediktor projekat.
Koristi fpdf2 biblioteku.
"""
import os
from fpdf import FPDF

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
DOCS = os.path.join(BASE, "docs")
os.makedirs(DOCS, exist_ok=True)


class DocPDF(FPDF):
    """Prilagodjen FPDF za dokumentaciju projekta."""

    def __init__(self):
        super().__init__()
        font_loaded = False
        for font_path, bold_path in [
            ("C:/Windows/Fonts/DejaVuSans.ttf", "C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
            ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
            ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/seguisb.ttf"),
        ]:
            if os.path.exists(font_path):
                self.add_font("Uni", "", font_path)
                bpath = bold_path if os.path.exists(bold_path) else font_path
                self.add_font("Uni", "B", bpath)
                font_loaded = True
                break
        if not font_loaded:
            raise RuntimeError("No Unicode font found (tried DejaVuSans, Calibri, Segoe UI)")
        self._font_name = "Uni"

    def header(self):
        self.set_font("Uni", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "MotoGP DNF Prediktor — SAUSAU 2026", align="L")
        self.cell(0, 6, f"Strana {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Uni", "", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, "Nikola Lukic RA47-2023", align="C")

    def chapter_title(self, title):
        self.set_font("Uni", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section_title(self, title):
        self.set_font("Uni", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Uni", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Uni", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, "  - " + text)
        self.ln(1)

    def metrics_table(self, rows, headers, col_widths=None):
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)
        self.set_font("Uni", "B", 9)
        self.set_fill_color(211, 47, 47)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Uni", "", 9)
        self.set_text_color(40, 40, 40)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, align="C")
            self.ln()
        self.ln(3)

    def add_image_centered(self, path, w=120):
        if os.path.exists(path):
            x = (self.w - w) / 2
            self.image(path, x=x, w=w)
            self.ln(3)
        else:
            self.body_text(f"[Slika nije pronadjena: {path}]")


pdf = DocPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ============================================================
# NASLOVNA STRANA
# ============================================================
pdf.ln(20)
pdf.set_font("Uni", "B", 26)
pdf.set_text_color(211, 47, 47)
pdf.cell(0, 12, "MotoGP DNF Prediktor", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Uni", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "Predikcija ishoda trke primenom masinskog ucenja", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Binarna klasifikacija: Zavrsio (0) / Odustajanje — DNF (1)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)
pdf.set_font("Uni", "", 11)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, "SAUSAU 2026 — Projekat iz Masinskog ucenja", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Student: Nikola Lukic  RA47-2023", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Jul 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# ============================================================
# 1. UVOD
# ============================================================
pdf.add_page()
pdf.chapter_title("1. Uvod")

pdf.body_text(
    "Ovaj projekat se bavi predikcijom ishoda MotoGP trke — da li ce vozac zavrsiti trku "
    "(Zavrsio) ili odustati (DNF — Did Not Finish). Problem je formulisan kao binarna "
    "klasifikacija sa jakim disbalansom klasa: svega ~10.9% trka se zavrsava DNF-om, "
    "dok ~89.1% vozaca zavrsava trku."
)

pdf.body_text(
    "Cilj projekta je da se na osnovu informacija dostupnih PRE starta trke (staza, vozac, "
    "tim, konstruktor, godina, redni broj trke u sezoni) predvidi da li ce vozac odustati. "
    "Ovo je izazovan problem jer su DNF dogadjaji retki i tesko predvidivi bez informacija "
    "sa same trke (brzina, vreme, pozicija), koje se ne smeju koristiti zbog data leakage-a."
)

pdf.section_title("1.1 Skup podataka")
pdf.body_text(
    "Glavni dataset (FILTERED_ROWS.csv) obuhvata 56.396 redova iz perioda 1949-2022. godine, "
    "sa 15 originalnih kolona. Nakon filtriranja (samo MotoGP i 500cc klasa), dataset se "
    "smanjuje na 14.944 reda. Dopunski podaci iz Qualifying.csv (2022-2025) koriste se za "
    "izvlacenje dodatnih karakteristika staza."
)

pdf.section_title("1.2 Specifikacija problema")
pdf.bullet("Tip problema: Binarna klasifikacija (Zavrsio=0, DNF=1)")
pdf.bullet("Metrika uspeha: Recall (DNF) — false negative (propusten DNF) je skup")
pdf.bullet("Izazovi: jak disbalans klasa, data leakage opasnost, ogranicene pre-race informacije")

# ============================================================
# 2. POCETNO PREPROCESIRANJE
# ============================================================
pdf.add_page()
pdf.chapter_title("2. Pocetno preprocesiranje podataka")

pdf.section_title("2.1 Nedostajuce vrednosti i anomalije")
pdf.body_text(
    "Proverom nedostajucih vrednosti utvrdjeno je da kolone number (56% missing) i speed "
    "(41% missing) imaju veliki broj nedostajucih vrednosti. Ove kolone su svakako uklonjene "
    "u kasnijim koracima iz drugih razloga (data leakage i redundantnost). Ostale kolone "
    "nemaju znacajan broj nedostajucih vrednosti."
)

pdf.section_title("2.2 Filtriranje kategorija")
pdf.body_text(
    "Iz originalnog dataseta sa vise takmicarskih kategorija (MotoGP, 500cc, Moto2, Moto3, "
    "MotoE), zadrzane su samo MotoGP (od 2002. godine) i 500cc (prethodna najvisa klasa, "
    "1949-2001). Ove dve kategorije predstavljaju ekvivalentan nivo takmicenja i imaju "
    "slicne karakteristike u pogledu DNF obrazaca. Nakon filtriranja, dataset broji 14.944 reda."
)

pdf.section_title("2.3 Kreiranje target varijable")
pdf.body_text(
    "Target varijabla Status_Zavrsetka se kreira iz originalne kolone position po pravilu: "
    "position > 0 => Zavrsio (0), position <= 0 => DNF (1). Vrednost 0 oznacava da vozac "
    "nije startovao, a negativne vrednosti oznacavaju odustajanje u odredjenom krugu."
)

pdf.section_title("2.4 Uklanjanje data leakage kolona")
pdf.body_text(
    "Data leakage je ozbiljan problem u vremenskim predikcijama. Kolone koje su poznate "
    "tek NAKON zavrsetka trke MORAJU biti uklonjene pre treniranja modela. U suprotnom, "
    "model bi dobio lazno dobre rezultate koristeci informacije koje u stvarnom svetu "
    "ne bi bile dostupne pre trke."
)
pdf.body_text(
    "Uklonjene data leakage kolone:\n"
    "  - position — pozicija na kraju trke (iz nje se izvodi target)\n"
    "  - points — broj osvojenih poena (poznat tek nakon trke)\n"
    "  - speed — prosecna brzina tokom trke (poznata tek nakon trke)\n"
    "  - time — ukupno vreme trke (poznato tek nakon trke)"
)

pdf.section_title("2.5 Uklanjanje redundantnih kolona")
pdf.body_text(
    "Sledece kolone su uklonjene kao redundantne ili nepotrebne:\n"
    "  - rider — numericki ID vozaca (redundantno sa rider_name)\n"
    "  - number — broj motocikla (56% missing, ne utice na ishod)\n"
    "  - country — zemlja vozaca (vozac vec identifikuje sebe)\n"
    "  - category — sve su MotoGP/500cc nakon filtriranja (konstantno)"
)

pdf.section_title("2.6 Enkodiranje podataka")
pdf.body_text(
    "Kategoricke varijable (shortname, circuit_name, rider_name, team_name, bike_name) "
    "enkodirane su TargetEncoder-om. Za razliku od standardnog LabelEncoder-a ili "
    "OneHotEncoder-a, TargetEncoder enkodira svaku kategoriju prosecnom vrednoscu target "
    "varijable za tu kategoriju, sto daje znatno vise informacija tree-based modelima. "
    "Numericke varijable su skalirane StandardScaler-om (mean=0, std=1)."
)

# ============================================================
# 3. EKSPLORATIVNA ANALIZA
# ============================================================
pdf.add_page()
pdf.chapter_title("3. Eksplorativna analiza skupa (EDA)")

pdf.section_title("3.1 Distribucija target varijable")
pdf.body_text(
    "Dataset je izrazito neuravnotezen: 13.318 trka je zavrseno (89.1%), dok je samo "
    "1.626 trka zavrseno DNF-om (10.9%). Ovaj disbalans zahteva posebne tehnike — "
    "u ovom projektu koriscen je SMOTE (Synthetic Minority Oversampling TEchnique) "
    "za balansiranje klasa na 60/40 odnos pre treniranja modela."
)
pdf.add_image_centered(os.path.join(RESULTS, "experiments", "figures", "target_distribution.png"), w=100)

pdf.section_title("3.2 Korelaciona analiza")
pdf.body_text(
    "Korelacionom analizom numerickih atributa sa target varijablom uocene su sledece "
    "najjace korelacije: Track_DNF_Rate (0.181), Team_DNF_Rate (0.178), year (0.173), "
    "Bike_DNF_Rate (0.152). Ovi atributi pokazuju da su istorijski obrasci (DNF stopa "
    "staze, tima i konstruktora) najbolji prediktori DNF-a. Pojedinacni numericki atributi "
    "nemaju izrazito jake korelacije sa targetom (najvisa je ~0.18), sto ukazuje na "
    "kompleksnost problema — DNF zavisi od kombinacije vise faktora."
)
pdf.add_image_centered(os.path.join(RESULTS, "experiments", "figures", "correlation_matrix.png"), w=130)

pdf.section_title("3.3 Boxplot analiza po klasama")
pdf.body_text(
    "Boxplotovi izvedenih feature-a po klasama (Zavrsio vs DNF) pokazuju da DNF klasa "
    "ima nesto vise prosecne vrednosti za vecinu feature-a (posebno za Team_DNF_Rate i "
    "Track_DNF_Rate), ali distribucije se znatno preklapaju. Ovo potvrdjuje da nema "
    "jednostavnog pravila za razdvajanje klasa — potrebni su kompleksniji modeli."
)

# ============================================================
# 4. ODABIR I TRENIRANJE MODELA
# ============================================================
pdf.add_page()
pdf.chapter_title("4. Odabir i treniranje modela")

pdf.section_title("4.1 Pregled probanih pristupa")
pdf.body_text(
    "Tokom projekta testirano je vise pristupa resavanju problema. U nastavku je dat "
    "pregled svih probanih metoda sa kratkim opisom i razlogom zasto su neki odbaceni:"
)

approaches = [
    ("XGBoost Standalone", "SMOTE 60/40 + threshold 0.80",
     "FN=20, Rec=91.8%", "Solidan baseline, polazna tacka"),
    ("Cost-sensitive", "Sample weight + cost scorer (FN/FP=5:1)",
     "Poboljsan F1, gori recall", "Cost optimizuje ukupni trosak, ne minimalni FN"),
    ("Anomaly detection", "Isolation Forest + One-Class SVM",
     "F1~0.17, Recall~13%", "DNF nije outlier u feature space-u — klase se preklapaju"),
    ("SVM tuning", "GridSearchCV + razni thresholdi",
     "Losa kalibracija verovatnoca", "SVM nije pogodan za ovaj problem"),
    ("Stacking", "XGB+LR+RF+MLP + meta-LR",
     "Recall=25.8%", "Meta-learner zagladjuje predikcije, gubi ekstremne DNF slucajeve"),
    ("Ensemble OR", "XGBoost + LR, OR logika",
     "FN=16, Rec=93.4%", "OR logika hvata DNF koje jedan model propusti — znacajno poboljsanje"),
    ("Platt kalibracija", "CalibratedClassifierCV za oba modela",
     "FN=9, Rec=96.3%", "XGBoost Brier score sa 0.62 na 0.33 — kljucno poboljsanje"),
    ("Feature selection", "SHAP + RFECV, 16->12 feature-a",
     "Recall pao na 89.8%", "I slabi feature-i doprinose — zadrzano svih 16"),
    ("Bagging K-voting", "5x XGBoost + 5x LR, bootstrap, K=2 konsenzus",
     "FN=6, Rec=97.5%", "Najbolji model — diverzitet kroz bootstrap + konsenzus"),
]

for name, what, result, note in approaches:
    pdf.set_font("Uni", "B", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(38, 5, name)
    pdf.set_font("Uni", "", 9)
    pdf.cell(55, 5, what)
    pdf.cell(38, 5, result)
    pdf.cell(0, 5, note, new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)

pdf.section_title("4.2 Tri produkcijska modela")
pdf.body_text(
    "Nakon svih eksperimenata, izdvojena su tri modela koja daju najbolje rezultate. "
    "Svi modeli su trenirani na SMOTE 60/40 podacima sa 16 feature-a (5 kategorickih "
    "TargetEncoded + 11 numerickih StandardScaled). Evaluacija je vrsena na test skupu "
    "od 2.242 reda (1.998 Zavrsio, 244 DNF)."
)

# ============================================================
# 4.3 Model 1
# ============================================================
pdf.add_page()
pdf.section_title("4.3 Model 1: XGBoost Standalone (SMOTE 60/40)")
pdf.body_text(
    "Prvi model je jednostavan XGBoost klasifikator treniran na SMOTE 60/40 podacima. "
    "Korisceni su hiperparametri: learning_rate=0.01, max_depth=6, n_estimators=200, "
    "scale_pos_weight=8. Threshold za DNF predikciju je 0.80."
)
pdf.body_text(
    "Rezultat: Ovaj model predstavlja baseline sa kojim se porede svi napredniji modeli. "
    "Propuska 20 od 244 DNF-a (8.2%), uz 1.536 laznih alarma."
)
pdf.metrics_table(
    [["20", "1536", "224", "91.8%", "12.7%", "0.224", "78.5%"]],
    ["FN", "FP", "TP", "Recall", "Prec", "F1", "DNF%"],
    [20, 20, 20, 25, 25, 20, 20],
)
pdf.add_image_centered(os.path.join(RESULTS, "model1_xgboost", "figures", "confusion_matrix.png"), w=95)

# ============================================================
# 4.4 Model 2
# ============================================================
pdf.add_page()
pdf.section_title("4.4 Model 2: OR Ensemble Kalibrisan")
pdf.body_text(
    "Drugi model koristi dva nezavisna modela — XGBoost i LogisticRegression — sa OR "
    "logikom: DNF se predvidja ako BILO KOJI od dva modela predvidi DNF. Oba modela su "
    "kalibrisana Platt scaling-om (CalibratedClassifierCV, method='sigmoid', cv=5)."
)
pdf.body_text(
    "Razlog za kalibraciju: XGBoost-ove sirove predict_proba vrednosti su lose "
    "kalibrisane (Brier score 0.62, sve verovatnoce > 0.80). Platt scaling popravlja "
    "Brier score na 0.33, dajuci realne verovatnoce koje omogucavaju precizniji "
    "threshold tuning. Pragovi su postavljeni na XGB=0.35, LR=0.30."
)
pdf.body_text(
    "Rezultat: OR logika u kombinaciji sa kalibracijom smanjuje broj propustenih "
    "DNF-ova sa 20 na svega 9 (poboljsanje od 55%). Ovo je postignuto jer LR hvata "
    "DNF-ove koje XGBoost propusta i obrnuto."
)
pdf.metrics_table(
    [["9", "1600", "235", "96.3%", "12.8%", "0.226", "81.8%"]],
    ["FN", "FP", "TP", "Recall", "Prec", "F1", "DNF%"],
    [20, 20, 20, 25, 25, 20, 20],
)
pdf.add_image_centered(os.path.join(RESULTS, "model2_or", "figures", "confusion_matrix.png"), w=95)
pdf.ln(2)
pdf.body_text(
    "Kalibracione krive pokazuju koliko su verovatnoce modela pouzdane. Idealna "
    "kalibracija znaci da predvidjena verovatnoca od X% odgovara stvarnoj ucestalosti "
    "DNF-a od X%. Kriva koja prati dijagonalu (isprekidana linija) oznacava savrsenu "
    "kalibraciju."
)
pdf.add_image_centered(os.path.join(RESULTS, "model2_or", "figures", "calibration_curves.png"), w=160)

# ============================================================
# 4.5 Model 3
# ============================================================
pdf.add_page()
pdf.section_title("4.5 Model 3: Bagging Ensemble sa K-voting-om (FINALNI)")
pdf.body_text(
    "Treci i finalni model koristi bagging (bootstrap aggregating) sa K-voting konsenzusom. "
    "Iz originalnog trening skupa generise se 5 bootstrap uzoraka (nasumicno izvlacenje "
    "SA VRACANJEM, ista velicina kao original). Na svakom bag-u se trenira po jedan "
    "XGBoost i jedan LR (oba Platt-kalibrisana), sto daje ukupno 10 nezavisnih modela."
)
pdf.body_text(
    "K-voting: Svaki od 10 modela glasa (DNF=1, Zavrsio=0) na osnovu thresholda 0.35. "
    "Konacna odluka: DNF ako bar K=2 modela glasa za DNF. Ovo filtrira pojedinacne "
    "random greske — ako jedan model nasumicno pogresi, drugi ga ispravljaju."
)
pdf.body_text(
    "Rezultat: Bagging sa K=2 daje najbolji rezultat od svih modela — samo 6 propustenih "
    "DNF-ova od 244 (2.5%). U odnosu na pocetni XGBoost baseline (FN=20), ovo je "
    "poboljsanje od 70% u broju propustenih DNF-ova."
)
pdf.metrics_table(
    [["6", "1625", "238", "97.5%", "12.8%", "0.226", "82.6%"]],
    ["FN", "FP", "TP", "Recall", "Prec", "F1", "DNF%"],
    [20, 20, 20, 25, 25, 20, 20],
)
pdf.add_image_centered(os.path.join(RESULTS, "model3_bagging", "figures", "confusion_matrix.png"), w=95)
pdf.ln(2)
pdf.body_text(
    "Grafik performansi po K vrednostima: Kako K raste (zahteva se vise glasova za DNF), "
    "Recall opada (vise propustenih DNF-ova), ali Precision raste (manje laznih alarma). "
    "K=2 predstavlja optimalan balans."
)
pdf.add_image_centered(os.path.join(RESULTS, "model3_bagging", "figures", "k_voting_performance.png"), w=135)

# ============================================================
# 5. POREDJENJE MODELA
# ============================================================
pdf.add_page()
pdf.chapter_title("5. Poredjenje modela i evolucija performansi")

pdf.section_title("5.1 Uporedna tabela")
pdf.metrics_table(
    [
        ["1. XGBoost Standalone", "20", "1536", "91.8%", "12.7%", "0.224"],
        ["2. OR Kalibrisan", "9", "1600", "96.3%", "12.8%", "0.226"],
        ["3. Bagging K=2 (finalni)", "6", "1625", "97.5%", "12.8%", "0.226"],
    ],
    ["Model", "FN", "FP", "Recall", "Prec", "F1"],
    [50, 15, 15, 25, 25, 20],
)

pdf.section_title("5.2 Evolucija kroz iteracije")
pdf.body_text(
    "Projekat je prosao kroz vise iteracija poboljsanja. Svaka iteracija je donela "
    "konkretno smanjenje broja propustenih DNF-ova (FN):"
)
pdf.metrics_table(
    [
        ["1", "XGBoost Standalone", "20", "1536", "91.8%", "--"],
        ["2", "+ Ensemble OR voting", "16", "1553", "93.4%", "-4 FN"],
        ["3", "+ Threshold tuning", "10", "1600", "95.9%", "-6 FN"],
        ["4", "+ Platt kalibracija", "9", "1600", "96.3%", "-1 FN"],
        ["5", "+ Bagging K-voting", "6", "1625", "97.5%", "-3 FN"],
    ],
    ["Faza", "Pristup", "FN", "FP", "Recall", "Poboljsanje"],
    [10, 42, 15, 15, 22, 28],
)

pdf.body_text(
    "Od pocetnih 20 propustenih DNF-ova do svega 6 u finalnom modelu — smanjenje "
    "false negative za 70%. Cena ovog poboljsanja je porast false positive sa 1.536 "
    "na 1.625 (povecanje od 5.8%). S obzirom da je u ovom problemu false negative "
    "(propusten DNF) znatno skuplji od false positive (lazni alarm), ovaj trade-off "
    "je prihvatljiv."
)

# ============================================================
# 6. HIPERPARAMETRI
# ============================================================
pdf.chapter_title("6. Podesavanje hiperparametara")

pdf.body_text(
    "Za svaki model, hiperparametri su optimizovani kroz GridSearchCV sa stratified "
    "5-fold cross-validacijom. Skoring metrika je bio F1-score (za balansiranje "
    "preciznosti i odziva), a za cost-sensitive modele koriscen je custom cost scorer."
)

pdf.section_title("6.1 XGBoost hiperparametri")
pdf.body_text(
    "Optimalni hiperparametri za XGBoost (pronadjeni GridSearch-om):\n"
    "  - learning_rate = 0.01 (najmanji testiran, daje stabilnije konvergenciju)\n"
    "  - max_depth = 6 (dovoljno duboko za kompleksne interakcije, bez overfitting-a)\n"
    "  - n_estimators = 200 (dovoljno stabala za konvergenciju gradijenta)\n"
    "  - scale_pos_weight = 8 (daje 8x vecu tezinu DNF klasi pri racunanju gradijenta)"
)

pdf.section_title("6.2 LogisticRegression hiperparametri")
pdf.body_text(
    "  - C = 10 (umerena regularizacija)\n"
    "  - class_weight = 'balanced' (automatski balansira tezine klasa)\n"
    "  - max_iter = 2000 (dovoljno iteracija za konvergenciju)"
)

pdf.section_title("6.3 Bagging hiperparametri")
pdf.body_text(
    "  - Broj bagova = 5 (dovoljan diverzitet, prihvatljivo vreme treniranja)\n"
    "  - Kalibracioni CV unutar bag-a = 3 folda\n"
    "  - Threshold po modelu = 0.35\n"
    "  - Default K = 2 (optimalan konsenzus — rezultati za sve K su u metrics_by_K.csv)"
)

# ============================================================
# 7. ODABIR NAJZNACAJNIJIH ATRIBUTA
# ============================================================
pdf.add_page()
pdf.chapter_title("7. Odabir najznacajnijih atributa")

pdf.section_title("7.1 SHAP analiza")
pdf.body_text(
    "SHAP (SHapley Additive exPlanations) vrednosti su koriscene za odredjivanje "
    "doprinosa svakog feature-a predikciji modela. SHAP analiza je izvrsena na "
    "XGBoost modelu (kao tree-based model, podrzava TreeSHAP)."
)
pdf.body_text(
    "Najznacajniji atributi prema kombinovanoj SHAP + Permutation importance analizi:\n\n"
    "  1. rider_name — identitet vozaca nosi najvise informacija (neki vozaci imaju\n"
    "     znacajno visu DNF stopu od drugih)\n"
    "  2. year — godina trke (starije trke imaju drugacije DNF obrasce)\n"
    "  3. team_name — neki timovi su pouzdaniji od drugih\n"
    "  4. Team_DNF_Rate — istorijska DNF stopa tima\n"
    "  5. Track_DNF_Rate — istorijska DNF stopa staze\n\n"
    "Najslabiji atributi (niska SHAP vaznost, negativna permutation importance):\n"
    "  - shortname (−0.08): redundantan sa circuit_name, uklanjanje poboljsava F1\n"
    "  - Quali_Spread, Quali_Top6_Gap, Quali_Time_Std: veoma niska vaznost\n"
    "  - Season_Phase: marginalan doprinos"
)

pdf.add_image_centered(os.path.join(RESULTS, "experiments", "figures", "shap_bar.png"), w=125)

pdf.section_title("7.2 Eksperiment sa redukovanim skupom atributa")
pdf.body_text(
    "Pokusano je uklanjanje 4 najslabija atributa (shortname + 3 Quali feature-a), "
    "smanjujuci broj feature-a sa 16 na 12. Medjutim, recall je opao sa 96.3% na "
    "89.8% — iako su ovi feature-i slabi, ipak doprinose detekciji dodatnih DNF-ova. "
    "Zakljucak: za ovaj problem (gde je FN prioritet), bolje je zadrzati sve feature-e."
)

# ============================================================
# 8. DEPLOYMENT
# ============================================================
pdf.add_page()
pdf.chapter_title("8. Deployment modela")

pdf.section_title("8.1 Streamlit aplikacija")
pdf.body_text(
    "Finalni model (Bagging K=2) je deployovan kroz Streamlit web aplikaciju. "
    "Aplikacija omogucava interaktivno testiranje modela kroz intuitivni korisnicki "
    "interfejs. Aplikacija se pokrece komandom:\n\n"
    "    streamlit run app/app.py"
)

pdf.body_text(
    "Funkcionalnosti aplikacije:\n"
    "  - Biranje staze, vozaca, tima, konstruktora, godine i rednog broja trke\n"
    "  - Slider za K parametar (1-10): minimalan broj glasova za DNF\n"
    "  - Slider za threshold (0.0-1.0): prag verovatnoce po modelu\n"
    "  - Gauge prikaz verovatnoce DNF-a\n"
    "  - Indikatori rizika (istorijski DNF rate, DNF rate tima, itd.)\n"
    "  - 5 preset scenarija za brzo testiranje (Marquez Jerez, Rossi Mugello...)\n"
    "  - Informacije o tome koliko je od 10 bagged modela glasalo za DNF"
)

pdf.section_title("8.2 Struktura projekta")
pdf.body_text(
    "Projekat je organizovan po principima cistog koda:\n\n"
    "  src/              — pipeline skripte (data_load, data_cleaning, preprocessing)\n"
    "  production/        — produkcijski modeli (svaki u svom folderu)\n"
    "  models/            — istrenirani modeli (.pkl)\n"
    "  results/           — metrike, grafikoni, matrice konfuzije\n"
    "  app/               — Streamlit deployment aplikacija\n"
    "  experiments/       — svi eksperimentalni pokusaji (10 foldera)\n"
    "  docs/              — dokumentacija\n"
    "  data/              — sirovi i procesirani podaci"
)

# ============================================================
# 9. ZAKLJUCAK
# ============================================================
pdf.add_page()
pdf.chapter_title("9. Zakljucak")

pdf.section_title("9.1 Postignuti rezultati")
pdf.body_text(
    "Projekat je uspesno implementirao sistem za predikciju DNF-a u MotoGP trkama. "
    "Kroz iterativni proces poboljsanja, broj propustenih DNF-ova (false negative) "
    "smanjen je sa pocetnih 20 na svega 6 od 244 test DNF-a — poboljsanje od 70%. "
    "Finalni recall iznosi 97.5%, sto znaci da model hvata skoro sve stvarne DNF-ove."
)

pdf.section_title("9.2 Kljucni nalazi")
pdf.bullet("OR logika (vise modela, bilo koji predvidi DNF) je superiorna za minimizaciju FN")
pdf.bullet("Platt kalibracija verovatnoca je neophodna za XGBoost (Brier score sa 0.62 na 0.33)")
pdf.bullet("Bagging sa K-voting konsenzusom dodatno filtrira random greske pojedinacnih modela")
pdf.bullet("DNF je inherentno tesko predvideti sa samo pre-race informacijama — cak i najbolji model ima nisku preciznost (12.8%)")
pdf.bullet("Data leakage je kritican — kolone sa post-race informacijama moraju biti uklonjene")
pdf.bullet("Feature-i sa niskom vaznoscu ipak doprinose — za FN-prioritet, bolje je zadrzati sve")

pdf.section_title("9.3 Ogranicenja i buduci rad")
pdf.body_text(
    "Glavno ogranicenje modela je niska preciznost (12.8%) — 87% DNF predikcija su lazni "
    "alarmi. Ovo je posledica fundamentalne tezine problema: sa samo pre-race informacijama, "
    "tesko je precizno razdvojiti trke koje ce se zavrsiti DNF-om od onih koje nece."
)
pdf.body_text(
    "Potencijalni pravci za poboljsanje:\n"
    "  - Grid position (startna pozicija) — ako bi bili dostupni za ceo dataset\n"
    "  - Vremenska prognoza za dan trke (kisa povecava DNF stopu)\n"
    "  - Pouzdanost motora po modelu i godini\n"
    "  - Broj padova vozaca u tekucoj sezoni"
)

pdf.section_title("9.4 Doprinos asistenta (vestacka inteligencija)")
pdf.body_text(
    "U izradi ovog projekta koriscen je AI asistent (OpenCode/Claude) za:\n"
    "  - Generisanje inicijalnog koda pipeline-a i modela\n"
    "  - Predlaganje i implementaciju razlicitih ML pristupa\n"
    "  - Debugging i optimizaciju hiperparametara\n"
    "  - Objasnjenje koncepata (bootstrap, Platt scaling, SHAP)\n"
    "  - Generisanje dokumentacije\n\n"
    "Sav kod je pregledan, testiran i prilagodjen od strane studenta. "
    "Sve odluke o izboru modela, thresholda i arhitekture su donete na osnovu "
    "eksperimentalnih rezultata i konsultacija sa asistentom."
)

# ============================================================
# SAVE
# ============================================================
output_path = os.path.join(DOCS, "Dokumentacija_MotoGP_DNF_Prediktor.pdf")
pdf.output(output_path)
print(f"Dokumentacija sacuvana: {output_path}")
print(f"Broj strana: {pdf.page_no()}")
