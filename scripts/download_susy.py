"""
download_susy.py
=================
Scarica e ricostruisce localmente i due file del dataset SUSY usati nella
valutazione (D35 in Table 2 del paper), che NON sono inclusi nel repository
per la loro dimensione (~750MB / ~340MB, oltre i limiti pratici di GitHub):

  imbalanced_datasets/susy_lowlevel8.csv       (5,000,000 righe)
  imbalanced_datasets/min/susy_lowlevel8_min.csv  (solo classe minoritaria, class == 1)

Fonte originale: UCI Machine Learning Repository - SUSY Data Set
https://archive.ics.uci.edu/dataset/279/susy
(Baldi, Sadowski, Whiteson, "Searching for Exotic Particles in High-Energy
Physics with Deep Learning", Nature Communications, 2014)

Il file originale SUSY.csv.gz contiene, per riga: 1 etichetta di classe seguita
da 18 feature (le prime 8 "low-level" cinematiche, le successive 10
"high-level" derivate). Questo script scarica il file, mantiene la label e
SOLO le prime 8 feature low-level (Attr0..Attr7), rinomina le colonne secondo
la convenzione usata nel resto del repository (colonne feature per prime,
"class" come ultima colonna) e produce sia il file completo sia il
sottoinsieme di sola classe minoritaria (class == 1, 2'287'827 righe attese,
si veda Table 2 del paper).

Uso:
    python scripts/download_susy.py

Richiede una connessione a Internet e ~2.5GB di spazio disco temporaneo per
il file .gz e il CSV decompresso.
"""
import gzip
import os
import shutil
import urllib.request

SUSY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"

DATA_DIR = "./imbalanced_datasets"
MIN_DIR = os.path.join(DATA_DIR, "min")
GZ_PATH = os.path.join(DATA_DIR, "SUSY.csv.gz")
FULL_OUT = os.path.join(DATA_DIR, "susy_lowlevel8.csv")
MIN_OUT = os.path.join(MIN_DIR, "susy_lowlevel8_min.csv")

N_LOWLEVEL_FEATURES = 8
HEADER = ",".join(f"Attr{i}" for i in range(N_LOWLEVEL_FEATURES)) + ",class"


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] {dest} già presente.")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def convert(gz_path: str, full_out: str, min_out: str) -> None:
    print(f"[convert] {gz_path} -> {full_out} (+ subset in {min_out})")
    os.makedirs(os.path.dirname(min_out), exist_ok=True)
    n_total = 0
    n_minority = 0
    with gzip.open(gz_path, "rt") as src, \
         open(full_out, "w", newline="") as f_full, \
         open(min_out, "w", newline="") as f_min:
        f_full.write(HEADER + "\n")
        f_min.write(HEADER + "\n")
        for line in src:
            fields = line.rstrip("\n").split(",")
            label = fields[0]
            # keep only the 8 low-level kinematic features (columns 1..8)
            lowlevel = fields[1:1 + N_LOWLEVEL_FEATURES]
            class_val = str(int(float(label)))
            row = ",".join(lowlevel + [class_val])
            f_full.write(row + "\n")
            n_total += 1
            if class_val == "1":
                f_min.write(row + "\n")
                n_minority += 1
    print(f"[done] {n_total} righe totali, {n_minority} nella classe minoritaria (class == 1).")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MIN_DIR, exist_ok=True)
    if os.path.exists(FULL_OUT) and os.path.exists(MIN_OUT):
        print("[skip] I file SUSY ricostruiti sono già presenti.")
        return
    download(SUSY_URL, GZ_PATH)
    convert(GZ_PATH, FULL_OUT, MIN_OUT)
    os.remove(GZ_PATH)
    print("[cleanup] rimosso l'archivio .gz intermedio.")


if __name__ == "__main__":
    main()
