"""
download_susy.py
=================
Downloads and locally reconstructs the two SUSY dataset files used in the
evaluation (D35 in Table 2 of the paper), which are NOT included in the
repository due to their size (~750MB / ~340MB, beyond GitHub's practical
limits):

  imbalanced_datasets/susy_lowlevel8.csv          (5,000,000 rows)
  imbalanced_datasets/min/susy_lowlevel8_min.csv  (minority class only, class == 1)

Original source: UCI Machine Learning Repository - SUSY Data Set
https://archive.ics.uci.edu/dataset/279/susy
(Baldi, Sadowski, Whiteson, "Searching for Exotic Particles in High-Energy
Physics with Deep Learning", Nature Communications, 2014)

The original SUSY.csv.gz file contains, per row: 1 class label followed by
18 features (the first 8 kinematic "low-level" features, the remaining 10
derived "high-level" features). This script downloads the file, keeps the
label and ONLY the first 8 low-level features (Attr0..Attr7), renames the
columns according to the convention used in the rest of the repository
(feature columns first, "class" as the last column), and produces both the
full file and the minority-class-only subset (class == 1, 2,287,827 rows
expected, see Table 2 of the paper).

Usage:
    python scripts/download_susy.py

Requires an internet connection and ~2.5GB of temporary disk space for the
.gz file and the decompressed CSV.
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
        print(f"[skip] {dest} already present.")
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
    print(f"[done] {n_total} total rows, {n_minority} in the minority class (class == 1).")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MIN_DIR, exist_ok=True)
    if os.path.exists(FULL_OUT) and os.path.exists(MIN_OUT):
        print("[skip] Reconstructed SUSY files are already present.")
        return
    download(SUSY_URL, GZ_PATH)
    convert(GZ_PATH, FULL_OUT, MIN_OUT)
    os.remove(GZ_PATH)
    print("[cleanup] removed intermediate .gz archive.")


if __name__ == "__main__":
    main()
