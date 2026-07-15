#!/usr/bin/env python3
# =============================================================================
# rebalance_datasets.py
#
# Legge i dataset sbilanciati da ./imbalanced_datasets/ e li ribilancia con le
# tuple generate da SyRFDres in ./augmentation_results/v7_strict.
#
# Per ogni dataset <nome>.csv e per ogni soglia thr in {2, 4, 8}:
#   1. individua il file generato <nome>_min_thr<thr>_strict_2000_resnet_<ts>.csv
#      (se ce n'e' piu' d'uno, prende il piu' recente per timestamp);
#   2. calcola quante tuple servono per pareggiare la classe minoritaria
#      alla maggioritaria (diff = #maggioranza - #minoranza);
#   3. estrae le prime `diff` tuple dal file generato e le accoda al dataset;
#   4. salva in ./augmented_datasets/SYRFD_thr<thr>_strict/<nome>_thr<thr>_strict_balanced.csv
#
# Uso:  python rebalance_datasets.py
# =============================================================================

import os
import re
import glob
import pandas as pd

# ----------------------------- CONFIGURAZIONE --------------------------------
IMBALANCED_DIR = "./imbalanced_datasets"
#MODE = "strict"
MODE = "relaxed"
#GENERATED_DIR = "./augmentation_results/v7_{}".format(MODE)
GENERATED_DIR = "./augmentation_results_variable"
OUTPUT_ROOT = "./augmented_datasets"
THRESHOLDS = [2, 4, 8]
CLASS_COL = "class"
# ------------------------------------------------------------------------------


def find_generated_file(base: str, thr: int) -> str | None:
    """Trova il csv generato per (dataset, soglia); il piu' recente se multipli."""
    pattern = os.path.join(GENERATED_DIR, f"{base}_min_thr{thr}_{MODE}_*_resnet_*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # il timestamp in ms e' l'ultimo campo del nome: ordina e prendi il maggiore
    def ts(path):
        m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
        return int(m.group(1)) if m else -1
    return max(candidates, key=ts)


def main():
    # dataset sbilanciati: i csv nella root (esclude sottocartelle come 'min'
    # e gli eventuali duplicati *_min.csv presenti nella root)
    dataset_paths = sorted(
        p for p in glob.glob(os.path.join(IMBALANCED_DIR, "*.csv"))
        if not os.path.basename(p).endswith("_min.csv")
    )
    if not dataset_paths:
        print(f"Nessun dataset trovato in {IMBALANCED_DIR}")
        return

    for thr in THRESHOLDS:
        os.makedirs(os.path.join(OUTPUT_ROOT, f"SYRFD_thr{thr}_{MODE}"), exist_ok=True)

    n_ok, n_warn = 0, 0
    for data_path in dataset_paths:
        base = os.path.splitext(os.path.basename(data_path))[0]
        df = pd.read_csv(data_path)
        if CLASS_COL not in df.columns:
            print(f"[WARN] {base}: colonna '{CLASS_COL}' assente - saltato.")
            n_warn += 1
            continue

        counts = df[CLASS_COL].value_counts()
        min_label = counts.idxmin()
        diff = int(counts.max() - counts.min())
        if diff <= 0:
            print(f"[WARN] {base}: gia' bilanciato - saltato.")
            n_warn += 1
            continue

        for thr in THRESHOLDS:
            gen_path = find_generated_file(base, thr)
            if gen_path is None:
                print(f"[WARN] {base} thr{thr}: nessun file generato trovato - saltato.")
                n_warn += 1
                continue

            gen_df = pd.read_csv(gen_path)

            # allinea le colonne del generato a quelle del dataset originale
            missing = [c for c in df.columns if c not in gen_df.columns]
            if missing:
                print(f"[WARN] {base} thr{thr}: colonne mancanti nel generato {missing} - saltato.")
                n_warn += 1
                continue
            gen_df = gen_df[df.columns]

            # forza l'etichetta minoritaria (coerenza di dtype/valore)
            gen_df[CLASS_COL] = min_label

            if len(gen_df) < diff:
                print(f"[WARN] {base} thr{thr}: servono {diff} tuple ma il generato "
                      f"ne contiene {len(gen_df)} - uso tutte quelle disponibili.")
                take = gen_df
            else:
                take = gen_df.head(diff)

            balanced = pd.concat([df, take], ignore_index=True)

            out_dir = os.path.join(OUTPUT_ROOT, f"SYRFD_thr{thr}_{MODE}")
            out_path = os.path.join(out_dir, f"{base}_thr{thr}_{MODE}_balanced.csv")
            balanced.to_csv(out_path, index=False)

            new_counts = balanced[CLASS_COL].value_counts()
            print(f"[ OK ] {base} thr{thr}: +{len(take)} tuple "
                  f"({dict(counts)} -> {dict(new_counts)}) -> {out_path}")
            n_ok += 1

    print("=" * 60)
    print(f"Dataset bilanciati creati: {n_ok}   (warning: {n_warn})")


if __name__ == "__main__":
    main()