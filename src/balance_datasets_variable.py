#!/usr/bin/env python3
# =============================================================================
# balance_datasets_variable.py
#
# Legge i dataset sbilanciati da ./imbalanced_datasets/ e li ribilancia con le
# tuple generate da SyRFDres in ./augmentation_results/v7_<MODE>.
#
# A differenza di balance_datasets.py, qui non si itera su piu' soglie: le
# tuple generate non dipendono da RFD_threshold (MODE = "variable_relaxed"
# oppure "variable_strict").
#
# Per ogni dataset <nome>.csv:
#   1. individua il file generato <nome>_min_relaxedthr_<MODE>_*_resnet_<ts>.csv
#      (se ce n'e' piu' d'uno, prende il piu' recente per timestamp);
#   2. calcola quante tuple servono per pareggiare la classe minoritaria
#      alla maggioritaria (diff = #maggioranza - #minoranza);
#   3. estrae le prime `diff` tuple dal file generato e le accoda al dataset;
#   4. salva in ./augmented_datasets/SYRFD_<MODE>/<nome>_<MODE>_balanced.csv
#
# Uso:  python balance_datasets_variable.py
# =============================================================================

import os
import re
import glob
import pandas as pd

# ----------------------------- CONFIGURAZIONE --------------------------------
IMBALANCED_DIR = "./imbalanced_datasets"
CLASS_COL = "class"

MODE = "variable_strict"  # oppure MODE = "variable_strict"
GENERATED_DIR = "./augmentation_results/v7_{}".format(MODE)
OUTPUT_ROOT = "./augmented_datasets"
# ------------------------------------------------------------------------------


def find_generated_file(base: str) -> str | None:
    """Trova il csv generato per il dataset; il piu' recente se multipli."""
    pattern = os.path.join(GENERATED_DIR, f"{base}_min_relaxedthr_{MODE}_*_resnet_*.csv")
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

    out_dir = os.path.join(OUTPUT_ROOT, f"SYRFD_{MODE}")
    os.makedirs(out_dir, exist_ok=True)

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

        gen_path = find_generated_file(base)
        if gen_path is None:
            print(f"[WARN] {base}: nessun file generato trovato - saltato.")
            n_warn += 1
            continue

        gen_df = pd.read_csv(gen_path)

        # allinea le colonne del generato a quelle del dataset originale
        missing = [c for c in df.columns if c not in gen_df.columns]
        if missing:
            print(f"[WARN] {base}: colonne mancanti nel generato {missing} - saltato.")
            n_warn += 1
            continue
        gen_df = gen_df[df.columns]

        # forza l'etichetta minoritaria (coerenza di dtype/valore)
        gen_df[CLASS_COL] = min_label

        if len(gen_df) < diff:
            print(f"[WARN] {base}: servono {diff} tuple ma il generato "
                  f"ne contiene {len(gen_df)} - uso tutte quelle disponibili.")
            take = gen_df
        else:
            take = gen_df.head(diff)

        balanced = pd.concat([df, take], ignore_index=True)

        out_path = os.path.join(out_dir, f"{base}_{MODE}_balanced.csv")
        balanced.to_csv(out_path, index=False)

        new_counts = balanced[CLASS_COL].value_counts()
        print(f"[ OK ] {base}: +{len(take)} tuple "
              f"({dict(counts)} -> {dict(new_counts)}) -> {out_path}")
        n_ok += 1

    print("=" * 60)
    print(f"Dataset bilanciati creati: {n_ok}   (warning: {n_warn})")


if __name__ == "__main__":
    main()
