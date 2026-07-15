"""
Analizza i file rfd_compliance_<dataset>.csv (uno per dataset, 36 in totale)
e produce:

1) PLOT DI SINTESI (aggregati sui 36 file) per ciascuna combinazione
   (metrica x RFD_threshold): boxplot con mediana (linea centrale),
   IQR/whisker, marker della media e barra di deviazione standard.
   -> ./plots/RFD_Compliance/summary/boxplot_<metrica>_thr<soglia>.png

2) PLOT DI DETTAGLIO per ciascun file (= dataset) e per ciascun
   RFD_threshold presente in quel file: barplot raggruppato che confronta
   RFD_tuple_compliance_relaxed vs strict per ogni Algoritmo/configurazione.
   -> ./plots/RFD_Compliance/per_dataset/<dataset>_thr<soglia>.png

Uso:
    python plot_rfd_compliance.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
INPUT_DIR = "./evaluation_metrics/RFDcompliance"
OUTPUT_DIR = "./plots/RFD_Compliance"
FILE_PATTERN = "rfd_compliance_*.csv"

METRICS = ["RFD_tuple_compliance_relaxed", "RFD_tuple_compliance_strict"]
NON_METRIC_COLS = {"Algoritmo", "File", "RFD_file", "RFD_threshold", "RFD_epsilon", "N_RFDs"}


def dataset_name_from_filename(path):
    base = os.path.basename(path)
    base = re.sub(r"^rfd_compliance_", "", base)
    base = re.sub(r"\.csv$", "", base)
    return base


def category_for_row(dataset, algoritmo, file_field):
    """Costruisce l'etichetta di categoria Algoritmo,suffix rimuovendo dal
    campo File il nome del dataset (che varia da file a file) in modo da
    poter confrontare/aggregare la stessa configurazione tra dataset diversi."""
    file_no_ext = re.sub(r"\.csv$", "", str(file_field))
    prefix = dataset + "_"
    suffix = file_no_ext[len(prefix):] if file_no_ext.startswith(prefix) else file_no_ext
    return f"{algoritmo},{suffix}"


def load_all(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, FILE_PATTERN)))
    if not files:
        raise FileNotFoundError(
            f"Nessun file trovato in '{input_dir}' con pattern '{FILE_PATTERN}'."
        )

    all_rows = []
    for fp in files:
        dataset = dataset_name_from_filename(fp)
        df = pd.read_csv(fp)
        df["dataset"] = dataset
        df["category"] = [
            category_for_row(dataset, alg, f) for alg, f in zip(df["Algoritmo"], df["File"])
        ]
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True), [dataset_name_from_filename(f) for f in files]


def boxplot_with_mean_std(ax, matrix, ylabel):
    """matrix: DataFrame (n_file x n_categorie). Disegna boxplot + mean marker
    + errorbar di deviazione standard sopra ogni box."""
    categories = matrix.columns.tolist()
    data_per_cat = [matrix[c].dropna().values for c in categories]

    positions = np.arange(1, len(categories) + 1)

    bp = ax.boxplot(
        data_per_cat,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=False,
        medianprops=dict(color="#1a2332", linewidth=1.6),
        boxprops=dict(facecolor="#8ecae6", edgecolor="#1a2332", linewidth=0.9),
        whiskerprops=dict(color="#1a2332", linewidth=0.9),
        capprops=dict(color="#1a2332", linewidth=0.9),
        flierprops=dict(
            marker="o", markersize=3, markerfacecolor="#e76f51", markeredgecolor="none", alpha=0.6
        ),
    )

    means = np.array([np.nanmean(v) if len(v) else np.nan for v in data_per_cat])
    stds = np.array([np.nanstd(v) if len(v) else np.nan for v in data_per_cat])

    ax.errorbar(
        positions, means, yerr=stds, fmt="D", color="#e76f51", ecolor="#e76f51",
        elinewidth=1.3, capsize=3, markersize=4, label="media ± dev. standard", zorder=5,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=90)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def make_summary_plots(long_df, output_dir):
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    thresholds = sorted(long_df["RFD_threshold"].dropna().unique())

    for metric in METRICS:
        for thr in thresholds:
            sub = long_df[long_df["RFD_threshold"] == thr]
            pivot = sub.pivot_table(index="dataset", columns="category", values=metric, aggfunc="first")
            # ordina le categorie alfabeticamente per leggibilità
            pivot = pivot[sorted(pivot.columns)]

            n_cat = pivot.shape[1]
            width = max(14, n_cat * 0.28)
            fig, ax = plt.subplots(figsize=(width, 8))
            boxplot_with_mean_std(ax, pivot, ylabel=metric)
            ax.set_title(
                f"{metric} — RFD_threshold={thr:g} — aggregato su {pivot.shape[0]} dataset"
            )
            plt.tight_layout()

            thr_label = f"{thr:g}".replace(".", "_")
            out_path = os.path.join(summary_dir, f"boxplot_{metric}_thr{thr_label}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Salvato: {out_path}")


def make_per_dataset_plots(long_df, output_dir):
    per_ds_dir = os.path.join(output_dir, "per_dataset")
    os.makedirs(per_ds_dir, exist_ok=True)

    for dataset, df_ds in long_df.groupby("dataset"):
        thresholds = sorted(df_ds["RFD_threshold"].dropna().unique())
        for thr in thresholds:
            sub = df_ds[df_ds["RFD_threshold"] == thr].drop_duplicates(subset=["category"])
            sub = sub.sort_values("category")

            categories = sub["category"].tolist()
            relaxed_vals = sub["RFD_tuple_compliance_relaxed"].values
            strict_vals = sub["RFD_tuple_compliance_strict"].values

            n_cat = len(categories)
            width = max(14, n_cat * 0.3)
            fig, ax = plt.subplots(figsize=(width, 8))

            x = np.arange(n_cat)
            bar_w = 0.38
            ax.bar(x - bar_w / 2, relaxed_vals, width=bar_w, label="RFD_tuple_compliance_relaxed", color="#457b9d")
            ax.bar(x + bar_w / 2, strict_vals, width=bar_w, label="RFD_tuple_compliance_strict", color="#e76f51")

            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=90)
            ax.set_ylabel("Compliance")
            ax.set_ylim(0, 1.05)
            thr_val = thr
            ax.set_title(f"{dataset} — RFD_threshold={thr_val:g}")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            plt.tight_layout()

            thr_label = f"{thr_val:g}".replace(".", "_")
            out_path = os.path.join(per_ds_dir, f"{dataset}_thr{thr_label}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Salvato: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    long_df, datasets = load_all(INPUT_DIR)
    print(f"Trovati {len(datasets)} dataset: {datasets}")
    print(f"Righe totali caricate: {len(long_df)}")
    print(f"Categorie uniche (Algoritmo,configurazione): {long_df['category'].nunique()}")

    make_summary_plots(long_df, OUTPUT_DIR)
    make_per_dataset_plots(long_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()