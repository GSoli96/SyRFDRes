"""
Plot RFD_tuple_compliance_strict (sopra) e RFD_tuple_compliance_relaxed
(sotto) in funzione di Label, raggruppati per RFD_threshold (2, 4, 8).

Stile:
- 2 subplot in verticale (2 righe, 1 colonna): strict sopra, relaxed sotto
- 3 serie per subplot, una per ogni RFD_threshold, con colori pastello diversi
- Stesso marker per tutte le serie
- Curva smussata (spline) che collega i punti di ogni serie
- Error bar = std
- linewidth = 0.5 ovunque (linee, error bar, bordi marker, assi, legenda)
- Legenda unica, centrata in alto, su 6 colonne, con ombra
- Output in PDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

LW = 0.5        
FRAME_LW = 0.5  

# ---------------------------------------------------------------
# 1. Carica il dataframe (modifica il path secondo le tue esigenze)
# ---------------------------------------------------------------
df = pd.read_csv('./evaluation_metrics/aggregated/aggregated_rfdcompliance_full_mean.csv')

# Colonne di interesse
cols_needed = [
    "Label",
    "RFD_threshold",
    "RFD_tuple_compliance_relaxed_mean",
    "RFD_tuple_compliance_relaxed_std",
    "RFD_tuple_compliance_strict_mean",
    "RFD_tuple_compliance_strict_std",
]
df = df[cols_needed].copy()

# ---------------------------------------------------------------
# 2. Setup: metriche (una per subplot), threshold, colori pastello, marker
# ---------------------------------------------------------------
# Ordine dei subplot: sopra "strict", sotto "relaxed"
subplot_metrics = [
    ("RFD_tuple_compliance_strict", "Strict"),
    ("RFD_tuple_compliance_relaxed", "Relaxed"),
]

MARKER = "o"  # stesso marker per tutte le serie

thresholds = sorted(df["RFD_threshold"].unique())  # [2, 4, 8]

# 3 colori pastello, uno per threshold
pastel_colors = ["#A8D8B9", "#F7C6C7", "#AEC6E8"]  # verde, rosa, azzurro pastello
if len(thresholds) > 3:
    cmap = plt.get_cmap("Pastel1")
    pastel_colors = [cmap(i) for i in range(len(thresholds))]
palette = {t: pastel_colors[i] for i, t in enumerate(thresholds)}

# ---------------------------------------------------------------
# 3. Ordinamento asse X: la posizione dei punti segue l'ordine
#    delle etichette (Label) così come compaiono nel dataframe
# ---------------------------------------------------------------
labels_order = list(dict.fromkeys(df["Label"]))       # ordine di comparsa, senza duplicati
x_positions = {lab: i for i, lab in enumerate(labels_order)}


# ---------------------------------------------------------------
# 4. Plot: 2 subplot in verticale (strict sopra, relaxed sotto)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)

legend_handles = []
legend_handles_done = False  # costruiamo la legenda una sola volta (soglie uguali in entrambi i subplot)

for ax, (metric_name, metric_label) in zip(axes, subplot_metrics):
    for t in thresholds:
        sub = df[df["RFD_threshold"] == t].copy()
        sub["x"] = sub["Label"].map(x_positions)
        sub = sub.sort_values("x")
        
        x = sub["x"].to_numpy(dtype=float)
        y = sub[f"{metric_name}_mean"].to_numpy(dtype=float)
        yerr = sub[f"{metric_name}_std"].to_numpy(dtype=float)

        color = palette[t]

        # --- curva smussata (spline) che collega i punti ---
        if len(x) >= 4:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=LW, zorder=1)
        else:
            ax.plot(x, y, color=color, linewidth=LW, zorder=1)

        # --- punti + error bar ---
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=MARKER,
            color=color,
            ecolor=color,
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=LW,
            capsize=3,
            elinewidth=LW,
            capthick=LW,
            linestyle="none",
            zorder=2,
        )

        if not legend_handles_done:
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker=MARKER,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=LW,
                    linestyle="-",
                    linewidth=LW,
                    markersize=5,
                    label=f"φ = {t}",
                )
            )

    ax.grid(visible=True, linestyle='dotted', alpha=0.5)

    legend_handles_done = True

    ax.set_ylabel(f"{metric_label} Compliance", fontsize=10)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # --- Rimuove lo spazio (margine) prima e dopo sull'asse X ---
    ax.margins(x=0.01)

    # --- Imposta le yticks da 0 a 1.25 con step 0.25 su entrambi i subplot ---
    for ax in axes:
        ax.set_yticks(np.arange(0, 1.25 + 0.25, 0.25))
        ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0, 1.25 + 0.25, 0.25)], fontsize=8)
        ax.set_ylim(0, 1.25)
    for spine in ax.spines.values():
        spine.set_linewidth(LW)
    ax.tick_params(width=LW)


# ---------------------------------------------------------------
# 5. Asse X con le etichette originali (condiviso, mostrato solo sotto)
# ---------------------------------------------------------------
axes[-1].set_xticks(list(x_positions.values()))
axes[-1].set_xticklabels(list(x_positions.keys()), rotation=90, fontsize=8)

for spine in ax.spines.values():
    spine.set_linewidth(FRAME_LW)


# ---------------------------------------------------------------
# 6. Legenda unica: centrata in alto (sopra il primo subplot),
#    su 6 colonne, con ombra
# ---------------------------------------------------------------
legend = fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=len(legend_handles),  # 3 serie (una per threshold), condivise tra i 2 subplot
    frameon=True,
    fontsize=9,
    columnspacing=1.0,
    handletextpad=0.5,
    shadow=True,
)
legend.get_frame().set_linewidth(LW)
fig.tight_layout()

# ---------------------------------------------------------------
# 7. Salvataggio in PDF
# ---------------------------------------------------------------
fig.savefig("rfd_tuple_compliance.pdf", bbox_inches="tight")
plt.show()