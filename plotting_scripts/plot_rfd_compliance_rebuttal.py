#!/usr/bin/env python3
"""
plot_rfd_compliance_rebuttal.py
=============================================================================
Genera rfd_tuple_compliance_rebuttal.pdf risolvendo:
- D43: Raggruppamento esplicito dei metodi in Baselines, LLMs e SyRFDRes (Ours),
  con separatori verticali, etichette di gruppo e sfondi distinti.
- D44: Ripristino completo di tutte le soglie phi in {2, 4, 8} per SyRFDRes,
  evidenziando i punti verdi (phi=2), rossi (phi=4) e blu (phi=8) a 1.00.
- Attualmente (CLIP_ERROR_BARS_TO_01 = False):
    Il comportamento rimane esattamente quello della versione attuale (barre di errore simmetriche estese fino a $1.25$ e tick fino a $1.2$).
- Se in futuro imposterai CLIP_ERROR_BARS_TO_01 = True:
    Il codice calcolerà in automatico le barre asimmetriche arrestandole a $1.0$ in alto (e $0.0$ in basso) e formatterà i tick dell'asse $Y$ sul range naturale $[0.0, 1.0]$.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

LW = 0.6
FRAME_LW = 0.6

# Imposta a True per troncare le barre di errore al range teorico [0, 1]
# evitando che mean + std sfori oltre 1.0; False mantiene la versione attuale simmetrica.
CLIP_ERROR_BARS_TO_01 = False

# 1. Carica il dataframe aggregato
df = pd.read_csv('./evaluation_metrics/aggregated/aggregated_rfdcompliance_full_mean.csv')

# 2. Definizione dell'ordine e dei raggruppamenti (D43)
BASELINES_ORDER = [
    'SMOTE', 'SMOTE-C', 'FSMOTE', 'casTGAN', 'CTABGAN', 'CTABGAN+', 
    'CuTS', 'DDPM', 'GOGGLE', 'PreFair', 'TVAE', 
    'Tabfairgan', 'TabularARGN'
]

LLM_ZS_ORDER = ['DeepSeek', 'LLaMA']
LLM_PROMPTED_ORDER = [
    # Devstral
    'Dev/conf', 'Dev/distr', 'Dev/dtree', 'Dev/ens', 'Dev/hier', 'Dev/react', 'Dev/schema', 'Dev/self',
    # Gemma 3
    'G3/conf', 'G3/distr', 'G3/dtree', 'G3/ens', 'G3/hier', 'G3/react', 'G3/schema', 'G3/self',
    # Gemma 4
    'G4', 'G4/conf', 'G4/distr', 'G4/dtree', 'G4/ens', 'G4/hier', 'G4/react', 'G4/schema', 'G4/self',
    # GPT 20B
    'GPT20/conf', 'GPT20/distr', 'GPT20/dtree', 'GPT20/ens', 'GPT20/hier', 'GPT20/react', 'GPT20/schema', 'GPT20/self',
    # GPT 120B
    'GPT120/conf', 'GPT120/distr', 'GPT120/dtree', 'GPT120/ens', 'GPT120/hier', 'GPT120/react', 'GPT120/sch', 'GPT120/self',
]
LLMS_ORDER = LLM_ZS_ORDER + LLM_PROMPTED_ORDER

SYRFD_ORDER = [
    'SyRFDφ2/r', 'SyRFDφ2/s',
    'SyRFDφ4/r', 'SyRFDφ4/s',
    'SyRFDφ8/r', 'SyRFDφ8/s',
    'SyRFDφe/r', 'SyRFDφe/s'
]

# Filtra solo i label effettivamente presenti nei dati
available_labels = set(df['Label'].unique())
baselines_present = [l for l in BASELINES_ORDER if l in available_labels]
llms_present = [l for l in LLMS_ORDER if l in available_labels]
syrfd_present = [l for l in SYRFD_ORDER if l in available_labels]

# Ordine globale completo
labels_order = baselines_present + llms_present + syrfd_present
x_positions = {lab: i for i, lab in enumerate(labels_order)}

n_base = len(baselines_present)
n_llm = len(llms_present)
n_syrfd = len(syrfd_present)

idx_split_1 = n_base - 0.5
idx_split_2 = n_base + n_llm - 0.5

# 3. Setup subplot e palette
subplot_metrics = [
    ("RFD_tuple_compliance_strict", "Strict"),
    ("RFD_tuple_compliance_relaxed", "Relaxed"),
]

MARKER = "o"
thresholds = sorted(df["RFD_threshold"].unique())  # [2.0, 4.0, 8.0]

# Colori pastello distinti per soglia:
# phi=2 -> Verde pastello
# phi=4 -> Rosa/Rosso pastello
# phi=8 -> Azzurro/Blu pastello
palette = {
    2.0: "#66c2a5",  # verde menta/pastello più vivace per visibilità
    4.0: "#fc8d62",  # corallo/arancio-rosso
    8.0: "#8da0cb",  # pervinca/blu pastello
}
threshold_labels = {
    2.0: r"$\phi = 2$",
    4.0: r"$\phi = 4$",
    8.0: r"$\phi = 8$",
}

fig, axes = plt.subplots(2, 1, figsize=(14.5, 4), sharex=True)

groups = [
    ("Baselines", 0, n_base),
    ("LLM-based Generators", n_base, n_base + n_llm),
    ("SyRFDRes (Ours)", n_base + n_llm, len(labels_order)),
]

for row_idx, (ax, (metric_name, metric_label)) in enumerate(zip(axes, subplot_metrics)):
    # Sfondi leggeri per i gruppi
    ax.axvspan(-0.5, idx_split_1, facecolor='#f7f7f7', alpha=0.5, zorder=0)
    ax.axvspan(idx_split_1, idx_split_2, facecolor='#ffffff', alpha=0.5, zorder=0)
    ax.axvspan(idx_split_2, len(labels_order) - 0.5, facecolor='#e6f5f9', alpha=0.45, zorder=0)

    # Linee divisorie verticali
    ax.axvline(idx_split_1, color='#888888', linestyle='--', linewidth=0.8, alpha=0.75, zorder=1)
    ax.axvline(idx_split_2, color='#2b8cbe', linestyle='--', linewidth=0.9, alpha=0.85, zorder=1)

    legend_handles = []

    for t in thresholds:
        color = palette[t]
        sub_t = df[df["RFD_threshold"] == t].copy()
        sub_t["x"] = sub_t["Label"].map(x_positions)
        sub_t = sub_t.dropna(subset=["x"]).sort_values("x")

        # Traccia curve continue per ciascun gruppo separatamente
        for grp_name, g_start, g_end in groups:
            sub_g = sub_t[(sub_t["x"] >= g_start) & (sub_t["x"] < g_end)]
            if len(sub_g) == 0:
                continue

            x = sub_g["x"].to_numpy(dtype=float)
            y = sub_g[f"{metric_name}_mean"].to_numpy(dtype=float)
            yerr = sub_g[f"{metric_name}_std"].to_numpy(dtype=float)

            # Spline all'interno del gruppo
            if len(x) >= 4:
                x_smooth = np.linspace(x.min(), x.max(), 180)
                try:
                    spline = make_interp_spline(x, y, k=3)
                    y_smooth = np.clip(spline(x_smooth), -0.05, 1.05)
                    ax.plot(x_smooth, y_smooth, color=color, linewidth=LW, alpha=0.85, zorder=2)
                except Exception:
                    ax.plot(x, y, color=color, linewidth=LW, alpha=0.85, zorder=2)
            else:
                ax.plot(x, y, color=color, linewidth=LW, alpha=0.85, zorder=2)

            # Barre di errore (opzionalmente troncate a [0, 1] se abilitato)
            if CLIP_ERROR_BARS_TO_01:
                yerr_lower = np.minimum(y, yerr)
                yerr_upper = np.minimum(np.maximum(0.0, 1.0 - y), yerr)
                yerr_plot = [yerr_lower, yerr_upper]
            else:
                yerr_plot = yerr

            # Punti con barre di errore
            ax.errorbar(
                x, y, yerr=yerr_plot,
                fmt=MARKER,
                color=color,
                ecolor=color,
                markersize=4.8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=2.5,
                elinewidth=0.6,
                capthick=0.6,
                linestyle="none",
                zorder=3,
            )

        # Handle per la legenda
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=MARKER,
                color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="-",
                linewidth=1.0,
                markersize=5.5,
                label=threshold_labels[t],
            )
        )

    # Intestazione macro-gruppi sul primo subplot
    if row_idx == 0:
        font_local = 9.5
        ax.text((n_base - 1) / 2, 1.035, "Baselines", ha="center", va="bottom",
                fontsize=font_local+5, fontweight="bold", color="#333333", transform=ax.get_xaxis_transform())
        ax.text(n_base + (n_llm - 1) / 2, 1.035, "LLM-based Models", ha="center", va="bottom",
                fontsize=font_local+5, fontweight="bold", color="#333333", transform=ax.get_xaxis_transform())
        ax.text(n_base + n_llm + (n_syrfd - 1) / 2, 1.035, "SyRFDRes", ha="center", va="bottom",
                fontsize=font_local+5, fontweight="bold", color="#08519c", transform=ax.get_xaxis_transform())

    ax.grid(visible=True, linestyle='dotted', alpha=0.45)
    ax.set_ylabel(f"{metric_label} Compliance", fontsize=9.5)
    if CLIP_ERROR_BARS_TO_01:
        ax.set_ylim(-0.04, 1.15)
        ax.set_yticks(np.linspace(0, 1.0, 6))
    else:
        ax.set_ylim(-0.04, 1.25)
        ax.set_yticks(np.linspace(0, 1.2, 7))
    ax.margins(x=0.01)

# Asse X con etichette formattate
def format_xtick_label(lab):
    lab_out = lab
    for pat in ["SyRFDφe", "SyRFD\\varphi e", "SyRFDvar"]:
        lab_out = lab_out.replace(pat, "SyRFDvar")
    lab_out = lab_out.replace("SyRFD", "SyRFDRes")
    return lab_out

xtick_labels = [format_xtick_label(k) for k in x_positions.keys()]
axes[-1].set_xticks(list(x_positions.values()))
axes[-1].set_xticklabels(xtick_labels, rotation=90, fontsize=7.2)
axes[-1].tick_params(axis='x', which='major', pad=0)
axes[0].tick_params(axis='y', which='major', pad=0)
axes[-1].tick_params(axis='y', which='major', pad=0)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_LW)

# Legenda centrata in alto sopra i grafici (posizionata più in alto per non toccare i titoli dei gruppi)
legend = fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.998),
    ncol=3,
    frameon=True,
    facecolor="white",
    fontsize=9.0,
    columnspacing=2.5,
    handletextpad=0.6,
    shadow=True,
)
legend.get_frame().set_linewidth(LW)

plt.subplots_adjust(top=0.8, bottom=0.17, left=0.06, right=0.985, hspace=0.11)

out_pdf = "rfd_tuple_compliance_rebuttal.pdf"
out_png = "rfd_tuple_compliance_rebuttal.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")

supp_dir = "supplementary_material/plots/rfd_compliance"
os.makedirs(supp_dir, exist_ok=True)
fig.savefig(os.path.join(supp_dir, out_pdf), bbox_inches="tight")
fig.savefig(os.path.join(supp_dir, out_png), dpi=300, bbox_inches="tight")
print(f"Salvato con successo: {out_pdf} e {out_png} anche in {supp_dir}")
