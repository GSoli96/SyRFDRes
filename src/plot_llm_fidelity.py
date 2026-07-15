import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
import matplotlib.cm as cm

# Leggi i dati
path = "evaluation_metrics/aggregated/aggregated_fidelity_full_mean.csv"
df = pd.read_csv(path)

# Filtra solo le famiglie di interesse
df_filtered = df[df['Family'].isin(['SyRFD', 'LLM', 'LLM-zs'])].copy()

# Seleziona solo le colonne necessarie
cols = ["Family", "Label", "KL_mean_mean", "JS_mean_mean", "KS_mean_mean",
        "Chi2_mean_mean", "WD_mean_mean", "pMSE_mean", "MMD_mean",
        "PCA_WD_mean_mean"]

df_filtered = df_filtered[cols]

# Arrotonda alla 3 cifra decimale
numeric_cols = [col for col in cols if col not in ["Family", "Label"]]
df_filtered[numeric_cols] = df_filtered[numeric_cols].round(3)

# Definisci le metriche da plottare (tutte "lower is better")
metrics = ["KL_mean_mean", "JS_mean_mean", "KS_mean_mean", "Chi2_mean_mean",
           "WD_mean_mean", "pMSE_mean", "MMD_mean", "PCA_WD_mean_mean"]

# Mappa dei nomi delle metriche in LaTeX (adattata per mathtext di matplotlib)
metric_names_map = {
    "KL_mean_mean": r"$\mathrm{KL}\downarrow$",
    "JS_mean_mean": r"$\mathrm{JS}\downarrow$",
    "KS_mean_mean": r"$\mathrm{KS}\downarrow$",
    "Chi2_mean_mean": r"$\chi^2\downarrow$",
    "WD_mean_mean": r"$\mathrm{WD}\downarrow$",
    "pMSE_mean": r"$\mathrm{pMSE}\downarrow$",
    "MMD_mean": r"$\mathrm{MMD}\downarrow$",
    "PCA_WD_mean_mean": r"$\mathrm{WD}_{PCA}\downarrow$",
}

# Mappa colori per le modalità LLM
color_map_llm = {
    'conf': '#FF6B6B',
    'distr': '#4ECDC4',
    'dtree': '#45B7D1',
    'ens': '#96CEB4',
    'hier': '#FFEAA7',
    'react': '#DDA0DD',
    'schema': '#FF8C94',
    'self': '#A8D8EA'
}

color_map_llm = {
    'conf':   '#66C2A5',  # verde acqua
    'distr':  '#FC8D62',  # arancione salmone
    'dtree':  '#8DA0CB',  # blu lavanda
    'ens':    '#E78AC3',  # rosa
    'hier':   '#A6D854',  # verde lime
    'react':  '#FFD92F',  # giallo
    'schema': '#E5C494',  # beige
    'self':   '#B3B3B3',  # grigio chiaro
}

color_map_llm = {
    'conf':   '#E78AC3',  # arancione
    'distr':  '#56B4E9',  # azzurro
    'dtree':  '#009E73',  # verde
    'ens':    '#F0E442',  # giallo
    'hier':   '#0072B2',  # blu scuro
    'react':  '#D55E00',  # rosso mattone
    'schema': '#7B1B02',  # granata
    'self':   '#999999',  # grigio
}

color_map_llm = {
    'conf':   '#56B4E9',  # azzurro
    'distr':  '#0072B2',  # blu scuro
    'dtree':  '#009E73',  # verde
    'ens':    '#F0E442',  # giallo
    'hier':   '#E78AC3',  # rosa
    'react':  '#D55E00',  # rosso mattone
    'schema': '#7B1B02',  # granata
    'self':   '#999999',  # grigio
}

group_names_map = {
    'G4': 'Gemma-4',
    'G3': 'Gemma-3',
    'GPT120': 'GPT-OSS 120',
    'GPT20': 'GPT-OSS 20',
    'Dev': 'Devstral'
}

# Mappa colori per SyRFD (gradiente da chiaro a scuro per le soglie)
def get_syrfd_color(variant):
    if 'φ2' in variant:
        return '#FED8B1'  # Arancione chiaro
    elif 'φ4' in variant:
        return '#FDB462'  # Arancione medio
    elif 'φ8' in variant:
        return '#E67E22'  # Arancione scuro
    else:
        return '#F39C12'

# Definisci i gruppi
def get_group_llm(label):
    """Estrae il gruppo LLM (G4, G3, GPT120, GPT20, Dev)"""
    if '/' in label:
        return label.split('/')[0]
    return label

def get_mode_llm(label):
    """Estrae la modalità LLM (conf, distr, dtree, ens, hier, react, schema, self)"""
    if '/' in label:
        return label.split('/')[-1]
    return ''

# Separa i dati
df_syrfd = df_filtered[df_filtered['Family'] == 'SyRFD']
df_llm = df_filtered[df_filtered['Family'] == 'LLM']
df_llm_zs = df_filtered[df_filtered['Family'] == 'LLM-zs']

# Ordine dei gruppi LLM
llm_groups_order = ['G4', 'G3', 'GPT120', 'GPT20', 'Dev']

# Ordine delle modalità LLM
llm_modes_order = ['conf', 'distr', 'dtree', 'ens', 'hier', 'react', 'schema', 'self']

# Crea una figura con 2 righe e 4 colonne (8 metriche di fidelity)
fig, axes = plt.subplots(2, 4, figsize=(13, 4.5), sharex=True)
fig.subplots_adjust(hspace=0.01, wspace=0.03)
plt.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.16, hspace=0.01, wspace=0.03)

axes_flat = axes.flatten()

# Indici dei subplot che vanno nella riga in basso (ultima riga: qui indici 4-7)
bottom_row_indices = [4, 5, 6, 7]

# Per ogni metrica, crea un subplot
for i, metric in enumerate(metrics):
    ax = axes_flat[i]

    all_x_positions = []
    group_ticks = []
    group_labels = []
    current_pos = 0

    bar_width = 0.1
    bar_step = 0.1  # Contiguo, nessun gap interno
    group_spacing = 0.25  # Spazio tra gruppi di barre

    # ============ 1. GRUPPO SyRFD ============
    syrfd_labels = df_syrfd['Label'].tolist()
    syrfd_values = []
    syrfd_colors = []
    syrfd_hatches = []

    for label in syrfd_labels:
        row = df_syrfd[df_syrfd['Label'] == label].iloc[0]
        syrfd_values.append(row[metric])
        syrfd_colors.append(get_syrfd_color(label))
        # Aggiungi hatch per rlx
        if '/rlx' in label:
            syrfd_hatches.append('//')
        else:
            syrfd_hatches.append('')

    # Barre SyRFD - distanziate per gruppo (assume ordine: φ2 [0,1], φ4 [2,3], φ8 [4,5])
    x_syrfd_phi2 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        ax.bar(x_syrfd_phi2[j], syrfd_values[j], bar_width, color=syrfd_colors[j],
               edgecolor='black', linewidth=0.5, hatch=syrfd_hatches[j])
    group_ticks.append(np.mean(x_syrfd_phi2))
    group_labels.append(r'SyRFDRes $\phi 2$')
    current_pos += 2 * bar_step + group_spacing

    x_syrfd_phi4 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        idx = j + 2
        ax.bar(x_syrfd_phi4[j], syrfd_values[idx], bar_width, color=syrfd_colors[idx],
               edgecolor='black', linewidth=0.5, hatch=syrfd_hatches[idx])
    group_ticks.append(np.mean(x_syrfd_phi4))
    group_labels.append(r'SyRFDRes $\phi 4$')
    current_pos += 2 * bar_step + group_spacing

    x_syrfd_phi8 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        idx = j + 4
        ax.bar(x_syrfd_phi8[j], syrfd_values[idx], bar_width, color=syrfd_colors[idx],
               edgecolor='black', linewidth=0.5, hatch=syrfd_hatches[idx])
    group_ticks.append(np.mean(x_syrfd_phi8))
    group_labels.append(r'SyRFDRes $\phi 8$')
    current_pos += 2 * bar_step + group_spacing

    all_x_positions.extend(x_syrfd_phi2)
    all_x_positions.extend(x_syrfd_phi4)
    all_x_positions.extend(x_syrfd_phi8)

    # ============ EVIDENZIA BACKGROUND SyRFD ============
    if len(syrfd_labels) >= 6:
        x_min_phi2 = min(x_syrfd_phi2) - bar_width / 2
        x_max_phi2 = max(x_syrfd_phi2) + bar_width / 2
        x_min_phi4 = min(x_syrfd_phi4) - bar_width / 2
        x_max_phi4 = max(x_syrfd_phi4) + bar_width / 2
        x_min_phi8 = min(x_syrfd_phi8) - bar_width / 2
        x_max_phi8 = max(x_syrfd_phi8) + bar_width / 2

        bg_color = '#9DD6AD'  # Verde chiaro (nessuna metrica di fidelity necessita del rosso)
        alpha = 0.9

        # Scegli il colore in base al subplot (penultimo = indice 8)
        if i == 5:  # secondo seconda riga
            bg_color = '#D26466'  # Rosso chiaro
            alpha = 0.9
        elif i == 2:
            bg_color = '#d29964'  # arancione
            bg_color = '#D26466'  # Rosso chiaro
            alpha = 0.9
        else:
            bg_color = '#9DD6AD'  # Verde chiaro
            alpha = 0.9

        for x_min, x_max in [(x_min_phi2, x_max_phi2), (x_min_phi4, x_max_phi4),
                              (x_min_phi8, x_max_phi8)]:
            rect = Rectangle((x_min, 0), x_max - x_min, 1,
                              facecolor=bg_color, alpha=alpha, zorder=0,
                              transform=ax.get_xaxis_transform())
            ax.add_patch(rect)

    # ============ 2. GRUPPI LLM ============
    for group in llm_groups_order:
        group_data = df_llm[df_llm['Label'].str.startswith(group + '/')]

        if len(group_data) == 0:
            continue

        group_values = []
        group_modes = []
        for mode in llm_modes_order:
            mode_label = f"{group}/{mode}"
            mode_row = group_data[group_data['Label'] == mode_label]
            if len(mode_row) > 0:
                group_values.append(mode_row.iloc[0][metric])
                group_modes.append(mode)
            else:
                group_values.append(0)
                group_modes.append(mode)

        x_group = current_pos + np.arange(len(group_modes)) * bar_step
        colors_group = [color_map_llm.get(mode, '#999999') for mode in group_modes]

        ax.bar(x_group, group_values, bar_width, color=colors_group,
               edgecolor='black', linewidth=0.5)

        group_ticks.append(np.mean(x_group))
        group_labels.append(group_names_map.get(group, group))

        all_x_positions.extend(x_group)
        current_pos += len(group_modes) * bar_step + 0.3

    # ============ 3. GRUPPO LLM-zs ============
    if len(df_llm_zs) > 0:
        zs_labels = df_llm_zs['Label'].tolist()
        zs_values = [df_llm_zs[df_llm_zs['Label'] == label].iloc[0][metric] for label in zs_labels]

        x_zs = current_pos + np.arange(len(zs_labels)) * (bar_step + 0.1)

        colors_zs = ['#8E44AD', '#2C3E50']  # viola e grigio scuro
        ax.bar(x_zs, zs_values, bar_width, color=colors_zs[:len(zs_labels)],
               edgecolor='black', linewidth=0.5)

        for idx, label in enumerate(zs_labels):
            group_ticks.append(x_zs[idx])
            group_labels.append(label)

        all_x_positions.extend(x_zs)
        current_pos += len(zs_labels) * (bar_step + 0.3) + group_spacing + 0.5

    ax.set_xlim(min(all_x_positions) - bar_width, max(all_x_positions) + bar_width)

    # Tutte le metriche di fidelity sono "lower is better" e non negative:
    # si parte da 0 e si lascia autoscale il top. Se serve zoomare su differenze
    # piccole tra i metodi, regola qui i bottom per singola metrica come nello
    # script diversity originale (if i == 0: ax.set_ylim(bottom=...), ecc.).
# ============ IMPOSTA YLIM SPECIFICI PER OGNI METRICA ============
    # Indici dei subplot (row 0: colonne 0-4, row 1: colonne 5-9)
    if i == 0:  # Silhouette_mean (row 1, col 1)
        ax.set_ylim(bottom=0.2)
    elif i == 1:  # Davies-Bouldin_mean (row 1, col 2)
        ax.set_ylim(bottom=0.050)
    elif i == 2:  # Compactness_cls_k_mean (row 1, col 3)
        ax.set_ylim(bottom=0.1)
    elif i == 4:  # Recall_mean (row 2, col 1)
        ax.set_ylim(bottom=47, top=200)
    #elif i == 5:  # Density_mean (row 2, col 2) ax.set_ylim(bottom=0.5)
    elif i == 6:  # Coverage_mean (row 2, col 3)
        ax.set_ylim(bottom=0.2)
    elif i == 7:  # mDCR_mean (row 2, col 4)
        ax.set_ylim(bottom=0.1)


    # Etichette dell'asse x solo per la riga in basso
    if i in bottom_row_indices:
        ax.set_xticks(group_ticks)
        ax.set_xticklabels(group_labels, rotation=90, ha='center', fontsize=8)
    else:
        ax.set_xticks([])

    ax.set_ylabel(metric_names_map[metric], fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

# Legenda esterna centrale posizionata in alto
syrfd_legend = [
    Patch(facecolor='#FED8B1', edgecolor='black', label=r'SyRFDRes $\phi 2$'),
    Patch(facecolor='#FDB462', edgecolor='black', label=r'SyRFDRes $\phi 4$'),
    Patch(facecolor='#E67E22', edgecolor='black', label=r'SyRFDRes $\phi 8$'),
    Patch(facecolor='#F39C12', edgecolor='black', hatch='//', label=r'SyRFDRes rlx')
]

llm_legend2 = [
    Patch(facecolor='#FF6B6B', edgecolor='black', label='Confidence CoT'),
    Patch(facecolor='#4ECDC4', edgecolor='black', label='Distribution Guidance'),
    Patch(facecolor='#45B7D1', edgecolor='black', label='Decision-Tree CoT'),
    Patch(facecolor='#96CEB4', edgecolor='black', label='Ensemble CoT'),
    Patch(facecolor='#FFEAA7', edgecolor='black', label='Hierarchical CoT'),
    Patch(facecolor='#DDA0DD', edgecolor='black', label='ReAct CoT'),
    
    Patch(facecolor='#56B4E9', edgecolor='black', label='Schema Constraints'),
    Patch(facecolor='#0072B2', edgecolor='black', label='Self-Consistency CoT')
]

llm_legend_names = {
    'conf': 'Confidence CoT',
    'distr': 'Distribution Guidance',
    'dtree': 'Decision-Tree CoT',
    'ens': 'Ensemble CoT',
    'hier': 'Hierarchical CoT',
    'react': 'ReAct CoT',
    'schema': 'Schema Constraints',
    'self': 'Self-Consistency CoT',
}

llm_legend = [
    Patch(facecolor=color_map_llm[mode], edgecolor='black', label=llm_legend_names[mode])
    for mode in llm_modes_order
]

zs_legend = [
    Patch(facecolor='#8E44AD', edgecolor='black', label='DeepSeek'),
    Patch(facecolor='#2C3E50', edgecolor='black', label='LLaMA')
]

all_legend = syrfd_legend + llm_legend + zs_legend
#fig.legend(handles=all_legend, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=7, shadow=True, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.82], pad=0.0)
plt.savefig('confronto_metriche_fidelity.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Stampa un riepilogo dei dati
print("Riepilogo dati:")
print(f"SyRFD: {len(df_syrfd)} varianti")
print(f"LLM: {len(df_llm)} configurazioni")
print(f"LLM-zs: {len(df_llm_zs)} modelli")
print(f"\nTotale: {len(df_filtered)} righe")

print("\nGruppi LLM presenti:", sorted(df_llm['Label'].str.split('/').str[0].unique()))
print("Modalità LLM:", llm_modes_order)