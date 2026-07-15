import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
import matplotlib.cm as cm

# Leggi i dati
path = "evaluation_metrics/aggregated/aggregated_diversity_full_mean.csv"
df = pd.read_csv(path)

# Filtra solo le famiglie di interesse
df_filtered = df[df['Family'].isin(['SyRFD', 'LLM', 'LLM-zs'])].copy()

# Seleziona tutte le colonne necessarie (incluse medie, mediane, std)
cols = ["Family", "Label", 
        "Silhouette_mean", "Silhouette_median", "Silhouette_std",
        "Davies-Bouldin_mean", "Davies-Bouldin_median", "Davies-Bouldin_std",
        "Compactness_cls_k_mean", "Compactness_cls_k_median", "Compactness_cls_k_std",
        "Precision_mean", "Precision_median", "Precision_std",
        "Recall_mean", "Recall_median", "Recall_std",
        "Density_mean", "Density_median", "Density_std",
        "Coverage_mean", "Coverage_median", "Coverage_std",
        "mDCR_mean", "mDCR_median", "mDCR_std",
        "DBI_div_mean", "DBI_div_median", "DBI_div_std",
        "Coverage_div_mean", "Coverage_div_median", "Coverage_div_std"]

df_filtered = df_filtered[cols]

# Arrotonda alla 3 cifra decimale
numeric_cols = [col for col in cols if col not in ["Family", "Label"]]
df_filtered[numeric_cols] = df_filtered[numeric_cols].round(3)

# Definisci le metriche da plottare con le relative colonne
metrics_config = [
    {"metric": "Silhouette_mean", "median": "Silhouette_median", "std": "Silhouette_std"},
    {"metric": "Davies-Bouldin_mean", "median": "Davies-Bouldin_median", "std": "Davies-Bouldin_std"},
    {"metric": "Compactness_cls_k_mean", "median": "Compactness_cls_k_median", "std": "Compactness_cls_k_std"},
    {"metric": "Precision_mean", "median": "Precision_median", "std": "Precision_std"},
    {"metric": "Recall_mean", "median": "Recall_median", "std": "Recall_std"},
    {"metric": "Density_mean", "median": "Density_median", "std": "Density_std"},
    {"metric": "Coverage_mean", "median": "Coverage_median", "std": "Coverage_std"},
    {"metric": "mDCR_mean", "median": "mDCR_median", "std": "mDCR_std"},
    {"metric": "DBI_div_mean", "median": "DBI_div_median", "std": "DBI_div_std"},
    {"metric": "Coverage_div_mean", "median": "Coverage_div_median", "std": "Coverage_div_std"}
]

# Definisci le metriche da plottare
metrics = ["Silhouette_mean", "Davies-Bouldin_mean", "Compactness_cls_k_mean", 
           "Precision_mean", "Recall_mean", "Density_mean", "Coverage_mean", 
           "mDCR_mean", "DBI_div_mean", "Coverage_div_mean"]

# Mappa dei nomi delle metriche in LaTeX come richiesto (adattata per mathtext di matplotlib)
metric_names_map = {
    "Silhouette_mean": r"$\mathrm{Silh}\rightarrow 0$",
    "Davies-Bouldin_mean": r"$\mathrm{DBI}\uparrow$",
    "Compactness_cls_k_mean": r"$\mathcal{I}_{Com}$",
    "Precision_mean": r"$\mathrm{Prec}\uparrow$",
    "Recall_mean": r"$\mathrm{Rec}\uparrow$",
    "Density_mean": r"$\mathrm{Dens}\uparrow$",
    "Coverage_mean": r"$\mathrm{Cov}\uparrow$",
    "mDCR_mean": r"$\mathrm{mDCR}$",
    "DBI_div_mean": r"$\mathrm{DBI}_q\uparrow$",
    "Coverage_div_mean": r"$\mathrm{Cov}_q\uparrow$"
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
'''

# Colori dalla colormap 'inferno' per le modalità LLM
inferno_cmap = cm.get_cmap('inferno')
# Estrai 8 colori dalla colormap per le 8 modalità
inferno_colors = [inferno_cmap(i / 7) for i in range(8)]  # 8 colori distribuiti uniformemente

# Mappa colori per le modalità LLM usando 'inferno'
color_map_llm = {
    'conf': inferno_colors[7],
    'distr': inferno_colors[6],
    'dtree': inferno_colors[5],
    'ens': inferno_colors[4],
    'hier': inferno_colors[3],
    'react': inferno_colors[2],
    'schema': inferno_colors[1],
    'self': inferno_colors[0]
}
'''

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

# Crea una figura con 2 righe e 5 colonne
fig, axes = plt.subplots(2, 5, figsize=(16, 4), sharex=True)
# Riduci lo spazio orizzontale e verticale tra i subplot
fig.subplots_adjust(hspace=0.01, wspace=0.005)  # wspace ridotto da 0.02 a 0.005
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, hspace=0.01, wspace=0.005)

axes_flat = axes.flatten()

# Per ogni metrica, crea un subplot
for i, config in enumerate(metrics_config):
    ax = axes_flat[i]
    
    metric = config["metric"]
    median_col = config["median"]
    std_col = config["std"]
    
    # Liste per tenere traccia delle posizioni delle barre, e per le xticks a livello di gruppo
    all_x_positions = []
    group_ticks = []
    group_labels = []
    current_pos = 0
    
    bar_width = 0.1
    bar_step = 0.1  # Contiguo, nessun gap interno
    
    '''
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
    
    # Barre SyRFD
    x_syrfd = current_pos + np.arange(len(syrfd_labels)) * bar_step
    for j, (val, color, hatch) in enumerate(zip(syrfd_values, syrfd_colors, syrfd_hatches)):
        bar = ax.bar(x_syrfd[j], val, bar_width, color=color, 
                    edgecolor='black', linewidth=0.5, hatch=hatch)
    
    # Ticks per gruppi SyRFD
    if len(syrfd_labels) >= 2:
        group_ticks.append(np.mean(x_syrfd[0:2]))
        group_labels.append(r'SyRFD $\phi 2$')
    if len(syrfd_labels) >= 4:
        group_ticks.append(np.mean(x_syrfd[2:4]))
        group_labels.append(r'SyRFD $\phi 4$')
    if len(syrfd_labels) >= 6:
        group_ticks.append(np.mean(x_syrfd[4:6]))
        group_labels.append(r'SyRFD $\phi 8$')
        
    all_x_positions.extend(x_syrfd)
    current_pos += len(syrfd_labels) * bar_step + 0.2  # ripristina lo spazio tra gruppi
    '''
    # ============ 1. GRUPPO SyRFD ============
    group_spacing = 0.25  # Spazio tra gruppi di barre    
    syrfd_labels = df_syrfd['Label'].tolist()
    syrfd_values = []
    syrfd_medians = []
    syrfd_stds = []
    syrfd_colors = []
    syrfd_hatches = []
    
    for label in syrfd_labels:
        row = df_syrfd[df_syrfd['Label'] == label].iloc[0]
        syrfd_values.append(row[metric])
        syrfd_medians.append(row[median_col])
        syrfd_stds.append(row[std_col])
        syrfd_colors.append(get_syrfd_color(label))
        if '/rlx' in label:
            syrfd_hatches.append('//')
        else:
            syrfd_hatches.append('')
    
    # Barre SyRFD - distanziate per gruppo
    # φ2 group (indici 0,1)
    x_syrfd_phi2 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        ax.bar(x_syrfd_phi2[j], syrfd_values[j], bar_width, color=syrfd_colors[j], 
               edgecolor='black', linewidth=0.5, hatch=syrfd_hatches[j])

        ax.errorbar(x_syrfd_phi2[j], syrfd_medians[j], yerr=syrfd_stds[j],
            fmt='none', ecolor='black', capsize=2, capthick=1, elinewidth=1, alpha=0.7)

    group_ticks.append(np.mean(x_syrfd_phi2))
    group_labels.append(r'SyRFDRes $\phi 2$')
    current_pos += 2 * bar_step + group_spacing
    
    # φ4 group (indici 2,3)
    x_syrfd_phi4 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        idx = j + 2
        ax.bar(x_syrfd_phi4[j], syrfd_values[idx], bar_width, color=syrfd_colors[idx], 
               edgecolor='black', linewidth=0.5, hatch=syrfd_hatches[idx])

        # Aggiungi error bar con std
        ax.errorbar(x_syrfd_phi4[j], syrfd_medians[idx], yerr=syrfd_stds[idx],
                    fmt='none', ecolor='black', capsize=2, capthick=1, elinewidth=1, alpha=0.7)
    

    group_ticks.append(np.mean(x_syrfd_phi4))
    group_labels.append(r'SyRFDRes $\phi 4$')
    current_pos += 2 * bar_step + group_spacing
    
    # φ8 group (indici 4,5)
    x_syrfd_phi8 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        idx = j + 4
        ax.bar(x_syrfd_phi8[j], syrfd_values[idx], bar_width, color=syrfd_colors[idx], 
               edgecolor='black', linewidth=0.5, hatch=syrfd_hatches[idx])

        # Aggiungi error bar con std
        ax.errorbar(x_syrfd_phi8[j], syrfd_medians[idx], yerr=syrfd_stds[idx],
                    fmt='none', ecolor='black', capsize=2, capthick=1, elinewidth=1, alpha=0.7)

    group_ticks.append(np.mean(x_syrfd_phi8))
    group_labels.append(r'SyRFDRes $\phi 8$')
    current_pos += 2 * bar_step + group_spacing
    
    all_x_positions.extend(x_syrfd_phi2)
    all_x_positions.extend(x_syrfd_phi4)
    all_x_positions.extend(x_syrfd_phi8)

    # ============ EVIDENZIA BACKGROUND SyRFD ============
    if len(syrfd_labels) >= 6:
        x_min_phi2 = min(x_syrfd_phi2) - bar_width/2
        x_max_phi2 = max(x_syrfd_phi2) + bar_width/2
        x_min_phi4 = min(x_syrfd_phi4) - bar_width/2
        x_max_phi4 = max(x_syrfd_phi4) + bar_width/2
        x_min_phi8 = min(x_syrfd_phi8) - bar_width/2
        x_max_phi8 = max(x_syrfd_phi8) + bar_width/2
        
        # Scegli il colore in base al subplot (penultimo = indice 8)
        if i == 8:  # Penultimo subplot (DBI_div_mean)
            bg_color = '#D26466'  # Rosso chiaro
            alpha = 0.9
        else:
            bg_color = '#9DD6AD'  # Verde chiaro
            alpha = 0.9
        
        # Aggiungi i rettangoli che coprono tutto il plot in verticale
        # Usa le coordinate dei dati per l'asse x e coordinate di figura per l'asse y
        rect = Rectangle((x_min_phi2, 0), x_max_phi2 - x_min_phi2, 1,
                         facecolor=bg_color, alpha=alpha, zorder=0,
                         transform=ax.get_xaxis_transform())
        ax.add_patch(rect)
        
        rect = Rectangle((x_min_phi4, 0), x_max_phi4 - x_min_phi4, 1,
                         facecolor=bg_color, alpha=alpha, zorder=0,
                         transform=ax.get_xaxis_transform())
        ax.add_patch(rect)
        
        rect = Rectangle((x_min_phi8, 0), x_max_phi8 - x_min_phi8, 1,
                         facecolor=bg_color, alpha=alpha, zorder=0,
                         transform=ax.get_xaxis_transform())
        ax.add_patch(rect)

    # ============ 2. GRUPPI LLM ============
    for group in llm_groups_order:
        # Filtra i dati per questo gruppo
        group_data = df_llm[df_llm['Label'].str.startswith(group + '/')]
        
        if len(group_data) == 0:
            continue
        
        group_values = []
        group_medians = []
        group_stds = []
        group_modes = []
        
        for mode in llm_modes_order:
            mode_label = f"{group}/{mode}"
            mode_row = group_data[group_data['Label'] == mode_label]
            if len(mode_row) > 0:
                group_values.append(mode_row.iloc[0][metric])
                group_medians.append(mode_row.iloc[0][median_col])
                group_stds.append(mode_row.iloc[0][std_col])
                group_modes.append(mode)
            else:
                group_values.append(0)
                group_medians.append(0)
                group_stds.append(0)
                group_modes.append(mode)
        # Barre per questo gruppo
        x_group = current_pos + np.arange(len(group_modes)) * bar_step
        colors_group = [color_map_llm.get(mode, '#999999') for mode in group_modes]
        
        # Barre per questo gruppo
        for j, (val, med, std, color) in enumerate(zip(group_values, group_medians, group_stds, colors_group)):
            bar = ax.bar(x_group[j], val, bar_width, color=color,
                        edgecolor='black', linewidth=0.5)
            # Aggiungi error bar con std
            if std > 0:  # Mostra solo se std > 0
                ax.errorbar(x_group[j], med, yerr=std,
                            fmt='none', ecolor='black', capsize=1.5, capthick=1, 
                            elinewidth=0.8, alpha=0.6)

        # Ticks per gruppo LLM
        group_ticks.append(np.mean(x_group))
        group_labels.append(group_names_map.get(group, group))
        
        all_x_positions.extend(x_group)
        current_pos += len(group_modes) * bar_step + 0.3  # ripristina lo spazio tra gruppi
    
        # ============ 3. GRUPPO LLM-zs ============
    if len(df_llm_zs) > 0:
        zs_labels = df_llm_zs['Label'].tolist()
        zs_values = [df_llm_zs[df_llm_zs['Label'] == label].iloc[0][metric] for label in zs_labels]
        zs_medians = []
        zs_stds = []
        
        for label in zs_labels:
            row = df_llm_zs[df_llm_zs['Label'] == label].iloc[0]
            zs_values.append(row[metric])
            zs_medians.append(row[median_col])
            zs_stds.append(row[std_col])

        # Distanzia le due barre di LLM-zs
        x_zs = current_pos + np.arange(len(zs_labels)) * (bar_step + 0.1)  # Aggiunto spazio extra
        
        # Colori per LLM-zs
        colors_zs = ['#8E44AD', '#2C3E50']
        for j, (val, med, std) in enumerate(zip(zs_values, zs_medians, zs_stds)):
            bar = ax.bar(x_zs[j], val, bar_width, color=colors_zs[j],
                        edgecolor='black', linewidth=0.5)
            # Aggiungi error bar con std
            if std > 0:
                ax.errorbar(x_zs[j], med, yerr=std,
                            fmt='none', ecolor='black', capsize=1.5, capthick=1,
                            elinewidth=0.8, alpha=0.6)
        
        
        # Ticks per LLM-zs - ogni barra ha il suo tick
        for idx, label in enumerate(zs_labels):
            group_ticks.append(x_zs[idx])
            group_labels.append(label)
            
        all_x_positions.extend(x_zs)
        current_pos += len(zs_labels) * (bar_step + 0.3) + group_spacing + 0.5
    
    # Rimuovi lo spazio bianco a destra e a sinistra impostando i limiti dell'asse x
    ax.set_xlim(min(all_x_positions) - bar_width, max(all_x_positions) + bar_width)

    # ============ IMPOSTA YLIM SPECIFICI PER OGNI METRICA ============
    # Indici dei subplot (row 0: colonne 0-4, row 1: colonne 5-9)
    if i == 0:  # Silhouette_mean (row 1, col 1)
        ax.set_ylim(bottom=0)
    elif i == 1:  # Davies-Bouldin_mean (row 1, col 2)
        ax.set_ylim(bottom=3)
    elif i == 2:  # Compactness_cls_k_mean (row 1, col 3)
        ax.set_ylim(bottom=0.06)
    elif i == 3:  # Precision_mean (row 1, col 4)
        ax.set_ylim(bottom=0.7)
    elif i == 4:  # Recall_mean (row 1, col 5)
        ax.set_ylim(bottom=0.9)
    elif i == 5:  # Density_mean (row 2, col 1)
        ax.set_ylim(bottom=0.5)
    elif i == 6:  # Coverage_mean (row 2, col 2)
        ax.set_ylim(bottom=0.9)
    elif i == 7:  # mDCR_mean (row 2, col 3)
        ax.set_ylim(bottom=0.35)
    elif i == 8:  # DBI_div_mean (row 2, col 4)
        ax.set_ylim(bottom=1.3)
    elif i == 9:  # Coverage_div_mean (row 2, col 5)
        ax.set_ylim(bottom=0.9)

    # Imposta etichette dell'asse x (solo per gli ultimi due subplot in basso)
    # Gli ultimi due subplot sono gli indici 8 e 9 (riga 2, colonne 4 e 5)
    if i in [5, 6, 7, 8, 9]:
        ax.set_xticks(group_ticks)
        ax.set_xticklabels(group_labels, rotation=90, ha='center', fontsize=8)
    else:
        ax.set_xticks([])
    
    # Imposta y-label con i nomi LaTeX richiesti
    ax.set_ylabel(metric_names_map[metric], fontsize=10)
    
    # Griglia
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

# Legenda esterna centrale con shadow=True posizionata in alto
syrfd_legend = [
    Patch(facecolor='#FED8B1', edgecolor='black', label=r'SyRFDRes $\phi 2$'),
    Patch(facecolor='#FDB462', edgecolor='black', label=r'SyRFDRes $\phi 4$'),
    Patch(facecolor='#E67E22', edgecolor='black', label=r'SyRFDRes $\phi 8$'),
    Patch(facecolor='#F39C12', edgecolor='black', hatch='//', label=r'SyRFDRes rlx')
]

llm_legend = [
    Patch(facecolor='#FF6B6B', edgecolor='black', label='Confidence CoT'),
    Patch(facecolor='#4ECDC4', edgecolor='black', label='Distribution Guidance'),
    Patch(facecolor='#45B7D1', edgecolor='black', label='Decision-Tree CoT'),
    Patch(facecolor='#96CEB4', edgecolor='black', label='Ensemble CoT'),
    Patch(facecolor='#FFEAA7', edgecolor='black', label='Hierarchical CoT'),
    Patch(facecolor='#DDA0DD', edgecolor='black', label='ReAct CoT'),
    Patch(facecolor='#FF8C94', edgecolor='black', label='Schema Constraints'),
    Patch(facecolor='#A8D8EA', edgecolor='black', label='Self-Consistency CoT')
]

zs_legend = [
    Patch(facecolor='#8E44AD', edgecolor='black', label='DeepSeek'),
    Patch(facecolor='#2C3E50', edgecolor='black', label='LLaMA')
]

all_legend = syrfd_legend + llm_legend + zs_legend
# Legenda in alto
fig.legend(handles=all_legend, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=7, shadow=True, fontsize=8)

# Salva il plot lasciando spazio per la legenda esterna in alto
plt.tight_layout(rect=[0, 0, 1, 0.8], pad=0.0)
plt.savefig('confronto_metriche_completo.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Stampa un riepilogo dei dati
print("Riepilogo dati:")
print(f"SyRFD: {len(df_syrfd)} varianti")
print(f"LLM: {len(df_llm)} configurazioni")
print(f"LLM-zs: {len(df_llm_zs)} modelli")
print(f"\nTotale: {len(df_filtered)} righe")

print("\nGruppi LLM presenti:", sorted(df_llm['Label'].str.split('/').str[0].unique()))
print("Modalità LLM:", llm_modes_order)