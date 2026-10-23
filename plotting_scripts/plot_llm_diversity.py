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

# Seleziona solo le colonne necessarie
cols = ["Family", "Label", "Precision_mean", "Recall_mean", 
        "Density_mean", "Coverage_mean", "mDCR_mean"]

df_filtered = df_filtered[cols]

# Arrotonda alla 3 cifra decimale
numeric_cols = [col for col in cols if col not in ["Family", "Label"]]
df_filtered[numeric_cols] = df_filtered[numeric_cols].round(3)

# Definisci le metriche da plottare
metrics = ["Precision_mean", "Recall_mean", "Density_mean", "Coverage_mean", "mDCR_mean"]

# Mappa dei nomi delle metriche in LaTeX come richiesto (adattata per mathtext di matplotlib)
metric_names_map = {
    "Precision_mean": r"$\mathrm{Prec}\uparrow$",
    "Recall_mean": r"$\mathrm{Rec}\uparrow$",
    "Density_mean": r"$\mathrm{Dens}\uparrow$",
    "Coverage_mean": r"$\mathrm{Cov}\uparrow$",
    "mDCR_mean": r"$\mathrm{mDCR}$"
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
    'conf':   '#56B4E9',  # azzurro
    'distr':  '#0072B2',  # blu scuro
    'dtree':  '#009E73',  # verde
    'ens':    '#F0E442',  # giallo
    'hier':   '#E78AC3',  # rosa
    'react':  '#D55E00',  # rosso mattone
    'schema': '#7B1B02',  # granata
    'self':   '#999999',  # grigio
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

# Crea una figura con 1 riga e 5 colonne
fig, axes = plt.subplots(1, 5, figsize=(13, 2), sharex=True)
fig.subplots_adjust(wspace=0.04)
plt.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.30, wspace=0.04)

axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

# Modifica lo spessore dei bordi per tutti i subplot
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)  # Puoi cambiare il valore (es. 0.5, 1, 2, 3, etc.)

# Per ogni metrica, crea un subplot
for i, metric in enumerate(metrics):
    ax = axes_flat[i]
    
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
    
    # Barre SyRFD - distanziate per gruppo
    # φ2 group (indici 0,1)
    x_syrfd_phi2 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        ax.bar(x_syrfd_phi2[j], syrfd_values[j], bar_width, color=syrfd_colors[j], 
               edgecolor='black', linewidth=0.3, hatch=syrfd_hatches[j])
    group_ticks.append(np.mean(x_syrfd_phi2))
    group_labels.append(r'SyRFDRes $\phi 2$')
    current_pos += 2 * bar_step + group_spacing
    
    # φ4 group (indici 2,3)
    x_syrfd_phi4 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        idx = j + 2
        ax.bar(x_syrfd_phi4[j], syrfd_values[idx], bar_width, color=syrfd_colors[idx], 
               edgecolor='black', linewidth=0.3, hatch=syrfd_hatches[idx])
    group_ticks.append(np.mean(x_syrfd_phi4))
    group_labels.append(r'SyRFDRes $\phi 4$')
    current_pos += 2 * bar_step + group_spacing
    
    # φ8 group (indici 4,5)
    x_syrfd_phi8 = current_pos + np.arange(2) * bar_step
    for j in range(2):
        idx = j + 4
        ax.bar(x_syrfd_phi8[j], syrfd_values[idx], bar_width, color=syrfd_colors[idx], 
               edgecolor='black', linewidth=0.3, hatch=syrfd_hatches[idx])
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
        '''
        if i == 8:  # Penultimo subplot (DBI_div_mean)
            bg_color = '#D26466'  # Rosso chiaro
            alpha = 0.9
        else:
        '''

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
        
        # Prepara i dati per le 8 modalità
        group_values = []
        group_modes = []
        for mode in llm_modes_order:
            mode_label = f"{group}/{mode}"
            mode_row = group_data[group_data['Label'] == mode_label]
            if len(mode_row) > 0:
                group_values.append(mode_row.iloc[0][metric])
                group_modes.append(mode)
            else:
                group_values.append(0)  # placeholder se manca
                group_modes.append(mode)
        
        # Barre per questo gruppo
        x_group = current_pos + np.arange(len(group_modes)) * bar_step
        colors_group = [color_map_llm.get(mode, '#999999') for mode in group_modes]
        
        bars_group = ax.bar(x_group, group_values, bar_width, color=colors_group,
                           edgecolor='black', linewidth=0.3)
        
        # Ticks per gruppo LLM
        group_ticks.append(np.mean(x_group))
        group_labels.append(group_names_map.get(group, group))
        
        all_x_positions.extend(x_group)
        current_pos += len(group_modes) * bar_step + 0.3  # ripristina lo spazio tra gruppi
    
        # ============ 3. GRUPPO LLM-zs ============
    if len(df_llm_zs) > 0:
        zs_labels = df_llm_zs['Label'].tolist()
        zs_values = [df_llm_zs[df_llm_zs['Label'] == label].iloc[0][metric] for label in zs_labels]
        
        # Distanzia le due barre di LLM-zs
        zs_step = 0.65  # Spazio per evitare l'overlap delle etichette DeepSeek e LLaMA
        x_zs = current_pos + np.arange(len(zs_labels)) * zs_step
        
        # Colori per LLM-zs
        colors_zs = ['#8E44AD', '#2C3E50']  # viola e grigio scuro
        bars_zs = ax.bar(x_zs, zs_values, bar_width, color=colors_zs[:len(zs_labels)],
                        edgecolor='black', linewidth=0.3)
        
        # Ticks per LLM-zs - ogni barra ha il suo tick
        for idx, label in enumerate(zs_labels):
            group_ticks.append(x_zs[idx])
            group_labels.append(label)
            
        all_x_positions.extend(x_zs)
        current_pos += len(zs_labels) * (bar_step + 0.3) + group_spacing + 0.5
    
    # Rimuovi lo spazio bianco a destra e a sinistra impostando i limiti dell'asse x
    ax.set_xlim(min(all_x_positions) - bar_width, max(all_x_positions) + bar_width)

    # ============ IMPOSTA YLIM SPECIFICI PER OGNI METRICA ============
    ylim_bottom_map = {
        "Precision_mean": 0.7,
        "Recall_mean": 0.9,
        "Density_mean": 0.5,
        "Coverage_mean": 0.9,
        "mDCR_mean": 0.35,
    }
    ylim_top_map = {
        "Precision_mean": 0.91,
        "Recall_mean": 1.08,
        "Density_mean": 1.01,
        "Coverage_mean": 1.05,
        "mDCR_mean": 0.8,
    }
    if metric in ylim_bottom_map:
        ax.set_ylim(bottom=ylim_bottom_map[metric], top=ylim_top_map[metric])

    # Imposta etichette dell'asse x per tutti i subplot (essendo 1 sola riga)
    ax.set_xticks(group_ticks)
    ax.set_xticklabels(group_labels, rotation=90, ha='center', fontsize=8)
    ax.tick_params(axis='x', which='major', pad=1)

    # Imposta y-label con i nomi LaTeX richiesti
    ax.set_ylabel(metric_names_map[metric], fontsize=10, labelpad=1)
    
    # Griglia
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    # Dopo aver creato il plot
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_ha('center')
        label.set_va('center')    


# Legenda esterna centrale con shadow=True posizionata in alto
syrfd_legend = [
    Patch(facecolor='#FED8B1', edgecolor='black', label=r'SyRFDRes$\phi 2$'),
    Patch(facecolor='#FDB462', edgecolor='black', label=r'SyRFDRes$\phi 4$'),
    Patch(facecolor='#E67E22', edgecolor='black', label=r'SyRFDRes$\phi 8$'),
    Patch(facecolor='#F39C12', edgecolor='black', hatch='//', label=r'SyRFDRes rlx')
]

llm_legend2 = [
    Patch(facecolor='#FF6B6B', edgecolor='black', label='Confidence CoT'),
    Patch(facecolor='#4ECDC4', edgecolor='black', label='Distribution Guidance'),
    Patch(facecolor='#45B7D1', edgecolor='black', label='Decision-Tree CoT'),
    Patch(facecolor='#96CEB4', edgecolor='black', label='Ensemble CoT'),
    Patch(facecolor='#FFEAA7', edgecolor='black', label='Hierarchical CoT'),
    Patch(facecolor='#DDA0DD', edgecolor='black', label='ReAct CoT'),
    Patch(facecolor='#FF8C94', edgecolor='black', label='Schema Constraints'),
    Patch(facecolor='#A8D8EA', edgecolor='black', label='Self-Consistency CoT')
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
# Legenda in alto
#fig.legend(handles=all_legend, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=7, shadow=True, fontsize=8)

# Salva il plot
plt.tight_layout(pad=0.2)
plt.savefig('confronto_metriche_diversity_2_5_v3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('confronto_metriche_diversity_v3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('confronto_metriche_diversity.pdf', dpi=300, bbox_inches='tight')
import os
os.makedirs('supplementary_material/plots', exist_ok=True)
plt.savefig('supplementary_material/plots/confronto_metriche_diversity_2_5_v3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('supplementary_material/plots/confronto_metriche_diversity_v3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('supplementary_material/plots/confronto_metriche_diversity.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Stampa un riepilogo dei dati
print("Riepilogo dati:")
print(f"SyRFD: {len(df_syrfd)} varianti")
print(f"LLM: {len(df_llm)} configurazioni")
print(f"LLM-zs: {len(df_llm_zs)} modelli")
print(f"\nTotale: {len(df_filtered)} righe")

print("\nGruppi LLM presenti:", sorted(df_llm['Label'].str.split('/').str[0].unique()))
print("Modalità LLM:", llm_modes_order)