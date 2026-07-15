import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

# Define paths
DIVERSITY_DIR = "./evaluation_metrics/diversity"
FIDELITY_DIR = "./evaluation_metrics/fidelity"
OUTPUT_DIR = "./evaluation_metrics/aggregated"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define algorithm name mapping (short names)
ALGORITHM_NAMES = {
    'SMOTE': 'SMOTE',
    'SMOTECDNN': 'SMOTE-CDNN',
    'tvae': 'TVAE',
    'tabdiff': 'TabDiff',
    'ctabgan': 'CTAB-GAN',
    'ctabganp': 'CTAB-GAN+',
    'ddpm': 'DDPM',
    'casTGAN': 'CasTGAN',
    'GOGGLE': 'GOGGLE',
    'deepseek': 'DeepSeek',
    'llama': 'LLaMA',
    'SYRFD_thr2_relaxed': 'SYRFD 2/rlx',
    'SYRFD_thr2_strict': 'SYRFD 2/str',
    'SYRFD_thr4_relaxed': 'SYRFD 4/rlx',
    'SYRFD_thr4_strict': 'SYRFD 4/str',
    'SYRFD_thr8_relaxed': 'SYRFD 8/rlx',
    'SYRFD_thr8_strict': 'SYRFD 8/str',
}

# LLM name mapping
LLM_NAME_MAPPING = {
    'devstral-small-2_24b-cloud': 'Dev',
    'gemma3_12b': 'G3',
    'gemma4_31b-cloud': 'G4',
    'gpt-oss_120b-cloud': 'GPT120',
    'gpt-oss_20b-cloud': 'GPT20'
}

CONFIG_SHORT_NAMES = {
    'confidence': 'conf',
    'decision_tree': 'dtree',
    'distribution_guidance': 'distr',
    'ensemble': 'ens',
    'hierarchical': 'hier',
    'react': 'react',
    'schema_constraints': 'schema',
    'self_consistency': 'self',
    'cot_confidence': 'conf',
    'cot_decision_tree': 'dtree',
    'cot_ensemble': 'ens',
    'cot_hierarchical': 'hier',
    'cot_react': 'react',
    'cot_self_consistency': 'self'
}

# Define state-of-the-art algorithms (include strict and relaxed versions)
SOTA_ALGORITHMS = [
    'GOGGLE', 'SMOTE', 'SMOTECDNN',
    'SYRFD_thr2_relaxed', 'SYRFD_thr2_strict',
    'SYRFD_thr4_relaxed', 'SYRFD_thr4_strict',
    'SYRFD_thr8_relaxed', 'SYRFD_thr8_strict',
    'casTGAN', 'ctabgan', 'ctabganp', 'ddpm', 'deepseek', 
    'llama', 'tabdiff', 'tvae'
]

# LLM algorithms with their configurations
LLM_CONFIGS = {
    'devstral-small-2_24b-cloud': [
        'confidence', 'decision_tree', 'distribution_guidance',
        'ensemble', 'hierarchical', 'react', 'schema_constraints', 'self_consistency'
    ],
    'gemma3_12b': [
        'cot_confidence', 'cot_decision_tree', 'cot_ensemble',
        'cot_hierarchical', 'cot_react', 'cot_self_consistency',
        'distribution_guidance', 'schema_constraints'
    ],
    'gemma4_31b-cloud': [
        'cot_confidence', 'cot_decision_tree', 'cot_ensemble',
        'cot_hierarchical', 'cot_react', 'cot_self_consistency',
        'distribution_guidance', 'schema_constraints'
    ],
    'gpt-oss_120b-cloud': [
        'confidence', 'decision_tree', 'distribution_guidance',
        'ensemble', 'hierarchical', 'react', 'schema_constraints', 'self_consistency'
    ],
    'gpt-oss_20b-cloud': [
        'confidence', 'decision_tree', 'distribution_guidance',
        'ensemble', 'hierarchical', 'react', 'schema_constraints', 'self_consistency'
    ]
}

# Define metric groups with LaTeX-friendly names
DIVERSITY_METRICS = {
    'Silhouette': 'Silhouette',
    'Davies-Bouldin': 'DBI$_{\\mathbb{R}\\cup\\mathbb{G}}$',
    'Compactness_cls_k': 'Comp',
    'Precision': 'Prec',
    'Recall': 'Rec',
    'Density': 'Dens',
    'Coverage': 'Cov',
    'mDCR': 'mDCR',
    'DBI_div': 'DBI$_{\\mathbb{G}}$',
    'Coverage_div': 'Cov$_q$'
}

FIDELITY_METRICS = {
    'KL_mean': 'KL',
    'JS_mean': 'JS',
    'Q_multi_attr_sim': 'Q',
    'KS_mean': 'KS',
    'Chi2_mean': '$\\chi^2$',
    'WD_mean': 'WD',
    'pMSE': 'pMSE',
    'MAD_mean': 'MAD',
    'MMD': 'MMD',
    'PCA_WD_mean': 'WD$_{\\mathrm{PCA}}$',
    'Moment_mean_diff': '$\\Delta_\\mu$',
    'Moment_var_diff': '$\\Delta_v$',
    'Moment_skew_diff': '$\\Delta_{\\mathrm{skew}}$',
    'Moment_kurt_diff': '$\\Delta_{\\mathrm{kurt}}$'
}

# Order for fidelity metrics (to maintain consistency)
FIDELITY_ORDER = ['KL_mean', 'JS_mean', 'Q_multi_attr_sim', 'KS_mean', 
                  'Chi2_mean', 'WD_mean', 'pMSE', 'MAD_mean', 'MMD', 
                  'PCA_WD_mean', 'Moment_mean_diff', 'Moment_var_diff', 
                  'Moment_skew_diff', 'Moment_kurt_diff']

def get_best_config_for_llm(llm_df, metric_type='diversity'):
    """Find the best configuration for an LLM"""
    if llm_df.empty:
        return None
    
    if metric_type == 'diversity' and 'Silhouette' in llm_df.columns:
        # For diversity: maximize Silhouette (or minimize Davies-Bouldin if Silhouette is not available)
        if 'Silhouette' in llm_df.columns and not llm_df['Silhouette'].isna().all():
            best_idx = llm_df['Silhouette'].idxmax()
        else:
            # Use Davies-Bouldin as fallback (minimize)
            best_idx = llm_df['Davies-Bouldin'].idxmin()
    elif metric_type == 'fidelity' and 'JS_mean' in llm_df.columns:
        # For fidelity: minimize JS_mean
        best_idx = llm_df['JS_mean'].idxmin()
    else:
        # If no selection criterion, take first
        best_idx = llm_df.index[0]
    
    return llm_df.loc[best_idx]

def get_short_algorithm_name(row, metric_type='diversity'):
    """Get short name for algorithm based on mapping"""
    algorithm = row['Algoritmo']
    file_name = row['File']
    
    # Check if it's an LLM with configuration
    for llm, short_name in LLM_NAME_MAPPING.items():
        if algorithm.startswith(llm):
            # Extract configuration from file name
            for config, short_config in CONFIG_SHORT_NAMES.items():
                if config in file_name:
                    return f"{short_name}/{short_config}"
            # If no config found, return just the short name
            return short_name
    
    # Check SOTA algorithms
    if algorithm in ALGORITHM_NAMES:
        return ALGORITHM_NAMES[algorithm]
    
    return algorithm

def load_and_aggregate_data(directory, metric_type='diversity'):
    """Load all CSV files and aggregate data"""
    files = glob.glob(os.path.join(directory, "*.csv"))
    all_data = []
    
    for file in files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Separate SOTA and LLM
    sota_df = combined_df[combined_df['Algoritmo'].isin(SOTA_ALGORITHMS)]
    
    # For LLMs, find best configuration per algorithm
    llm_data = []
    for llm in LLM_CONFIGS.keys():
        llm_df = combined_df[combined_df['Algoritmo'] == llm]
        if not llm_df.empty:
            best_config = get_best_config_for_llm(llm_df, metric_type)
            if best_config is not None:
                llm_data.append(best_config)
    
    if llm_data:
        llm_df = pd.DataFrame(llm_data)
        combined_filtered = pd.concat([sota_df, llm_df], ignore_index=True)
    else:
        combined_filtered = sota_df
    
    return combined_filtered

def create_aggregated_table(df, metric_columns, agg_func='mean', metric_type='diversity'):
    """Create aggregated table with mean or median values"""
    if agg_func == 'mean':
        table = df.groupby('Algoritmo')[metric_columns].mean()
    else:  # median
        table = df.groupby('Algoritmo')[metric_columns].median()
    
    # Round to 4 decimal places
    table = table.round(4)

    # Rename index using short names (with configs for LLMs)
    new_index = []
    for idx in table.index:
        # Find the corresponding row in the original df to get file name
        row_data = df[df['Algoritmo'] == idx]
        if not row_data.empty:
            # Use the first occurrence to get file name
            row = row_data.iloc[0]
            short_name = get_short_algorithm_name(row, metric_type)
            new_index.append(short_name)
        else:
            new_index.append(idx)
    
    table.index = new_index
    return table

def table_to_latex(df, title, caption, label, metric_mapping):
    """Convert DataFrame to LaTeX table with nice formatting"""
    # Rename columns using metric mapping
    df_renamed = df.copy()
    new_columns = []
    for col in df_renamed.columns:
        if col in metric_mapping:
            new_columns.append(metric_mapping[col])
        else:
            new_columns.append(col)
    df_renamed.columns = new_columns
    
    # Sort index for better readability
    df_renamed = df_renamed.sort_index()
    
    # Generate LaTeX table
    latex_str = f"\\begin{{table}}[htbp]\n"
    latex_str += f"\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{tab:{label}}}\n"
    latex_str += f"\\resizebox{{\\textwidth}}{{!}}{{\n"
    latex_str += f"\\begin{{tabular}}{{l" + "c"*len(df_renamed.columns) + "}}\n"
    latex_str += f"\\toprule\n"
    
    # Column headers
    latex_str += f"Algorithm & " + " & ".join(df_renamed.columns) + " \\\\\n"
    latex_str += f"\\midrule\n"
    
    # Data rows
    for idx, row in df_renamed.iterrows():
        row_str = f"{idx} & " + " & ".join([f"{val:.4f}" for val in row]) + " \\\\\n"
        latex_str += row_str
    
    latex_str += f"\\bottomrule\n"
    latex_str += f"\\end{{tabular}}\n"
    latex_str += f"}}\n"
    latex_str += f"\\end{{table}}\n\n"
    
    return latex_str

def create_metric_barplots(df, metrics, metric_type, output_dir):
    """Create grouped barplots for each metric"""
    # Melt the dataframe for easier plotting
    df_long = df.melt(id_vars=['Algoritmo', 'File'], 
                      value_vars=metrics,
                      var_name='Metric', value_name='Value')
    
    for metric in metrics:
        metric_data = df_long[df_long['Metric'] == metric]
        
        # Pivot to get algorithms as columns and files as index
        pivot_data = metric_data.pivot_table(index='File', columns='Algoritmo', values='Value')
        
        # Rename columns using short names with configs
        new_columns = []
        for col in pivot_data.columns:
            row_data = df[df['Algoritmo'] == col]
            if not row_data.empty:
                row = row_data.iloc[0]
                short_name = get_short_algorithm_name(row, metric_type)
                new_columns.append(short_name)
            else:
                new_columns.append(col)
        
        pivot_data.columns = new_columns
        
        # Sort columns for better readability
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        
        # Create barplot
        fig, ax = plt.subplots(figsize=(16, 7))
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric_type} Metric: {metric}', fontsize=14)
        ax.set_xlabel('Dataset/File', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_type.lower()}_{metric}_barplot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_type.lower()}_{metric}_barplot.png")

print("Loading diversity data...")
diversity_df = load_and_aggregate_data(DIVERSITY_DIR, 'diversity')

print("\nLoading fidelity data...")
fidelity_df = load_and_aggregate_data(FIDELITY_DIR, 'fidelity')

# Determine fidelity metrics from the data
available_fidelity_metrics = [col for col in fidelity_df.columns if col not in ['Algoritmo', 'File']]
fidelity_metrics = [col for col in FIDELITY_ORDER if col in available_fidelity_metrics]
print(f"Fidelity metrics found: {fidelity_metrics}")

# Get actual diversity metrics available
available_diversity_metrics = [col for col in diversity_df.columns if col not in ['Algoritmo', 'File']]
diversity_metrics = [col for col in DIVERSITY_METRICS.keys() if col in available_diversity_metrics]
print(f"Diversity metrics found: {diversity_metrics}")

# Create tables
print("\n" + "="*100)
print("DIVERSITY METRICS - MEAN")
print("="*100)
diversity_mean = create_aggregated_table(diversity_df, diversity_metrics, 'mean', 'diversity')
print(diversity_mean)

print("\n" + "="*100)
print("DIVERSITY METRICS - MEDIAN")
print("="*100)
diversity_median = create_aggregated_table(diversity_df, diversity_metrics, 'median', 'diversity')
print(diversity_median)

print("\n" + "="*100)
print("FIDELITY METRICS - MEAN")
print("="*100)
fidelity_mean = create_aggregated_table(fidelity_df, fidelity_metrics, 'mean', 'fidelity')
print(fidelity_mean)

print("\n" + "="*100)
print("FIDELITY METRICS - MEDIAN")
print("="*100)
fidelity_median = create_aggregated_table(fidelity_df, fidelity_metrics, 'median', 'fidelity')
print(fidelity_median)

# Save tables to CSV
diversity_mean.to_csv(os.path.join(OUTPUT_DIR, 'diversity_mean_table.csv'))
diversity_median.to_csv(os.path.join(OUTPUT_DIR, 'diversity_median_table.csv'))
fidelity_mean.to_csv(os.path.join(OUTPUT_DIR, 'fidelity_mean_table.csv'))
fidelity_median.to_csv(os.path.join(OUTPUT_DIR, 'fidelity_median_table.csv'))

# Generate LaTeX tables
print("\nGenerating LaTeX tables...")
latex_tables = []

# Diversity Mean table
latex_tables.append(table_to_latex(
    diversity_mean, 
    "Diversity Metrics - Mean Values", 
    "Mean values of diversity metrics across all datasets.",
    "diversity_mean",
    DIVERSITY_METRICS
))

# Diversity Median table
latex_tables.append(table_to_latex(
    diversity_median, 
    "Diversity Metrics - Median Values", 
    "Median values of diversity metrics across all datasets.",
    "diversity_median",
    DIVERSITY_METRICS
))

# Fidelity Mean table
latex_tables.append(table_to_latex(
    fidelity_mean, 
    "Fidelity Metrics - Mean Values", 
    "Mean values of fidelity metrics across all datasets.",
    "fidelity_mean",
    {k: v for k, v in FIDELITY_METRICS.items() if k in fidelity_metrics}
))

# Fidelity Median table
latex_tables.append(table_to_latex(
    fidelity_median, 
    "Fidelity Metrics - Median Values", 
    "Median values of fidelity metrics across all datasets.",
    "fidelity_median",
    {k: v for k, v in FIDELITY_METRICS.items() if k in fidelity_metrics}
))

# Save all LaTeX tables to a single file
latex_file = os.path.join(OUTPUT_DIR, 'tables.tex')
with open(latex_file, 'w') as f:
    f.write("% LaTeX tables for aggregated metrics\n")
    f.write("% Generated by evaluation_metrics_aggregator.py\n\n")
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{booktabs}\n")
    f.write("\\usepackage{geometry}\n")
    f.write("\\usepackage{graphicx}\n")
    f.write("\\geometry{a4paper, margin=0.5in}\n")
    f.write("\\begin{document}\n\n")
    
    for table in latex_tables:
        f.write(table)
    
    f.write("\\end{document}\n")

print(f"LaTeX tables saved to {latex_file}")

# Create barplots
print("\nCreating diversity barplots...")
create_metric_barplots(diversity_df, diversity_metrics, 'Diversity', OUTPUT_DIR)

print("\nCreating fidelity barplots...")
create_metric_barplots(fidelity_df, fidelity_metrics, 'Fidelity', OUTPUT_DIR)

print(f"\nAll tables and plots saved to {OUTPUT_DIR}!")

# Print summary of algorithms included
print("\n" + "="*100)
print("ALGORITHMS INCLUDED IN TABLES")
print("="*100)
print("\nSOTA Algorithms (with short names):")
for algo in sorted(SOTA_ALGORITHMS):
    # Find corresponding row to get short name
    row_data = diversity_df[diversity_df['Algoritmo'] == algo]
    if not row_data.empty:
        row = row_data.iloc[0]
        short_name = get_short_algorithm_name(row, 'diversity')
        print(f"  - {algo} -> {short_name}")

print("\nLLM Algorithms (best configurations):")
for llm in LLM_CONFIGS.keys():
    llm_data = diversity_df[diversity_df['Algoritmo'] == llm]
    if not llm_data.empty:
        best_config = get_best_config_for_llm(llm_data, 'diversity')
        if best_config is not None:
            short_name = get_short_algorithm_name(best_config, 'diversity')
            print(f"  - {llm} -> {short_name}")