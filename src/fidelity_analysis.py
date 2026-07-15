import pandas as pd
import glob
import os
from pathlib import Path
import numpy as np

def clean_numeric_value(val):
    """Pulisce e converte un valore in float se possibile"""
    if pd.isna(val) or val == '' or val == 'None':
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Rimuovi spazi e sostituisci virgola con punto (per decimali)
        val = val.strip().replace(',', '.')
        # Rimuovi eventuali caratteri non numerici tranne punto e segno meno
        val = ''.join(c for c in val if c.isdigit() or c in '.-')
        if val == '' or val == '-':
            return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan
    return np.nan

def process_fidelity_files():
    # Trova tutti i file CSV nella directory
    file_pattern = "./evaluation_metrics/fidelity/fidelity_*.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"Nessun file trovato con pattern: {file_pattern}")
        return None, None
    
    # Lista per memorizzare tutti i DataFrame
    all_dfs = []
    
    # Leggi ogni file
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Applica pulizia a tutte le colonne tranne 'Algoritmo'
            for col in df.columns:
                if col != 'Algoritmo':
                    df[col] = df[col].apply(clean_numeric_value)
            
            all_dfs.append(df)
            print(f"Letto: {Path(file).name} - {len(df)} righe")
        except Exception as e:
            print(f"Errore leggendo {file}: {e}")
    
    if not all_dfs:
        print("Nessun dato letto!")
        return None, None
    
    # Combina tutti i DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nTotale righe combinate: {len(combined_df)}")
    print(f"Strategie uniche: {sorted(combined_df['Algoritmo'].unique())}")
    
    # Colonne delle metriche (escludi 'Algoritmo')
    metric_columns = [col for col in combined_df.columns if col != 'Algoritmo']
    
    # Raggruppa per Algoritmo e calcola media e std
    results = []
    
    for algo in combined_df['Algoritmo'].unique():
        algo_data = combined_df[combined_df['Algoritmo'] == algo]
        row = {'Algoritmo': algo}
        
        for col in metric_columns:
            values = algo_data[col].dropna()
            if len(values) > 0:
                row[f'{col}_mean'] = values.mean()
                row[f'{col}_std'] = values.std()
            else:
                row[f'{col}_mean'] = float('nan')
                row[f'{col}_std'] = float('nan')
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Ordina per Algoritmo
    results_df = results_df.sort_values('Algoritmo')
    
    return results_df, metric_columns

def generate_latex_table(results_df, metric_columns):
    """Genera tabella LaTeX con media ± devstd"""
    
    # Nomi più leggibili per le metriche
    metric_names = {
        'KL_mean': 'KL',
        'JS_mean': 'JS',
        'Q_multi_attr_sim': 'Q',
        'KS_mean': 'KS',
        'Chi2_mean': 'χ²',
        'WD_mean': 'WD',
        'pMSE': 'pMSE',
        'MAD_mean': 'MAD',
        'MMD': 'MMD',
        'PCA_WD_mean': 'PCA-WD',
        'Moment_mean_diff': 'Δμ',
        'Moment_var_diff': 'Δσ²',
        'Moment_skew_diff': 'Δγ',
        'Moment_kurt_diff': 'Δκ'
    }
    
    # Nomi abbreviati per gli algoritmi
    algo_names = {
        'GOGGLE': 'GOGGLE',
        'SMOTE': 'SMOTE',
        'SMOTECDNN': 'SMOTE-CDNN',
        'SYRFD_thr2': 'SYRFD-thr2',
        'SYRFD_thr2_relaxed': 'SYRFD-thr2-rel',
        'SYRFD_thr2_strict': 'SYRFD-thr2-str',
        'SYRFD_thr4': 'SYRFD-thr4',
        'SYRFD_thr4_relaxed': 'SYRFD-thr4-rel',
        'SYRFD_thr4_strict': 'SYRFD-thr4-str',
        'SYRFD_thr8': 'SYRFD-thr8',
        'SYRFD_thr8_relaxed': 'SYRFD-thr8-rel',
        'SYRFD_thr8_strict': 'SYRFD-thr8-str',
        'casTGAN': 'casTGAN',
        'ctabgan': 'CTAB-GAN',
        'ctabganp': 'CTAB-GAN+',
        'ddpm': 'DDPM',
        'deepseek': 'DeepSeek',
        'devstral-small-2_24b-cloud': 'Devstral-24B',
        'gemma3_12b': 'Gemma3-12B',
        'gemma4_31b-cloud': 'Gemma4-31B',
        'gpt-oss_120b-cloud': 'GPT-OSS-120B',
        'gpt-oss_20b-cloud': 'GPT-OSS-20B',
        'llama': 'Llama',
        'tabdiff': 'TabDiff',
        'tvae': 'TVAE'
    }
    
    # Costruisci la tabella
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l' + 'c' * len(metric_columns) + '}')
    lines.append('\\hline')
    
    # Intestazione
    header = ['\\textbf{Metodo}'] + [f'\\textbf{{{metric_names.get(m, m)}}}' for m in metric_columns]
    lines.append(' & '.join(header) + ' \\\\')
    lines.append('\\hline')
    
    # Righe dei dati
    for _, row in results_df.iterrows():
        algo = row['Algoritmo']
        algo_label = algo_names.get(algo, algo)
        
        cells = [algo_label]
        for col in metric_columns:
            mean_val = row[f'{col}_mean']
            std_val = row[f'{col}_std']
            
            if pd.isna(mean_val) or pd.isna(std_val):
                cells.append('-')
            else:
                # Formatta: media ± devstd
                cells.append(f'{mean_val:.3f}$\\pm${std_val:.3f}')
        
        lines.append(' & '.join(cells) + ' \\\\')
    
    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\caption{Confronto delle metriche di fedeltà per diversi algoritmi di generazione di dati tabulari.}')
    lines.append('\\label{tab:fidelity_comparison}')
    lines.append('\\end{table}')
    
    return '\n'.join(lines)

def save_results(results_df, metric_columns, output_csv="fidelity_summary.csv"):
    """Salva i risultati in un file CSV"""
    # Crea un DataFrame con media e std in colonne separate
    summary_data = []
    for _, row in results_df.iterrows():
        record = {'Algoritmo': row['Algoritmo']}
        for col in metric_columns:
            record[f'{col}_mean'] = row[f'{col}_mean']
            record[f'{col}_std'] = row[f'{col}_std']
        summary_data.append(record)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nRisultati salvati in {output_csv}")

def generate_simplified_latex(results_df, metric_columns):
    """Genera una tabella LaTeX più compatta come nell'immagine"""
    
    # Metriche da mostrare (in ordine)
    metrics_to_show = [
        'KL_mean', 'JS_mean', 'Q_multi_attr_sim', 'KS_mean', 
        'WD_mean', 'pMSE', 'MAD_mean', 'MMD'
    ]
    
    # Filtra solo le metriche disponibili
    available_metrics = [m for m in metrics_to_show if m in metric_columns]
    
    if not available_metrics:
        available_metrics = metric_columns[:8]  # fallback
    
    metric_names = {
        'KL_mean': 'KL',
        'JS_mean': 'JS',
        'Q_multi_attr_sim': 'Q',
        'KS_mean': 'KS',
        'WD_mean': 'WD',
        'pMSE': 'pMSE',
        'MAD_mean': 'MAD',
        'MMD': 'MMD'
    }
    
    # Seleziona solo alcuni algoritmi chiave come nell'immagine
    key_algorithms = ['GOGGLE', 'SMOTE', 'deepseek', 'gemma3_12b', 
                      'gemma4_31b-cloud', 'gpt-oss_20b-cloud', 'tabdiff', 'tvae']
    
    algo_names = {
        'GOGGLE': 'GOGGLE',
        'SMOTE': 'SMOTE',
        'deepseek': 'DeepSeek',
        'gemma3_12b': 'Gemma3-12B',
        'gemma4_31b-cloud': 'Gemma4-31B',
        'gpt-oss_20b-cloud': 'GPT-OSS',
        'tabdiff': 'TabDiff',
        'tvae': 'TVAE'
    }
    
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l' + 'c' * len(available_metrics) + '}')
    lines.append('\\hline')
    
    # Intestazione con due righe (come nell'immagine)
    header1 = ['\\textbf{Metodo}'] + ['' for _ in available_metrics]
    lines.append(' & '.join(header1) + ' \\\\')
    
    header2 = [''] + [f'\\textbf{{{metric_names.get(m, m)}}}' for m in available_metrics]
    lines.append(' & '.join(header2) + ' \\\\')
    lines.append('\\hline')
    
    # Filtra i dati per gli algoritmi chiave
    filtered_df = results_df[results_df['Algoritmo'].isin(key_algorithms)]
    filtered_df = filtered_df.set_index('Algoritmo').reindex(key_algorithms).reset_index()
    
    for _, row in filtered_df.iterrows():
        algo = row['Algoritmo']
        if pd.isna(algo):
            continue
        algo_label = algo_names.get(algo, algo)
        
        cells = [algo_label]
        for col in available_metrics:
            mean_val = row[f'{col}_mean']
            if pd.isna(mean_val):
                cells.append('-')
            else:
                cells.append(f'{mean_val:.3f}')
        
        lines.append(' & '.join(cells) + ' \\\\')
    
    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\caption{Confronto delle metriche di fedeltà.}')
    lines.append('\\label{tab:fidelity_simple}')
    lines.append('\\end{table}')
    
    return '\n'.join(lines)

def main():
    # Processa i file
    results_df, metric_columns = process_fidelity_files()
    
    if results_df is None or len(results_df) == 0:
        print("Nessun risultato ottenuto!")
        return
    
    # Salva i risultati
    save_results(results_df, metric_columns)
    
    # Genera tabella LaTeX completa
    latex_table = generate_latex_table(results_df, metric_columns)
    with open("fidelity_table_full.tex", "w") as f:
        f.write(latex_table)
    print("\nTabella LaTeX completa generata in fidelity_table_full.tex")
    
    # Genera tabella LaTeX semplificata (simile all'immagine)
    latex_simple = generate_simplified_latex(results_df, metric_columns)
    with open("fidelity_table_simple.tex", "w") as f:
        f.write(latex_simple)
    print("Tabella LaTeX semplificata generata in fidelity_table_simple.tex")
    
    # Mostra un'anteprima dei risultati
    print("\n" + "="*80)
    print("ANTEPRIMA DEI RISULTATI (medie per algoritmo):")
    print("="*80)
    
    # Mostra prime 5 metriche
    display_cols = ['Algoritmo'] + [f'{col}_mean' for col in metric_columns[:5]]
    print(results_df[display_cols].round(3).to_string(index=False))
    
    # Mostra statistiche
    print("\n" + "="*80)
    print("STATISTICHE:")
    print("="*80)
    print(f"Numero di algoritmi: {len(results_df)}")
    print(f"Numero di metriche: {len(metric_columns)}")
    print(f"Metriche: {', '.join(metric_columns)}")

if __name__ == "__main__":
    main()