#!/usr/bin/env python3
"""
generate_rebuttal_artifacts_runtime_comparison.py
=============================================================================
Publication-ready runtime & memory comparison plots and tables for rebuttal.
Compares State-of-the-Art (SOTA) data augmentation algorithms with SyRFDRes
(both SyRFDRes-Relaxed and SyRFDRes-Strict).

Designed specifically for tight paper space constraints (SIGMOD / VLDB / ACM / IEEE):
- Ultra-compact dimensions (fits in single column or tight span)
- (a) Execution Time (mean ± std) plot with non-overlapping error bars
- (b) Peak Memory Usage (mean ± std) plot
- Combined 1x2 multi-panel figure + standalone separate figures
- Standalone clean legend export (PDF & PNG)
- Both Log-scale (recommended for wide dynamic range 0.3s - 1000s) and Linear-scale
- LaTeX and CSV tables exported to paper_rebuttal_artifacts/tables/
=============================================================================
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# =============================================================================
from pathlib import Path

# Resolve project root dynamically
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'plotting_scripts' else Path('.').resolve()

OUTPUT_DIR = str(REPO_ROOT / 'supplementary_material')
PLOTS_DIR  = os.path.join(OUTPUT_DIR, 'plots', 'paper_rebuttal')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Publication typography & styling matching generate_rebuttal_artifacts.py
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':        8.0,
    'axes.labelsize':   8.0,
    'axes.titlesize':   8.5,
    'xtick.labelsize':  7.0,
    'ytick.labelsize':  7.0,
    'legend.fontsize':  7.0,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
})

# Color palette: SyRFDRes colors match the paper rebuttal palette; SOTA in elegant slate
COLOR_RELAXED   = '#44A5C2'  # Paper SyRFDRes Relaxed
COLOR_STRICT    = '#ffaf4a'  # Paper SyRFDRes Strict
COLOR_SOTA_DL   = '#5d6d7e'  # Deep Learning / GAN baselines
COLOR_SOTA_SM   = '#85929e'  # SMOTE-family baselines

NAME_MAPPING = {
    'castgan':      'CasTGAN',
    'ctabgan':      'CTAB-GAN',
    'ctabgan-plus': 'CTAB-GAN+',
    'cuts':         'CUTS',
    'ddpm':         'TabDDPM',
    'fair-smote':   'Fair-SMOTE',
    'goggle':       'GOGGLE',
    'prefair':      'PreFair',
    'smote':        'SMOTE',
    'smote-cdnn':   'SMOTE-CDNN',
    'tabfairgan':   'TabFairGAN',
    'tabular-argn': 'Tabular-ARGN',
    'tvae':         'TVAE',
    'syrfd_relaxed':'SyRFDRes (Relaxed)',
    'syrfd_strict': 'SyRFDRes (Strict)',
}


def load_sota_data(summary_path, raw_fallback_path=None, use_raw_directly=False):
    """
    Load SOTA benchmark data.
    If summary_path has NaN std, enriches std from raw multi-dataset benchmark.
    If use_raw_directly is True, aggregates directly from raw_fallback_path.
    """
    if use_raw_directly and raw_fallback_path and os.path.exists(raw_fallback_path):
        df_raw = pd.read_csv(raw_fallback_path)
        succ = df_raw[df_raw['status'] == 'SUCCESS']
        raw_stats = succ.groupby('algorithm').agg(
            time_mean=('total_time_s', 'mean'),
            time_std=('total_time_s', 'std'),
            mem_mean=('total_mem_peak_mb', 'mean'),
            mem_std=('total_mem_peak_mb', 'std')
        ).reset_index()
        records = []
        for _, r in raw_stats.iterrows():
            algo = r['algorithm']
            records.append({
                'algorithm_raw': algo,
                'algorithm': NAME_MAPPING.get(algo, algo),
                'time_mean': float(r['time_mean']),
                'time_std': float(r['time_std']) if not pd.isna(r['time_std']) else 0.0,
                'mem_mean': float(r['mem_mean']),
                'mem_std': float(r['mem_std']) if not pd.isna(r['mem_std']) else 0.0,
                'category': 'SOTA Baseline',
            })
        print(f"  [i] Aggregated SOTA metrics directly from raw benchmark ({len(records)} algorithms).")
        return pd.DataFrame(records)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"SOTA summary file not found: {summary_path}")

    df_sum = pd.read_csv(summary_path)

    # Check for raw fallback to populate missing std values
    raw_stats = None
    if raw_fallback_path and os.path.exists(raw_fallback_path):
        try:
            df_raw = pd.read_csv(raw_fallback_path)
            succ = df_raw[df_raw['status'] == 'SUCCESS']
            raw_stats = succ.groupby('algorithm').agg(
                time_std_raw=('total_time_s', 'std'),
                mem_std_raw=('total_mem_peak_mb', 'std')
            ).reset_index()
        except Exception as e:
            print(f"Warning: could not process raw fallback file: {e}")

    records = []
    enriched_count = 0
    for _, row in df_sum.iterrows():
        algo = row['algorithm']
        time_mean = row.get('total_time_s_mean', np.nan)
        time_std  = row.get('total_time_s_std', np.nan)
        mem_mean  = row.get('total_mem_peak_mb_mean', np.nan)
        mem_std   = row.get('total_mem_peak_mb_std', np.nan)

        # If std is NaN or 0 in summary, check raw fallback
        if (pd.isna(time_std) or time_std == 0) and raw_stats is not None:
            match = raw_stats[raw_stats['algorithm'] == algo]
            if not match.empty:
                time_std = match['time_std_raw'].values[0]
                if pd.isna(mem_std) or mem_std == 0:
                    mem_std = match['mem_std_raw'].values[0]
                enriched_count += 1

        records.append({
            'algorithm_raw': algo,
            'algorithm': NAME_MAPPING.get(algo, algo),
            'time_mean': float(time_mean) if not pd.isna(time_mean) else 0.0,
            'time_std': float(time_std) if not pd.isna(time_std) else 0.0,
            'mem_mean': float(mem_mean) if not pd.isna(mem_mean) else 0.0,
            'mem_std': float(mem_std) if not pd.isna(mem_std) else 0.0,
            'category': 'SOTA Baseline',
        })

    if enriched_count > 0:
        print(f"  [i] Enriched standard deviations for {enriched_count} SOTA algorithms from multi-dataset benchmark.")

    return pd.DataFrame(records)


def load_syrfd_data(strict_dir=None, relaxed_dir=None, filter_datasets=None):
    """
    Load SyRFDRes data from ablation folders.
    Extracts L0_full completed baseline runs.
    """
    if strict_dir is None:
        strict_dir = str(REPO_ROOT / 'ablation/strict') if (REPO_ROOT / 'ablation/strict').exists() else 'ablation_strict'
    if relaxed_dir is None:
        relaxed_dir = str(REPO_ROOT / 'ablation/relaxed') if (REPO_ROOT / 'ablation/relaxed').exists() else 'ablation_relaxed'

    rel_files = glob.glob(os.path.join(relaxed_dir, 'results', '*.csv'))
    stt_files = glob.glob(os.path.join(strict_dir, 'results', '*.csv'))

    if not rel_files or not stt_files:
        raise FileNotFoundError(f"Could not find SyRFDRes results in ablation folders ({relaxed_dir}, {strict_dir}).")

    df_rel = pd.concat([pd.read_csv(f) for f in rel_files], ignore_index=True)
    df_stt = pd.concat([pd.read_csv(f) for f in stt_files], ignore_index=True)

    df_rel = df_rel[(df_rel['status'] == 'completed') & (df_rel['config_name'] == 'L0_full')]
    df_stt = df_stt[(df_stt['status'] == 'completed') & (df_stt['config_name'] == 'L0_full')]

    if filter_datasets is not None:
        df_rel = df_rel[df_rel['dataset'].isin(filter_datasets)]
        df_stt = df_stt[df_stt['dataset'].isin(filter_datasets)]

    records = [
        {
            'algorithm_raw': 'syrfd_relaxed',
            'algorithm': NAME_MAPPING['syrfd_relaxed'],
            'time_mean': float(df_rel['time_t_wall_total'].mean()),
            'time_std': float(df_rel['time_t_wall_total'].std()),
            'mem_mean': float(df_rel['peak_rss_mb'].mean()),
            'mem_std': float(df_rel['peak_rss_mb'].std()),
            'category': 'SyRFDRes (Ours)',
        },
        {
            'algorithm_raw': 'syrfd_strict',
            'algorithm': NAME_MAPPING['syrfd_strict'],
            'time_mean': float(df_stt['time_t_wall_total'].mean()),
            'time_std': float(df_stt['time_t_wall_total'].std()),
            'mem_mean': float(df_stt['peak_rss_mb'].mean()),
            'mem_std': float(df_stt['peak_rss_mb'].std()),
            'category': 'SyRFDRes (Ours)',
        }
    ]
    return pd.DataFrame(records)


def get_bar_colors(df):
    """Assign colors distinguishing SOTA baselines vs SyRFDRes variants."""
    colors = []
    for _, row in df.iterrows():
        raw = row['algorithm_raw']
        if raw == 'syrfd_relaxed':
            colors.append(COLOR_RELAXED)
        elif raw == 'syrfd_strict':
            colors.append(COLOR_STRICT)
        elif raw in ['smote', 'fair-smote', 'smote-cdnn', 'prefair']:
            colors.append(COLOR_SOTA_SM)
        else:
            colors.append(COLOR_SOTA_DL)
    return colors


def compute_xerr(values, stds, log_scale=True, min_bound=0.01):
    """
    Computes asymmetric error bars clipped to prevent negative bounds or zero on log scale.
    """
    v = np.array(values)
    s = np.nan_to_num(np.array(stds), nan=0.0)
    if log_scale:
        lower = np.minimum(np.maximum(0.0, v - min_bound), s)
    else:
        lower = np.minimum(v, s)
    upper = s
    return [lower, upper]


# =============================================================================
# EXPORT STANDALONE LEGEND
# =============================================================================
def save_standalone_legend(name='fig_runtime_legend'):
    """
    Exports a standalone tight legend PDF and PNG for paper flexibility.
    """
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_RELAXED, edgecolor='black', linewidth=0.4, label='SyRFDRes (Relaxed)'),
        mpatches.Patch(facecolor=COLOR_STRICT,  edgecolor='black', linewidth=0.4, label='SyRFDRes (Strict)'),
        #mpatches.Patch(facecolor=COLOR_SOTA_DL, edgecolor='black', linewidth=0.4, label='SOTA Baselines'),
    ]
    fig_leg, ax_leg = plt.subplots(figsize=(3.4, 0.45))
    ax_leg.axis('off')
    ax_leg.legend(handles=legend_handles, loc='center', ncol=3, frameon=True,
                  framealpha=0.95, edgecolor='gray', facecolor='white',
                  fontsize=7.2, handlelength=1.1, handletextpad=0.4, borderpad=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig_leg.savefig(os.path.join(PLOTS_DIR, f"{name}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig_leg)
    print(f"  ✓ Saved standalone legend: {name}.pdf/.png")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_horizontal_comparison(df, filename_prefix='fig_runtime_memory_comparison', log_scale=True, include_legend=True):
    """
    Ultra-compact horizontal bar plot (1x2 side-by-side).
    Horizontal bars are ideal for paper layout: names are read left-to-right with zero overlap.
    """
    df_sorted = df.sort_values(by='time_mean', ascending=True).reset_index(drop=True)
    colors = get_bar_colors(df_sorted)
    y_pos = np.arange(len(df_sorted))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 1.5), sharey=True)

    # Panel (a): Time
    xerr1 = compute_xerr(df_sorted['time_mean'], df_sorted['time_std'], log_scale=log_scale, min_bound=0.05)
    ax1.barh(y_pos, df_sorted['time_mean'], xerr=xerr1,
             color=colors, height=0.68, edgecolor='black', linewidth=0.35,
             capsize=1.2, error_kw={'elinewidth': 0.55, 'ecolor': '#111111'})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['algorithm'], fontsize=5)
    ax1.set_xlabel('Execution Time (s)', fontsize=6)
    #ax1.set_title('(a) Execution Time', fontsize=6.0, pad=3)
    if log_scale:
        ax1.set_xscale('log')
        ax1.set_xlim(left=0.1, right=max(df_sorted['time_mean'] + df_sorted['time_std']) * 1.8)
    else:
        ax1.set_xlim(left=0)
    ax1.grid(True, axis='x')
    ax1.tick_params(axis='both', pad=1.2, length=1.8)

    # Panel (b): Memory
    xerr2 = compute_xerr(df_sorted['mem_mean'], df_sorted['mem_std'], log_scale=log_scale, min_bound=0.05)
    ax2.barh(y_pos, df_sorted['mem_mean'], xerr=xerr2,
             color=colors, height=0.68, edgecolor='black', linewidth=0.35,
             capsize=1.2, error_kw={'elinewidth': 0.55, 'ecolor': '#111111'})
    ax2.set_xlabel('Peak Memory (MB)', fontsize=6)
    #ax2.set_title('(b) Peak Memory Usage', fontsize=6.0, pad=3)
    if log_scale:
        ax2.set_xscale('log')
        ax2.set_xlim(left=0.1, right=max(df_sorted['mem_mean'] + df_sorted['mem_std']) * 1.8)
    else:
        ax2.set_xlim(left=0)
    ax2.grid(True, axis='x')
    ax2.tick_params(axis='both', pad=1.2, length=1.8)

    # In-plot legend in lower right of Panel (a) where values are < 1s (completely open space)
    if include_legend:
        legend_handles = [
            mpatches.Patch(facecolor=COLOR_RELAXED, edgecolor='black', linewidth=0.35, label='SyRFDRes (Relaxed)'),
            mpatches.Patch(facecolor=COLOR_STRICT,  edgecolor='black', linewidth=0.35, label='SyRFDRes (Strict)'),
            #mpatches.Patch(facecolor=COLOR_SOTA_DL, edgecolor='black', linewidth=0.35, label='SOTA Baselines'),
        ]
        ax1.legend(handles=legend_handles, loc='lower right', frameon=True,
                   framealpha=0.92, facecolor='white', edgecolor='#cccccc',
                   fontsize=6.2, handlelength=0.9, borderpad=0.25, handletextpad=0.35)

    plt.subplots_adjust(wspace=0.10, left=0.22, right=0.98, top=0.91, bottom=0.15)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f"{filename_prefix}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved combined 1x2 plot: {filename_prefix}.pdf/.png")


def plot_single_figures(df, log_scale=True):
    """
    Generates two completely separate individual ultra-compact plots:
    1. Execution time plot (with std)
    2. Peak memory plot (with std)
    Dimensions: width=2.9in, height=2.3in (perfect for single-column insertion).
    """
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_RELAXED, edgecolor='black', linewidth=0.35, label='SyRFDRes (Relaxed)'),
        mpatches.Patch(facecolor=COLOR_STRICT,  edgecolor='black', linewidth=0.35, label='SyRFDRes (Strict)'),
        #mpatches.Patch(facecolor=COLOR_SOTA_DL, edgecolor='black', linewidth=0.35, label='SOTA Baselines'),
    ]

    # --- 1. Single Time Plot ---
    df_sorted_time = df.sort_values(by='time_mean', ascending=True).reset_index(drop=True)
    colors_time = get_bar_colors(df_sorted_time)
    y_pos_time = np.arange(len(df_sorted_time))
    xerr_time = compute_xerr(df_sorted_time['time_mean'], df_sorted_time['time_std'], log_scale=log_scale, min_bound=0.05)

    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    ax.barh(y_pos_time, df_sorted_time['time_mean'], xerr=xerr_time,
            color=colors_time, height=0.68, edgecolor='black', linewidth=0.35,
            capsize=1.2, error_kw={'elinewidth': 0.55, 'ecolor': '#111111'})
    ax.set_yticks(y_pos_time)
    ax.set_yticklabels(df_sorted_time['algorithm'], fontsize=6.6)
    ax.set_xlabel('Execution Time (s)' + (' [log scale]' if log_scale else ''), fontsize=7.2)
    ax.set_title('Average Execution Time (Mean ± Std)', fontsize=7.8, pad=3)
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim(left=0.1, right=max(df_sorted_time['time_mean'] + df_sorted_time['time_std']) * 2.0)
    else:
        ax.set_xlim(left=0)
    ax.grid(True, axis='x')
    ax.tick_params(axis='both', pad=1.2, length=1.8)

    ax.legend(handles=legend_handles, loc='lower right', frameon=True,
              framealpha=0.92, facecolor='white', edgecolor='#cccccc',
              fontsize=5.8, handlelength=0.9, borderpad=0.25, handletextpad=0.35)

    plt.tight_layout()
    name_time = f"fig_runtime_time_{'log' if log_scale else 'linear'}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f"{name_time}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved single time plot: {name_time}.pdf/.png")

    # --- 2. Single Memory Plot ---
    df_sorted_mem = df.sort_values(by='mem_mean', ascending=True).reset_index(drop=True)
    colors_mem = get_bar_colors(df_sorted_mem)
    y_pos_mem = np.arange(len(df_sorted_mem))
    xerr_mem = compute_xerr(df_sorted_mem['mem_mean'], df_sorted_mem['mem_std'], log_scale=log_scale, min_bound=0.05)

    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    ax.barh(y_pos_mem, df_sorted_mem['mem_mean'], xerr=xerr_mem,
            color=colors_mem, height=0.68, edgecolor='black', linewidth=0.35,
            capsize=1.2, error_kw={'elinewidth': 0.55, 'ecolor': '#111111'})
    ax.set_yticks(y_pos_mem)
    ax.set_yticklabels(df_sorted_mem['algorithm'], fontsize=6.6)
    ax.set_xlabel('Peak Memory (MB)' + (' [log scale]' if log_scale else ''), fontsize=7.2)
    ax.set_title('Peak Memory Usage (Mean ± Std)', fontsize=7.8, pad=3)
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim(left=0.1, right=max(df_sorted_mem['mem_mean'] + df_sorted_mem['mem_std']) * 2.0)
    else:
        ax.set_xlim(left=0)
    ax.grid(True, axis='x')
    ax.tick_params(axis='both', pad=1.2, length=1.8)

    ax.legend(handles=legend_handles, loc='lower right', frameon=True,
              framealpha=0.92, facecolor='white', edgecolor='#cccccc',
              fontsize=5.8, handlelength=0.9, borderpad=0.25, handletextpad=0.35)

    plt.tight_layout()
    name_mem = f"fig_runtime_memory_{'log' if log_scale else 'linear'}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f"{name_mem}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved single memory plot: {name_mem}.pdf/.png")


def plot_vertical_grouped(df, filename_prefix='fig_runtime_comparison_vertical', log_scale=True):
    """
    Alternative vertical bar plots for column layout.
    """
    df_sorted = df.sort_values(by='time_mean', ascending=False).reset_index(drop=True)
    colors = get_bar_colors(df_sorted)
    x_pos = np.arange(len(df_sorted))

    # Time
    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    yerr_time = compute_xerr(df_sorted['time_mean'], df_sorted['time_std'], log_scale=log_scale, min_bound=0.05)
    ax.bar(x_pos, df_sorted['time_mean'], yerr=yerr_time,
           color=colors, width=0.65, edgecolor='black', linewidth=0.35,
           capsize=1.2, error_kw={'elinewidth': 0.55, 'ecolor': '#111111'})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['algorithm'], rotation=45, ha='right', fontsize=6.4)
    ax.set_ylabel('Time (s)' + (' [log]' if log_scale else ''), fontsize=7.2)
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, axis='y')
    ax.tick_params(axis='both', pad=1.0, length=1.8)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f"{filename_prefix}_time.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Memory
    df_mem_sorted = df.sort_values(by='mem_mean', ascending=False).reset_index(drop=True)
    colors_mem = get_bar_colors(df_mem_sorted)
    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    yerr_mem = compute_xerr(df_mem_sorted['mem_mean'], df_mem_sorted['mem_std'], log_scale=log_scale, min_bound=0.05)
    ax.bar(x_pos, df_mem_sorted['mem_mean'], yerr=yerr_mem,
           color=colors_mem, width=0.65, edgecolor='black', linewidth=0.35,
           capsize=1.2, error_kw={'elinewidth': 0.55, 'ecolor': '#111111'})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_mem_sorted['algorithm'], rotation=45, ha='right', fontsize=6.4)
    ax.set_ylabel('Peak Mem (MB)' + (' [log]' if log_scale else ''), fontsize=7.2)
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, axis='y')
    ax.tick_params(axis='both', pad=1.0, length=1.8)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f"{filename_prefix}_memory.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved vertical comparison plots: {filename_prefix}_*")


def export_tables(df):
    """Export formatted CSV and LaTeX summary tables."""
    table_df = df.copy()
    table_df['Time (s)'] = table_df.apply(
        lambda r: f"{r['time_mean']:.2f} ± {r['time_std']:.2f}" if r['time_std'] > 0 else f"{r['time_mean']:.2f}", axis=1
    )
    table_df['Peak Memory (MB)'] = table_df.apply(
        lambda r: f"{r['mem_mean']:.2f} ± {r['mem_std']:.2f}" if r['mem_std'] > 0 else f"{r['mem_mean']:.2f}", axis=1
    )

    out_csv = os.path.join(TABLES_DIR, "table_runtime_memory_comparison.csv")
    table_df[['algorithm', 'category', 'Time (s)', 'Peak Memory (MB)']].to_csv(out_csv, index=False)
    print(f"  ✓ Saved CSV summary table: {out_csv}")

    # LaTeX Table
    tex_rows = []
    for _, r in table_df.iterrows():
        is_ours = 'SyRFDRes' in r['algorithm']
        name_str = f"\\textbf{{{r['algorithm']}}}" if is_ours else r['algorithm']
        time_str = f"${r['time_mean']:.2f} \\pm {r['time_std']:.2f}$" if r['time_std'] > 0 else f"${r['time_mean']:.2f}$"
        mem_str  = f"${r['mem_mean']:.2f} \\pm {r['mem_std']:.2f}$" if r['mem_std'] > 0 else f"${r['mem_mean']:.2f}$"
        tex_rows.append(f"{name_str} & {r['category']} & {time_str} & {mem_str} \\\\")

    latex_code = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Runtime and Memory Comparison between SyRFDRes and SOTA Baselines.}\n"
        "\\label{tab:runtime_memory_comparison}\n"
        "\\resizebox{\\columnwidth}{!}{\n"
        "\\begin{tabular}{llcc}\n"
        "\\toprule\n"
        "\\textbf{Algorithm} & \\textbf{Type} & \\textbf{Execution Time (s)} & \\textbf{Peak Memory (MB)} \\\\\n"
        "\\midrule\n"
        + "\n".join(tex_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )

    out_tex = os.path.join(TABLES_DIR, "table_runtime_memory_comparison.tex")
    with open(out_tex, 'w') as f:
        f.write(latex_code)
    print(f"  ✓ Saved LaTeX summary table: {out_tex}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready runtime & memory comparison artifacts.")
    # Resolve default paths relative to REPO_ROOT
    default_runtime_dir = REPO_ROOT / 'ablation/runtime_results' if (REPO_ROOT / 'ablation/runtime_results').exists() else (REPO_ROOT / 'ablation_runtime_results')
    default_strict_dir  = str(REPO_ROOT / 'ablation/strict') if (REPO_ROOT / 'ablation/strict').exists() else str(REPO_ROOT / 'ablation_strict')
    default_relaxed_dir = str(REPO_ROOT / 'ablation/relaxed') if (REPO_ROOT / 'ablation/relaxed').exists() else str(REPO_ROOT / 'ablation_relaxed')

    default_sota_summary = str(default_runtime_dir / 'augmentation_benchmark_summary_20261019_130216.csv')
    default_sota_raw     = str(default_runtime_dir / 'augmentation_benchmark_20261019_130216.csv')

    parser.add_argument('--sota-file', type=str,
                        default=default_sota_summary,
                        help='Path to SOTA summary benchmark CSV')
    parser.add_argument('--raw-fallback', type=str,
                        default=default_sota_raw,
                        help='Path to multi-dataset benchmark CSV to supply std if missing in summary')
    parser.add_argument('--use-raw-sota', action='store_true',
                        help='Aggregate SOTA metrics directly from raw multi-dataset benchmark')
    parser.add_argument('--dataset-scope', choices=['all', 'common'], default='all',
                        help='Dataset scope for SyRFDRes: "all" (all datasets) or "common" (only datasets common to SOTA benchmark)')
    parser.add_argument('--strict-dir', type=str, default=default_strict_dir,
                        help='Path to ablation_strict directory')
    parser.add_argument('--relaxed-dir', type=str, default=default_relaxed_dir,
                        help='Path to ablation_relaxed directory')
    args = parser.parse_args()

    print("======================================================================")
    print("🚀 GENERATING RUNTIME & MEMORY COMPARISON REBUTTAL ARTIFACTS")
    print("======================================================================")
    print(f"• SOTA Summary File: {args.sota_file}")
    print(f"• Raw Fallback File: {args.raw_fallback}")
    print(f"• Dataset Scope:     {args.dataset_scope}")
    print(f"• Strict Directory:  {args.strict_dir}")
    print(f"• Relaxed Directory: {args.relaxed_dir}")
    print("----------------------------------------------------------------------")

    # 1. Load SOTA Data
    df_sota = load_sota_data(args.sota_file, args.raw_fallback, use_raw_directly=args.use_raw_sota)

    # 2. Determine Filter Datasets if scope is common
    filter_datasets = None
    if args.dataset_scope == 'common' and os.path.exists(args.raw_fallback):
        df_raw = pd.read_csv(args.raw_fallback)
        filter_datasets = set(df_raw['dataset'].unique())
        print(f"  [i] Filtering SyRFDRes runs to {len(filter_datasets)} datasets common to SOTA benchmark.")

    # 3. Load SyRFDRes Data
    df_syrfd = load_syrfd_data(args.strict_dir, args.relaxed_dir, filter_datasets=filter_datasets)

    df_all = pd.concat([df_sota, df_syrfd], ignore_index=True)

    print(f"Loaded {len(df_sota)} SOTA baselines and {len(df_syrfd)} SyRFDRes configurations.")
    print("Summary of values:")
    print(df_all[['algorithm', 'time_mean', 'time_std', 'mem_mean', 'mem_std']].to_string(index=False))
    print("----------------------------------------------------------------------")

    # 4. Generate Plots
    print("Generating publication plots...")
    # Standalone Legend
    save_standalone_legend('fig_runtime_legend')

    # Combined 1x2 Horizontal (Log & Linear)
    plot_horizontal_comparison(df_all, filename_prefix='fig_runtime_memory_comparison_log', log_scale=True)
    plot_horizontal_comparison(df_all, filename_prefix='fig_runtime_memory_comparison_linear', log_scale=False)

    # Separate Single Plots (Log & Linear)
    plot_single_figures(df_all, log_scale=True)
    plot_single_figures(df_all, log_scale=False)

    # Vertical Alternative Plots
    plot_vertical_grouped(df_all, filename_prefix='fig_runtime_comparison_vertical', log_scale=True)

    # 5. Export Tables
    print("Exporting tables...")
    export_tables(df_all)

    print("======================================================================")
    print("✨ ALL ARTIFACTS GENERATED SUCCESSFULLY!")
    print(f"📁 Plots saved to:  {PLOTS_DIR}")
    print(f"📁 Tables saved to: {TABLES_DIR}")
    print("======================================================================")


if __name__ == '__main__':
    main()
