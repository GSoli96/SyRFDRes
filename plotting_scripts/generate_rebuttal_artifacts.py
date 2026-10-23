#!/usr/bin/env python3
"""
generate_rebuttal_artifacts.py
Publication-ready PDF plots + LaTeX/CSV tables for VLDB rebuttal.

ALL PLOTS show mean ± 68% CI (bootstrap) across 36 datasets.
  - Bar plots: bar height = mean, error bar = 68% CI (≈ ±1 SEM)
  - Line plots: curve = mean, shaded band = 68% CI

Covers reviewer comments: D4, D5, W1, W3, D26, D39, D46.

Output:
  paper_rebuttal_artifacts/plots/   ← PDF + PNG
  paper_rebuttal_artifacts/tables/  ← CSV + LaTeX
"""

from matplotlib import axes
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns   # requires >= 0.13 for gap= parameter

# =============================================================================
# CUSTOMISABLE PARAMETERS
# =============================================================================
BAR_GROUP_WIDTH = 0.50   # total width of an x-category (both bars together)
BAR_GAP         = 0.02   # gap between Relaxed/Strict bars within each group
BAR_EDGE_WIDTH  = 0.40   # thin black border on each bar
ERR_CAPSIZE     = 0.8    # error-bar cap size (pt) — narrow
ERR_LINE_WIDTH  = 0.65   # error-bar line width
LINE_WIDTH      = 1.2
MARKER_SIZE     = 5.5
FIG_DPI         = 300
YLABEL_FS       = 6    # y-axis label font size

# Layout per Fig 1 (Ablation): "2x3" (2 righe × 3 colonne) oppure "3x2" (3 righe × 2 colonne)
FIG1_LAYOUT          = "2x3"

# Se True, usa sigle compatte (L0..L6, C0..C9) per le xticks di Fig 1
# e aggiunge una legenda estesa a 7 colonne che spiega i simboli.
# Se False, mantiene le etichette testuali complete ruotate.
FIG1_USE_SHORT_TICKS = True

PALETTE = {'Relaxed': '#44A5C2', 'Strict': '#ffaf4a'}

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':        9.5,
    'axes.labelsize':   9.5,
    'axes.titlesize':  10.5,
    'xtick.labelsize':  8.0,
    'ytick.labelsize':  8.0,
    'legend.fontsize':  8.5,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
    'grid.alpha':       0.30,
    'grid.linestyle':  '--',
})

from pathlib import Path

# Resolve project root dynamically
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'plotting_scripts' else Path('.').resolve()

OUTPUT_DIR = str(REPO_ROOT / 'supplementary_material')
PLOTS_DIR  = os.path.join(OUTPUT_DIR, 'plots', 'paper_rebuttal')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================
def _find_files(pattern_relative):
    for base in [REPO_ROOT, Path('.')]:
        files = glob.glob(str(base / pattern_relative))
        if files:
            return files
    return []

def load_data():
    rel_files = _find_files('ablation/relaxed/results/*.csv') or _find_files('ablation_relaxed/results/*.csv')
    if not rel_files:
        raise FileNotFoundError("Could not find relaxed ablation results.")
    rel = pd.concat([pd.read_csv(f) for f in rel_files], ignore_index=True)
    rel['mode'] = 'Relaxed'

    stt_files = _find_files('ablation/strict/results/*.csv') or _find_files('ablation_strict/results/*.csv')
    if not stt_files:
        raise FileNotFoundError("Could not find strict ablation results.")
    stt = pd.concat([pd.read_csv(f) for f in stt_files], ignore_index=True)
    stt['mode'] = 'Strict'
    return pd.concat([rel, stt], ignore_index=True)

df_all  = load_data()
df_comp = df_all[df_all['status'] == 'completed'].copy()


# =============================================================================
# HELPERS
# =============================================================================
def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f'{name}.{ext}'),
                    dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}.pdf/.png')


def save_legend_pdf(name):
    """Export a standalone legend as a tight PDF/PNG.

    Style: ncol=2, centered title, white background, shadow=True.
    """
    handles = [mpatches.Patch(facecolor=PALETTE[m], edgecolor='black',
                               linewidth=BAR_EDGE_WIDTH, label=m)
               for m in ('Relaxed', 'Strict')]
    fig, ax = plt.subplots(figsize=(2.6, 0.65))
    ax.axis('off')
    leg = ax.legend(handles=handles, labels=['Relaxed', 'Strict'],
                    #title=title,
                     ncol=2, loc='center',
                    frameon=True, facecolor='white', shadow=True,
                    fontsize=8.5, title_fontsize=8.5,
                    borderpad=0.6, handlelength=1.2, handletextpad=0.5)
    leg.get_title().set_ha('center')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS_DIR, f'{name}.{ext}'),
                    dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved legend: {name}.pdf/.png')


def rm_legend(ax):
    if ax.legend_:
        ax.legend_.remove()


def bar_plot(ax, data, x, y, hue='mode', order=None,
             hue_order=('Relaxed', 'Strict')):
    """Bar plot: mean ± 68% CI (bootstrap) across datasets.

    Each bar gets a thin black edge (BAR_EDGE_WIDTH).
    Gap between Relaxed/Strict bars within a group = BAR_GAP.
    """
    sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=list(hue_order),
                order=order, palette=PALETTE, ax=ax,
                errorbar=('ci', 68), capsize=ERR_CAPSIZE,
                width=BAR_GROUP_WIDTH, gap=BAR_GAP,
                err_kws={'linewidth': ERR_LINE_WIDTH},
                edgecolor='black', linewidth=BAR_EDGE_WIDTH)


def line_plot(ax, data, x, y, xvals, hue='mode',
              hue_order=('Relaxed', 'Strict')):
    """Line plot: mean ± 68% CI (bootstrap), x-ticks forced to actual data values.

    xvals: list of x-axis values to use as ticks (must match data).
    """
    sns.lineplot(data=data, x=x, y=y, hue=hue, style=hue,
                 hue_order=list(hue_order), palette=PALETTE,
                 markers=True, dashes=False, lw=LINE_WIDTH,
                 markersize=MARKER_SIZE, ax=ax, errorbar=('ci', 68))
    # Force ticks to actual data points to avoid misalignment
    ax.set_xticks(xvals)
    ax.set_xticklabels([str(v) for v in xvals])


# =============================================================================
# FIG 1 — Ablation: ALL Losses + ALL Components
# Layout: 3 rows × 2 cols (NO sharex — axes swapped in row 2)
# Row 0: (a) SWD/Loss       | (b) SWD/Component
# Row 1: (c) Coverage/Loss   | (d) Coverage/Component  [ylim_bot=0.5 on (d)]
# Row 2: (e) Violation/Comp  | (f) Violation/Loss      ← SWAPPED so Loss is last
# Legend exported separately.
# All bars: mean ± 68% CI across 36 datasets.
# Addresses: W3, W1, D39
# =============================================================================
def plot_ablation():
    loss_map = {
        'L0_full':            'Full',
        'L1_no_MMD':          'w/o MMD',
        'L2_no_softRFD':      'w/o SoftRFD',
        'L3_no_composite':    'w/o Composite',
        'L4_no_SWD':          'w/o SWD',
        'L5_no_coverage':     'w/o Coverage',
        'L6_no_anticollapse': 'w/o Anti-col.',
    }
    comp_map = {
        'C0_full':              'Full',
        'C1_no_RFD_validator':  'w/o Valid.',
        'C2_validator_relaxed': 'Relax. Valid.',
        'C4_no_rolling_buffer': 'w/o Buffer',
        'C5_no_warmup':         'w/o Warmup',
        'C6_MLP_no_residual':   'MLP (No Res.)',
        'C9_no_early_stopping': 'w/o EarlyStop',
    }
    loss_order = list(loss_map.values())
    comp_order = list(comp_map.values())

    sub_l = df_comp[df_comp['config_name'].isin(loss_map)].copy()
    sub_l['Label'] = sub_l['config_name'].map(loss_map)

    sub_c = df_comp[df_comp['config_name'].isin(comp_map)].copy()
    if 'C0_full' not in sub_c['config_name'].unique():
        base = df_comp[df_comp['config_name'] == 'L0_full'].copy()
        base['config_name'] = 'C0_full'
        sub_c = pd.concat([sub_c, base], ignore_index=True)
    sub_c['Label'] = sub_c['config_name'].map(comp_map)

    # Mappe per sigle corte
    loss_short_map = {
        'Full': r'$L_0$',
        'w/o MMD': r'$L_1$',
        'w/o SoftRFD': r'$L_2$',
        'w/o Composite': r'$L_3$',
        'w/o SWD': r'$L_4$',
        'w/o Coverage': r'$L_5$',
        'w/o Anti-col.': r'$L_6$',
    }
    comp_short_map = {
        'Full': r'$C_0$',
        'w/o Valid.': r'$C_1$',
        'Relax. Valid.': r'$C_2$',
        'w/o Buffer': r'$C_4$',
        'w/o Warmup': r'$C_5$',
        'MLP (No Res.)': r'$C_6$',
        'w/o EarlyStop': r'$C_9$',
    }

    if FIG1_USE_SHORT_TICKS:
        sub_l['Label'] = sub_l['Label'].map(loss_short_map)
        sub_c['Label'] = sub_c['Label'].map(comp_short_map)
        loss_order_plot = [loss_short_map[x] for x in loss_order]
        comp_order_plot = [comp_short_map[x] for x in comp_order]
    else:
        loss_order_plot = loss_order
        comp_order_plot = comp_order

    if FIG1_LAYOUT == "2x3":
        # =====================================================================
        # LAYOUT 2x3: 2 righe × 3 colonne
        # Riga 0: Losses (Fidelity, Diversity, Compliance)
        # Riga 1: Components (Fidelity, Diversity, Compliance)
        # Y-labels ripristinate su tutti i subplot
        # =====================================================================
        panels_2x3 = [
            # Riga 0: Losses
            [
                ('eval_swd_eval', 'SWD (↓)', '(a) Fidelity: Loss', sub_l, loss_order_plot, None, None, None),
                ('eval_coverage', 'Coverage (↑)', '(b) Diversity: Loss', sub_l, loss_order_plot, None, 0.9, [0, 0.30, 0.60, 0.90]),
                ('eval_rfd_violation_rate', 'Violation Rate (↓)', '(c) Compliance: Loss', sub_l, loss_order_plot, None, None, None),
            ],
            # Riga 1: Components
            [
                ('eval_swd_eval', 'SWD (↓)', '(d) Fidelity: Component', sub_c, comp_order_plot, None, 0.4, None),
                ('eval_coverage', 'Coverage (↑)', '(e) Diversity: Component', sub_c, comp_order_plot, 0.5, None, None),
                ('eval_rfd_violation_rate', 'Violation Rate (↓)', '(f) Compliance: Component', sub_c, comp_order_plot, None, 0.03, ['0', '0.01', '0.02', '0.03']),
            ]
        ]

        fig, axes = plt.subplots(2, 3, figsize=(5.2, 2))

        for r in range(2):
            for c in range(3):
                col, ylabel, title, data, order, ylim_bot, ylim_top, yticks_custom = panels_2x3[r][c]
                ax = axes[r, c]

                bar_plot(ax, data, 'Label', col, order=order)
                ax.set_title(title, fontsize=7, pad=2)
                ax.set_ylabel(ylabel, fontsize=YLABEL_FS, labelpad=1)
                ax.set_xlabel('')
                ax.grid(True, axis='y')
                rm_legend(ax)
                ax.tick_params(axis='y', which='major', labelsize=6.8, pad=0.3)
                ax.tick_params(axis='x', which='major', pad=0.3)

                if ylim_bot is not None:
                    ax.set_ylim(bottom=ylim_bot)
                if ylim_top is not None:
                    ax.set_ylim(top=ylim_top)
                if yticks_custom is not None:
                    if isinstance(yticks_custom[0], (int, float)):
                        ax.set_yticks(yticks_custom)
                    else:
                        ax.set_yticks(np.linspace(0, ylim_top, len(yticks_custom)))
                        ax.set_yticklabels(yticks_custom)

                ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(order))))
                if FIG1_USE_SHORT_TICKS:
                    ax.set_xticklabels(order, rotation=0, ha='center', fontsize=7.0)
                else:
                    ax.set_xticklabels(order, rotation=0, ha='right', fontsize=6.0)

        # Se attiva la codifica corta, inserisci la legenda a 7 colonne in basso
        if FIG1_USE_SHORT_TICKS:
            loss_legend_items = [
                r'$L_0$: Full', r'$L_1$: w/o MMD', r'$L_2$: w/o SoftRFD',
                r'$L_3$: w/o Comp.', r'$L_4$: w/o SWD', r'$L_5$: w/o Cov.',
                r'$L_6$: w/o Anti-col.'
            ]
            comp_legend_items = [
                r'$C_0$: Full', r'$C_1$: w/o Valid.', r'$C_2$: Rel. Valid.',
                r'$C_4$: w/o Buffer', r'$C_5$: w/o Warmup', r'$C_6$: MLP (No Res.)',
                r'$C_9$: w/o EarlyStop'
            ]
            # Interleave per riempire riga 0 con Losses e riga 1 con Components in ncol=7
            interleaved_labels = []
            for l, comp in zip(loss_legend_items, comp_legend_items):
                interleaved_labels.append(l)
                interleaved_labels.append(comp)

            legend_handles = [Line2D([0], [0], color='none', label=lab) for lab in interleaved_labels]
            fig.legend(
                handles=legend_handles,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.06),
                ncol=7,
                fontsize=5.6,
                handlelength=0,
                handletextpad=0,
                columnspacing=0.8,
                frameon=True,
                facecolor='white',
                edgecolor='#cccccc',
            )
            plt.subplots_adjust(wspace=0.38, hspace=0.62, left=0.08, right=0.98, top=0.92, bottom=0.20)
        else:
            plt.subplots_adjust(wspace=0.38, hspace=0.50, left=0.08, right=0.98, top=0.92, bottom=0.18)

    else:
        # =====================================================================
        # LAYOUT 3x2 (Originale): 3 righe × 2 colonne
        # =====================================================================
        panels = [
            # Row 0
            ('eval_swd_eval', 'SWD (↓)',
             '(a) Fidelity: Loss', '(b) Fidelity: Component',
             sub_l, loss_order_plot, sub_c, comp_order_plot, None),
            # Row 1
            ('eval_coverage', 'Coverage (↑)',
             '(c) Diversity: Loss', '(d) Diversity: Component',
             sub_l, loss_order_plot, sub_c, comp_order_plot, 0.5),  # ylim bottom on (d)
            # Row 2 — Loss left (e), Component right (f)
            ('eval_rfd_violation_rate', 'Violation Rate (↓)',
             '(e) Compliance: Loss', '(f) Compliance: Component',
             sub_l, loss_order_plot, sub_c, comp_order_plot, None),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(5, 3.8), sharex='col')
        plt.subplots_adjust(wspace=0, hspace=0)

        for row, (col, ylabel, tl, tr, dl, ol, dr, or_, ylim_bot) in enumerate(panels):
            ax_l, ax_r = axes[row, 0], axes[row, 1]

            bar_plot(ax_l, dl, 'Label', col, order=ol)
            ax_l.set_title(tl, fontsize=7)
            ax_l.set_ylabel(ylabel, fontsize=YLABEL_FS)
            ax_l.set_xlabel('')
            ax_l.grid(True, axis='y')
            rm_legend(ax_l)
            ax_l.tick_params(axis='y', which='major', labelsize=7, rotation=90)
            for label in ax_l.get_yticklabels():
                label.set_verticalalignment('center')
                label.set_horizontalalignment('center')

            bar_plot(ax_r, dr, 'Label', col, order=or_)
            ax_r.set_title(tr, fontsize=7)
            ax_r.set_ylabel("", fontsize=YLABEL_FS)
            ax_r.set_xlabel('')
            ax_r.grid(True, axis='y')
            rm_legend(ax_r)
            ax_r.tick_params(axis='y', which='major', labelsize=7, rotation=90)
            for label in ax_r.get_yticklabels():
                label.set_verticalalignment('center')
                label.set_horizontalalignment('center')
            if ylim_bot is not None:
                ax_r.set_ylim(bottom=ylim_bot)

        # Secondo plot, prima riga → top ylim = 0.3
        axes[0, 1].set_ylim(top=0.4)

        # Primo plot, seconda riga → top ylim = 0.75 e ticks [0, 0.25, 0.50, 0.75]
        axes[1, 0].set_ylim(top=0.9)
        axes[1, 0].set_yticks([0, 0.30, 0.60, 0.90])

        # Secondo plot, ultima riga → top ylim = 0.03
        axes[-1, 1].set_ylim(top=0.03)
        axes[-1, 1].set_yticklabels(['0','0.01','0.02','0.03'])

        # Only the bottom row needs x-tick labels (sharex hides upper rows).
        for col_idx, order in [(0, loss_order_plot), (1, comp_order_plot)]:
            ax_bot = axes[-1, col_idx]
            ax_bot.xaxis.set_major_locator(mticker.FixedLocator(range(len(order))))
            if FIG1_USE_SHORT_TICKS:
                ax_bot.set_xticklabels(order, rotation=0, ha='center', fontsize=7)
            else:
                ax_bot.set_xticklabels(order, rotation=35, ha='right', fontsize=6)

        plt.subplots_adjust(wspace=0.2, hspace=0.4, left=0.42, right=0.98, top=0.93, bottom=0.22)

    save_fig(fig, 'fig1_ablation_losses_components')
    save_legend_pdf('fig1_legend')



# =============================================================================
# FIG 2 — Acceptance Rate, Budget Exhaustion & Adaptive Sampling Scaling
# Layout: 2 rows × 3 cols
# Row 0: (a) Acceptance Rate  | (b) Budget Completion  | (c) Rounds vs s
# Row 1: (d) Rounds vs B      | (e) Runtime vs s        | (f) Runtime vs B
# X-ticks explicitly set to actual data values for line plots.
# Legend exported separately.
# Addresses: D4, D5, D26, W1
# =============================================================================
def plot_sampling_diagnostics():
    YLABEL_FS       = 8    # y-axis label font size
    l0    = df_comp[df_comp['config_name'] == 'L0_full'].copy()

    r_sub = df_comp[df_comp['config_name'].str.startswith('R_nsamples')].copy()
    r_sub['s'] = r_sub['gen_requested'].astype(int)
    r_sub['exhausted'] = (r_sub['gen_obtained'] < r_sub['gen_requested']).astype(float)
    S_VALS = sorted(r_sub['s'].unique().tolist())   # [50, 100, 200, 500]

    b_sub = df_comp[df_comp['config_name'].str.startswith('R_batch')].copy()
    b_sub['B'] = b_sub['config_name'].str.extract(r'(\d+)').astype(int)
    B_VALS = sorted(b_sub['B'].unique().tolist())   # [32, 64, 128, 256]

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3))
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 2.5))

    # ── (a) Acceptance rate boxplot ──────────────────────────────────────────
    ax = axes[0, 0]
    sns.boxplot(data=l0, x='mode', y='gen_acceptance_rate',
                hue='mode', palette=PALETTE, ax=ax,
                width=BAR_GROUP_WIDTH, gap=0,
                boxprops=dict(alpha=0.85, edgecolor='black', linewidth=BAR_EDGE_WIDTH),
                whiskerprops=dict(linewidth=0.7, color='black'),
                capprops=dict(linewidth=0.7, color='black'),
                medianprops=dict(linewidth=1.0, color='black'),
                showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'w',
                           'markeredgecolor': 'k', 'markersize': MARKER_SIZE})
    ax.set_title('(a) Acceptance Rate')
    ax.set_title('(a)',fontsize=YLABEL_FS, pad=3)
    ax.set_ylabel('Acceptance Rate', fontsize=YLABEL_FS)
    ax.set_xlabel('Mode', fontsize=YLABEL_FS)
    ax.set_yticks([0, 0.5, 0.8, 1])
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, axis='y')
    ax.tick_params(axis='x', pad=0.4)
    ax.tick_params(axis='y', pad=0.3)  # Distanza numeri asse Y dal grafico (default matplotlib ~3.5-4)

    rm_legend(ax)

    # ── (b) Budget completion % vs s ─────────────────────────────────────────
    ax = axes[0, 1]
    exh = r_sub.groupby(['mode', 's'])['exhausted'].mean().reset_index()
    exh['success_pct'] = (1 - exh['exhausted']) * 100
    bar_plot(ax, exh, 's', 'success_pct')
    ax.set_title('(b) Budget Completion (%)')
    ax.set_title('(b)',fontsize=YLABEL_FS, pad=3)
    ax.set_yticks([80, 90, 100])
    ax.set_ylabel('Runs meeting\ntarget (%)', fontsize=YLABEL_FS)
    ax.set_xlabel('Target tuples', fontsize=YLABEL_FS)
    ax.set_ylim(80, 105)
    ax.grid(True, axis='y')
    ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(S_VALS))))
    ax.set_xticklabels([str(v) for v in S_VALS])
    ax.tick_params(axis='x', pad=0.4)
    ax.tick_params(axis='y', pad=0.3)  # Distanza numeri asse Y dal grafico (default matplotlib ~3.5-4)
    rm_legend(ax)

    # ── (c) Rounds vs s ──────────────────────────────────────────────────────
    ax = axes[0, 2]
    line_plot(ax, r_sub, 's', 'gen_attempts', xvals=S_VALS)
    ax.set_title('(c) Sampling Rounds vs. Target Tuples')
    ax.set_title('(c)',fontsize=YLABEL_FS, pad=3)
    ax.set_yticks([1.0, 1.3, 1.5, 1.7])
    ax.set_ylabel('Avg. Sampling\nRounds', fontsize=YLABEL_FS)
    ax.set_xlabel('Target tuples', fontsize=YLABEL_FS)
    ax.tick_params(axis='x', pad=0.4)
    ax.tick_params(axis='y', pad=0.3)
    ax.grid(True)
    rm_legend(ax)

    # ── (d) Rounds vs B ──────────────────────────────────────────────────────
    ax = axes[1, 0]
    line_plot(ax, b_sub, 'B', 'gen_attempts', xvals=B_VALS)
    ax.set_title('(d) Sampling Rounds vs. Batch')
    ax.set_title('(d)',fontsize=YLABEL_FS, pad=3)
    ax.set_yticks([1.0, 1.3, 1.5, 1.7])
    ax.set_ylabel('Avg. Sampling\nRounds', fontsize=YLABEL_FS)
    ax.set_xlabel('Batch size', fontsize=YLABEL_FS)
    ax.tick_params(axis='x', pad=0.4)
    ax.tick_params(axis='y', pad=0.3)
    ax.grid(True)
    rm_legend(ax)

    # ── (e) Runtime vs s ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    line_plot(ax, r_sub, 's', 'time_t_generation_total', xvals=S_VALS)
    ax.set_title('(e) Generation Runtime vs. Target Tuples')
    ax.set_title('(e)',fontsize=YLABEL_FS, pad=3)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel('Runtime (s)', fontsize=YLABEL_FS)
    ax.set_xlabel('Target tuples', fontsize=YLABEL_FS)
    ax.tick_params(axis='y', pad=0.3)
    ax.tick_params(axis='x', pad=0.4)
    ax.grid(True)
    rm_legend(ax)

    # ── (f) Runtime vs B ─────────────────────────────────────────────────────
    ax = axes[1, 2]
    line_plot(ax, b_sub, 'B', 'time_t_generation_total', xvals=B_VALS)
    ax.set_title('(f) Generation Runtime vs. Batch')
    ax.set_title('(f)',fontsize=YLABEL_FS, pad=3)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel('Runtime (s)', fontsize=YLABEL_FS)
    ax.set_xlabel('Batch size', fontsize=YLABEL_FS)
    ax.tick_params(axis='y', pad=0.3)
    ax.tick_params(axis='x', pad=0.4)
    ax.grid(True)
    rm_legend(ax)

    plt.subplots_adjust(wspace=0.01, hspace=0.02, left=0.05, right=0.58, top=0.01, bottom=0.0)
    plt.tight_layout()
    save_fig(fig, 'fig2_sampling_diagnostics')
    save_legend_pdf('fig2_legend')


# =============================================================================
# FIG 3 — Runtime Breakdown  (W1, D46)
# Stacked horizontal bar; title removed (defined in caption).
# Legend exported separately.
# Bar height reduced to decrease gap between Relaxed and Strict.
# =============================================================================
def plot_runtime_breakdown():
    phases = [
        ('Warmup',      'time_t_warmup',           '#92cce0'),
        ('Training',    'time_t_phase2',            '#44A5C2'),
        ('Generation',  'time_t_generation_total',  '#ffaf4a'),
    ]

    modes  = ['Relaxed', 'Strict']
    y_pos  = np.array([0.0, 0.45])   # closer y positions → less gap between bars

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 1.2))
    lefts = np.zeros(len(modes))

    for phase, col, color in phases:
        means = [
            df_comp[(df_comp['mode'] == m) &
                    (df_comp['config_name'] == 'L0_full')][col].mean()
            for m in modes
        ]
        bars = ax.barh(y_pos, means, left=lefts, color=color, label=phase,
                       height=0.30, edgecolor='black', linewidth=0.4)
        for bar, val in zip(bars, means):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}s', ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')
        lefts = lefts + np.array(means)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modes)
    ax.set_xlabel('Wall-clock time (s)')
    # ax.set_title('Runtime breakdown: Relaxed vs. Strict')  # defined in caption
    ax.grid(True, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig3_runtime_breakdown')

    # Legend with phase colours (different from mode palette)
    phase_handles = [mpatches.Patch(facecolor=c, edgecolor='black',
                                    linewidth=0.4, label=p)
                     for p, _, c in phases]
    fig_leg, ax_leg = plt.subplots(figsize=(3.2, 0.65))
    ax_leg.axis('off')
    leg = ax_leg.legend(handles=phase_handles, labels=[p for p, _, _ in phases],
                        #title='Phase',
                         ncol=3, loc='center',
                        frameon=True, facecolor='white', shadow=True,
                        fontsize=8.5, title_fontsize=8.5,
                        borderpad=0.6, handlelength=1.2)
    leg.get_title().set_ha('center')
    fig_leg.tight_layout()
    for ext in ('pdf', 'png'):
        fig_leg.savefig(os.path.join(PLOTS_DIR, f'fig3_legend.{ext}'),
                        dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig_leg)
    print('  Saved legend: fig3_legend.pdf/.png')


# =============================================================================
# FIG 4 — Sensitivity Analysis  (W3) — all 10 hyperparams, 34 configs
# Layout: 2 rows × 5 cols
# X-ticks forced to actual data values on every panel.
# Legend exported separately.
# =============================================================================
def plot_sensitivity():
    hp_list = [
        # (title, config_prefix, xcol, xvals, xscale)
        ('(a) SoftRFD Weight',        'S_lambda_rfd=',       'lam',   [0.0, 5.0, 10.0, 20.0], 'linear'),
        ('(b) Temperature',           'S_tau=',              'tau',   [0.1, 0.25, 0.5, 1.0, 2.0], 'linear'),
        ('(c) RFD Threshold',         'S_rfd_threshold=',    'T',     [2, 4, 8],               'linear'),
        ('(d) Coverage Weight',       'S_w_coverage=',       'wcov',  [0.0, 0.2, 0.5, 1.0],   'linear'),
        ('(e) SWD Weight',            'S_w_swd=',            'wswd',  [0.0, 1.0, 2.0],         'linear'),
        ('(f) Composite Weight',      'S_lambda_composite=', 'lcomp', [0.0, 1.0, 2.0],         'linear'),
        ('(g) Warmup Ratio',          'S_warmup_ratio=',     'warm',  [0.0, 0.2, 0.4],         'linear'),
        ('(h) Learning Rate',         'S_lr=',               'lr',    [0.0001, 0.001, 0.005],  'log'),
        ('(i) Batch Size',            'S_batch_size=',       'bs',    [32, 64, 128],            'linear'),
        ('(j) SWD Projections',       'S_n_projections=',    'proj',  [10, 50, 200],            'linear'),
    ]

    # Panels where ylabel is shown: (a)=idx 0, (f)=idx 5
    YLABEL_PANELS = {0, 5}
    # Panel (b) tau [idx=1]: x-values 0.1 and 0.25 are close → rotate labels
    ROTATE_PANELS = {1}

    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 5.2))
    axes_flat = axes.flatten()

    for idx, (title, prefix, xcol, xvals, xscale) in enumerate(hp_list):
        ax = axes_flat[idx]
        cfgs = [f'{prefix}{v}' for v in xvals]
        sub  = df_comp[df_comp['config_name'].isin(cfgs)].copy()
        sub[xcol] = sub['config_name'].str.replace(prefix, '', regex=False).astype(float)

        line_plot(ax, sub, xcol, 'eval_swd_eval', xvals=xvals)
        ax.set_title(title)
        ax.set_xlabel('')
        # ylabel only on designated panels to save horizontal space
        if idx in YLABEL_PANELS:
            ax.set_ylabel('SWD ($\\downarrow$)', fontsize=YLABEL_FS+3)
        else:
            ax.set_ylabel('')
        ax.grid(True)
        if xscale == 'log':
            ax.set_xscale('log')
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(v) for v in xvals])
        # Rotate x-tick labels on panels where values are densely packed
        if idx in ROTATE_PANELS:
            ax.tick_params(axis='x', labelrotation=40)
            for lbl in ax.get_xticklabels():
                lbl.set_ha('right')
        rm_legend(ax)

    plt.tight_layout()
    save_fig(fig, 'fig4_sensitivity_analysis')

    # Legend ncol=3 for Fig4
    import matplotlib.patches as _mp
    handles4 = [_mp.Patch(facecolor=PALETTE[m], edgecolor='black',
                           linewidth=BAR_EDGE_WIDTH, label=m)
                for m in ('Relaxed', 'Strict')]
    fig_leg4, ax_leg4 = plt.subplots(figsize=(3.2, 0.65))
    ax_leg4.axis('off')
    leg4 = ax_leg4.legend(handles=handles4, labels=['Relaxed', 'Strict'],
                          ncol=3, loc='center',
                          frameon=True, facecolor='white', shadow=True,
                          fontsize=8.5,
                          borderpad=0.6, handlelength=1.2, handletextpad=0.5)

    plt.subplots_adjust(wspace=0.2, hspace=0.01, left=0.05, right=0.98, top=0.1, bottom=0.0)
    #fig_leg4.tight_layout()
    for ext in ('pdf', 'png'):
        fig_leg4.savefig(os.path.join(PLOTS_DIR, f'fig4_legend.{ext}'),
                         dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig_leg4)
    print('  Saved legend: fig4_legend.pdf/.png')


# =============================================================================
# TABLES
# =============================================================================
def generate_tables():
    abl_map = {
        'L0_full':            'Full (SyRFDRes)',
        'L1_no_MMD':          'w/o MMD Loss',
        'L2_no_softRFD':      'w/o SoftRFD Loss',
        'L3_no_composite':    'w/o Composite Reg.',
        'L4_no_SWD':          'w/o SWD Loss',
        'L5_no_coverage':     'w/o Coverage Loss',
        'L6_no_anticollapse': 'w/o Anti-collapse Loss',
        'C1_no_RFD_validator': 'w/o RFD Validator',
        'C2_validator_relaxed':'Relaxed Validator',
        'C4_no_rolling_buffer':'w/o Rolling Buffer',
        'C5_no_warmup':        'w/o Warmup',
        'C6_MLP_no_residual':  'w/o ResNet (MLP)',
        'C9_no_early_stopping':'w/o Early Stopping',
    }
    rows = []
    for mode in ('Relaxed', 'Strict'):
        for key, label in abl_map.items():
            sub = df_comp[(df_comp['mode'] == mode) & (df_comp['config_name'] == key)]
            if len(sub) == 0:
                continue
            rows.append({
                'Mode':        mode,
                'Variant':     label,
                'SWD↓':       f"{sub['eval_swd_eval'].mean():.4f}",
                'Coverage↑':  f"{sub['eval_coverage'].mean():.4f}",
                'RFD Viol.↓': f"{sub['eval_rfd_violation_rate'].mean():.4f}",
                'Acc. Rate':  f"{sub['gen_acceptance_rate'].mean():.4f}",
                'Rounds':     f"{sub['gen_attempts'].mean():.3f}",
                'Train (s)':  f"{sub['time_t_train_total'].mean():.1f}",
                'Gen (s)':    f"{sub['time_t_generation_total'].mean():.2f}",
            })
    df_t1 = pd.DataFrame(rows)
    df_t1.to_csv(os.path.join(TABLES_DIR, 'table1_ablation.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'table1_ablation.tex'), 'w') as f:
        f.write(df_t1.to_latex(index=False, escape=False))
    print('  Saved table1_ablation')

    rows2 = []
    for mode in ('Relaxed', 'Strict'):
        m   = df_comp[df_comp['mode'] == mode]
        n   = len(m)
        exh = int((m['gen_obtained'] < m['gen_requested']).sum())
        rows2.append({
            'Mode': mode, 'Setting': 'All configs', 'n': n,
            'Exhausted': f'{exh}/{n} ({exh/n*100:.1f}%)',
            'Acc. Rate': f"{m['gen_acceptance_rate'].mean():.4f} ± {m['gen_acceptance_rate'].std():.4f}",
            'Acc. Min':  f"{m['gen_acceptance_rate'].min():.4f}",
            'Rounds':    f"{m['gen_attempts'].mean():.3f}",
        })
    for mode in ('Relaxed', 'Strict'):
        for s in [50, 100, 200, 500]:
            sub = df_comp[(df_comp['mode'] == mode) &
                          (df_comp['config_name'] == f'R_nsamples={s}')]
            if len(sub) == 0:
                continue
            n   = len(sub)
            exh = int((sub['gen_obtained'] < sub['gen_requested']).sum())
            rows2.append({
                'Mode': mode, 'Setting': f's={s}', 'n': n,
                'Exhausted': f'{exh}/{n} ({exh/n*100:.1f}%)',
                'Acc. Rate': f"{sub['gen_acceptance_rate'].mean():.4f}",
                'Acc. Min':  f"{sub['gen_acceptance_rate'].min():.4f}",
                'Rounds':    f"{sub['gen_attempts'].mean():.3f}",
            })
    df_t2 = pd.DataFrame(rows2)
    df_t2.to_csv(os.path.join(TABLES_DIR, 'table2_sampling_diagnostics.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'table2_sampling_diagnostics.tex'), 'w') as f:
        f.write(df_t2.to_latex(index=False, escape=False))
    print('  Saved table2_sampling_diagnostics')

    rows3 = []
    for mode in ('Relaxed', 'Strict'):
        m = df_comp[(df_comp['mode'] == mode) & (df_comp['config_name'] == 'L0_full')]
        rows3.append({
            'Mode':        mode,
            'Warmup (s)':  f"{m['time_t_warmup'].mean():.1f}",
            'Phase-2 (s)': f"{m['time_t_phase2'].mean():.1f}",
            'Train (s)':   f"{m['time_t_train_total'].mean():.1f}",
            'Gen (s)':     f"{m['time_t_generation_total'].mean():.2f}",
            'E2E (s)':     f"{m['time_t_end_to_end'].mean():.1f}",
        })
    df_t3 = pd.DataFrame(rows3)
    df_t3.to_csv(os.path.join(TABLES_DIR, 'table3_runtime.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'table3_runtime.tex'), 'w') as f:
        f.write(df_t3.to_latex(index=False, escape=False))
    print('  Saved table3_runtime')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f'Loaded {len(df_comp)} completed runs | '
          f'{df_comp["config_name"].nunique()} configs | '
          f'modes: {list(df_comp["mode"].unique())}')
    print('NOTE: all plots show mean ± 68% CI (bootstrap) across 36 datasets.\n')

    print('[Fig 1] Ablation: Loss & Component ...')
    plot_ablation()

    print('[Fig 2] Sampling diagnostics (2 rows × 3 cols) ...')
    plot_sampling_diagnostics()

    print('[Fig 3] Runtime breakdown ...')
    plot_runtime_breakdown()

    print('[Fig 4] Sensitivity analysis (2 rows × 5 cols) ...')
    plot_sensitivity()

    print('\n[Tables] ...')
    generate_tables()

    print('\nDone →', OUTPUT_DIR)
