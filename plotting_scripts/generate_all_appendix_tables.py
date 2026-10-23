#!/usr/bin/env python3
"""
Generate complete set of publication-ready appendix tables (LaTeX and CSV)
for the VLDB paper rebuttal / supplementary material.
"""

import os
import glob
import pandas as pd
import numpy as np

from pathlib import Path

# Resolve project root dynamically
REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'plotting_scripts' else Path('.').resolve()

OUTPUT_DIR = str(REPO_ROOT / "supplementary_material/tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _find_files(pattern_relative):
    # Try relative to repo root, current dir, and fallback
    for base in [REPO_ROOT, Path('.')]:
        files = glob.glob(str(base / pattern_relative))
        if files:
            return files
    return []

def load_data():
    rel_files = _find_files('ablation/relaxed/results/*.csv') or _find_files('ablation_relaxed/results/*.csv')
    if not rel_files:
        raise FileNotFoundError("Could not find relaxed ablation results.")
    df_rel = pd.concat([pd.read_csv(f) for f in rel_files], ignore_index=True)
    df_rel['mode'] = 'Relaxed'

    str_files = _find_files('ablation/strict/results/*.csv') or _find_files('ablation_strict/results/*.csv')
    if not str_files:
        raise FileNotFoundError("Could not find strict ablation results.")
    df_str = pd.concat([pd.read_csv(f) for f in str_files], ignore_index=True)
    df_str['mode'] = 'Strict'

    df_all = pd.concat([df_rel, df_str], ignore_index=True)
    return df_all

df_all = load_data()
df_comp = df_all[df_all['status'] == 'completed'].copy()

# ==============================================================================
# TABLE A1: Full Loss & Component Ablation (Table 1 enriched with end-to-end time)
# ==============================================================================
def gen_table_a1():
    configs = [
        ('L0_full', 'Full Framework (SyRFDRes)'),
        ('L2_no_softRFD', 'w/o SoftRFD Loss ($\\mathcal{L}_{\\mathrm{RFD}}=0$)'),
        ('L1_no_MMD', 'w/o MMD Loss ($\\mathcal{L}_{\\mathrm{MMD}}=0$)'),
        ('L3_no_composite', 'w/o Composite Regularizer ($\\mathcal{L}_{\\mathrm{comp}}=0$)'),
        ('L4_no_SWD', 'w/o SWD Loss ($\\mathcal{L}_{\\mathrm{SWD}}=0$)'),
        ('L5_no_coverage', 'w/o Coverage Loss ($\\mathcal{L}_{\\mathrm{cov}}=0$)'),
        ('L6_no_anticollapse', 'w/o Anti-collapse Penalty'),
        ('C1_no_RFD_validator', 'w/o RFD Validator Filter'),
        ('C2_validator_relaxed', 'Validator in Relaxed Mode'),
        ('C4_no_rolling_buffer', 'w/o Rolling Buffer'),
        ('C5_no_warmup', 'w/o Warmup Phase (Phase 1)'),
        ('C6_MLP_no_residual', 'w/o ResNet (Plain MLP Generator)'),
        ('C9_no_early_stopping', 'w/o Early Stopping')
    ]
    cfg_keys = [c[0] for c in configs]
    name_map = dict(configs)

    rows = []
    for mode in ['Relaxed', 'Strict']:
        sub = df_comp[(df_comp['mode'] == mode) & (df_comp['config_name'].isin(cfg_keys))]
        mean_df = sub.groupby('config_name')[['eval_swd_eval', 'eval_coverage', 'eval_rfd_violation_rate', 
                                              'gen_acceptance_rate', 'time_t_train_total', 
                                              'time_t_generation_total', 'time_t_end_to_end']].mean()
        for k in cfg_keys:
            if k in mean_df.index:
                r = mean_df.loc[k]
                rows.append({
                    'Mode': mode,
                    'Model Variant': name_map[k],
                    'SWD ($\\downarrow$)': f"{r['eval_swd_eval']:.4f}",
                    'Coverage ($\\uparrow$)': f"{r['eval_coverage']:.4f}",
                    'RFD Viol. ($\\downarrow$)': f"{r['eval_rfd_violation_rate']:.4f}",
                    'Acceptance Rate': f"{r['gen_acceptance_rate']:.4f}",
                    'Train Time (s)': f"{r['time_t_train_total']:.1f}",
                    'Gen Time (s)': f"{r['time_t_generation_total']:.2f}",
                    'Total Time (s)': f"{r['time_t_end_to_end']:.1f}"
                })
    df_t = pd.DataFrame(rows)
    df_t.to_csv(os.path.join(OUTPUT_DIR, "table_A1_ablation_all_variants.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "table_A1_ablation_all_variants.tex"), "w") as f:
        f.write(df_t.to_latex(index=False, escape=False))
    print("Table A1 generated.")

# ==============================================================================
# TABLE A2: Adaptive Sampling & Acceptance Diagnostics
# ==============================================================================
def gen_table_a2():
    rows = []
    for mode in ['Relaxed', 'Strict']:
        m_comp = df_comp[df_comp['mode'] == mode]
        exhausted = (m_comp['gen_obtained'] < m_comp['gen_requested']).sum()
        total_eval = len(m_comp)
        rows.append({
            'Mode': mode,
            'Evaluation Cohort': 'Benchmark Total (All Settings)',
            'Budget Exhausted': f"{exhausted} / {total_eval} ({exhausted/total_eval*100:.2f}\\%)",
            'Mean Acc. Rate': f"{m_comp['gen_acceptance_rate'].mean():.4f}",
            'Min Acc. Rate': f"{m_comp['gen_acceptance_rate'].min():.4f}",
            'Avg Rounds': f"{m_comp['gen_attempts'].mean():.2f}",
            'Gen Time (s)': f"{m_comp['time_t_generation_total'].mean():.2f}",
            'Total Time (s)': f"{m_comp['time_t_end_to_end'].mean():.1f}"
        })
        for s_val in [50, 100, 200, 500]:
            sub = df_comp[(df_comp['mode'] == mode) & (df_comp['config_name'] == f'R_nsamples={s_val}')]
            if len(sub) > 0:
                s_exh = (sub['gen_obtained'] < sub['gen_requested']).sum()
                rows.append({
                    'Mode': mode,
                    'Evaluation Cohort': f'Target $s={s_val}$ Tuples',
                    'Budget Exhausted': f"{s_exh} / {len(sub)} ({s_exh/len(sub)*100:.2f}\\%)",
                    'Mean Acc. Rate': f"{sub['gen_acceptance_rate'].mean():.4f}",
                    'Min Acc. Rate': f"{sub['gen_acceptance_rate'].min():.4f}",
                    'Avg Rounds': f"{sub['gen_attempts'].mean():.2f}",
                    'Gen Time (s)': f"{sub['time_t_generation_total'].mean():.2f}",
                    'Total Time (s)': f"{sub['time_t_end_to_end'].mean():.1f}"
                })
    df_t = pd.DataFrame(rows)
    df_t.to_csv(os.path.join(OUTPUT_DIR, "table_A2_adaptive_sampling_diagnostics.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "table_A2_adaptive_sampling_diagnostics.tex"), "w") as f:
        f.write(df_t.to_latex(index=False, escape=False))
    print("Table A2 generated.")

# ==============================================================================
# TABLE A3: Comprehensive Sensitivity Analysis
# ==============================================================================
def gen_table_a3():
    sens_groups = [
        ('Threshold Multiplier $T$', [('S_rfd_threshold=2', '$T=2$'), ('S_rfd_threshold=4', '$T=4$'), ('S_rfd_threshold=8', '$T=8$')]),
        ('Temperature $\\tau$', [('S_tau=0.1', '$\\tau=0.10$'), ('S_tau=0.25', '$\\tau=0.25$'), ('S_tau=0.5', '$\\tau=0.50$ (Default)'), ('S_tau=1.0', '$\\tau=1.00$'), ('S_tau=2.0', '$\\tau=2.00$')]),
        ('Weight $\\lambda_{\\mathrm{RFD}}$', [('S_lambda_rfd=0.0', '$\\lambda_{\\mathrm{RFD}}=0.0$'), ('S_lambda_rfd=5.0', '$\\lambda_{\\mathrm{RFD}}=5.0$'), ('S_lambda_rfd=10.0', '$\\lambda_{\\mathrm{RFD}}=10.0$ (Default)'), ('S_lambda_rfd=20.0', '$\\lambda_{\\mathrm{RFD}}=20.0$')]),
        ('Weight $\\lambda_{\\mathrm{comp}}$', [('S_lambda_composite=0.0', '$\\lambda_{\\mathrm{comp}}=0.0$'), ('S_lambda_composite=1.0', '$\\lambda_{\\mathrm{comp}}=1.0$ (Default)'), ('S_lambda_composite=2.0', '$\\lambda_{\\mathrm{comp}}=2.0$')]),
        ('Weight $w_{\\mathrm{cov}}$', [('S_w_coverage=0.0', '$w_{\\mathrm{cov}}=0.0$'), ('S_w_coverage=0.2', '$w_{\\mathrm{cov}}=0.2$'), ('S_w_coverage=0.5', '$w_{\\mathrm{cov}}=0.5$ (Default)'), ('S_w_coverage=1.0', '$w_{\\mathrm{cov}}=1.0$')]),
        ('Weight $w_{\\mathrm{SWD}}$', [('S_w_swd=0.0', '$w_{\\mathrm{SWD}}=0.0$'), ('S_w_swd=1.0', '$w_{\\mathrm{SWD}}=1.0$ (Default)'), ('S_w_swd=2.0', '$w_{\\mathrm{SWD}}=2.0$')]),
        ('SWD Projections $P$', [('S_n_projections=10', '$P=10$'), ('S_n_projections=50', '$P=50$ (Default)'), ('S_n_projections=200', '$P=200$')]),
        ('Batch Size $B$', [('S_batch_size=32', '$B=32$'), ('S_batch_size=64', '$B=64$ (Default)'), ('S_batch_size=128', '$B=128$')]),
        ('Learning Rate $\\eta$', [('S_lr=0.0001', '$\\eta=10^{-4}$'), ('S_lr=0.001', '$\\eta=10^{-3}$ (Default)'), ('S_lr=0.005', '$\\eta=5\\cdot 10^{-3}$')]),
        ('Warmup Ratio', [('S_warmup_ratio=0.0', 'Ratio $=0.0$'), ('S_warmup_ratio=0.2', 'Ratio $=0.2$ (Default)'), ('S_warmup_ratio=0.4', 'Ratio $=0.4$')])
    ]

    rows = []
    for group_name, cfg_list in sens_groups:
        for cfg_id, val_label in cfg_list:
            for mode in ['Relaxed', 'Strict']:
                sub = df_comp[(df_comp['mode'] == mode) & (df_comp['config_name'] == cfg_id)]
                if len(sub) > 0:
                    rows.append({
                        'Hyperparameter': group_name,
                        'Setting': val_label,
                        'Mode': mode,
                        'SWD ($\\downarrow$)': f"{sub['eval_swd_eval'].mean():.4f}",
                        'Coverage ($\\uparrow$)': f"{sub['eval_coverage'].mean():.4f}",
                        'Acc. Rate': f"{sub['gen_acceptance_rate'].mean():.4f}",
                        'Train Time (s)': f"{sub['time_t_train_total'].mean():.1f}"
                    })
    df_t = pd.DataFrame(rows)
    df_t.to_csv(os.path.join(OUTPUT_DIR, "table_A3_hyperparameter_sensitivity.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "table_A3_hyperparameter_sensitivity.tex"), "w") as f:
        f.write(df_t.to_latex(index=False, escape=False))
    print("Table A3 generated.")

# ==============================================================================
# TABLE A4: Grid Search on Joint Loss Weights
# ==============================================================================
def gen_table_a4():
    g_cfgs = [c for c in df_comp['config_name'].unique() if c.startswith('G_')]
    import re
    rows = []
    for cfg in sorted(g_cfgs):
        m = re.search(r'lambda_rfd=([0-9.]+)_w_coverage=([0-9.]+)', cfg)
        if not m:
            continue
        lam_val, cov_val = m.group(1), m.group(2)
        for mode in ['Relaxed', 'Strict']:
            sub = df_comp[(df_comp['mode'] == mode) & (df_comp['config_name'] == cfg)]
            if len(sub) > 0:
                rows.append({
                    '$\\lambda_{\\mathrm{RFD}}$': lam_val,
                    '$w_{\\mathrm{cov}}$': cov_val,
                    'Mode': mode,
                    'SWD ($\\downarrow$)': f"{sub['eval_swd_eval'].mean():.4f}",
                    'Coverage ($\\uparrow$)': f"{sub['eval_coverage'].mean():.4f}",
                    'Acc. Rate': f"{sub['gen_acceptance_rate'].mean():.4f}",
                    'Train Time (s)': f"{sub['time_t_train_total'].mean():.1f}"
                })
    df_t = pd.DataFrame(rows)
    df_t.to_csv(os.path.join(OUTPUT_DIR, "table_A4_gridsearch_joint_weights.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "table_A4_gridsearch_joint_weights.tex"), "w") as f:
        f.write(df_t.to_latex(index=False, escape=False))
    print("Table A4 generated.")

# ==============================================================================
# TABLE A5: Per-Dataset Diagnostic Breakdown (Default SyRFDRes L0_full)
# ==============================================================================
def gen_table_a5():
    l0 = df_comp[df_comp['config_name'] == 'L0_full']
    datasets = sorted(l0['dataset'].unique())

    rows = []
    for ds in datasets:
        sub_r = l0[(l0['dataset'] == ds) & (l0['mode'] == 'Relaxed')]
        sub_s = l0[(l0['dataset'] == ds) & (l0['mode'] == 'Strict')]

        row = {'Dataset': ds}
        if len(sub_r) > 0:
            r = sub_r.iloc[0]
            row['SWD (Rlx)'] = f"{r['eval_swd_eval']:.4f}"
            row['Cov (Rlx)'] = f"{r['eval_coverage']:.4f}"
            row['Acc (Rlx)'] = f"{r['gen_acceptance_rate']:.4f}"
            row['Gen Time (Rlx)'] = f"{r['time_t_generation_total']:.2f}s"
            row['Total Time (Rlx)'] = f"{r['time_t_end_to_end']:.1f}s"
        else:
            row['SWD (Rlx)'] = row['Cov (Rlx)'] = row['Acc (Rlx)'] = row['Gen Time (Rlx)'] = row['Total Time (Rlx)'] = '-'

        if len(sub_s) > 0:
            s = sub_s.iloc[0]
            row['SWD (Str)'] = f"{s['eval_swd_eval']:.4f}"
            row['Cov (Str)'] = f"{s['eval_coverage']:.4f}"
            row['Acc (Str)'] = f"{s['gen_acceptance_rate']:.4f}"
            row['Gen Time (Str)'] = f"{s['time_t_generation_total']:.2f}s"
            row['Total Time (Str)'] = f"{s['time_t_end_to_end']:.1f}s"
        else:
            row['SWD (Str)'] = row['Cov (Str)'] = row['Acc (Str)'] = row['Gen Time (Str)'] = row['Total Time (Str)'] = 'N/A'
        rows.append(row)

    df_t = pd.DataFrame(rows)
    df_t.to_csv(os.path.join(OUTPUT_DIR, "table_A5_per_dataset_breakdown.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "table_A5_per_dataset_breakdown.tex"), "w") as f:
        f.write(df_t.to_latex(index=False, escape=False))
    print("Table A5 generated.")

if __name__ == '__main__':
    gen_table_a1()
    gen_table_a2()
    gen_table_a3()
    gen_table_a4()
    gen_table_a5()
    print("All appendix tables generated successfully.")
