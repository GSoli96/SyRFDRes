#!/usr/bin/env python3
"""
generate_latex_table_with_std.py
=============================================================================
Fonde automaticamente le metriche di Media e Deviazione Standard da:
  - ./evaluation_metrics/aggregated/aggregated_diversity_full_mean.csv
  - ./evaluation_metrics/aggregated/aggregated_fidelity_full_mean.csv

Genera il codice LaTeX della Tabella con le metriche nel formato:
  $mean_{\pm std}$
=============================================================================
"""

import os
import pandas as pd
import numpy as np

DIV_CSV = "./evaluation_metrics/aggregated/aggregated_diversity_full_mean.csv"
FID_CSV = "./evaluation_metrics/aggregated/aggregated_fidelity_full_mean.csv"
OUT_TEX = "./evaluation_metrics/aggregated/table_main_with_std.tex"

# Lista ordinata dei metodi come compaiono nella tabella del paper
METHODS = [
    # (Label in CSV, Nome visualizzato nel paper)
    ("casTGAN", r"CasTGAN \cite{ctgan}"),
    ("CTABGAN", r"CTAB-GAN \cite{zhao21}"),
    ("CTABGAN+", r"CTAB-GAN+~\cite{zhao2023ctab}"),
    ("cuts", r"CuTS \cite{vero2023cuts}"),
    ("DDPM", r"DDPM \cite{ho2020denoising}"),
    ("fsmote", r"FSMOTE \cite{sonoda2023fair}"),
    ("GOGGLE", r"GOGGLE \cite{liu2023goggle}"),
    ("prefair", r"PreFair \cite{pujol2022prefair}"),
    ("SMOTE", r"SMOTE \cite{chawla2002smote}"),
    ("SMOTE-C", r"SMOTE-CDNN \cite{wang2023synthetic}"),
    ("tabfairgan", r"Tabfairgan \cite{rajabi2022tabfairgan}"),
    ("tabularargn", r"TabularARGN \cite{tiwald2025tabularargn}"),
    ("TVAE", r"TVAE \cite{ctgan}"),
    ("MIDRULE", None),
    ("SyRFDφ2/r", r"\system $\phi=2$/\emph{rlx}"),
    ("SyRFDφ2/s", r"\system $\phi=2$/\emph{str}"),
    ("SyRFDφ4/r", r"\system $\phi=4$/\emph{rlx}"),
    ("SyRFDφ4/s", r"\system $\phi=4$/str"),
    ("SyRFDφ8/r", r"\system $\phi=8$/\emph{rlx}"),
    ("SyRFDφ8/s", r"\system $\phi=8$/\emph{str}"),
    ("SyRFDφe/r", r"\system \emph{var}/\emph{rlx}"),
    ("SyRFDφe/s", r"\system \emph{var}/\emph{str}"),
]

# Metriche di Diversity
DIVERSITY_COLS = [
    # (col_base, decimals, macro_name)
    ("Silhouette", 4, "gSilh"),
    ("Davies-Bouldin", 4, "gDBI"),
    ("Compactness_cls_k", 4, None),
    ("Precision", 4, "gPrec"),
    ("Recall", 4, "gRec"),
    ("Density", 4, "gDens"),
    ("Coverage", 4, "gCov"),
    ("mDCR", 4, None),
    ("Coverage_div", 4, "gCovq"),
]

# Metriche di Fidelity
FIDELITY_COLS = [
    ("KL_mean", 4, "gKL"),
    ("JS_mean", 4, "gJS"),
    ("KS_mean", 4, "gKS"),
    ("Chi2_mean", 4, "gChi"),
    ("WD_mean", 4, "gWD"),
    ("pMSE", 4, "gpMSE"),
    ("MMD", 4, "gMMD"),
    ("PCA_WD_mean", 4, "gWDPCA"),
    ("Moment_mean_diff", 4, "gDmu"),
    ("Moment_var_diff", 4, "gDv"),
    ("Moment_skew_diff", 4, "gDskew"),
]

def fmt_val_std(mean_val, std_val, dec=4, macro=None, use_macro=False):
    if pd.isna(mean_val):
        return "--"
    std_str = f"{std_val:.{dec}f}" if not pd.isna(std_val) else "0.0000"
    mean_str = f"{mean_val:.{dec}f}"
    
    # Formato $mean_{\pm std}$
    core = f"${mean_str}_{{ \\pm {std_str} }}$"
    if use_macro and macro:
        return f"\\{macro}{{{core}}}"
    return core

def generate_table(use_macros=False):
    df_div = pd.read_csv(DIV_CSV).set_index("Label")
    df_fid = pd.read_csv(FID_CSV).set_index("Label")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength\tabcolsep{1.5pt}")
    lines.append(r"\renewcommand*{\arraystretch}{1.05}")
    lines.append(r"\scalebox{0.55}{")
    lines.append(r"\begin{tabular}{lccccccccc|ccccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{1}{l}{\multirow{2}{*}{\textbf{Algorithm}}} & \multicolumn{9}{c|}{\textbf{Diversity}} & \multicolumn{11}{c}{\textbf{Fidelity}} \\")
    lines.append(r"& $\mathrm{Silh}\xrightarrow 0$ & $\mathrm{DBI}\uparrow$ & $\mathcal{I}_{Com}$ & $\mathrm{Prec}\uparrow$ & $\mathrm{Rec}\uparrow$ & $\mathrm{Dens}\uparrow$ & $\mathrm{Cov}\uparrow$ & $\mathrm{mDCR}$ & $\mathrm{Cov}_q\uparrow$ & KL$\downarrow$ & JS$\downarrow$ & KS$\downarrow$ & $\chi^2\downarrow$ & WD$\downarrow$ & pMSE$\downarrow$ & MMD$\downarrow$ & $\mathrm{WD}_{\mathrm{PCA}}\downarrow$ & $\Delta_\mu\downarrow$ & $\Delta_v \downarrow$ & $\Delta_{\mathrm{skew}}\downarrow$ \\")
    lines.append(r"\midrule")

    for lab, display_name in METHODS:
        if lab == "MIDRULE":
            lines.append(r"\midrule")
            continue

        row_cells = [display_name]

        # Diversity metrics
        for col_name, dec, macro in DIVERSITY_COLS:
            m_col = f"{col_name}_mean"
            s_col = f"{col_name}_std"
            mean_v = df_div.loc[lab, m_col] if lab in df_div.index and m_col in df_div.columns else np.nan
            std_v = df_div.loc[lab, s_col] if lab in df_div.index and s_col in df_div.columns else np.nan
            row_cells.append(fmt_val_std(mean_v, std_v, dec=dec, macro=macro, use_macro=use_macros))

        # Fidelity metrics
        for col_name, dec, macro in FIDELITY_COLS:
            m_col = f"{col_name}_mean"
            s_col = f"{col_name}_std"
            mean_v = df_fid.loc[lab, m_col] if lab in df_fid.index and m_col in df_fid.columns else np.nan
            std_v = df_fid.loc[lab, s_col] if lab in df_fid.index and s_col in df_fid.columns else np.nan
            row_cells.append(fmt_val_std(mean_v, std_v, dec=dec, macro=macro, use_macro=use_macros))

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{Diversity and fidelity metrics reported as $\text{mean}_{\pm \text{std}}$ across all benchmark datasets.}")
    lines.append(r"\label{tab:diversity_fidelity_mean_std}")
    lines.append(r"\end{table*}")

    tex_content = "\n".join(lines)
    with open(OUT_TEX, "w") as f:
        f.write(tex_content)
    print(f"Table successfully saved to {OUT_TEX}")
    return tex_content

if __name__ == "__main__":
    generate_table(use_macros=False)
