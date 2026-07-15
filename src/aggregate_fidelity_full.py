#!/usr/bin/env python3
# =============================================================================
# aggregate_fidelity.py
#
# Aggrega i file ./evaluation_metrics/fidelity/fidelity_<dataset>.csv
# in tabelle pronte per il paper:
#
#   out/aggregated_fidelity_full_<stat>.csv   tutte le configurazioni x tutte
#                                             le metriche: media, mediana, std,
#                                             rank medio
#   out/table_main_<stat>.tex                 tabella per il main paper (LLM
#                                             collassati al best-per-modello o
#                                             per esteso, rank medio finale)
#   out/table_llm_appendix_<stat>.tex         dettaglio delle 8 strategie per LLM
#
# Aggregazione direction-aware:
#   lower   = valori bassi migliori   (KL, JS, KS, Chi2, WD, pMSE, MAD, MMD,
#                                       PCA_WD, Moment_*_diff: tutte misure di
#                                       divergenza/distanza/differenza dal reale)
#   higher  = valori alti migliori    (Q_multi_attr_sim: e' una similarita')
#   report  = senza direzione monotona: riportata ma esclusa dai rank
#
# Il rank e' calcolato PER DATASET tra tutte le configurazioni, poi mediato:
# robusto a outlier e a scale diverse tra dataset (a differenza della media).
#
# Struttura e convenzioni identiche a aggregate_diversity.py, per coerenza
# tra le due pipeline di aggregazione (stesso schema di Label/Family, stesso
# meccanismo di rank e la stessa generazione di tabelle LaTeX).
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd

# ----------------------------- CONFIGURAZIONE --------------------------------
INPUT_DIR = "./evaluation_metrics/fidelity"
OUT_DIR = "./evaluation_metrics/aggregated"
# Le righe SYRFD_thrX SENZA suffisso di modalita' (naming vecchio) sono escluse
# di default: sembrano run precedenti all'introduzione di strict/relaxed.
EXCLUDE_LEGACY_SYRFD = True
# statistica mostrata nelle celle delle tabelle LaTeX: "median" (robusta agli
# outlier, es. picchi di MMD/pMSE su generatori instabili) oppure "mean"
CELL_STAT = "mean"
# True: la tabella main mostra TUTTE le strategie LLM (tabella lunga, consigliato
# table* a due colonne); False: LLM collassati al best-per-modello
FULL_LLM_IN_MAIN = True

# direzione per metrica (vedi header)
METRIC_DIR = {
    "KL_mean": "lower",
    "JS_mean": "lower",
    #"Q_multi_attr_sim": "higher",
    "KS_mean": "lower",
    "Chi2_mean": "lower",
    "WD_mean": "lower",
    "pMSE": "lower",
    "MAD_mean": "lower",
    "MMD": "lower",
    "PCA_WD_mean": "lower",
    "Moment_mean_diff": "lower",
    "Moment_var_diff": "lower",
    "Moment_skew_diff": "lower",
    "Moment_kurt_diff": "lower",
}
# metriche nella tabella del main (le altre restano nel CSV completo)
MAIN_METRICS = ["KL_mean", "JS_mean", 
#"Q_multi_attr_sim", 
"KS_mean", "Chi2_mean", "WD_mean", "pMSE", "MMD", "PCA_WD_mean", "Moment_mean_diff","Moment_var_diff","Moment_skew_diff" ]
# ------------------------------------------------------------------------------

STRATEGY_KEYS = ["confidence", "decision_tree", "distribution", "ensemble",
                 "hierarchical", "react", "schema", "self"]
STRATEGY_SHORT = {"confidence": "conf", "decision_tree": "dtree",
                  "distribution": "distr", "ensemble": "ens",
                  "hierarchical": "hier", "react": "react",
                  "schema": "schema", "self": "self"}


def model_prefix(algo_lower: str):
    if "devstral" in algo_lower:
        return "Dev"
    if "gemma3" in algo_lower:
        return "G3"
    if "gemma4" in algo_lower:
        return "G4"
    if "gpt" in algo_lower:
        # distingue le due taglie di GPT-OSS
        return "GPT120" if "120b" in algo_lower else "GPT20"
    return None


def get_label(algo: str, file: str):
    """Etichetta breve e non ambigua (fix della collisione SyRFD + GPT 20/120)."""
    a, f = algo.lower(), str(file).lower()
    fixed = {"SMOTE": "SMOTE", "SMOTECDNN": "SMOTE-C", "casTGAN": "casTGAN",
             "ddpm": "DDPM", "tabdiff": "TabDiff", "tvae": "TVAE",
             "ctabgan": "CTABGAN", "ctabganp": "CTABGAN+", "GOGGLE": "GOGGLE",
             "deepseek": "DeepSeek", "llama": "LLaMA",
              "Dataset_balanced_TABFAIRGDT" : "TABFAIRGDT", "Dataset_balanced_cuts" : "cuts", 
              "Dataset_balanced_fsmote" : "fsmote", "Dataset_balanced_prefair" : "prefair",
               "Dataset_balanced_tabfairgan" : "tabfairgan", "Dataset_balanced_tabularargn" : "tabularargn"
             }
    if algo in fixed:
        return fixed[algo], "LLM-zs" if algo in ("deepseek", "llama") else (
            "Geometric" if algo.startswith("SMOTE") else "DeepGen")
    if algo.startswith("SYRFD"):
        parts = algo.split("_")
        thr = parts[1][-1]
        mode = parts[2] if len(parts) > 2 else "legacy"
        suffix = {"strict": "s", "relaxed": "r", "legacy": "*"}[mode]
        return f"SyRFD\u03C6{thr}/{suffix}", "SyRFD"
    mp = model_prefix(a)
    if mp:
        for k in STRATEGY_KEYS:
            if k in f:
                return f"{mp}/{STRATEGY_SHORT[k]}", "LLM"
        return mp, "LLM"
    return algo, "Other"


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(INPUT_DIR, "fidelity_*.csv"))):
        dataset = os.path.basename(path)[len("fidelity_"):-len(".csv")]
        df = pd.read_csv(path)
        if EXCLUDE_LEGACY_SYRFD:
            legacy = df["Algoritmo"].str.match(r"SYRFD_thr\d+$")
            df = df[~legacy]
        lab = df.apply(lambda r: get_label(r["Algoritmo"], r["File"]), axis=1)
        df = df.assign(Label=[l[0] for l in lab], Family=[l[1] for l in lab],
                       Dataset=dataset)
        rows.append(df)
    if not rows:
        raise SystemExit(f"Nessun file fidelity_*.csv in {INPUT_DIR}")
    return pd.concat(rows, ignore_index=True)


def oriented(series: pd.Series, mode: str) -> pd.Series:
    """Trasforma i valori cosi' che 'piu' piccolo = migliore' per il ranking."""
    if mode == "lower":
        return series
    if mode == "higher":
        return -series
    raise ValueError(mode)


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank per (dataset, metrica) tra tutte le configurazioni presenti."""
    for m, mode in METRIC_DIR.items():
        if mode == "report" or m not in df.columns:
            continue
        df[f"rank_{m}"] = (df.groupby("Dataset")[m]
                             .transform(lambda s: oriented(s, mode).rank(method="average")))
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["rank_avg"] = df[rank_cols].mean(axis=1)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRIC_DIR if m in df.columns]
    aggs = {m: ["mean", "median", "std"] for m in metrics}
    aggs["rank_avg"] = ["mean"]
    g = df.groupby(["Family", "Label"]).agg(aggs)
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    g["n_datasets"] = df.groupby(["Family", "Label"])["Dataset"].nunique()
    return g.sort_values("rank_avg_mean").reset_index()


def collapse_llm_best(df: pd.DataFrame) -> pd.DataFrame:
    """Per il main: per ogni (modello LLM, dataset) tiene la strategia con
    rank medio migliore su quel dataset -> riga '<Model> (best)'."""
    is_llm = df["Family"] == "LLM"
    llm = df[is_llm].copy()
    llm["Model"] = llm["Label"].str.split("/").str[0]
    idx = llm.groupby(["Model", "Dataset"])["rank_avg"].idxmin()
    best = llm.loc[idx].copy()
    best["Label"] = best["Model"] + " (best)"
    best.drop(columns="Model", inplace=True)
    return pd.concat([df[~is_llm], best], ignore_index=True)


def fmt(x, digits=4):
    return "--" if pd.isna(x) else f"{x:.{digits}f}"


ARROW = {"lower": r"$\downarrow$", "higher": r"$\uparrow$", "report": ""}


def to_latex(agg: pd.DataFrame, metrics, caption, label, star=False) -> str:
    env = "table*" if star else "table"
    header = " & ".join(
        [r"\textbf{Method}"] +
        [rf"\textbf{{{m.replace('_mean','').replace('_',' ')}}} {ARROW[METRIC_DIR[m]]}"
         for m in metrics] +
        [r"\textbf{Avg.\ rank} $\downarrow$"]) + r" \\"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", r"\scriptsize",
             r"\setlength\tabcolsep{3pt}",
             r"\begin{tabular}{l" + "r" * (len(metrics) + 1) + "}",
             r"\toprule", header, r"\midrule"]
    best_rank = agg["rank_avg_mean"].min()
    prev_family = None
    for _, r in agg.iterrows():
        if prev_family is not None and r["Family"] != prev_family:
            lines.append(r"\midrule")
        prev_family = r["Family"]
        cells = [r["Label"].replace("_", r"\_")]
        for m in metrics:
            cells.append(fmt(r[f"{m}_{CELL_STAT}"]))
        rank_txt = fmt(r["rank_avg_mean"], 2)
        if r["rank_avg_mean"] == best_rank:
            rank_txt = rf"\textbf{{{rank_txt}}}"
        cells.append(rank_txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\end{{{env}}}"]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = add_ranks(load_all())

    # 1) CSV completo (tutte le configurazioni, tutte le statistiche)
    full = aggregate(df)
    full.to_csv(os.path.join(OUT_DIR, "aggregated_fidelity_full_{}.csv".format(CELL_STAT)), index=False)

    # 2) tabella main: famiglie ordinate; LLM per esteso (tutte le strategie)
    #    oppure collassati al best-per-modello, secondo FULL_LLM_IN_MAIN
    src = df if FULL_LLM_IN_MAIN else collapse_llm_best(df)
    main_df = aggregate(src)
    order = {"Geometric": 0, "DeepGen": 1, "LLM": 2, "LLM-zs": 3, "Other":4,"SyRFD": 5}
    main_df = main_df.sort_values(["Family", "rank_avg_mean"],
                                  key=lambda s: s.map(order) if s.name == "Family" else s)
    stat_txt = "median" if CELL_STAT == "median" else "mean"
    llm_txt = ("all prompting strategies are reported per LLM"
               if FULL_LLM_IN_MAIN else
               "for LLMs, the best prompting strategy per dataset is selected")
    tex = to_latex(main_df, MAIN_METRICS,
                   caption=(rf"Fidelity results: per-dataset {stat_txt} over the "
                            rf"benchmarks and average rank (computed per dataset, "
                            rf"direction-aware); {llm_txt}."),
                   label="tab:fidelity_main",
                   star=FULL_LLM_IN_MAIN)
    with open(os.path.join(OUT_DIR, "table_main_fidelity_{}.tex".format(CELL_STAT)), "w") as f:
        f.write(tex)

    # 3) appendice: dettaglio strategie LLM
    llm_df = full[full["Family"].isin(["LLM", "LLM-zs"])]
    tex_llm = to_latex(llm_df, MAIN_METRICS,
                       caption=(r"Fidelity results of the LLM-based generators, "
                                r"per model and prompting strategy (mean over "
                                r"the benchmarks; average rank as in "
                                r"Table~\ref{tab:fidelity_main})."),
                       label="tab:fidelity_llm")
    with open(os.path.join(OUT_DIR, "table_llm_appendix_fidelity_{}.tex".format(CELL_STAT)), "w") as f:
        f.write(tex_llm)

    print(f"Dataset letti: {df['Dataset'].nunique()}  |  configurazioni: {df['Label'].nunique()}")
    print(f"Output in {OUT_DIR}/: aggregated_fidelity_full_{CELL_STAT}.csv, table_main_{CELL_STAT}.tex, table_llm_appendix_{CELL_STAT}.tex")
    print("\nTop 10 per rank medio:")
    print(full[["Family", "Label", "rank_avg_mean", "n_datasets"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()