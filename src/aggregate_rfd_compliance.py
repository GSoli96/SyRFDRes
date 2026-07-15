import os
import glob
import re
import numpy as np
import pandas as pd

# ----------------------------- CONFIGURAZIONE --------------------------------
INPUT_DIR = "./evaluation_metrics/RFDcompliance"
OUT_DIR = "./evaluation_metrics/aggregated"
# Le righe SYRFD_thrX SENZA suffisso di modalita' (naming vecchio) sono escluse
# di default: sembrano run precedenti all'introduzione di strict/relaxed.
EXCLUDE_LEGACY_SYRFD = True
# statistica mostrata nelle celle delle tabelle LaTeX: "median" (robusta agli
# outlier) oppure "mean"
CELL_STAT = "mean"
# True: la tabella main mostra TUTTE le strategie LLM (tabella lunga, consigliato
# table* a due colonne); False non e' supportato per queste metriche (vedi nota
# sotto) e viene comunque trattato come True.
FULL_LLM_IN_MAIN = True

# Le metriche di RFD compliance NON hanno una direzione univoca di
# miglioramento (non sono, come per la fidelity, tutte "lower is better" o
# "higher is better"): per questo motivo qui non si calcola alcun ranking
# direction-aware ne' si evidenzia un "best" in grassetto. Le tabelle
# riportano solo i valori aggregati (mean/median/std) per configurazione.
MAIN_METRICS = [
    "RFD_tuple_compliance_relaxed",
    "RFD_tuple_compliance_strict",
    "RFD_pair_violation_mixed",
    "RFD_pair_violation_synth",
    "RFD_preserved_deps",
]
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
             "deepseek": "DeepSeek", "llama": "LLaMA"}
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


def extract_dataset(file_name: str) -> str:
    """Ricava il nome del dataset dalla colonna 'File', ripulendo i suffissi
    introdotti dai vari generatori/strategie (es. '_balanced',
    '_balanced_confidence', '_thr2_relaxed_balanced', ...)."""
    f = str(file_name)
    if f.endswith(".csv"):
        f = f[:-4]
    # naming SyRFD: <dataset>_thrN_(relaxed|strict)_balanced
    m = re.match(r"^(.*?)_thr\d+_(?:relaxed|strict)_balanced", f)
    if m:
        return m.group(1)
    # naming generico (incluse le strategie LLM): <dataset>_balanced[...]
    m = re.match(r"^(.*?)_balanced", f)
    if m:
        return m.group(1)
    return f


def load_all() -> pd.DataFrame:
    rows = []
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not paths:
        raise SystemExit(f"Nessun file .csv in {INPUT_DIR}")
    for path in paths:
        df = pd.read_csv(path)
        if EXCLUDE_LEGACY_SYRFD and "Algoritmo" in df.columns:
            legacy = df["Algoritmo"].str.match(r"SYRFD_thr\d+$")
            df = df[~legacy]
        lab = df.apply(lambda r: get_label(r["Algoritmo"], r["File"]), axis=1)
        df = df.assign(Label=[l[0] for l in lab],
                       Family=[l[1] for l in lab],
                       Dataset=df["File"].apply(extract_dataset))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def aggregate(df: pd.DataFrame, group_cols=("Family", "Label", "RFD_threshold")) -> pd.DataFrame:
    """Aggrega per (Family, Label[, RFD_threshold]). La soglia RFD_threshold e'
    un campo discriminante a se stante: righe con soglie diverse NON vengono
    mai mescolate insieme in una stessa cella."""
    group_cols = list(group_cols)
    metrics = [m for m in MAIN_METRICS if m in df.columns]
    # includi anche eventuali altre colonne RFD_* per il csv completo
    extra_metrics = [c for c in df.columns
                     if c.startswith("RFD_") and c not in metrics
                     and c not in ("RFD_threshold", "RFD_epsilon")
                     and pd.api.types.is_numeric_dtype(df[c])]
    all_metrics = metrics + extra_metrics
    aggs = {m: ["mean", "median", "std"] for m in all_metrics}
    g = df.groupby(group_cols).agg(aggs)
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    g["n_datasets"] = df.groupby(group_cols)["Dataset"].nunique()
    g["n_rows"] = df.groupby(group_cols).size()
    return g.reset_index()


def fmt(x, digits=3):
    return "--" if pd.isna(x) else f"{x:.{digits}f}"


def to_latex(agg: pd.DataFrame, metrics, caption, label, star=False) -> str:
    env = "table*" if star else "table"
    header = " & ".join(
        [r"\textbf{Method}"] +
        [rf"\textbf{{{m.replace('RFD_', '').replace('_', ' ')}}}" for m in metrics]
    ) + r" \\"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", r"\scriptsize",
             r"\setlength\tabcolsep{3pt}",
             r"\begin{tabular}{l" + "r" * len(metrics) + "}",
             r"\toprule", header, r"\midrule"]
    prev_family = None
    for _, r in agg.iterrows():
        if prev_family is not None and r["Family"] != prev_family:
            lines.append(r"\midrule")
        prev_family = r["Family"]
        cells = [str(r["Label"]).replace("_", r"\_")]
        for m in metrics:
            cells.append(fmt(r[f"{m}_{CELL_STAT}"]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\end{{{env}}}"]
    return "\n".join(lines)


def thr_tag(thr) -> str:
    return f"thr{int(thr)}" if float(thr).is_integer() else f"thr{thr}".replace(".", "p")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_all()

    # 1) CSV completo (tutte le configurazioni x soglia, tutte le statistiche)
    full = aggregate(df)
    full = full.sort_values(["RFD_threshold", "Family", "Label"])
    full.to_csv(os.path.join(OUT_DIR, f"aggregated_rfdcompliance_full_{CELL_STAT}.csv"), index=False)

    order = {"Geometric": 0, "DeepGen": 1, "LLM": 2, "LLM-zs": 3, "SyRFD": 4, "Other": 5}
    stat_txt = "median" if CELL_STAT == "median" else "mean"

    thresholds = sorted(df["RFD_threshold"].dropna().unique())
    for thr in thresholds:
        tag = thr_tag(thr)
        sub = df[df["RFD_threshold"] == thr]

        # 2) tabella main per questa soglia: tutte le strategie LLM per esteso
        main_df = aggregate(sub, group_cols=("Family", "Label"))
        main_df = main_df.sort_values(["Family", "Label"],
                                      key=lambda s: s.map(order) if s.name == "Family" else s)
        tex = to_latex(
            main_df, MAIN_METRICS,
            caption=(rf"RFD compliance results at threshold {thr:g}: per-dataset "
                     rf"{stat_txt} over the benchmarks (all prompting strategies "
                     rf"are reported per LLM). Metrics have no assumed direction "
                     rf"of improvement and are reported as-is."),
            label=f"tab:rfdcompliance_main_{tag}",
            star=True,
        )
        with open(os.path.join(OUT_DIR, f"table_main_rfdcompliance_{tag}_{CELL_STAT}.tex"), "w") as f:
            f.write(tex)

        # 3) appendice: dettaglio strategie LLM per questa soglia
        llm_df = full[(full["RFD_threshold"] == thr) & (full["Family"].isin(["LLM", "LLM-zs"]))]
        llm_df = llm_df.sort_values(["Family", "Label"])
        tex_llm = to_latex(
            llm_df, MAIN_METRICS,
            caption=(rf"RFD compliance results of the LLM-based generators at "
                     rf"threshold {thr:g}, per model and prompting strategy "
                     rf"({stat_txt} over the benchmarks)."),
            label=f"tab:rfdcompliance_llm_{tag}",
        )
        with open(os.path.join(OUT_DIR, f"table_llm_appendix_rfdcompliance_{tag}_{CELL_STAT}.tex"), "w") as f:
            f.write(tex_llm)

    print(f"Dataset letti: {df['Dataset'].nunique()}  |  configurazioni: {df['Label'].nunique()}  |  soglie: {thresholds}")
    print(f"Output in {OUT_DIR}/: aggregated_rfdcompliance_full_{CELL_STAT}.csv, "
          f"table_main_rfdcompliance_<thr>_{CELL_STAT}.tex, "
          f"table_llm_appendix_rfdcompliance_<thr>_{CELL_STAT}.tex")
    print("\nAnteprima aggregato completo:")
    print(full.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
