"""
evaluate_metrics_macro.py
=============================================================================
Valuta in modo esaustivo e confronta il dataset ORIGINALE (reale minoritario)
rispetto a TUTTI i dataset bilanciati trovati nelle sottodirectory di
"./augmented_datasets/" (una sottodirectory per ciascun algoritmo di data
augmentation, vedi target_folders in find_matching_datasets).

A differenza delle versioni precedenti, l'output NON e' piu' un'unica
macro-tabella: le metriche vengono divise in 3 CSV distinti, salvati in 3
sottocartelle separate di "./evaluation_metrics/":

    ./evaluation_metrics/fidelity/       -> metriche di FIDELITY
    ./evaluation_metrics/diversity/      -> metriche di DIVERSITY
    ./evaluation_metrics/RFDcompliance/  -> metriche di RFD COMPLIANCE

Sezione "2bis. Metriche di RFD compliance":

parse_rfd_file: formato 'A@..., B@... -> C@...', soglia dal nome file con
pattern RFD<k> o thr<k> (coerente con le cartelle SYRFD_thr2/4/8 e con i
file scoperti in ./discovered_rfds_processed/).
find_rfd_files: individua automaticamente, nella directory
"./discovered_rfds_processed/", tutti i file di RFD relativi al dataset in
esame (es. "RFD2_E0.0_abalone9-18_min.txt", "RFD4_E0.0_...", "RFD8_E0.0_...").
Il numero e il valore delle soglie disponibili sono arbitrari: lo script non
assume che esistano solo le soglie 2/4/8, ma scopre dinamicamente tutti i
file presenti che matchano il dataset.
_rfd_violations: test di violazione vettorizzato, con chunking sulle righe
generate per non esplodere in memoria sui dataset grandi, ed esclusione
della diagonale nel caso sintetico-vs-sintetico.
compute_rfd_compliance: produce cinque metriche, allineate una-a-una con la
Sezione 3 del paper:

  - RFD_tuple_compliance_relaxed: frazione di tuple sintetiche senza
    violazioni verso D+ (= Definizione di relaxed compliance);
  - RFD_tuple_compliance_strict: pulite anche verso le altre sintetiche
    (= strict compliance);
  - RFD_pair_violation_mixed / RFD_pair_violation_synth: tassi a livello di
    coppia, la vista piu' fine per dire di quanto un competitor viola;
  - RFD_preserved_deps: frazione delle RFD di Sigma ancora valide su
    D+ U S (vista a livello di dipendenza).

Per ogni dataset bilanciato trovato, le metriche di RFD compliance vengono
calcolate UNA VOLTA PER OGNI soglia/file di RFD scoperto automaticamente:
il CSV di RFD compliance e' quindi in formato "long" (una riga per ogni
combinazione algoritmo/file bilanciato x soglia RFD).

Due scelte di design da conoscere: (1) i file aumentati dei competitor
contengono anche le righe originali, quindi il set sintetico S viene
ricavato filtrando la classe minoritaria e rimuovendo i duplicati esatti
delle tuple reali - altrimenti le righe originali (compliant per
costruzione) gonfierebbero le metriche di tutti. (2) Tutto e' calcolato su
valori grezzi, come nel validatore, perche' le soglie sono in unita'
originali.

Uso:
    python evaluate_metrics_v5.py \
        --real ./imbalanced_datasets/min/abalone9-18_min.csv \
        --imbalanced ./imbalanced_datasets/abalone9-18.csv \
        --augmented_dir ./augmented_datasets \
        --rfd_dir ./discovered_rfds_processed \
        --output_dir ./evaluation_metrics
"""

import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp, chisquare, skew, kurtosis
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# =========================================================================
# 1. METRICHE DI FIDELITY
# =========================================================================

def compute_kl_divergence(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 30) -> float:
    n_attrs = real_vals.shape[1]
    kl_per_attr = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        # Salta attributi costanti nel reale: tutti i punti reali cadono in un bin -> distribuzione degenere
        if np.std(r) < 1e-6:
            kl_per_attr.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(g, bins=bins)
        q, _ = np.histogram(r, bins=bins)
        p = (p + 1).astype(float); p /= p.sum()
        q = (q + 1).astype(float); q /= q.sum()
        kl_per_attr.append(float(entropy(p, q)))
    return float(np.mean(kl_per_attr))

def compute_js_divergence(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 30) -> float:
    n_attrs = real_vals.shape[1]
    js_per_attr = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        # Salta attributi costanti nel reale
        if np.std(r) < 1e-6:
            js_per_attr.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(g, bins=bins)
        q, _ = np.histogram(r, bins=bins)
        p = (p + 1).astype(float); p /= p.sum()
        q = (q + 1).astype(float); q /= q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        js_per_attr.append(float(js))
    return float(np.mean(js_per_attr))

def compute_q_similarity(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    def l2_norm(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms < 1e-10, 1.0, norms)
    gen_n, real_n = l2_norm(gen_vals), l2_norm(real_vals)
    sim_matrix = gen_n @ real_n.T
    max_sim = sim_matrix.max(axis=1)
    return float(((max_sim + 1.0) / 2.0).mean())

def compute_ks_mean(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    ks_stats = []
    for d in range(real_vals.shape[1]):
        stat, _ = ks_2samp(real_vals[:, d], gen_vals[:, d])
        ks_stats.append(stat)
    return float(np.mean(ks_stats))

def compute_chi2_mean(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 20) -> float:
    chi2_stats = []
    for d in range(real_vals.shape[1]):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        # Salta attributi costanti nel reale: tutti nel primo bin -> chi2 non significativo
        if np.std(r) < 1e-6:
            chi2_stats.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        f_obs, _ = np.histogram(g, bins=bins)
        f_exp, _ = np.histogram(r, bins=bins)
        f_obs = f_obs + 1
        f_exp = f_exp + 1
        f_obs = f_obs * (f_exp.sum() / f_obs.sum())
        stat, _ = chisquare(f_obs, f_exp=f_exp)
        chi2_stats.append(stat / len(g))
    return float(np.mean(chi2_stats))

def compute_wd_mean(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    wd_stats = []
    for d in range(real_vals.shape[1]):
        r_sorted = np.sort(real_vals[:, d])
        g_sorted = np.sort(gen_vals[:, d])
        grid = np.linspace(0, 1, 100)
        r_quant = np.quantile(r_sorted, grid)
        g_quant = np.quantile(g_sorted, grid)
        wd_stats.append(np.mean(np.abs(r_quant - g_quant)))
    return float(np.mean(wd_stats))

def compute_pmse(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    n_real, n_gen = len(real_vals), len(gen_vals)
    X = np.vstack([real_vals, gen_vals])
    y = np.array([0] * n_real + [1] * n_gen)
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    try:
        beta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        preds = np.clip(X_bias @ beta, 0.0, 1.0)
    except:
        preds = np.full(y.shape, n_gen / (n_real + n_gen))
    c = n_gen / (n_real + n_gen)
    pmse = np.mean((preds - c) ** 2)
    return float(pmse)

def compute_mad_mean(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    mu_real = np.mean(real_vals, axis=0)
    mu_gen = np.mean(gen_vals, axis=0)
    std_real = np.std(real_vals, axis=0)
    # Salta attributi a varianza zero nel reale (costanti nella classe minoritaria):
    # il denominatore 1e-8 causerebbe overflow se il generato produce valori != 0
    valid = std_real > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs(mu_real[valid] - mu_gen[valid]) / std_real[valid]))

def compute_mmd(real_vals: np.ndarray, gen_vals: np.ndarray, scales=[0.05, 0.2, 0.5, 1.0, 2.0]) -> float:
    def rbf_kernel_matrix(x, y):
        dist_matrix = cdist(x, y, 'sqeuclidean')
        dim = x.shape[1]
        return sum(np.exp(-dist_matrix / (s * dim + 1e-10)) for s in scales)
    k_xx = rbf_kernel_matrix(gen_vals, gen_vals).mean()
    k_yy = rbf_kernel_matrix(real_vals, real_vals).mean()
    k_xy = rbf_kernel_matrix(gen_vals, real_vals).mean()
    return float(np.clip(k_xx - 2.0 * k_xy + k_yy, a_min=0.0, a_max=None))

def compute_pca_wd_mean(real_vals: np.ndarray, gen_vals: np.ndarray, n_components: int = 3) -> float:
    n_comp = min(n_components, real_vals.shape[1])
    pca = PCA(n_components=n_comp)
    real_pca = pca.fit_transform(real_vals)
    gen_pca = pca.transform(gen_vals)
    wd_stats = []
    for d in range(n_comp):
        r_quant = np.quantile(real_pca[:, d], np.linspace(0, 1, 100))
        g_quant = np.quantile(gen_pca[:, d], np.linspace(0, 1, 100))
        wd_stats.append(np.mean(np.abs(r_quant - g_quant)))
    return float(np.mean(wd_stats))

def compute_moment_diffs(real_vals: np.ndarray, gen_vals: np.ndarray) -> tuple:
    m1_r, m1_g = np.mean(real_vals, axis=0), np.mean(gen_vals, axis=0)
    m2_r, m2_g = np.var(real_vals, axis=0), np.var(gen_vals, axis=0)
    m3_r, m3_g = skew(real_vals, axis=0, bias=False), skew(gen_vals, axis=0, bias=False)
    m4_r, m4_g = kurtosis(real_vals, axis=0, bias=False), kurtosis(gen_vals, axis=0, bias=False)
    m3_r, m3_g = np.nan_to_num(m3_r), np.nan_to_num(m3_g)
    m4_r, m4_g = np.nan_to_num(m4_r), np.nan_to_num(m4_g)
    std_r = np.std(real_vals, axis=0)
    var_r = m2_r

    # Salta attributi a varianza zero (costanti nel reale): denominatore ~0 causa overflow
    valid_std = std_r > 1e-6
    if np.any(valid_std):
        diff_m1 = float(np.mean(np.abs(m1_r[valid_std] - m1_g[valid_std]) / std_r[valid_std]))
    else:
        diff_m1 = 0.0

    valid_var = var_r > 1e-12
    if np.any(valid_var):
        diff_m2 = float(np.mean(np.abs(m2_r[valid_var] - m2_g[valid_var]) / var_r[valid_var]))
    else:
        diff_m2 = 0.0
    diff_m3 = float(np.mean(np.abs(m3_r - m3_g)))
    diff_m4 = float(np.mean(np.abs(m4_r - m4_g)))
    return diff_m1, diff_m2, diff_m3, diff_m4

# =========================================================================
# 2. METRICHE DI DIVERSITY & SPAZIALI
# =========================================================================

def compute_silhouette(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    n_gen, n_real = len(gen_vals), len(real_vals)
    if n_gen < 2 or n_real < 2: return float('nan')
    X = np.vstack([gen_vals, real_vals])
    labels = np.array([0] * n_gen + [1] * n_real)
    try: return float(silhouette_score(X, labels, metric='euclidean'))
    except: return float('nan')

def compute_dbi(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    n_gen, n_real = len(gen_vals), len(real_vals)
    if n_gen < 2 or n_real < 2: return float('nan')
    X = np.vstack([gen_vals, real_vals])
    labels = np.array([0] * n_gen + [1] * n_real)
    try: return float(davies_bouldin_score(X, labels))
    except: return float('nan')

def compute_intra_compactness(gen_vals: np.ndarray) -> float:
    return float(np.mean(np.var(gen_vals, axis=0)))

def _get_kth_nn_dists(X: np.ndarray, k: int) -> np.ndarray:
    dists = cdist(X, X, 'euclidean')
    dists.sort(axis=1)
    return dists[:, k]

def compute_precision_recall_density_coverage(real_vals: np.ndarray, gen_vals: np.ndarray, k: int = 5) -> tuple:
    n_real, n_gen = len(real_vals), len(gen_vals)
    k_r = min(k, n_real - 1)
    k_g = min(k, n_gen - 1)
    r_real = _get_kth_nn_dists(real_vals, k_r)
    r_gen  = _get_kth_nn_dists(gen_vals, k_g)
    d_real_to_gen = cdist(real_vals, gen_vals, 'euclidean')
    d_gen_to_real = d_real_to_gen.T
    
    precision_count = 0
    for j in range(n_gen):
        if np.any(d_gen_to_real[j, :] <= r_real):
            precision_count += 1
    precision = precision_count / n_gen
    
    recall_count = 0
    for i in range(n_real):
        if np.any(d_real_to_gen[i, :] <= r_gen):
            recall_count += 1
    recall = recall_count / n_real
    
    density_per_gen = []
    for j in range(n_gen):
        count = np.sum(d_gen_to_real[j, :] <= r_real)
        density_per_gen.append(count / k_r)
    density = float(np.mean(density_per_gen))
    
    coverage_count = 0
    for i in range(n_real):
        if np.any(d_real_to_gen[i, :] <= r_real[i]):
            coverage_count += 1
    coverage = coverage_count / n_real
    return precision, recall, density, coverage

def compute_mdcr(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    d_matrix = cdist(gen_vals, real_vals, 'euclidean')
    min_dists = d_matrix.min(axis=1)
    return float(np.mean(min_dists))

def compute_fe_ratio_mean(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 10) -> float:
    n_attrs = real_vals.shape[1]
    ratios = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        # Salta se il reale è costante: l'entropy del reale sarebbe ~0 -> divisione esplode.
        # Non si usa il range combinato perché il generato potrebbe avere valori fuori dal range reale.
        if np.std(r) < 1e-6:
            ratios.append(1.0)
            continue
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(g, bins=bins, density=True)
        q, _ = np.histogram(r, bins=bins, density=True)
        h_g = entropy(p + 1e-12)
        h_r = entropy(q + 1e-12)
        ratios.append(h_g / (h_r + 1e-12))
    return float(np.mean(ratios))

def compute_dbi_div(gen_vals: np.ndarray) -> float:
    if len(gen_vals) < 10: return float('nan')
    median = np.median(gen_vals, axis=0)
    labels = np.any(gen_vals > median, axis=1).astype(int)
    if len(np.unique(labels)) < 2: return 0.0
    try: return float(davies_bouldin_score(gen_vals, labels))
    except: return float('nan')

def compute_coverage_div(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    n_attrs = real_vals.shape[1]
    covered_fractions = []
    for d in range(n_attrs):
        r = real_vals[:, d]
        g = gen_vals[:, d]
        quantiles = np.quantile(r, np.linspace(0, 1, 11))
        unique_q = np.unique(quantiles)
        if len(unique_q) < 2:
            covered_fractions.append(1.0)
            continue
        counts, _ = np.histogram(g, bins=unique_q)
        active_bins = np.sum(counts > 0)
        total_bins = len(unique_q) - 1
        covered_fractions.append(active_bins / total_bins)
    return float(np.mean(covered_fractions))

# =========================================================================
# 2bis. METRICHE DI RFD COMPLIANCE
# =========================================================================
# Le soglie sono definite in unità ORIGINALI: tutte le metriche di questa
# sezione operano su valori grezzi (non normalizzati), coerentemente con il
# validatore di SYRFD. Si assume che Sigma sia stato scoperto su D+ (quindi
# D+ soddisfa Sigma e le coppie reale-reale non vanno ricontrollate).

def parse_rfd_file(rfd_path: str):
    """
    Parsa il file di RFD nel formato 'A@..., B@... -> C@...' (una per riga).
    La soglia viene estratta dal nome del file (pattern 'RFD<k>' o 'thr<k>'),
    con fallback 4.0; puo' essere sovrascritta da CLI con --rfd_threshold.
    Ritorna (lista di (lhs_attrs, rhs_attr), soglia).
    """
    rfds = []
    with open(rfd_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            lhs_str, rhs_str = line.split("->", 1)
            lhs = [tok.split("@")[0].strip() for tok in lhs_str.split(",") if tok.strip()]
            rhs = rhs_str.split(",")[0].split("@")[0].strip()
            if lhs and rhs:
                rfds.append((lhs, rhs))
    m = re.search(r"(?:RFD|thr)(\d+(?:\.\d+)?)", os.path.basename(rfd_path), flags=re.IGNORECASE)
    threshold = float(m.group(1)) if m else 4.0
    return rfds, threshold


def _rfd_violations(gen_arr: np.ndarray, ref_arr: np.ndarray,
                    lhs_idx: list, rhs_idx: int, eps: float,
                    same_set: bool = False, chunk: int = 512):
    """
    Test di violazione vettorizzato (broadcast) per UNA dipendenza X -> Y:
    la coppia (g, r) viola sse |g[A]-r[A]| <= eps per ogni A in X
    e |g[Y]-r[Y]| > eps. Chunking sulle righe generate per limitare la memoria.
    Se same_set=True (gen_arr e' ref_arr), esclude le coppie (i, i).
    Ritorna (viol_any per riga generata, numero di coppie ordinate violanti).
    """
    n_g = len(gen_arr)
    viol_any = np.zeros(n_g, dtype=bool)
    n_viol_pairs = 0
    R_l = ref_arr[:, lhs_idx]                       # (N, |X|)
    R_r = ref_arr[:, rhs_idx]                       # (N,)
    for s in range(0, n_g, chunk):
        e = min(s + chunk, n_g)
        G_l = gen_arr[s:e][:, lhs_idx]              # (c, |X|)
        G_r = gen_arr[s:e, rhs_idx]                 # (c,)
        d_lhs = np.abs(G_l[:, None, :] - R_l[None, :, :])   # (c, N, |X|)
        d_rhs = np.abs(G_r[:, None] - R_r[None, :])         # (c, N)
        V = np.all(d_lhs <= eps, axis=2) & (d_rhs > eps)
        if same_set:
            V[np.arange(e - s), np.arange(s, e)] = False    # esclude (i, i)
        viol_any[s:e] |= V.any(axis=1)
        n_viol_pairs += int(V.sum())
    return viol_any, n_viol_pairs


def compute_rfd_compliance(real_df: pd.DataFrame, gen_df: pd.DataFrame,
                           rfds: list, eps: float) -> dict:
    """
    Calcola le metriche di RFD compliance del set sintetico S rispetto a D+:

      - RFD_tuple_compliance_relaxed : frazione di tuple sintetiche che non
        introducono alcuna violazione rispetto alle tuple reali minoritarie
        (scope della modalita' relaxed / Definition Compliance-relaxed).
      - RFD_tuple_compliance_strict  : frazione pulita anche rispetto alle
        ALTRE tuple sintetiche (scope della modalita' strict).
      - RFD_pair_violation_mixed     : coppie violanti / coppie controllate,
        sulle coppie miste sintetico-reale (aggregato su tutte le RFD).
      - RFD_pair_violation_synth     : idem sulle coppie sintetico-sintetico.
      - RFD_preserved_deps           : frazione di RFD di Sigma prive di
        violazioni su D+ U S (equivale a: la RFD resta valida sull'aumentato,
        assumendo che D+ la soddisfi, come garantito dalla discovery su D+).

    Il set S viene ricavato da gen_df filtrando la classe minoritaria (se la
    colonna 'class' esiste) e rimuovendo i duplicati esatti delle tuple reali:
    i file aumentati dei competitor contengono anche le righe originali, che
    non vanno conteggiate come sintetiche.
    """
    out = {
        "RFD_tuple_compliance_relaxed": float("nan"),
        "RFD_tuple_compliance_strict": float("nan"),
        "RFD_pair_violation_mixed": float("nan"),
        "RFD_pair_violation_synth": float("nan"),
        "RFD_preserved_deps": float("nan"),
    }
    if not rfds:
        return out

    attrs = [c for c in real_df.columns if c != "class"]
    common = [a for a in attrs if a in gen_df.columns]

    # --- estrazione del set sintetico S ---
    gen_min = gen_df
    if "class" in gen_df.columns and "class" in real_df.columns:
        min_label = real_df["class"].iloc[0]
        mask = gen_df["class"].astype(str) == str(min_label)
        if mask.any():
            gen_min = gen_df[mask]
    # rimozione dei duplicati esatti delle righe reali (righe originali incluse
    # nei file aumentati): merge su tutti gli attributi comuni
    real_keys = set(map(tuple, np.round(real_df[common].values.astype(float), 10)))
    gen_vals_all = gen_min[common].values.astype(float)
    keep = np.array([tuple(np.round(row, 10)) not in real_keys for row in gen_vals_all])
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"    [RFD] {n_dropped} righe identiche a tuple reali escluse dal set sintetico")
    gen_vals = gen_vals_all[keep]
    real_vals = real_df[common].values.astype(float)

    n_s, n_r = len(gen_vals), len(real_vals)
    if n_s == 0:
        print("    [RFD] nessuna tupla sintetica individuata: metriche a NaN")
        return out

    col_idx = {a: i for i, a in enumerate(common)}

    clean_mixed = np.ones(n_s, dtype=bool)
    clean_synth = np.ones(n_s, dtype=bool)
    tot_viol_mixed = tot_pairs_mixed = 0
    tot_viol_synth = tot_pairs_synth = 0
    preserved = 0
    n_checked_deps = 0

    for lhs, rhs in rfds:
        if rhs not in col_idx or any(a not in col_idx for a in lhs):
            missing = [a for a in lhs + [rhs] if a not in col_idx]
            print(f"    [RFD] dipendenza saltata, attributi assenti: {missing}")
            continue
        n_checked_deps += 1
        lhs_idx = [col_idx[a] for a in lhs]
        rhs_idx = col_idx[rhs]

        v_mixed, p_mixed = _rfd_violations(gen_vals, real_vals, lhs_idx, rhs_idx, eps)
        clean_mixed &= ~v_mixed
        tot_viol_mixed += p_mixed
        tot_pairs_mixed += n_s * n_r

        p_synth = 0
        if n_s > 1:
            v_synth, p_synth = _rfd_violations(gen_vals, gen_vals, lhs_idx, rhs_idx,
                                               eps, same_set=True)
            clean_synth &= ~v_synth
            tot_pairs_synth += n_s * (n_s - 1)
            tot_viol_synth += p_synth

        if p_mixed == 0 and p_synth == 0:
            preserved += 1

    if n_checked_deps == 0:
        print("    [RFD] nessuna dipendenza valutabile sugli attributi comuni")
        return out

    out["RFD_tuple_compliance_relaxed"] = float(np.mean(clean_mixed))
    out["RFD_tuple_compliance_strict"] = float(np.mean(clean_mixed & clean_synth))
    out["RFD_pair_violation_mixed"] = (
        tot_viol_mixed / tot_pairs_mixed if tot_pairs_mixed else float("nan"))
    out["RFD_pair_violation_synth"] = (
        tot_viol_synth / tot_pairs_synth if tot_pairs_synth else float("nan"))
    out["RFD_preserved_deps"] = preserved / n_checked_deps
    return out

# =========================================================================
# 3. ENGINE DI VALUTAZIONE CORE
# =========================================================================
# Elenco esplicito delle chiavi prodotte da evaluate_all, usato dalla pipeline
# main per instradare ciascuna metrica nel CSV corretto (fidelity/diversity).
# Le metriche di RFD compliance sono calcolate a parte da compute_rfd_compliance
# (una volta per ciascuna soglia/file di RFD scoperto automaticamente), perche'
# non dipendono dalla normalizzazione ma dal file di RFD e dalla relativa soglia.
FIDELITY_METRIC_KEYS = [
    'KL_mean', 'JS_mean', 'Q_multi_attr_sim', 'KS_mean', 'Chi2_mean',
    'WD_mean', 'pMSE', 'MAD_mean', 'MMD', 'PCA_WD_mean',
    'Moment_mean_diff', 'Moment_var_diff', 'Moment_skew_diff', 'Moment_kurt_diff',
]

DIVERSITY_METRIC_KEYS = [
    'Silhouette', 'Davies-Bouldin', 'Compactness_cls_k',
    'Precision', 'Recall', 'Density', 'Coverage',
    'mDCR', 'FE_ratio_mean', 'DBI_div', 'Coverage_div',
]

RFD_METRIC_KEYS = [
    'RFD_tuple_compliance_relaxed', 'RFD_tuple_compliance_strict',
    'RFD_pair_violation_mixed', 'RFD_pair_violation_synth', 'RFD_preserved_deps',
]

def evaluate_all(real_df: pd.DataFrame, gen_df: pd.DataFrame, n_bins: int = 30) -> dict:
    attrs = [c for c in real_df.columns if c != 'class']
    gen_attrs = [c for c in gen_df.columns if c != 'class']
    common = [a for a in attrs if a in gen_attrs]

    if not common:
        raise ValueError("Nessun attributo in comune tra i due dataset.")

    # ---- Imputation NaN ----
    NAN_FILL_VALUE = 0.0000001
    nan_real = real_df[common].isna().any().any()
    nan_gen  = gen_df[common].isna().any().any()
    if nan_real:
        n_nan_real = real_df[common].isna().sum().sum()
        print(f"    [INFO] Imputation NaN nel dataset reale: {n_nan_real} valori -> {NAN_FILL_VALUE}")
        real_df = real_df.copy()
        real_df[common] = real_df[common].fillna(NAN_FILL_VALUE)
    if nan_gen:
        n_nan_gen = gen_df[common].isna().sum().sum()
        print(f"    [INFO] Imputation NaN nel dataset generato: {n_nan_gen} valori -> {NAN_FILL_VALUE}")
        gen_df = gen_df.copy()
        gen_df[common] = gen_df[common].fillna(NAN_FILL_VALUE)

    real_vals = real_df[common].values.astype(float)
    gen_vals  = gen_df[common].values.astype(float)

    # Identifica e rimuove attributi a varianza zero nel reale (costanti nella classe minoritaria).
    # MinMaxScaler su range=0 produce scale_=inf e inflaziona tutte le distanze euclidee
    # (mDCR, MMD, Precision/Recall/Density/Coverage, Silhouette, DBI).
    # Gli attributi costanti vengono esclusi da tutte le metriche che usano dati normalizzati.
    col_std = np.std(real_vals, axis=0)
    valid_cols = col_std > 1e-6
    n_zero_var = int(np.sum(~valid_cols))
    if n_zero_var > 0:
        zero_var_names = [common[i] for i in range(len(common)) if not valid_cols[i]]
        print(f"    [INFO] {n_zero_var} attributo/i a varianza zero escluso/i dalla normalizzazione: {zero_var_names}")
        real_vals_scaled = real_vals[:, valid_cols]
        gen_vals_scaled  = gen_vals[:, valid_cols]
    else:
        real_vals_scaled = real_vals
        gen_vals_scaled  = gen_vals

    scaler = MinMaxScaler()
    scaler.fit(real_vals_scaled)
    real_n = scaler.transform(real_vals_scaled)
    gen_n  = np.clip(scaler.transform(gen_vals_scaled), 0.0, 1.0)

    res = {}
    
    # Fidelity - metriche basate su istogrammi: usano real_vals/gen_vals grezzi
    # (gestiscono internamente attributi costanti con il check hi-lo < 1e-10)
    res['KL_mean'] = compute_kl_divergence(real_vals, gen_vals, n_bins)
    res['JS_mean'] = compute_js_divergence(real_vals, gen_vals, n_bins)
    res['Q_multi_attr_sim'] = compute_q_similarity(real_n, gen_n)
    res['KS_mean'] = compute_ks_mean(real_vals, gen_vals)
    res['Chi2_mean'] = compute_chi2_mean(real_vals, gen_vals)
    res['WD_mean'] = compute_wd_mean(real_vals, gen_vals)
    res['pMSE'] = compute_pmse(real_n, gen_n)
    res['MAD_mean'] = compute_mad_mean(real_n, gen_n)
    res['MMD'] = compute_mmd(real_n, gen_n)
    res['PCA_WD_mean'] = compute_pca_wd_mean(real_n, gen_n)
    
    m1, m2, m3, m4 = compute_moment_diffs(real_n, gen_n)
    res['Moment_mean_diff'] = m1
    res['Moment_var_diff'] = m2
    res['Moment_skew_diff'] = m3
    res['Moment_kurt_diff'] = m4

    # Diversity - tutte le metriche spaziali usano dati normalizzati (senza colonne costanti)
    res['Silhouette'] = compute_silhouette(real_n, gen_n)
    res['Davies-Bouldin'] = compute_dbi(real_n, gen_n)
    res['Compactness_cls_k'] = compute_intra_compactness(gen_n)
    
    prec, rec, dens, cov = compute_precision_recall_density_coverage(real_n, gen_n, k=5)
    res['Precision'] = prec
    res['Recall'] = rec
    res['Density'] = dens
    res['Coverage'] = cov
    
    res['mDCR'] = compute_mdcr(real_n, gen_n)
    res['FE_ratio_mean'] = compute_fe_ratio_mean(real_vals, gen_vals)
    res['DBI_div'] = compute_dbi_div(gen_n)
    res['Coverage_div'] = compute_coverage_div(real_vals, gen_vals)

    return res

# =========================================================================
# 4. LOGICA DI MATCHING ADATTIVO AD ALTA PRECISIONE
# =========================================================================

def clean_dataset_name(filename: str) -> str:
    """Ritorna la radice pura del dataset eliminando l'estensione e la dicitura _min."""
    ref_base, _ = os.path.splitext(os.path.basename(filename))
    return re.sub(r'_min$', '', ref_base, flags=re.IGNORECASE).lower().strip()

def find_matching_datasets(base_dataset_name: str, search_dir: str) -> list:
    """Scansiona ricorsivamente le directory cercando corrispondenze basate su prefisso."""
    matched_files = []
    core_ref = clean_dataset_name(base_dataset_name)
    
    if not os.path.exists(search_dir):
        print(f"[ATTENZIONE] La directory '{search_dir}' non esiste sul file system.")
        return matched_files

    # Array fisso e tassativo delle cartelle target fornite
    target_folders = [ "SYRFD_variable_relaxed", "SYRFD_variable_strict"]

    # Modelli che organizzano i file in sottocartelle per strategia:
    # per questi, il campo "filename" include il nome della sottocartella come suffisso
    # (es. "abalone9-18_balanced_confidence.csv") per distinguere le occorrenze.
    models_with_subfolders = {
        "devstral-small-2_24b-cloud",
        "gpt-oss_120b-cloud",
        "gpt-oss_20b-cloud",
    }

    for folder in target_folders:
        folder_path = os.path.join(search_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        # Ricerca ricorsiva (os.walk) per penetrare nelle cartelle interne (es. /hierarchical/, /ensemble/)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_base, _ = os.path.splitext(file)
                    file_base_lower = file_base.lower()
                    
                    # MATCH CRITICO ADATTIVO: verifica se il file inizia esattamente con il nome radice del dataset
                    if file_base_lower.startswith(core_ref):
                        full_path = os.path.join(root, file)

                        # Per i modelli con sottocartelle, arricchisce il filename con
                        # il nome della sottocartella diretta che contiene il file,
                        # in modo da distinguere le occorrenze dello stesso file
                        # (es. confidence/abalone9-18_balanced.csv -> abalone9-18_balanced_confidence.csv)
                        if folder in models_with_subfolders:
                            subfolder = os.path.relpath(root, folder_path)
                            if subfolder != '.':
                                # Prende solo il primo livello di sottocartella (ignora eventuali nesting profondi)
                                top_subfolder = subfolder.split(os.sep)[0]
                                display_name = f"{file_base}_{top_subfolder}.csv"
                            else:
                                display_name = file
                        else:
                            display_name = file

                        matched_files.append({
                            "algorithm": folder,
                            "filename": display_name,
                            "full_path": full_path
                        })
    return matched_files

def load_minority(imbalanced_path: str) -> pd.DataFrame:
    df = pd.read_csv(imbalanced_path)
    min_class = df['class'].value_counts().idxmin()
    return df[df['class'] == min_class].copy()

def find_rfd_files(base_dataset_name: str, rfd_dir: str) -> list:
    """
    Scansiona automaticamente 'rfd_dir' (default: ./discovered_rfds_processed/)
    cercando tutti i file .txt di RFD relativi al dataset in esame.

    Formato tipico dei file: 'RFD<soglia>_E<epsilon>_<nome_dataset>[_min].txt'
    (es. 'RFD2_E0.0_abalone9-18_min.txt', 'RFD4_E0.0_abalone9-18_min.txt',
    'RFD8_E0.0_abalone9-18_min.txt'), ma il numero e il valore delle soglie
    disponibili sono ARBITRARI: non ci si limita alle sole soglie 2/4/8, si
    scoprono dinamicamente tutti i file presenti il cui nome contiene la
    radice del dataset (stessa logica di match usata per i CSV bilanciati).

    Ritorna una lista di dict, uno per ciascun file trovato, ordinata per
    soglia crescente:
        {
          "filename": "RFD2_E0.0_abalone9-18_min.txt",
          "full_path": ".../RFD2_E0.0_abalone9-18_min.txt",
          "rfds": [(lhs_attrs, rhs_attr), ...],
          "threshold": 2.0,
          "epsilon": 0.0,
        }
    """
    found = []
    core_ref = clean_dataset_name(base_dataset_name)

    if not os.path.exists(rfd_dir):
        print(f"[ATTENZIONE] La directory delle RFD '{rfd_dir}' non esiste sul file system.")
        return found

    for fname in sorted(os.listdir(rfd_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        full_path = os.path.join(rfd_dir, fname)
        if not os.path.isfile(full_path):
            continue

        # MATCH: la radice pulita del dataset deve comparire nel nome del file RFD
        fname_clean = clean_dataset_name(fname)
        if core_ref not in fname_clean:
            continue

        try:
            rfds, threshold = parse_rfd_file(full_path)
        except Exception as e:
            print(f"    [ATTENZIONE] Impossibile parsare il file RFD '{fname}': {e}")
            continue

        if not rfds:
            print(f"    [ATTENZIONE] Nessuna RFD valida trovata in '{fname}', file ignorato.")
            continue

        eps_match = re.search(r'_E(\d+(?:\.\d+)?)', fname, flags=re.IGNORECASE)
        epsilon = float(eps_match.group(1)) if eps_match else None

        found.append({
            "filename": fname,
            "full_path": full_path,
            "rfds": rfds,
            "threshold": threshold,
            "epsilon": epsilon,
        })

    # Ordina per soglia crescente (piu' leggibile nel CSV di output)
    found.sort(key=lambda x: (x['threshold'] if x['threshold'] is not None else float('inf')))
    return found

# =========================================================================
# 5. MAIN PIPELINE
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo del dataset originale vs. tutti i dataset bilanciati "
                     "trovati nelle sottodirectory di augmented_dir; produce 3 CSV separati "
                     "(fidelity, diversity, RFD compliance)."
    )
    parser.add_argument("--real", type=str, help="CSV con sole tuple reali minoritarie")
    parser.add_argument("--imbalanced", type=str, help="CSV imbilanciato originale di riferimento")
    parser.add_argument("--augmented_dir", type=str, default="./augmented_datasets",
                        help="Path della root dei dataset bilanciati concorrenti")
    parser.add_argument("--rfd_dir", type=str, default="./discovered_rfds_processed",
                        help="Path della directory con i file di RFD scoperti (es. RFD2_E0.0_<dataset>_min.txt)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_metrics",
                        help="Directory radice di output; verranno create le sottocartelle "
                             "fidelity/, diversity/, RFDcompliance/")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    # ---- Identificazione del dataset ground truth ----
    if args.real:
        real_df = pd.read_csv(args.real)
        base_name = os.path.basename(args.real)
    elif args.imbalanced:
        real_df = load_minority(args.imbalanced)
        base_name = os.path.basename(args.imbalanced)
    else:
        print("[ERRORE] Specificare tassativamente il file --real o il file --imbalanced.")
        sys.exit(1)

    clean_target = clean_dataset_name(base_name)

    # ---- [1] Scouting automatico dei dataset bilanciati concorrenti ----
    print(f"[1] Ricerca dataset bilanciati per la radice '{clean_target}' in '{args.augmented_dir}'...")
    matching_datasets = find_matching_datasets(base_name, args.augmented_dir)
    print(f"    Trovati {len(matching_datasets)} dataset di confronto compatibili.")
    if not matching_datasets:
        print("[ERRORE] Nessun dataset bilanciato trovato: impossibile procedere.")
        sys.exit(1)

    # ---- [2] Scouting automatico dei file di RFD (per la RFD compliance) ----
    print(f"[2] Ricerca file di RFD per la radice '{clean_target}' in '{args.rfd_dir}'...")
    rfd_files = find_rfd_files(base_name, args.rfd_dir)
    if rfd_files:
        soglie = ", ".join(str(rf['threshold']) for rf in rfd_files)
        print(f"    Trovati {len(rfd_files)} file di RFD (soglie: {soglie}).")
    else:
        print("    [ATTENZIONE] Nessun file di RFD trovato: il CSV di RFD compliance sara' vuoto.")

    # ---- [3] Loop di computazione: fidelity + diversity (una volta per dataset) ----
    #          e RFD compliance (una volta per dataset x soglia di RFD trovata)
    fidelity_results = []
    diversity_results = []
    rfd_results = []

    print(f"[3] Calcolo delle metriche per {len(matching_datasets)} dataset bilanciati...")
    for item in matching_datasets:
        print(f"[-] Elaborazione di [{item['algorithm']}] -> {item['filename']}...")
        try:
            comp_df = pd.read_csv(item['full_path'])
            metrics = evaluate_all(real_df, comp_df, n_bins=args.bins)

            fid_row = {"Algoritmo": item['algorithm'], "File": item['filename']}
            fid_row.update({k: metrics[k] for k in FIDELITY_METRIC_KEYS})
            fidelity_results.append(fid_row)

            div_row = {"Algoritmo": item['algorithm'], "File": item['filename']}
            div_row.update({k: metrics[k] for k in DIVERSITY_METRIC_KEYS})
            diversity_results.append(div_row)
        except Exception as e:
            print(f"    [SALTATO] Errore durante il calcolo fidelity/diversity di {item['filename']}: {e}")
            continue

        # RFD compliance: una riga per ciascuna soglia/file di RFD scoperto
        for rfd_item in rfd_files:
            try:
                rfd_metrics = compute_rfd_compliance(
                    real_df, comp_df, rfd_item['rfds'], rfd_item['threshold']
                )
                rfd_row = {
                    "Algoritmo": item['algorithm'],
                    "File": item['filename'],
                    "RFD_file": rfd_item['filename'],
                    "RFD_threshold": rfd_item['threshold'],
                    "RFD_epsilon": rfd_item['epsilon'],
                    "N_RFDs": len(rfd_item['rfds']),
                }
                rfd_row.update(rfd_metrics)
                rfd_results.append(rfd_row)
            except Exception as e:
                print(f"    [SALTATO] Errore RFD compliance ({rfd_item['filename']}) su {item['filename']}: {e}")

    # ---- [4] Costruzione delle 3 tabelle finali ----
    fidelity_table = pd.DataFrame(fidelity_results)
    diversity_table = pd.DataFrame(diversity_results)
    rfd_table = pd.DataFrame(rfd_results)

    for table in (fidelity_table, diversity_table):
        if not table.empty:
            table.sort_values(by=['Algoritmo', 'File'], inplace=True)
            table.reset_index(drop=True, inplace=True)
    if not rfd_table.empty:
        rfd_table.sort_values(by=['RFD_threshold', 'Algoritmo', 'File'], inplace=True)
        rfd_table.reset_index(drop=True, inplace=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n" + "="*120)
    print(f" TABELLA FIDELITY (Dataset Radice: {clean_target})")
    print("="*120)
    print(fidelity_table.to_string(index=False) if not fidelity_table.empty else "(nessun risultato)")

    print("\n" + "="*120)
    print(f" TABELLA DIVERSITY (Dataset Radice: {clean_target})")
    print("="*120)
    print(diversity_table.to_string(index=False) if not diversity_table.empty else "(nessun risultato)")

    print("\n" + "="*120)
    print(f" TABELLA RFD COMPLIANCE (Dataset Radice: {clean_target})")
    print("="*120)
    print(rfd_table.to_string(index=False) if not rfd_table.empty else "(nessun risultato: nessun file di RFD trovato)")
    print("="*120 + "\n")

    # ---- [5] Salvataggio dei 3 CSV nelle rispettive sottocartelle ----
    fidelity_dir = os.path.join(args.output_dir, "fidelity")
    diversity_dir = os.path.join(args.output_dir, "diversity")
    rfd_dir_out = os.path.join(args.output_dir, "RFDcompliance")
    os.makedirs(fidelity_dir, exist_ok=True)
    os.makedirs(diversity_dir, exist_ok=True)
    os.makedirs(rfd_dir_out, exist_ok=True)

    fidelity_path = os.path.join(fidelity_dir, f"fidelity_{clean_target}.csv")
    diversity_path = os.path.join(diversity_dir, f"diversity_{clean_target}.csv")
    rfd_path = os.path.join(rfd_dir_out, f"rfd_compliance_{clean_target}.csv")

    fidelity_table.to_csv(fidelity_path, index=False)
    diversity_table.to_csv(diversity_path, index=False)
    rfd_table.to_csv(rfd_path, index=False)

    print(f"[OK] Fidelity esportata in:       {fidelity_path}")
    print(f"[OK] Diversity esportata in:      {diversity_path}")
    print(f"[OK] RFD compliance esportata in: {rfd_path}")

if __name__ == "__main__":
    main()