"""
evaluate_metrics_macro.py
Exhaustively evaluates and compares the ORIGINAL dataset (real minority
class) against ALL balanced datasets found in the subdirectories of
"./augmented_datasets/" (one subdirectory per data augmentation algorithm,
see target_folders in find_matching_datasets).

Unlike earlier versions, the output is no longer a single macro-table: the
metrics are split into 3 separate CSVs, saved in 3 separate subfolders of
"./evaluation_metrics/":

    ./evaluation_metrics/fidelity/       -> FIDELITY metrics
    ./evaluation_metrics/diversity/      -> DIVERSITY metrics
    ./evaluation_metrics/RFDcompliance/  -> RFD COMPLIANCE metrics

Section "2bis. RFD compliance metrics":

parse_rfd_file: format 'A@..., B@... -> C@...', threshold taken from the
file name using the pattern RFD<k> or thr<k> (consistent with the
SYRFD_thr2/4/8 folders and with the files discovered in
./discovered_rfds_processed/).
find_rfd_files: automatically locates, in the directory
"./discovered_rfds_processed/", all RFD files related to the dataset under
examination (e.g. "RFD2_E0.0_abalone9-18_min.txt", "RFD4_E0.0_...",
"RFD8_E0.0_..."). The number and value of the available thresholds are
arbitrary: the script does not assume that only thresholds 2/4/8 exist, it
dynamically discovers all files present that match the dataset.
_rfd_violations: vectorized violation test, with chunking over the
generated rows to avoid exploding memory usage on large datasets, and
exclusion of the diagonal in the synthetic-vs-synthetic case.
compute_rfd_compliance: produces five metrics, aligned one-to-one with
Section 3 of the paper:

  - RFD_tuple_compliance_relaxed: fraction of synthetic tuples with no
    violations against D+ (= relaxed compliance definition);
  - RFD_tuple_compliance_strict: also clean with respect to the other
    synthetic tuples (= strict compliance);
  - RFD_pair_violation_mixed / RFD_pair_violation_synth: pair-level rates,
    the finer-grained view for showing how much a competitor violates;
  - RFD_preserved_deps: fraction of Sigma's RFDs still valid on D+ U S
    (dependency-level view).

For each balanced dataset found, the RFD compliance metrics are computed
ONCE FOR EACH automatically discovered RFD threshold/file: the RFD
compliance CSV is therefore in "long" format (one row for each
algorithm/balanced-file x RFD-threshold combination).

Two design choices worth knowing: (1) the competitors' augmented files also
contain the original rows, so the synthetic set S is obtained by filtering
the minority class and removing exact duplicates of the real tuples -
otherwise the original rows (compliant by construction) would inflate
everyone's metrics. (2) Everything is computed on raw values, as in the
validator, because the thresholds are in original units.

Usage:
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

# 1. Fidelity metrics

def compute_kl_divergence(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 30) -> float:
    n_attrs = real_vals.shape[1]
    kl_per_attr = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        # Skip constant attributes in the real data: all real points fall into one bin -> degenerate distribution
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
        # Skip constant attributes in the real data
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
        # Skip constant attributes in the real data: all in the first bin -> chi2 not meaningful
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
    # Skip zero-variance attributes in the real data (constant within the minority class):
    # a 1e-8 denominator would overflow if the generated data produces values != 0
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

    # Skip zero-variance attributes (constant in the real data): denominator ~0 causes overflow
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

# 2. Diversity and spatial metrics

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
        # Skip if the real data is constant: real entropy would be ~0 -> division blows up.
        # The combined range is not used because the generated data may have values outside the real range.
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

# 2bis. RFD compliance metrics
# Thresholds are defined in ORIGINAL units: all metrics in this section
# operate on raw (non-normalized) values, consistently with the SYRFD
# validator. It is assumed that Sigma was discovered on D+ (so D+ satisfies
# Sigma and real-real pairs do not need to be rechecked).

def parse_rfd_file(rfd_path: str):
    """
    Parses the RFD file in the format 'A@..., B@... -> C@...' (one per line).
    The threshold is extracted from the file name (pattern 'RFD<k>' or
    'thr<k>'), with fallback 4.0; it can be overridden from the CLI with
    --rfd_threshold.
    Returns (list of (lhs_attrs, rhs_attr), threshold).
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
    Vectorized (broadcast) violation test for ONE dependency X -> Y:
    the pair (g, r) violates iff |g[A]-r[A]| <= eps for every A in X
    and |g[Y]-r[Y]| > eps. Chunking over the generated rows to limit memory
    usage. If same_set=True (gen_arr is ref_arr), excludes pairs (i, i).
    Returns (viol_any per generated row, number of violating ordered pairs).
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
            V[np.arange(e - s), np.arange(s, e)] = False    # exclude (i, i)
        viol_any[s:e] |= V.any(axis=1)
        n_viol_pairs += int(V.sum())
    return viol_any, n_viol_pairs


def compute_rfd_compliance(real_df: pd.DataFrame, gen_df: pd.DataFrame,
                           rfds: list, eps: float) -> dict:
    """
    Computes the RFD compliance metrics of the synthetic set S with respect
    to D+:

      - RFD_tuple_compliance_relaxed : fraction of synthetic tuples that do
        not introduce any violation with respect to the real minority
        tuples (scope of the relaxed mode / Definition Compliance-relaxed).
      - RFD_tuple_compliance_strict  : fraction also clean with respect to
        the OTHER synthetic tuples (scope of the strict mode).
      - RFD_pair_violation_mixed     : violating pairs / checked pairs, on
        the mixed synthetic-real pairs (aggregated over all RFDs).
      - RFD_pair_violation_synth     : same, on synthetic-synthetic pairs.
      - RFD_preserved_deps           : fraction of Sigma's RFDs with no
        violations on D+ U S (equivalent to: the RFD remains valid on the
        augmented data, assuming D+ satisfies it, as guaranteed by
        discovery on D+).

    The set S is obtained from gen_df by filtering the minority class (if
    the 'class' column exists) and removing exact duplicates of the real
    tuples: the competitors' augmented files also contain the original
    rows, which must not be counted as synthetic.
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

    # --- extraction of the synthetic set S ---
    gen_min = gen_df
    if "class" in gen_df.columns and "class" in real_df.columns:
        min_label = real_df["class"].iloc[0]
        mask = gen_df["class"].astype(str) == str(min_label)
        if mask.any():
            gen_min = gen_df[mask]
    # removal of exact duplicates of the real rows (original rows included
    # in the augmented files): merge on all common attributes
    real_keys = set(map(tuple, np.round(real_df[common].values.astype(float), 10)))
    gen_vals_all = gen_min[common].values.astype(float)
    keep = np.array([tuple(np.round(row, 10)) not in real_keys for row in gen_vals_all])
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"    [RFD] {n_dropped} rows identical to real tuples excluded from the synthetic set")
    gen_vals = gen_vals_all[keep]
    real_vals = real_df[common].values.astype(float)

    n_s, n_r = len(gen_vals), len(real_vals)
    if n_s == 0:
        print("    [RFD] no synthetic tuple found: metrics set to NaN")
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
            print(f"    [RFD] dependency skipped, missing attributes: {missing}")
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
        print("    [RFD] no dependency evaluable on the common attributes")
        return out

    out["RFD_tuple_compliance_relaxed"] = float(np.mean(clean_mixed))
    out["RFD_tuple_compliance_strict"] = float(np.mean(clean_mixed & clean_synth))
    out["RFD_pair_violation_mixed"] = (
        tot_viol_mixed / tot_pairs_mixed if tot_pairs_mixed else float("nan"))
    out["RFD_pair_violation_synth"] = (
        tot_viol_synth / tot_pairs_synth if tot_pairs_synth else float("nan"))
    out["RFD_preserved_deps"] = preserved / n_checked_deps
    return out

# 3. Core evaluation engine
# Explicit list of the keys produced by evaluate_all, used by the main
# pipeline to route each metric into the correct CSV (fidelity/diversity).
# The RFD compliance metrics are computed separately by compute_rfd_compliance
# (once for each automatically discovered RFD threshold/file), because they
# don't depend on normalization but on the RFD file and its threshold.
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
        raise ValueError("No attribute in common between the two datasets.")

    # ---- NaN imputation ----
    NAN_FILL_VALUE = 0.0000001
    nan_real = real_df[common].isna().any().any()
    nan_gen  = gen_df[common].isna().any().any()
    if nan_real:
        n_nan_real = real_df[common].isna().sum().sum()
        print(f"    [INFO] NaN imputation in the real dataset: {n_nan_real} values -> {NAN_FILL_VALUE}")
        real_df = real_df.copy()
        real_df[common] = real_df[common].fillna(NAN_FILL_VALUE)
    if nan_gen:
        n_nan_gen = gen_df[common].isna().sum().sum()
        print(f"    [INFO] NaN imputation in the generated dataset: {n_nan_gen} values -> {NAN_FILL_VALUE}")
        gen_df = gen_df.copy()
        gen_df[common] = gen_df[common].fillna(NAN_FILL_VALUE)

    real_vals = real_df[common].values.astype(float)
    gen_vals  = gen_df[common].values.astype(float)

    # Identify and remove zero-variance attributes in the real data (constant within the minority class).
    # MinMaxScaler on a zero range produces scale_=inf and inflates all Euclidean distances
    # (mDCR, MMD, Precision/Recall/Density/Coverage, Silhouette, DBI).
    # Constant attributes are excluded from all metrics that use normalized data.
    col_std = np.std(real_vals, axis=0)
    valid_cols = col_std > 1e-6
    n_zero_var = int(np.sum(~valid_cols))
    if n_zero_var > 0:
        zero_var_names = [common[i] for i in range(len(common)) if not valid_cols[i]]
        print(f"    [INFO] {n_zero_var} zero-variance attribute(s) excluded from normalization: {zero_var_names}")
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

    # Fidelity - histogram-based metrics: use raw real_vals/gen_vals
    # (constant attributes are handled internally via the hi-lo < 1e-10 check)
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

    # Diversity - all spatial metrics use normalized data (without constant columns)
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

# 4. Adaptive high-precision matching logic

def clean_dataset_name(filename: str) -> str:
    """Returns the pure dataset root by removing the extension and the '_min' suffix."""
    ref_base, _ = os.path.splitext(os.path.basename(filename))
    return re.sub(r'_min$', '', ref_base, flags=re.IGNORECASE).lower().strip()

def find_matching_datasets(base_dataset_name: str, search_dir: str) -> list:
    """Recursively scans the directories looking for prefix-based matches."""
    matched_files = []
    core_ref = clean_dataset_name(base_dataset_name)

    if not os.path.exists(search_dir):
        print(f"[WARNING] Directory '{search_dir}' does not exist on the file system.")
        return matched_files

    # Fixed, exhaustive array of the given target folders
    target_folders = [ "SYRFD_variable_relaxed", "SYRFD_variable_strict"]

    # Models that organize their files into per-strategy subfolders:
    # for these, the "filename" field includes the subfolder name as a suffix
    # (e.g. "abalone9-18_balanced_confidence.csv") to distinguish occurrences.
    models_with_subfolders = {
        "devstral-small-2_24b-cloud",
        "gpt-oss_120b-cloud",
        "gpt-oss_20b-cloud",
    }

    for folder in target_folders:
        folder_path = os.path.join(search_dir, folder)
        if not os.path.exists(folder_path):
            continue

        # Recursive search (os.walk) to descend into nested folders (e.g. /hierarchical/, /ensemble/)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_base, _ = os.path.splitext(file)
                    file_base_lower = file_base.lower()

                    # KEY ADAPTIVE MATCH: check whether the file name starts exactly with the dataset root name
                    if file_base_lower.startswith(core_ref):
                        full_path = os.path.join(root, file)

                        # For models with subfolders, enrich the filename with
                        # the name of the direct subfolder containing the file,
                        # to distinguish occurrences of the same file
                        # (e.g. confidence/abalone9-18_balanced.csv -> abalone9-18_balanced_confidence.csv)
                        if folder in models_with_subfolders:
                            subfolder = os.path.relpath(root, folder_path)
                            if subfolder != '.':
                                # Take only the first-level subfolder (ignore any deeper nesting)
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
    Automatically scans 'rfd_dir' (default: ./discovered_rfds_processed/)
    looking for all RFD .txt files related to the dataset under examination.

    Typical file format: 'RFD<threshold>_E<epsilon>_<dataset_name>[_min].txt'
    (e.g. 'RFD2_E0.0_abalone9-18_min.txt', 'RFD4_E0.0_abalone9-18_min.txt',
    'RFD8_E0.0_abalone9-18_min.txt'), but the number and value of the
    available thresholds are ARBITRARY: it is not limited to thresholds
    2/4/8 only, it dynamically discovers all files present whose name
    contains the dataset root (same matching logic used for the balanced
    CSVs).

    Returns a list of dicts, one per file found, sorted by increasing
    threshold:
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
        print(f"[WARNING] RFD directory '{rfd_dir}' does not exist on the file system.")
        return found

    for fname in sorted(os.listdir(rfd_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        full_path = os.path.join(rfd_dir, fname)
        if not os.path.isfile(full_path):
            continue

        # MATCH: the cleaned dataset root must appear in the RFD file name
        fname_clean = clean_dataset_name(fname)
        if core_ref not in fname_clean:
            continue

        try:
            rfds, threshold = parse_rfd_file(full_path)
        except Exception as e:
            print(f"    [WARNING] Could not parse RFD file '{fname}': {e}")
            continue

        if not rfds:
            print(f"    [WARNING] No valid RFD found in '{fname}', file ignored.")
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

    # Sort by increasing threshold (more readable in the output CSV)
    found.sort(key=lambda x: (x['threshold'] if x['threshold'] is not None else float('inf')))
    return found

# 5. Main pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Comparative benchmark of the original dataset vs. all balanced datasets "
                     "found in the subdirectories of augmented_dir; produces 3 separate CSVs "
                     "(fidelity, diversity, RFD compliance)."
    )
    parser.add_argument("--real", type=str, help="CSV containing only real minority tuples")
    parser.add_argument("--imbalanced", type=str, help="Original imbalanced reference CSV")
    parser.add_argument("--augmented_dir", type=str, default="./augmented_datasets",
                        help="Path to the root of the competing balanced datasets")
    parser.add_argument("--rfd_dir", type=str, default="./discovered_rfds_processed",
                        help="Path to the directory with the discovered RFD files (e.g. RFD2_E0.0_<dataset>_min.txt)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_metrics",
                        help="Root output directory; the subfolders "
                             "fidelity/, diversity/, RFDcompliance/ will be created")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    # ---- Ground-truth dataset identification ----
    if args.real:
        real_df = pd.read_csv(args.real)
        base_name = os.path.basename(args.real)
    elif args.imbalanced:
        real_df = load_minority(args.imbalanced)
        base_name = os.path.basename(args.imbalanced)
    else:
        print("[ERROR] Either --real or --imbalanced must be specified.")
        sys.exit(1)

    clean_target = clean_dataset_name(base_name)

    # ---- [1] Automatic scouting of competing balanced datasets ----
    print(f"[1] Searching for balanced datasets for root '{clean_target}' in '{args.augmented_dir}'...")
    matching_datasets = find_matching_datasets(base_name, args.augmented_dir)
    print(f"    Found {len(matching_datasets)} compatible comparison datasets.")
    if not matching_datasets:
        print("[ERROR] No balanced dataset found: cannot proceed.")
        sys.exit(1)

    # ---- [2] Automatic scouting of RFD files (for RFD compliance) ----
    print(f"[2] Searching for RFD files for root '{clean_target}' in '{args.rfd_dir}'...")
    rfd_files = find_rfd_files(base_name, args.rfd_dir)
    if rfd_files:
        soglie = ", ".join(str(rf['threshold']) for rf in rfd_files)
        print(f"    Found {len(rfd_files)} RFD files (thresholds: {soglie}).")
    else:
        print("    [WARNING] No RFD file found: the RFD compliance CSV will be empty.")

    # ---- [3] Computation loop: fidelity + diversity (once per dataset) ----
    #          and RFD compliance (once per dataset x RFD threshold found)
    fidelity_results = []
    diversity_results = []
    rfd_results = []

    print(f"[3] Computing metrics for {len(matching_datasets)} balanced datasets...")
    for item in matching_datasets:
        print(f"[-] Processing [{item['algorithm']}] -> {item['filename']}...")
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
            print(f"    [SKIPPED] Error while computing fidelity/diversity for {item['filename']}: {e}")
            continue

        # RFD compliance: one row for each discovered RFD threshold/file
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
                print(f"    [SKIPPED] RFD compliance error ({rfd_item['filename']}) on {item['filename']}: {e}")

    # ---- [4] Building the 3 final tables ----
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
    print(f" FIDELITY TABLE (Root dataset: {clean_target})")
    print("="*120)
    print(fidelity_table.to_string(index=False) if not fidelity_table.empty else "(no results)")

    print("\n" + "="*120)
    print(f" DIVERSITY TABLE (Root dataset: {clean_target})")
    print("="*120)
    print(diversity_table.to_string(index=False) if not diversity_table.empty else "(no results)")

    print("\n" + "="*120)
    print(f" RFD COMPLIANCE TABLE (Root dataset: {clean_target})")
    print("="*120)
    print(rfd_table.to_string(index=False) if not rfd_table.empty else "(no results: no RFD file found)")
    print("="*120 + "\n")

    # ---- [5] Saving the 3 CSVs into their respective subfolders ----
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

    print(f"[OK] Fidelity exported to:        {fidelity_path}")
    print(f"[OK] Diversity exported to:       {diversity_path}")
    print(f"[OK] RFD compliance exported to:  {rfd_path}")

if __name__ == "__main__":
    main()