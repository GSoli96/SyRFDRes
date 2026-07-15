"""
evaluate_metrics.py
====================
Valuta tutte le 6 metriche di qualità sui dati generati da SyRFD,
confrontandoli con i dati reali della classe minoritaria.

Le metriche sono calcolate su DUE livelli:
  1. COMPARISON  (gen vs real): KL, JS, Q-sim, DBI, Silhouette
     → misurano quanto i dati generati si avvicinano alla distribuzione reale
  2. INTRINSIC    (solo gen):   Intra-class Compactness
     → misura la coesione interna del cluster generato

Uso
---
  # Con CSV separati (tipico output di SyRFD)
  python evaluate_metrics.py \\
      --real   augmentation_results/dataset_min.csv \\
      --gen    augmentation_results/syrfd_v2_augmented.csv \\
      --output metrics_report.csv

  # Con dataset imbilanciato originale + file generato
  python evaluate_metrics.py \\
      --imbalanced path/to/full_dataset.csv \\
      --gen        augmentation_results/syrfd_v2_augmented.csv \\
      --output     metrics_report.csv

  # Mock (nessun file, usa dati fittizi)
  python evaluate_metrics.py
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler


# =========================================================================
# METRICHE
# =========================================================================

def compute_kl_divergence(real_vals: np.ndarray, gen_vals: np.ndarray,
                           n_bins: int = 30) -> float:
    """
    KL(gen || real) per-attributo, media sugli attributi.
    Obiettivo: → 0  (0 = distribuzioni identiche)

    Usa istogrammi con smoothing di Laplace per evitare log(0).
    """
    eps = 1e-10
    n_attrs = real_vals.shape[1]
    kl_per_attr = []

    for d in range(n_attrs):
        r = real_vals[:, d]
        g = gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        if hi - lo < 1e-10:
            kl_per_attr.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(gen_vals[:, d],  bins=bins, density=False)
        q, _ = np.histogram(real_vals[:, d], bins=bins, density=False)
        # Smoothing di Laplace
        p = (p + 1).astype(float)
        q = (q + 1).astype(float)
        p /= p.sum()
        q /= q.sum()
        kl_per_attr.append(float(entropy(p, q)))

    return float(np.mean(kl_per_attr))


def compute_js_divergence(real_vals: np.ndarray, gen_vals: np.ndarray,
                           n_bins: int = 30) -> float:
    """
    JS(gen || real) per-attributo, media sugli attributi.
    Obiettivo: → 0  (0 = distribuzioni identiche)
    JS è sempre in [0, ln2] ≈ [0, 0.693].
    """
    eps = 1e-10
    n_attrs = real_vals.shape[1]
    js_per_attr = []

    for d in range(n_attrs):
        r = real_vals[:, d]
        g = gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        if hi - lo < 1e-10:
            js_per_attr.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(gen_vals[:, d],  bins=bins, density=False)
        q, _ = np.histogram(real_vals[:, d], bins=bins, density=False)
        p = (p + 1).astype(float); p /= p.sum()
        q = (q + 1).astype(float); q /= q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        js_per_attr.append(float(js))

    return float(np.mean(js_per_attr))


def compute_q_similarity(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    """
    Q-function Multi-Attribute Similarity.
    Per ogni tupla generata trova la tupla reale più simile (coseno)
    e calcola la media delle similarità massime.
    Obiettivo: → 1  (1 = massima similarità strutturale)

    Nota: similarità coseno ∈ [-1, 1], mappata in [0, 1].
    """
    # Normalizzazione L2 per riga
    def l2_norm(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms < 1e-10, 1.0, norms)

    gen_n  = l2_norm(gen_vals)
    real_n = l2_norm(real_vals)
    sim_matrix = gen_n @ real_n.T          # (n_gen, n_real)
    max_sim = sim_matrix.max(axis=1)       # per ogni generata: max similarità
    q_score = (max_sim + 1.0) / 2.0       # mappa [-1,1] → [0,1]
    return float(q_score.mean())


def compute_intra_compactness(gen_vals: np.ndarray) -> float:
    """
    Intra-class Compactness: varianza media degli attributi delle tuple generate.
    Misura quanto è compatto (coeso) il cluster generato.
    Obiettivo: → 0  (0 = cluster perfettamente compatto)
    """
    return float(np.mean(np.var(gen_vals, axis=0)))


def compute_silhouette(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    """
    Silhouette Score su dati combinati (gen=0, real=1).
    Misura quanto i due cluster (gen vs real) sono ben separati.
    ∈ [-1, 1]. Obiettivo: → 1  (cluster ben separati e compatti)

    ATTENZIONE SULL'INTERPRETAZIONE:
    In questo contesto un silhouette BASSO è in realtà BUONO per
    l'oversampling, perché significa che i generati si mescolano ai reali
    (alta fidelity).  Tenerlo a mente durante la valutazione.
    """
    n_gen, n_real = len(gen_vals), len(real_vals)
    if n_gen < 2 or n_real < 2:
        return float('nan')
    X      = np.vstack([gen_vals, real_vals])
    labels = np.array([0] * n_gen + [1] * n_real)
    try:
        return float(silhouette_score(X, labels, metric='euclidean'))
    except Exception as e:
        print(f"  [WARN] Silhouette non calcolabile: {e}")
        return float('nan')


def compute_dbi(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    """
    Davies–Bouldin Index su dati combinati (gen=0, real=1).
    Misura il rapporto tra dispersione intra-cluster e separazione inter-cluster.
    ≥ 0. Obiettivo: più basso è meglio.
    """
    n_gen, n_real = len(gen_vals), len(real_vals)
    if n_gen < 2 or n_real < 2:
        return float('nan')
    X      = np.vstack([gen_vals, real_vals])
    labels = np.array([0] * n_gen + [1] * n_real)
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception as e:
        print(f"  [WARN] DBI non calcolabile: {e}")
        return float('nan')


# =========================================================================
# REPORT
# =========================================================================

def evaluate_all(real_df: pd.DataFrame, gen_df: pd.DataFrame,
                 n_bins: int = 30) -> dict:
    """
    Esegue tutte le metriche e restituisce un dizionario con i risultati.
    Normalizza i dati in [0,1] prima del calcolo per rendere le metriche
    indipendenti dalla scala degli attributi.
    """
    # Allinea le colonne (esclude 'class')
    attrs = [c for c in real_df.columns if c != 'class']
    gen_attrs = [c for c in gen_df.columns if c != 'class']
    common = [a for a in attrs if a in gen_attrs]

    if not common:
        raise ValueError("Nessun attributo in comune tra real e gen DataFrame.")

    real_vals = real_df[common].values.astype(float)
    gen_vals  = gen_df[common].values.astype(float)

    # Normalizzazione [0,1] basata sui reali (evita data leakage)
    scaler = MinMaxScaler()
    scaler.fit(real_vals)
    real_n = scaler.transform(real_vals)
    gen_n  = np.clip(scaler.transform(gen_vals), 0.0, 1.0)

    print(f"\n  Attributi valutati : {len(common)}")
    print(f"  Tuple reali        : {len(real_n)}")
    print(f"  Tuple generate     : {len(gen_n)}")

    results = {}

    print("\n  Calcolo KL Divergence...")
    results['kl_divergence'] = compute_kl_divergence(real_n, gen_n, n_bins)

    print("  Calcolo JS Divergence...")
    results['js_divergence'] = compute_js_divergence(real_n, gen_n, n_bins)

    print("  Calcolo Q-function Similarity...")
    results['q_similarity'] = compute_q_similarity(real_n, gen_n)

    print("  Calcolo Intra-class Compactness...")
    results['intra_compactness'] = compute_intra_compactness(gen_n)

    print("  Calcolo Silhouette Score...")
    results['silhouette'] = compute_silhouette(real_n, gen_n)

    print("  Calcolo Davies–Bouldin Index...")
    results['dbi'] = compute_dbi(real_n, gen_n)

    return results


def print_report(results: dict, real_count: int, gen_count: int):
    """Stampa il report formattato a terminale."""
    sep = "=" * 62

    def status(metric, value):
        """Valutazione qualitativa del valore della metrica."""
        thresholds = {
            'kl_divergence':   [(0.05, '✓ Ottimo'), (0.15, '~ Buono'), (0.30, '△ Accettabile'), (None, '✗ Scarso')],
            'js_divergence':   [(0.02, '✓ Ottimo'), (0.08, '~ Buono'), (0.15, '△ Accettabile'), (None, '✗ Scarso')],
            'q_similarity':    [(None, None)],  # gestita sotto
            'intra_compactness': [(0.005, '✓ Ottimo'), (0.02, '~ Buono'), (0.06, '△ Accettabile'), (None, '✗ Scarso')],
            'silhouette':      [(None, None)],  # invertita
            'dbi':             [(1.0, '✓ Ottimo'), (2.0, '~ Buono'), (4.0, '△ Accettabile'), (None, '✗ Scarso')],
        }
        if metric == 'q_similarity':
            if value >= 0.97: return '✓ Ottimo'
            if value >= 0.92: return '~ Buono'
            if value >= 0.85: return '△ Accettabile'
            return '✗ Scarso'
        if metric == 'silhouette':
            # Per oversampling: basso sil = gen mischiato ai reali = buono
            if value <= 0.1:  return '✓ Ottimo (ben mischiato)'
            if value <= 0.25: return '~ Buono'
            if value <= 0.45: return '△ Parzialmente separato'
            return '✗ Cluster troppo separato'
        for thresh, label in thresholds.get(metric, []):
            if thresh is None or value <= thresh:
                return label
        return '?'

    print(f"\n{sep}")
    print(f"  REPORT VALUTAZIONE METRICHE — SyRFD")
    print(f"{sep}")
    print(f"  Tuple reali    : {real_count}")
    print(f"  Tuple generate : {gen_count}")
    print(f"{sep}")

    rows = [
        ("KL Divergence",         "kl_divergence",    "→ 0",  "minimizzata"),
        ("JS Divergence",         "js_divergence",    "→ 0",  "minimizzata"),
        ("Q-sim Multi-Attr",      "q_similarity",     "→ 1",  "massimizzata"),
        ("Intra-class Compact.",  "intra_compactness","→ 0",  "minimizzata"),
        ("Silhouette Score",      "silhouette",       "vedi*","interpretaz. inversa"),
        ("Davies–Bouldin Index",  "dbi",              "basso","minimizzato"),
    ]

    for label, key, target, direction in rows:
        val = results[key]
        if val != val:  # NaN
            val_str = "   N/A  "
            st = "?"
        else:
            val_str = f"{val:8.6f}"
            st = status(key, val)
        print(f"  {label:<24} {val_str}   target {target:<6}  [{direction}]  {st}")

    print(f"{sep}")
    print("  * Silhouette basso (<0.1) = generati ben mischiati ai reali = buono")
    print(f"{sep}\n")


# =========================================================================
# MAIN
# =========================================================================

def load_minority(imbalanced_path: str) -> pd.DataFrame:
    df = pd.read_csv(imbalanced_path)
    min_class = df['class'].value_counts().idxmin()
    print(f"  Classe minoritaria identificata: '{min_class}'")
    return df[df['class'] == min_class].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Valutazione metriche su dati generati da SyRFD"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--real",       type=str, help="CSV con sole tuple reali minoritarie")
    group.add_argument("--imbalanced", type=str, help="CSV imbilanciato originale (auto-identifica minoritaria)")

    parser.add_argument("--gen",    type=str, default=None, help="CSV con tuple generate da SyRFD")
    parser.add_argument("--output", type=str, default="metrics_report.csv",
                        help="File CSV di output del report (default: metrics_report.csv)")
    parser.add_argument("--bins",   type=int, default=30,
                        help="Numero di bin per istogrammi KL/JS (default: 30)")
    args = parser.parse_args()

    # --- Caricamento dati reali ---
    if args.real:
        print(f"\nCaricamento dati reali: {args.real}")
        real_df = pd.read_csv(args.real)
    elif args.imbalanced:
        print(f"\nCaricamento dataset imbilanciato: {args.imbalanced}")
        real_df = load_minority(args.imbalanced)
    else:
        print("\nNessun file fornito — uso dati Mock per dimostrazione.")
        np.random.seed(42)
        attrs = [f"Attr{i}" for i in range(1, 11)]
        real_data = {a: np.random.normal(50 + i * 2, 5, 80) for i, a in enumerate(attrs)}
        real_data['class'] = 1
        real_df = pd.DataFrame(real_data)

    # --- Caricamento dati generati ---
    if args.gen:
        print(f"Caricamento dati generati: {args.gen}")
        gen_df = pd.read_csv(args.gen)
    else:
        print("Nessun file generato fornito — genero dati fittizi simili ai reali.")
        np.random.seed(99)
        attrs = [c for c in real_df.columns if c != 'class']
        gen_data = {}
        for a in attrs:
            mu, sigma = real_df[a].mean(), real_df[a].std()
            gen_data[a] = np.random.normal(mu, sigma * 1.1, 50)
        gen_data['class'] = 1
        gen_df = pd.DataFrame(gen_data)

    print(f"\nDati reali  : {real_df.shape}")
    print(f"Dati generati: {gen_df.shape}")

    # --- Valutazione ---
    print("\nCalcolo metriche in corso...")
    results = evaluate_all(real_df, gen_df, n_bins=args.bins)

    # --- Report a terminale ---
    print_report(results, len(real_df), len(gen_df))

    # --- Salvataggio CSV ---
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    report_df = pd.DataFrame([{
        'metric':    k,
        'value':     v,
        'objective': {
            'kl_divergence':    'minimize (→ 0)',
            'js_divergence':    'minimize (→ 0)',
            'q_similarity':     'maximize (→ 1)',
            'intra_compactness':'minimize (→ 0)',
            'silhouette':       'maximize (→ 1) | low = gen mixed with real',
            'dbi':              'minimize (lower = better)',
        }.get(k, '')
    } for k, v in results.items()])
    report_df.to_csv(args.output, index=False)
    print(f"Report salvato in: {args.output}")


if __name__ == "__main__":
    main()
