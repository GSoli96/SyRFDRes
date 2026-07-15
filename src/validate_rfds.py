"""
validate_rfds.py
================
Valida le Relaxed Functional Dependencies (RFD) su un dataset che combina
tuple originali e tuple generate da SyRFD.

Cosa controlla
--------------
Per ogni RFD  LHS → RHS  con soglia θ:
  una violazione esiste se esiste una COPPIA (ti, tj) tale che:
    |ti[Lk] - tj[Lk]| ≤ θ  per ogni Lk ∈ LHS   (LHS simile)
    |ti[RHS] - tj[RHS]| > θ                       (RHS dissimile)

Il report distingue:
  - Violazioni tra coppie (real, real)  → baseline: il dataset originale
  - Violazioni tra coppie (gen, real)   → introdotte dalle tuple generate
  - Violazioni tra coppie (gen, gen)    → interne al set generato
  - Violazioni totali

Uso
---
  python validate_rfds.py \\
      --imbalanced path/to/full_dataset.csv \\
      --gen        augmentation_results/syrfd_v2_augmented.csv \\
      --rfd        path/to/rfds.txt \\
      --threshold  4.0 \\
      --output     rfd_validation_report.csv

  # Solo dati reali (per controllare la baseline)
  python validate_rfds.py \\
      --imbalanced path/to/full_dataset.csv \\
      --rfd        path/to/rfds.txt

  # Mock (nessun file)
  python validate_rfds.py
"""

import re
import argparse
import os
import numpy as np
import pandas as pd
from itertools import product


# =========================================================================
# PARSING RFD
# =========================================================================

def parse_rfds(rfd_path: str) -> list[tuple[list[str], str]]:
    """
    Legge un file di RFD nel formato:
        Attr1@0, Attr2@0 -> Attr3@0
    Restituisce lista di (lhs_list, rhs_attr).
    """
    deps = []
    with open(rfd_path) as f:
        for line in f:
            line = line.strip()
            if '->' not in line:
                continue
            lhs, rhs = line.split('->')
            lhs_list = [a.split('@')[0] for a in re.split('[, ]+', lhs) if a and '@' in a]
            m = re.match(r"(\w+)@", rhs.strip())
            if m:
                deps.append((lhs_list, m.group(1)))
    return deps


def mock_rfds(attributes: list[str]) -> list[tuple[list[str], str]]:
    """Genera RFD fittizie per test."""
    if len(attributes) >= 5:
        return [
            ([attributes[0], attributes[1]], attributes[2]),
            ([attributes[3]], attributes[4]),
        ]
    return [([attributes[0]], attributes[1])]


# =========================================================================
# VIOLATION CHECK — NumPy vettorizzato
# =========================================================================

def check_violations_matrix(
    vals_a: np.ndarray,  # (Na, D)
    vals_b: np.ndarray,  # (Nb, D)
    lhs_cols: list[int],
    rhs_col: int,
    threshold: float,
    pair_type: str,
    label_a: list[str],
    label_b: list[str],
    max_reported: int = 10,
) -> tuple[int, list[dict]]:
    """
    Controlla violazioni tra ogni coppia (a, b) con a ∈ vals_a, b ∈ vals_b.
    Usa broadcasting NumPy per evitare loop Python sulle coppie.

    Restituisce (numero_violazioni, lista_esempi).
    """
    Na, Nb = len(vals_a), len(vals_b)
    if Na == 0 or Nb == 0:
        return 0, []

    # diffs[i, j, k] = |vals_a[i, k] - vals_b[j, k]|
    diffs = np.abs(vals_a[:, np.newaxis, :] - vals_b[np.newaxis, :, :])  # (Na, Nb, D)

    lhs_sim    = np.all(diffs[:, :, lhs_cols] <= threshold, axis=2)   # (Na, Nb)
    rhs_dissim = diffs[:, :, rhs_col] > threshold                      # (Na, Nb)

    violation_mask = lhs_sim & rhs_dissim  # (Na, Nb)
    n_violations   = int(violation_mask.sum())

    examples = []
    if n_violations > 0:
        rows, cols = np.where(violation_mask)
        for k in range(min(max_reported, len(rows))):
            i, j = rows[k], cols[k]
            examples.append({
                'pair_type': pair_type,
                'label_a':   label_a[i],
                'label_b':   label_b[j],
                'lhs_diffs': diffs[i, j, lhs_cols].tolist(),
                'rhs_diff':  float(diffs[i, j, rhs_col]),
            })

    return n_violations, examples


# =========================================================================
# VALIDAZIONE COMPLETA
# =========================================================================

def validate_rfds(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame | None,
    dependencies: list[tuple[list[str], str]],
    threshold: float,
    attrs: list[str],
) -> pd.DataFrame:
    """
    Esegue la validazione completa per ogni RFD.
    Restituisce un DataFrame con il report per ogni RFD.
    """
    attr_idx = {a: i for i, a in enumerate(attrs)}

    real_vals = real_df[attrs].values.astype(float)
    real_labels = [f"real_{i}" for i in range(len(real_vals))]

    has_gen = gen_df is not None and len(gen_df) > 0
    if has_gen:
        gen_vals = gen_df[attrs].values.astype(float)
        gen_labels = [f"gen_{i}" for i in range(len(gen_vals))]
    else:
        gen_vals   = np.empty((0, len(attrs)))
        gen_labels = []

    report_rows = []

    total_real_pairs = len(real_vals) * (len(real_vals) - 1) // 2
    total_gr_pairs   = len(real_vals) * len(gen_vals)
    total_gen_pairs  = len(gen_vals) * (len(gen_vals) - 1) // 2 if has_gen else 0

    print(f"\n  Coppie (real,real) : {total_real_pairs:,}")
    print(f"  Coppie (gen,real)  : {total_gr_pairs:,}")
    print(f"  Coppie (gen,gen)   : {total_gen_pairs:,}")
    print(f"  RFD da controllare : {len(dependencies)}\n")

    for dep_idx, (lhs_list, rhs_attr) in enumerate(dependencies):
        lhs_cols_idx = [attr_idx[a] for a in lhs_list if a in attr_idx]
        rhs_col_idx  = attr_idx.get(rhs_attr, -1)

        if not lhs_cols_idx or rhs_col_idx == -1:
            print(f"  [SKIP] RFD {dep_idx+1}: attributi non trovati ({lhs_list} -> {rhs_attr})")
            continue

        dep_str = f"{', '.join(lhs_list)} → {rhs_attr}"
        print(f"  Checking RFD {dep_idx+1}/{len(dependencies)}: {dep_str}")

        # ---- (real, real) — triangolare superiore ----
        n_rr, ex_rr = 0, []
        for i in range(len(real_vals)):
            diffs_i = np.abs(real_vals[i] - real_vals[i+1:])
            lhs_sim    = np.all(diffs_i[:, lhs_cols_idx] <= threshold, axis=1)
            rhs_dissim = diffs_i[:, rhs_col_idx] > threshold
            viol_mask  = lhs_sim & rhs_dissim
            n_rr += int(viol_mask.sum())
            if n_rr > 0 and len(ex_rr) < 5:
                viol_indices = np.where(viol_mask)[0]
                for vi in viol_indices[:max(0, 5 - len(ex_rr))]:
                    j = i + 1 + vi
                    ex_rr.append({
                        'pair_type': 'real-real',
                        'label_a': real_labels[i], 'label_b': real_labels[j],
                        'lhs_diffs': diffs_i[vi, lhs_cols_idx].tolist(),
                        'rhs_diff': float(diffs_i[vi, rhs_col_idx]),
                    })

        # ---- (gen, real) ----
        n_gr, ex_gr = 0, []
        if has_gen:
            n_gr, ex_gr = check_violations_matrix(
                gen_vals, real_vals, lhs_cols_idx, rhs_col_idx,
                threshold, 'gen-real', gen_labels, real_labels
            )

        # ---- (gen, gen) — triangolare superiore ----
        n_gg, ex_gg = 0, []
        if has_gen and len(gen_vals) > 1:
            for i in range(len(gen_vals)):
                diffs_i = np.abs(gen_vals[i] - gen_vals[i+1:])
                lhs_sim    = np.all(diffs_i[:, lhs_cols_idx] <= threshold, axis=1)
                rhs_dissim = diffs_i[:, rhs_col_idx] > threshold
                viol_mask  = lhs_sim & rhs_dissim
                n_gg += int(viol_mask.sum())
                if n_gg > 0 and len(ex_gg) < 5:
                    viol_indices = np.where(viol_mask)[0]
                    for vi in viol_indices[:max(0, 5 - len(ex_gg))]:
                        j = i + 1 + vi
                        ex_gg.append({
                            'pair_type': 'gen-gen',
                            'label_a': gen_labels[i], 'label_b': gen_labels[j],
                            'lhs_diffs': diffs_i[vi, lhs_cols_idx].tolist(),
                            'rhs_diff': float(diffs_i[vi, rhs_col_idx]),
                        })

        n_total = n_rr + n_gr + n_gg
        rfd_valid = (n_gr + n_gg) == 0  # le violazioni real-real erano già presenti
        status = "✓ VALIDA" if rfd_valid else "✗ VIOLATA"

        print(f"    {status}  |  rr={n_rr}  gr={n_gr}  gg={n_gg}  total={n_total}")

        # Esempi di violazioni per il report
        all_examples = ex_rr[:3] + ex_gr[:3] + ex_gg[:3]

        report_rows.append({
            'rfd':                   dep_str,
            'lhs':                   str(lhs_list),
            'rhs':                   rhs_attr,
            'threshold':             threshold,
            'violations_real_real':  n_rr,
            'violations_gen_real':   n_gr,
            'violations_gen_gen':    n_gg,
            'violations_total':      n_total,
            'rfd_valid_after_gen':   rfd_valid,
            'baseline_real_violations': n_rr,
            'new_violations':        n_gr + n_gg,
            'example_violations':    str(all_examples[:6]) if all_examples else 'none',
        })

    return pd.DataFrame(report_rows)


# =========================================================================
# SUMMARY REPORT
# =========================================================================

def print_summary(report_df: pd.DataFrame, threshold: float,
                  real_count: int, gen_count: int):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  REPORT VALIDAZIONE RFD — SyRFD")
    print(f"{sep}")
    print(f"  Tuple reali    : {real_count}")
    print(f"  Tuple generate : {gen_count}")
    print(f"  Soglia θ       : {threshold}")
    print(f"  RFD analizzate : {len(report_df)}")
    print(f"{sep}")

    valid_count   = report_df['rfd_valid_after_gen'].sum()
    invalid_count = len(report_df) - valid_count
    print(f"  RFD valide dopo generazione  : {valid_count}/{len(report_df)}")
    print(f"  RFD violate dopo generazione : {invalid_count}/{len(report_df)}")

    total_new_violations = report_df['new_violations'].sum()
    total_rr_violations  = report_df['violations_real_real'].sum()
    print(f"\n  Violazioni baseline (real-real)     : {total_rr_violations}")
    print(f"  Nuove violazioni introdotte (gen-*) : {total_new_violations}")

    if total_new_violations == 0:
        print(f"\n  ✓ Nessuna violazione RFD introdotta dalle tuple generate.")
    else:
        print(f"\n  ✗ Attenzione: {total_new_violations} violazioni introdotte.")
        print(f"    Rivedere le tuple generate o abbassare lambda_rfd nel training.")

    print(f"\n{sep}")
    print(f"  Dettaglio per RFD:")
    print(f"  {'RFD':<40} {'rr':>6} {'gr':>6} {'gg':>6} {'stato'}")
    print(f"  {'-'*62}")
    for _, row in report_df.iterrows():
        rfd_short = row['rfd'][:38] + '..' if len(row['rfd']) > 40 else row['rfd']
        st = '✓' if row['rfd_valid_after_gen'] else '✗'
        print(f"  {rfd_short:<40} {row['violations_real_real']:>6} "
              f"{row['violations_gen_real']:>6} {row['violations_gen_gen']:>6} "
              f"  {st}")
    print(f"{sep}\n")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validazione RFD su dati originali + generati"
    )
    parser.add_argument("--imbalanced", type=str, default=None,
                        help="CSV imbilanciato originale (auto-identifica classe minoritaria)")
    parser.add_argument("--real",       type=str, default=None,
                        help="CSV con sole tuple reali minoritarie (alternativa a --imbalanced)")
    parser.add_argument("--gen",        type=str, default=None,
                        help="CSV con tuple generate da SyRFD (opzionale)")
    parser.add_argument("--rfd",        type=str, default=None,
                        help="File TXT con le RFD")
    parser.add_argument("--threshold",  type=float, default=4.0,
                        help="Soglia θ per le RFD (default: 4.0)")
    parser.add_argument("--output",     type=str, default="rfd_validation_report.csv")
    args = parser.parse_args()

    # --- Caricamento dati reali ---
    if args.imbalanced:
        print(f"\nCaricamento dataset imbilanciato: {args.imbalanced}")
        df_all = pd.read_csv(args.imbalanced)
        min_class = df_all['class'].value_counts().idxmin()
        print(f"Classe minoritaria: '{min_class}'")
        real_df = df_all[df_all['class'] == min_class].copy().reset_index(drop=True)
    elif args.real:
        print(f"\nCaricamento dati reali: {args.real}")
        real_df = pd.read_csv(args.real).reset_index(drop=True)
    else:
        print("\nNessun file fornito — uso dati Mock.")
        np.random.seed(42)
        attrs = [f"Attr{i}" for i in range(1, 8)]
        real_data = {a: np.random.uniform(10, 60, 60) for a in attrs}
        real_data['class'] = 1
        real_df = pd.DataFrame(real_data)

    # --- Caricamento dati generati ---
    if args.gen:
        print(f"Caricamento dati generati: {args.gen}")
        gen_df = pd.read_csv(args.gen).reset_index(drop=True)
    else:
        print("Nessun file generato — validazione solo sui dati reali.")
        gen_df = None

    # --- Attributi comuni ---
    all_attrs = [c for c in real_df.columns if c != 'class']
    if gen_df is not None:
        gen_attrs = [c for c in gen_df.columns if c != 'class']
        all_attrs = [a for a in all_attrs if a in gen_attrs]

    print(f"\nAttributi: {all_attrs}")
    print(f"Dati reali   : {len(real_df)} tuple")
    print(f"Dati generati: {len(gen_df) if gen_df is not None else 0} tuple")

    # --- Caricamento RFD ---
    if args.rfd:
        print(f"\nCaricamento RFD da: {args.rfd}")
        dependencies = parse_rfds(args.rfd)
        print(f"Trovate {len(dependencies)} RFD.")
    else:
        print("\nNessun file RFD — uso RFD fittizie per test.")
        dependencies = mock_rfds(all_attrs)

    print(f"\nRFD da validare:")
    for lhs, rhs in dependencies:
        print(f"  {', '.join(lhs)} → {rhs}")

    # --- Validazione ---
    report_df = validate_rfds(
        real_df   = real_df,
        gen_df    = gen_df,
        dependencies = dependencies,
        threshold = args.threshold,
        attrs     = all_attrs,
    )

    # --- Summary ---
    print_summary(report_df, args.threshold, len(real_df),
                  len(gen_df) if gen_df is not None else 0)

    # --- Salvataggio ---
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    report_df.to_csv(args.output, index=False)
    print(f"Report salvato in: {args.output}")


if __name__ == "__main__":
    main()
