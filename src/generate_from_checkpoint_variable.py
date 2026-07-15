"""
use_checkpoint.py
==================
Script di utilità per RIUSARE un checkpoint già addestrato da SyRFDLearner
senza dover rifare il training.

Richiede che il modulo originale (quello con SyRFDLearner, TabResNetGenerator,
RFDValidator, ecc.) sia importabile, ossia:

    from syrfd_learning_v7_resnet_variable import SyRFDLearner

Funzionalità:
  1. --list                 -> mostra tutti i run salvati in checkpoint/configuration_parameters.csv
                                con le metriche principali, ordinati dal migliore al peggiore
  2. --run_id <ID>           -> carica quello specifico checkpoint e genera
  3. --best                  -> seleziona automaticamente il run con la metrica migliore
                                (default: metric_swd più basso; personalizzabile con --metric)

Nota: --data e --rfd servono comunque, perché il generatore da solo non sa
quali siano gli attributi/min/max (normalizzazione) né le dipendenze RFD
usate per validare le tuple generate. Devono essere PRESI DALLO STESSO
dataset/file RFD usati durante il training del checkpoint scelto.
"""

import os
import argparse
import pandas as pd

# --- MODIFICA QUESTA RIGA con il nome reale del tuo file/modulo ---
from syrfd_learning_v7_resnet_variable import SyRFDLearner
# --------------------------------------------------------------------

CKPT_DIR = "checkpoint_soglie_variable"
CONFIG_CSV = os.path.join(CKPT_DIR, "configuration_parameters.csv")


def list_runs(metric: str = "metric_swd", ascending: bool = True) -> pd.DataFrame:
    """Legge il CSV dei checkpoint e mostra i run ordinati per una metrica."""
    if not os.path.exists(CONFIG_CSV):
        raise FileNotFoundError(f"Nessun file di configurazione trovato: {CONFIG_CSV}")

    df = pd.read_csv(CONFIG_CSV)
    if metric not in df.columns:
        disponibili = [c for c in df.columns if c.startswith("metric_")]
        raise ValueError(f"Metrica '{metric}' non trovata. Disponibili: {disponibili}")

    df_sorted = df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
    cols_show = ["run_id", "base_name", "timestamp", metric] + \
                [c for c in ["metric_silhouette", "metric_coverage_loss", "metric_dbi_sklearn"] if c in df.columns and c != metric]
    print("\n=== Checkpoint disponibili (ordinati per {}) ===".format(metric))
    print(df_sorted[cols_show].to_string(index=False))
    return df_sorted


def pick_best_run_id(metric: str = "metric_swd", ascending: bool = True) -> str:
    df_sorted = list_runs(metric=metric, ascending=ascending)
    if df_sorted.empty:
        raise RuntimeError("Nessun run disponibile nel CSV.")
    best = df_sorted.iloc[0]
    print(f"\n[SELEZIONATO] run_id={best['run_id']} ({metric}={best[metric]:.5f})")
    return str(best["run_id"])


def main():
    parser = argparse.ArgumentParser(description="Riusa un checkpoint SyRFD già addestrato")
    parser.add_argument("--data", type=str, required=True, help="CSV del dataset originale usato in training")
    parser.add_argument("--rfd", type=str, required=True, help="File delle dipendenze RFD usato in training")
    parser.add_argument("--rfd_mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--oversampling", type=int, default=100, help="Numero di tuple sintetiche da generare")

    parser.add_argument("--list", action="store_true", help="Elenca i checkpoint disponibili ed esci")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID specifico da caricare")
    parser.add_argument("--best", action="store_true", help="Seleziona automaticamente il run migliore")
    parser.add_argument("--metric", type=str, default="metric_swd", help="Metrica per ordinare/selezionare i run (es. metric_swd, metric_silhouette)")
    parser.add_argument("--descending", action="store_true", help="Ordina la metrica in modo decrescente (usalo per metriche 'più alto è meglio', es. metric_silhouette)")

    args = parser.parse_args()

    if args.list:
        list_runs(metric=args.metric, ascending=not args.descending)
        return

    if args.best:
        run_id = pick_best_run_id(metric=args.metric, ascending=not args.descending)
    elif args.run_id:
        run_id = args.run_id
    else:
        parser.error("Specifica --run_id <ID>, oppure --best, oppure --list")
        return

    # Ricostruisce il learner (attributi, min/max, dipendenze RFD) dallo stesso
    # dataset/file RFD usati per addestrare il checkpoint scelto
    learner = SyRFDLearner(data_path=args.data, rfd_path=args.rfd, rfd_mode=args.rfd_mode)
    learner.load_checkpoint(run_id)

    print(f"\nGenerazione di {args.oversampling} tuple sintetiche dal checkpoint '{run_id}'...")
    df_gen = learner.generate(num_samples=args.oversampling)
    print(df_gen.head())

    output_dir = "augmentation_results"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{learner.base_name}_from_checkpoint_{run_id}.csv")
    df_gen.to_csv(out_path, index=False)
    print(f"\n[SALVATO] {out_path}")


if __name__ == "__main__":
    main()