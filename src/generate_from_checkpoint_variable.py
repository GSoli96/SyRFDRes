"""
use_checkpoint.py
==================
Utility script to REUSE a checkpoint already trained by SyRFDLearner
without redoing the training.

Requires the original module (the one with SyRFDLearner, TabResNetGenerator,
RFDValidator, etc.) to be importable, i.e.:

    from syrfd_learning_v7_resnet_variable import SyRFDLearner

Functionality:
  1. --list                 -> shows all runs saved in checkpoint/configuration_parameters.csv
                                with the main metrics, sorted from best to worst
  2. --run_id <ID>           -> loads that specific checkpoint and generates
  3. --best                  -> automatically selects the run with the best metric
                                (default: lowest metric_swd; customizable with --metric)

Note: --data and --rfd are still required, because the generator alone doesn't know
the attributes/min/max (normalization) or the RFD dependencies used to validate
the generated tuples. They must be TAKEN FROM THE SAME dataset/RFD file used during
training of the chosen checkpoint.
"""

import os
import argparse
import pandas as pd

# --- EDIT THIS LINE with the actual name of your file/module ---
from syrfd_learning_v7_resnet_variable import SyRFDLearner
# --------------------------------------------------------------------

CKPT_DIR = "checkpoint_soglie_variable"
CONFIG_CSV = os.path.join(CKPT_DIR, "configuration_parameters.csv")


def list_runs(metric: str = "metric_swd", ascending: bool = True) -> pd.DataFrame:
    """Read the checkpoint CSV and show the runs sorted by a metric."""
    if not os.path.exists(CONFIG_CSV):
        raise FileNotFoundError(f"No configuration file found: {CONFIG_CSV}")

    df = pd.read_csv(CONFIG_CSV)
    if metric not in df.columns:
        available = [c for c in df.columns if c.startswith("metric_")]
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    df_sorted = df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
    cols_show = ["run_id", "base_name", "timestamp", metric] + \
                [c for c in ["metric_silhouette", "metric_coverage_loss", "metric_dbi_sklearn"] if c in df.columns and c != metric]
    print("\n=== Available checkpoints (sorted by {}) ===".format(metric))
    print(df_sorted[cols_show].to_string(index=False))
    return df_sorted


def pick_best_run_id(metric: str = "metric_swd", ascending: bool = True) -> str:
    df_sorted = list_runs(metric=metric, ascending=ascending)
    if df_sorted.empty:
        raise RuntimeError("No runs available in the CSV.")
    best = df_sorted.iloc[0]
    print(f"\n[SELECTED] run_id={best['run_id']} ({metric}={best[metric]:.5f})")
    return str(best["run_id"])


def main():
    parser = argparse.ArgumentParser(description="Reuse an already-trained SyRFD checkpoint")
    parser.add_argument("--data", type=str, required=True, help="CSV of the original dataset used in training")
    parser.add_argument("--rfd", type=str, required=True, help="RFD dependency file used in training")
    parser.add_argument("--rfd_mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--oversampling", type=int, default=100, help="Number of synthetic tuples to generate")

    parser.add_argument("--list", action="store_true", help="List available checkpoints and exit")
    parser.add_argument("--run_id", type=str, default=None, help="Specific run ID to load")
    parser.add_argument("--best", action="store_true", help="Automatically select the best run")
    parser.add_argument("--metric", type=str, default="metric_swd", help="Metric used to sort/select runs (e.g. metric_swd, metric_silhouette)")
    parser.add_argument("--descending", action="store_true", help="Sort the metric in descending order (use for 'higher is better' metrics, e.g. metric_silhouette)")

    args = parser.parse_args()

    if args.list:
        list_runs(metric=args.metric, ascending=not args.descending)
        return

    if args.best:
        run_id = pick_best_run_id(metric=args.metric, ascending=not args.descending)
    elif args.run_id:
        run_id = args.run_id
    else:
        parser.error("Specify --run_id <ID>, or --best, or --list")
        return

    # Rebuild the learner (attributes, min/max, RFD dependencies) from the same
    # dataset/RFD file used to train the chosen checkpoint
    learner = SyRFDLearner(data_path=args.data, rfd_path=args.rfd, rfd_mode=args.rfd_mode)
    learner.load_checkpoint(run_id)

    print(f"\nGenerating {args.oversampling} synthetic tuples from checkpoint '{run_id}'...")
    df_gen = learner.generate(num_samples=args.oversampling)
    print(df_gen.head())

    output_dir = "augmentation_results"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{learner.base_name}_from_checkpoint_{run_id}.csv")
    df_gen.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()