"""
generate_from_checkpoint.py
============================
Standalone script to load a saved SyRFD checkpoint and generate
synthetic tuples without running training.

Typical usage
----------
# Basic generation - reads dataset, RFD and checkpoint from the standard folder
python generate_from_checkpoint.py \\
    --data   abalone9-18_min.csv \\
    --rfd    RFD4_E0.0_abalone9-18_min.txt \\
    --run_id a3f2bc91 \\
    --n      1000

# With relaxed mode and custom threshold
python generate_from_checkpoint.py \\
    --data      kddcup-guess_passwd_vs_satan_min.csv \\
    --rfd       rfds.txt \\
    --run_id    gs_007_4a2f1c \\
    --n         500 \\
    --rfd_mode  relaxed \\
    --threshold 4.0

# List all available checkpoints for a dataset
python generate_from_checkpoint.py \\
    --data  abalone9-18_min.csv \\
    --list

Expected structure of the checkpoint/ folder
-------------------------------------------
    checkpoint/
      <base>_<run_id>.pt              - model weights
      configuration_parameters.csv   - parameter log + metrics

where <base> = CSV file name without extension.
"""

import re
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Minimal components (self-contained copies, independent of syrfd_learning_v4)

class MLPGenerator(nn.Module):
    def __init__(self, latent_dim, data_dim, hidden_dims=[128, 256, 128]):
        super().__init__()
        layers, in_dim = [], latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, data_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class RFDValidator:
    """
    Hard validation of RFDs (identical to syrfd_learning_v4).

    strict  : each tuple is checked against originals + already generated ones
    relaxed : each tuple is checked only against the originals
    """

    def __init__(self, dependencies, attributes, threshold, mode="strict"):
        self.dependencies = dependencies
        self.attributes   = attributes
        self.threshold    = threshold
        self.mode         = mode
        self._generated_rows = []

    def reset_generated(self):
        self._generated_rows = []

    def _check_vs(self, new_vals, ref_vals, attr_idx):
        diffs = np.abs(ref_vals - new_vals)
        for lhs_list, rhs_attr in self.dependencies:
            lhs_cols = [attr_idx[a] for a in lhs_list if a in attr_idx]
            rhs_col  = attr_idx.get(rhs_attr, -1)
            if not lhs_cols or rhs_col == -1:
                continue
            if np.any(
                np.all(diffs[:, lhs_cols] <= self.threshold, axis=1)
                & (diffs[:, rhs_col] > self.threshold)
            ):
                return False
        return True

    def is_valid(self, row_dict, reference_df):
        attr_idx = {a: i for i, a in enumerate(self.attributes)}
        new_vals = np.array([float(row_dict.get(a, 0.0)) for a in self.attributes])
        ref_vals = reference_df[self.attributes].values.astype(float)
        if not self._check_vs(new_vals, ref_vals, attr_idx):
            return False
        if self.mode == "strict" and self._generated_rows:
            gen_vals = np.array([[float(r.get(a, 0.0)) for a in self.attributes]
                                  for r in self._generated_rows])
            if not self._check_vs(new_vals, gen_vals, attr_idx):
                return False
        return True

    def filter_valid(self, generated_df, reference_df):
        valid_mask = []
        for _, row in generated_df.iterrows():
            row_dict = {a: row[a] for a in self.attributes if a in row.index}
            ok = self.is_valid(row_dict, reference_df)
            valid_mask.append(ok)
            if ok and self.mode == "strict":
                self._generated_rows.append(row_dict)
        return generated_df[valid_mask].reset_index(drop=True)


# RFD parsing

def parse_rfds(rfd_path):
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


# Main functions

def find_checkpoint(base_name, run_id, ckpt_dir="checkpoint"):
    """Find the checkpoint .pt file. Raises FileNotFoundError if missing."""
    path = os.path.join(ckpt_dir, f"{base_name}_{run_id}.pt")
    if not os.path.exists(path):
        # Also search without base_name (for partial run_ids)
        candidates = [
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".pt") and run_id in f
        ] if os.path.isdir(ckpt_dir) else []
        if len(candidates) == 1:
            path = os.path.join(ckpt_dir, candidates[0])
            print(f"Found checkpoint: {path}")
        elif len(candidates) > 1:
            print(f"Found {len(candidates)} checkpoints with run_id '{run_id}':")
            for c in candidates:
                print(f"  {c}")
            raise FileNotFoundError(
                f"Specify base_name to disambiguate (use --list to see available ones).")
        else:
            raise FileNotFoundError(
                f"No checkpoint found for run_id='{run_id}' in '{ckpt_dir}'.\n"
                f"Use --list to see available checkpoints.")
    return path


def list_checkpoints(base_name, ckpt_dir="checkpoint"):
    """Print the available checkpoints for the given dataset."""
    config_csv = os.path.join(ckpt_dir, "configuration_parameters.csv")
    if not os.path.exists(config_csv):
        print(f"No configuration_parameters.csv file in '{ckpt_dir}'.")
        return

    df = pd.read_csv(config_csv)
    mask = df['base_name'] == base_name if 'base_name' in df.columns else pd.Series([True]*len(df))
    subset = df[mask]

    if len(subset) == 0:
        print(f"No checkpoint found for '{base_name}'.")
        return

    print(f"\n{'='*70}")
    print(f"  Available checkpoints for: {base_name}")
    print(f"{'='*70}")

    show_cols = ['run_id', 'timestamp', 'model_file']
    metric_cols = [c for c in subset.columns if c.startswith('metric_')]
    param_cols  = [c for c in subset.columns if c.startswith('param_')]

    key_metrics = ['metric_kl', 'metric_js', 'metric_q_sim',
                   'metric_dbi_local', 'metric_silhouette', 'metric_loss_composite']
    key_params  = ['param_epochs', 'param_lr', 'param_lambda_composite',
                   'param_warmup_epochs', 'param_rfd_mode']

    display = show_cols + [c for c in key_params if c in subset.columns] \
                        + [c for c in key_metrics if c in subset.columns]
    display = [c for c in display if c in subset.columns]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(subset[display].to_string(index=False))
    print(f"{'='*70}\n")


def generate(
    data_path:  str,
    rfd_path:   str,
    run_id:     str,
    n_samples:  int   = 100,
    threshold:  float = 4.0,
    latent_dim: int   = 32,
    rfd_mode:   str   = "strict",
    max_attempts: int = 5,
    ckpt_dir:   str   = "checkpoint",
    hidden_dims: list = None,
):
    """
    Load the checkpoint and generate n_samples valid tuples.

    Parameters
    ---------
    data_path    : CSV of the original dataset (used to derive the minority class,
                   attributes, min/max for denormalization)
    rfd_path     : TXT file with the RFDs
    run_id       : checkpoint identifier (from configuration_parameters.csv)
    n_samples    : number of tuples to generate
    threshold    : threshold theta for the RFDs
    latent_dim   : latent space dimension (must match the one used in training)
    rfd_mode     : 'strict' or 'relaxed'
    max_attempts : maximum number of attempts if few valid tuples are produced
    ckpt_dir     : checkpoint folder
    hidden_dims  : MLP architecture (default [128,256,128])
    """
    if hidden_dims is None:
        hidden_dims = [128, 256, 128]

    # --- Load dataset ---
    print(f"\nLoading dataset: {data_path}")
    df_all    = pd.read_csv(data_path)
    min_class = df_all['class'].value_counts().idxmin()
    df_min    = df_all[df_all['class'] == min_class].copy().reset_index(drop=True)
    attributes = [c for c in df_min.columns if c != 'class']
    print(f"Minority class: '{min_class}' ({len(df_min)} samples)")
    print(f"Attributes: {len(attributes)}")

    # --- Load RFD ---
    print(f"Loading RFD: {rfd_path}")
    dependencies = parse_rfds(rfd_path)
    print(f"Found {len(dependencies)} RFDs.")

    # --- Statistics for denormalization ---
    mins_np   = df_min[attributes].min().values.astype(np.float32)
    maxs_np   = df_min[attributes].max().values.astype(np.float32)
    ranges_np = np.clip(maxs_np - mins_np, 1e-5, None)
    mins_t    = torch.tensor(mins_np)
    ranges_t  = torch.tensor(ranges_np)

    # --- Load checkpoint ---
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    ckpt_path = find_checkpoint(base_name, run_id, ckpt_dir)

    data_dim  = len(attributes)
    generator = MLPGenerator(latent_dim, data_dim, hidden_dims)
    state     = torch.load(ckpt_path, map_location="cpu")
    generator.load_state_dict(state)
    generator.eval()
    print(f"Model loaded: {ckpt_path}")
    print(f"Architecture: latent={latent_dim} -> {hidden_dims} -> {data_dim}")

    # --- Validator ---
    validator = RFDValidator(dependencies, attributes, threshold, mode=rfd_mode)

    # --- Generation ---
    print(f"\nGenerating {n_samples} tuples (rfd_mode={rfd_mode}, threshold={threshold})...")
    valid_rows, attempt = [], 0

    while sum(len(d) for d in valid_rows) < n_samples and attempt < max_attempts:
        attempt += 1
        bs = max(n_samples * 2, 256)
        with torch.no_grad():
            z         = torch.randn(bs, latent_dim)
            x_gen_raw = generator(z)
            x_gen_n   = torch.sigmoid(x_gen_raw)
            x_gen_den = torch.clamp(x_gen_n, 0.0, 1.0) * ranges_t + mins_t

        df_gen = pd.DataFrame(x_gen_den.numpy(), columns=attributes)

        # Round integer attributes
        for attr in attributes:
            if np.issubdtype(df_min[attr].dtype, np.integer):
                df_gen[attr] = df_gen[attr].round().astype(int)

        df_valid = validator.filter_valid(df_gen, df_min)
        valid_rows.append(df_valid)
        n_valid = sum(len(d) for d in valid_rows)
        n_disc  = bs - len(df_valid)
        print(f"  Attempt {attempt}: {len(df_valid)}/{bs} valid "
              f"({n_disc} discarded by RFD). Total: {n_valid}/{n_samples}")

    result_df = pd.concat(valid_rows, ignore_index=True)
    if len(result_df) >= n_samples:
        result_df = result_df.head(n_samples)
    else:
        print(f"\nWARNING: only generated {len(result_df)}/{n_samples} valid tuples "
              f"after {max_attempts} attempts.")

    result_df['class'] = min_class
    return result_df


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Load a SyRFD checkpoint and generate synthetic tuples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data",       type=str, required=True,
                        help="CSV of the original dataset")
    parser.add_argument("--rfd",        type=str, default=None,
                        help="TXT file with the RFDs (required unless --list)")
    parser.add_argument("--run_id",     type=str, default=None,
                        help="run_id of the checkpoint to load")
    parser.add_argument("--n",          type=int, default=100,
                        help="Number of tuples to generate (default: 100)")
    parser.add_argument("--threshold",  type=float, default=4.0,
                        help="Threshold theta for the RFDs (default: 4.0)")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent space dimension (default: 32, must match training)")
    parser.add_argument("--rfd_mode",   type=str, default="strict",
                        choices=["strict", "relaxed"],
                        help="RFD validation mode (default: strict)")
    parser.add_argument("--max_attempts", type=int, default=5,
                        help="Maximum generation attempts (default: 5)")
    parser.add_argument("--ckpt_dir",   type=str, default="checkpoint",
                        help="Checkpoint folder (default: checkpoint/)")
    parser.add_argument("--output_dir", type=str, default="augmentation_results",
                        help="Output CSV folder (default: augmentation_results/)")
    parser.add_argument("--list",       action="store_true",
                        help="List available checkpoints for the dataset and exit")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.data))[0]

    # ---- List mode ----
    if args.list:
        list_checkpoints(base_name, args.ckpt_dir)
        return

    # ---- Argument validation ----
    if not args.run_id:
        print("ERROR: --run_id is required (use --list to see available ones)")
        sys.exit(1)
    if not args.rfd:
        print("ERROR: --rfd is required for generation")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"ERROR: dataset file not found: {args.data}")
        sys.exit(1)
    if not os.path.exists(args.rfd):
        print(f"ERROR: RFD file not found: {args.rfd}")
        sys.exit(1)

    # ---- Generation ----
    df_gen = generate(
        data_path    = args.data,
        rfd_path     = args.rfd,
        run_id       = args.run_id,
        n_samples    = args.n,
        threshold    = args.threshold,
        latent_dim   = args.latent_dim,
        rfd_mode     = args.rfd_mode,
        max_attempts = args.max_attempts,
        ckpt_dir     = args.ckpt_dir,
    )

    # ---- Save with timestamp ----
    os.makedirs(args.output_dir, exist_ok=True)
    ts       = int(time.time() * 1000)
    out_path = os.path.join(args.output_dir, f"{base_name}_{ts}.csv")
    df_gen.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"  Tuples generated : {len(df_gen)}")
    print(f"  Saved to         : {out_path}")
    print(f"{'='*50}")
    print(df_gen.head(5).to_string())


if __name__ == "__main__":
    main()
