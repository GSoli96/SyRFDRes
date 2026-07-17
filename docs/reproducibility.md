# Reproducibility Guide

This document describes the software/hardware environment used to produce
the results in the paper, the random-seed policy, and the end-to-end steps to
reproduce every table and figure.

## Environment

| Component | Version |
|---|---|
| OS | Windows 10/11 (x86_64) |
| Python | 3.12 |
| PyTorch | 2.6.0 (CUDA 12.4 build) |
| NumPy | 2.3.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| SciPy | 1.16.3 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8GB VRAM (CUDA 12.4, driver 581.95) |

The full pinned dependency list is in [`../requirements.txt`](../requirements.txt).
Training runs on the largest datasets (Covtype, SUSY one-vs-rest partitions)
were run on the same single-GPU workstation; no distributed or multi-GPU
training is used anywhere in the pipeline. CPU-only execution is supported
(PyTorch falls back automatically) but proportionally slower.

> If experiments for the camera-ready version are re-run on different
> hardware (e.g., a shared cluster or a cloud GPU instance), update this table
> accordingly before submission.

## Random seeds

Every entry point in `src/syrfd_learning_v7_resnet*.py` accepts a `--seed`
argument (default `42`), which fixes `random`, `numpy.random`, and
`torch` (including all CUDA devices) at the start of `main()`. All results
reported in the paper use the default seed unless stated otherwise. Note that
full bitwise determinism on GPU is not guaranteed by PyTorch across different
GPU models/driver versions even with fixed seeds (cuDNN non-deterministic
kernels); metric values may vary by a small margin from those reported in the
paper when re-run on different hardware.

## Datasets

35 of the 36 datasets (all except the SUSY one-vs-rest partition, D35 in
Table 2 of the paper) are included directly in `imbalanced_datasets/`. SUSY is
not committed due to its size (~750MB / ~340MB for the minority partition);
run:

```bash
python scripts/download_susy.py
```

to download it from the UCI Machine Learning Repository and reconstruct the
exact low-level-feature subset used in the paper.

RFDs used for training/validation were discovered upstream with the DiMe
(fixed similarity thresholds φ ∈ {2, 4, 8}) and Domino (per-attribute inferred
thresholds) algorithms; their outputs are included in
`discovered_rfds_processed/`. The discovery tools themselves are external and
not part of this repository (see paper references).

## End-to-end reproduction

1. **Install dependencies** — see the "Installation" section of the main
   `README.md`.
2. **Download SUSY** (only needed for D35) — `python scripts/download_susy.py`.
3. **Train SyRFDRes** for each dataset × threshold × mode combination. Ready
   runners are provided in `scripts/`:
   - `scripts/execute_variable_thresholds.bat` / `..._strict.bat` — variable
     (Domino) thresholds, relaxed/strict mode
   - `scripts/run_syrfd_relaxed.bat` / `scripts/run_syrfd_strict.bat` — fixed
     thresholds, relaxed/strict mode
   Each run saves a checkpoint under `checkpoint/` and prints the generated
   tuples; see `src/syrfd_learning_v7_resnet.py --help` for all options
   (`--oversampling`, `--epochs`, `--rfd_mode`, `--seed`).
4. **Rebalance datasets** with the generated tuples by merging real and
   synthetic tuples into the layout used under `augmented_datasets/`. The
   already-merged datasets are included directly in this artifact; the
   merging script itself is an internal utility not part of this artifact.
5. **Run baselines** (SMOTE family, deep generative models, LLM-based
   generators) using their respective official implementations/libraries;
   outputs are expected under `augmented_datasets/<baseline_name>/`, matching
   the layout already produced by the runs used in the paper. LLM baselines
   can be reproduced with `src/llm_generation_advanced2.py` (prompt templates
   in `prompts/llm_prompt_templates.md`).
6. **Evaluate** fidelity, diversity, RFD compliance and utility:
   `scripts/evaluate_metrics.sh`, `scripts/evaluate_metrics2.sh`, or directly
   `src/evaluate_metrics_v5.py` for a single dataset (see its `--help`).
7. **Validate RFD compliance** independently with `src/validate_rfds.py`.
8. **Aggregate metrics into paper-ready tables** and **regenerate figures**.
   The aggregation and plotting utilities used to produce the paper's tables
   from raw per-run metrics are internal utilities not part of this
   artifact; their outputs are included directly under
   `evaluation_metrics/aggregated/` and `plots/`.

## Known limitations

- The exact preprocessing step that extracts each dataset's minority-class
  partition (the `*_min.csv` files under `imbalanced_datasets/min/`) is not
  itself part of this repository; the already-extracted partitions are
  provided directly so this step does not need to be reproduced, except for
  SUSY (see `scripts/download_susy.py`, which performs this extraction
  inline).
- Dataset-merging, metric-aggregation, and plotting utility scripts are not
  included in this artifact (see the note on utility scripts in `README.md`);
  their outputs are included directly.
- A handful of raw result files under `augmentation_results/v7_variable_*/`
  have long auto-generated filenames (up to ~130 characters). On Windows,
  `git clone` may fail on these files if the repository is cloned into an
  already deeply nested destination path, due to the OS's legacy 260-character
  path limit. If this happens, clone closer to a drive root (e.g. `C:\src\...`)
  or enable Windows long-path support (`git config --global core.longpaths true`
  and the corresponding Windows policy setting).
