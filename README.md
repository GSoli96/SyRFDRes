# SyRFDRes: Constraint-Aware Minority Oversampling via Relaxed Functional Dependencies

This repository contains the reference implementation, experimental pipeline,
and results for **SyRFDRes** (**Sy**nthetic **R**elaxed **F**unctional
**D**ependency-aware oversampling with **Res**idual networks), a generative
framework for minority-class oversampling on imbalanced tabular data that
treats compliance with Relaxed Functional Dependencies (RFDs) as a first-class
objective, alongside statistical fidelity and diversity.

## Abstract

Class imbalance is a pervasive challenge in real-world tabular datasets,
where the minority class is systematically underrepresented, causing standard
classifiers to exhibit strong bias toward the majority. Existing oversampling
approaches, from SMOTE-based geometric interpolation to deep generative
models, focus exclusively on statistical fidelity and diversity, and remain
unaware of the quality constraints underlying data. We present SyRFDRes, a
generative framework for constrained minority oversampling that exploits
Relaxed Functional Dependencies (RFDs) at two complementary levels. At
training time, a new differentiable *SoftRFD loss*, based on sigmoid
approximations, provides a continuous gradient signal that steers the
generator away from the RFD-violation region. At inference time, a vectorized
*RFD Validator* checks entire batches of generated tuples against the
original data through broadcast tensor comparisons, discarding those that
introduce RFD violations and guaranteeing that the augmented dataset
preserves the validity of all input RFDs. SyRFDRes builds on a new residual
generator architecture, namely *TabResNet*, tailored for low-cardinalities of
minority classes, trained with a two-phase training that combines Maximum
Mean Discrepancy fidelity, RFD constraint enforcement, Sliced Wasserstein
distance, and an anti-collapse coverage objective. We evaluate SyRFDRes on 36
benchmark datasets, comparing against 9 geometric and deep generative methods
and 7 large language models, the latter prompted with 8 different strategies.
SyRFDRes achieves full RFD compliance by its very nature, matching and in
most cases exceeding existing oversampling approaches in terms of the
fidelity, diversity, and utility of the data generated.

## Method overview

- **TabResNet generator**: a residual MLP (input projection + 3 residual
  blocks + output projection) mapping Gaussian noise to synthetic tuples,
  designed to remain stable when trained on the small sample sizes typical of
  minority classes.
- **SoftRFD loss**: a differentiable, sigmoid-based relaxation of the RFD
  violation predicate, giving the generator a gradient signal that steers it
  away from the violation region during training.
- **Two-phase training**: a warmup phase (MMD fidelity + SoftRFD) anchors the
  generator to the real minority distribution and away from RFD violations;
  a refinement phase adds a Sliced Wasserstein Distance term and a
  coverage/anti-collapse term to spread generated tuples over the support of
  the real data.
- **Vectorized RFD Validator**: at generation time, a broadcast tensor
  comparison checks every candidate tuple against the minority partition
  (*relaxed* mode) or against the minority partition and all previously
  accepted synthetic tuples (*strict* mode), discarding violating tuples and
  triggering adaptive resampling until the oversampling budget is reached.

See the paper for the full formalization, correctness proofs of the
validator, and the complete experimental evaluation.

## Repository structure

```
.
├── src/                        Source code (see table below)
├── prompts/                    Verbatim LLM prompting-strategy templates
├── scripts/                    Shell/batch runners for the experimental pipeline
├── docs/
│   └── reproducibility.md      Environment, seeds, hardware, and step-by-step reproduction guide
├── imbalanced_datasets/        Input datasets (36 KEEL/imbalanced-learn + UCI benchmarks)
│   └── min/                    Minority-class partitions used for training
├── discovered_rfds_processed/  RFDs discovered on each minority partition (DiMe, Domino)
├── augmented_datasets/         Balanced datasets produced by SyRFDRes and every baseline
├── augmentation_results/       Raw SyRFDRes generation runs, grouped by threshold/mode
├── evaluation_metrics/         Per-dataset and aggregated fidelity/diversity/RFD-compliance metrics
├── evaluation_plots/           Per-dataset evaluation plots
├── plots/                      Aggregated summary plots (e.g. RFD compliance)
├── requirements.txt
└── LICENSE
```

### Source code (`src/`)

| Script | Role |
|---|---|
| `syrfd_learning_v7_resnet.py` | Main SyRFDRes training/generation entry point (fixed similarity thresholds) |
| `syrfd_learning_v7_resnet_variable.py` | Variant using per-attribute, automatically inferred thresholds (Domino) |
| `syrfd_learning_v7_resnet_perattr.py` | Variant with per-attribute threshold configuration |
| `generate_from_checkpoint.py` | Generate synthetic tuples from a saved checkpoint without retraining |
| `generate_from_checkpoint_variable.py` | Same, for the variable-threshold variant |
| `generate_from_grid_checkpoints.py` | Batch generation across a grid of saved checkpoints |
| `balance_datasets.py` / `balance_datasets_variable.py` | Merge real + synthetic tuples into balanced datasets |
| `validate_rfds.py` | Validate RFDs on real + synthetic data (compliance metrics) |
| `evaluate_metrics.py` / `evaluate_metrics_v5.py` | Fidelity/diversity/utility evaluation of augmented datasets |
| `aggregate_diversity_full.py`, `aggregate_fidelity_full.py`, `aggregate_fidelity_rfd_compliance.py`, `aggregate_rfd_compliance.py` | Aggregate per-dataset metrics into paper-ready tables |
| `fidelity_analysis.py` | Additional fidelity analysis utilities |
| `barplot_diversity.py`, `heatmap_rfd_compliance.py`, `plot_llm_diversity.py`, `plot_llm_diversity_with_error_bars.py`, `plot_llm_fidelity.py`, `plot_rfd_compliance.py` | Generate the paper's figures |
| `llm_generation_advanced2.py` | LLM baselines: 8 prompting strategies over 5 local/cloud models |

All scripts expect to be run **from the repository root** (they resolve data
paths such as `./imbalanced_datasets/` relative to the current working
directory).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required (developed and tested with Python 3.12). A CUDA-capable
GPU is recommended but not required (falls back to CPU automatically); see
`docs/reproducibility.md` for the exact environment used to produce the
paper's results.

Two datasets (SUSY-derived files, D35 in the paper) are not committed to this
repository due to their size. Run the following once before using them:

```bash
python scripts/download_susy.py
```

## Quick start

```bash
# 1. Train SyRFDRes on a minority partition with fixed-threshold RFDs
python src/syrfd_learning_v7_resnet.py \
    --data ./imbalanced_datasets/min/abalone9-18_min.csv \
    --rfd ./discovered_rfds_processed/RFD4_E0.0_abalone9-18.txt \
    --oversampling 1000 --epochs 2000 --rfd_mode strict --seed 42

# 2. Rebalance the full datasets with the generated tuples
python src/balance_datasets.py

# 3. Evaluate fidelity / diversity / RFD compliance against all baselines
python src/evaluate_metrics_v5.py --real ... --imbalanced ... --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

# 4. Aggregate metrics and regenerate the paper's tables/plots
python src/aggregate_fidelity_full.py
python src/aggregate_diversity_full.py
python src/plot_rfd_compliance.py
```

Ready-made runners for the full 36-dataset sweep are in `scripts/` (see
`docs/reproducibility.md` for the complete end-to-end procedure).

> **Note on checkpoints.** Trained generator checkpoints are not included in
> this artifact to keep it lightweight. Since every training entry point
> accepts a fixed `--seed` (default `42`), re-running `src/syrfd_learning_v7_resnet*.py`
> with the same arguments reproduces the same generator (up to the
> non-determinism of some CUDA kernels across GPU models/driver versions, see
> `docs/reproducibility.md`).

## Datasets

36 imbalanced binary/one-vs-rest classification datasets from
KEEL/imbalanced-learn and the UCI repository (Iris0, Cleveland-0-vs-4,
New-Thyroid1/2, Ecoli-0-vs-1, Dermatology-6, Migraine (6 one-vs-rest tasks),
Abalone9-18, Transfusion, Pima, Vowel0, Yeast1/3, KDDCup-guess, Obesity (7
one-vs-rest tasks), Page-blocks, Covtype (7 one-vs-rest tasks), SUSY). RFDs
were discovered on each minority partition with DiMe (fixed similarity
thresholds φ ∈ {2, 4, 8}) and Domino (automatically inferred per-attribute
thresholds). See Table 2 of the paper for the full per-dataset statistics.

## Baselines

- **Geometric**: SMOTE, SMOTE-CDNN, FSMOTE
- **Deep generative**: CasTGAN, CTAB-GAN, CTAB-GAN+, CuTS+, DDPM, GOGGLE,
  PreFair, TabFairGAN, TabularARGN, TVAE
- **LLM-based** (5 models × 8 prompting strategies, see `prompts/`):
  Devstral-Small-24B, Gemma-3-12B, Gemma-4-31B, GPT-OSS-20B, GPT-OSS-120B,
  plus zero-shot DeepSeek3 and Llama3

## License

Released under the MIT License (see `LICENSE`).

## Reproducibility

See [`docs/reproducibility.md`](docs/reproducibility.md) for the exact
software/hardware environment, random seeds, and a full step-by-step
procedure to reproduce every table and figure in the paper.
