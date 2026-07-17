#!/bin/bash

# Runs src/evaluate_metrics_v5.py for every dataset
# One explicit call per dataset (no loop)

# Base directories
BASE_DIR="./imbalanced_datasets"
MIN_DIR="${BASE_DIR}/min"
AUGMENTED_DIR="./augmented_datasets"
RFD_DIR="./discovered_rfds_processed"
OUTPUT_DIR="./evaluation_metrics"

# Create the output directory
mkdir -p "${OUTPUT_DIR}"

# Dataset 1: abalone9-18
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/abalone9-18_min.csv" --imbalanced "${BASE_DIR}/abalone9-18.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 2: cleveland-0_vs_4
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/cleveland-0_vs_4_min.csv" --imbalanced "${BASE_DIR}/cleveland-0_vs_4.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 3: dermatology-6
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/dermatology-6_min.csv" --imbalanced "${BASE_DIR}/dermatology-6.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 4: ecoli-0_vs_1
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/ecoli-0_vs_1_min.csv" --imbalanced "${BASE_DIR}/ecoli-0_vs_1.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 5: ecoli1
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/ecoli1_min.csv" --imbalanced "${BASE_DIR}/ecoli1.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 6: iris0
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/iris0_min.csv" --imbalanced "${BASE_DIR}/iris0.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 7: kddcup-guess_passwd_vs_satan
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/kddcup-guess_passwd_vs_satan_min.csv" --imbalanced "${BASE_DIR}/kddcup-guess_passwd_vs_satan.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 8: Migraine_onevsrest_0
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Migraine_onevsrest_0_min.csv" --imbalanced "${BASE_DIR}/Migraine_onevsrest_0.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 9: Migraine_onevsrest_1
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Migraine_onevsrest_1_min.csv" --imbalanced "${BASE_DIR}/Migraine_onevsrest_1.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 10: Migraine_onevsrest_2
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Migraine_onevsrest_2_min.csv" --imbalanced "${BASE_DIR}/Migraine_onevsrest_2.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 11: Migraine_onevsrest_3
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Migraine_onevsrest_3_min.csv" --imbalanced "${BASE_DIR}/Migraine_onevsrest_3.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 12: Migraine_onevsrest_4
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Migraine_onevsrest_4_min.csv" --imbalanced "${BASE_DIR}/Migraine_onevsrest_4.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 13: Migraine_onevsrest_5
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Migraine_onevsrest_5_min.csv" --imbalanced "${BASE_DIR}/Migraine_onevsrest_5.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

echo "=== Elaborazione completata per tutti i 28 dataset ==="