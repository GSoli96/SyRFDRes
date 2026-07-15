#!/bin/bash

# Script per eseguire src/evaluate_metrics_v5.py su tutti i dataset
# Esecuzione riga per riga

# Directory base
BASE_DIR="./imbalanced_datasets"
MIN_DIR="${BASE_DIR}/min"
AUGMENTED_DIR="./augmented_datasets"
RFD_DIR="./discovered_rfds_processed/Domino"
OUTPUT_DIR="./evaluation_metrics"

# Crea la directory di output
mkdir -p "${OUTPUT_DIR}"

# Dataset 14: new-thyroid1
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/new-thyroid1_min.csv" --imbalanced "${BASE_DIR}/new-thyroid1.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 15: newthyroid2
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/newthyroid2_min.csv" --imbalanced "${BASE_DIR}/newthyroid2.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 16: Obesity_onevsrest_0
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_0_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_0.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 17: Obesity_onevsrest_1
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_1_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_1.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 18: Obesity_onevsrest_2
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_2_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_2.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 19: Obesity_onevsrest_3
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_3_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_3.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 20: Obesity_onevsrest_4
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_4_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_4.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 21: Obesity_onevsrest_5
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_5_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_5.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 22: Obesity_onevsrest_6
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/Obesity_onevsrest_6_min.csv" --imbalanced "${BASE_DIR}/Obesity_onevsrest_6.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 23: page-blocks-1-3_vs_4
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/page-blocks-1-3_vs_4_min.csv" --imbalanced "${BASE_DIR}/page-blocks-1-3_vs_4.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 24: pima
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/pima_min.csv" --imbalanced "${BASE_DIR}/pima.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 25: transfusion
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/transfusion_min.csv" --imbalanced "${BASE_DIR}/transfusion.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 26: vowel0
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/vowel0_min.csv" --imbalanced "${BASE_DIR}/vowel0.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 27: yeast1
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/yeast1_min.csv" --imbalanced "${BASE_DIR}/yeast1.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

# Dataset 28: yeast3
python src/evaluate_metrics_v5.py --real "${MIN_DIR}/yeast3_min.csv" --imbalanced "${BASE_DIR}/yeast3.csv" --augmented_dir "${AUGMENTED_DIR}" --rfd_dir "${RFD_DIR}" --output_dir "${OUTPUT_DIR}" --bins 30

echo "=== Elaborazione completata per tutti i 28 dataset ==="