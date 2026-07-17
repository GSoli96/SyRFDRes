#!/usr/bin/env bash
# =============================================================================
# run_evaluate_metrics_v5.sh
# Runs src/evaluate_metrics_v5.py for each dataset, with one explicit call
# per dataset (NO loop/for/while: commands are listed one by one).
# =============================================================================
#python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/abalone9-18_min.csv --imbalanced ./imbalanced_datasets/abalone9-18.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

#python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/kddcup-guess_passwd_vs_satan_min.csv --imbalanced ./imbalanced_datasets/kddcup-guess_passwd_vs_satan.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Migraine_onevsrest_5_min.csv --imbalanced ./imbalanced_datasets/Migraine_onevsrest_5.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_2_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_2.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/transfusion_min.csv --imbalanced ./imbalanced_datasets/transfusion.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/cleveland-0_vs_4_min.csv --imbalanced ./imbalanced_datasets/cleveland-0_vs_4.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_3_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_3.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/vowel0_min.csv --imbalanced ./imbalanced_datasets/vowel0.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/dermatology-6_min.csv --imbalanced ./imbalanced_datasets/dermatology-6.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Migraine_onevsrest_0_min.csv --imbalanced ./imbalanced_datasets/Migraine_onevsrest_0.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/new-thyroid1_min.csv --imbalanced ./imbalanced_datasets/new-thyroid1.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_4_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_4.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/yeast1_min.csv --imbalanced ./imbalanced_datasets/yeast1.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/ecoli-0_vs_1_min.csv --imbalanced ./imbalanced_datasets/ecoli-0_vs_1.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Migraine_onevsrest_1_min.csv --imbalanced ./imbalanced_datasets/Migraine_onevsrest_1.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/newthyroid2_min.csv --imbalanced ./imbalanced_datasets/newthyroid2.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_5_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_5.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/yeast3_min.csv --imbalanced ./imbalanced_datasets/yeast3.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/ecoli1_min.csv --imbalanced ./imbalanced_datasets/ecoli1.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Migraine_onevsrest_2_min.csv --imbalanced ./imbalanced_datasets/Migraine_onevsrest_2.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_6_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_6.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Migraine_onevsrest_3_min.csv --imbalanced ./imbalanced_datasets/Migraine_onevsrest_3.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_0_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_0.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/page-blocks-1-3_vs_4_min.csv --imbalanced ./imbalanced_datasets/page-blocks-1-3_vs_4.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/iris0_min.csv --imbalanced ./imbalanced_datasets/iris0.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Migraine_onevsrest_4_min.csv --imbalanced ./imbalanced_datasets/Migraine_onevsrest_4.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/Obesity_onevsrest_1_min.csv --imbalanced ./imbalanced_datasets/Obesity_onevsrest_1.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics

python src/evaluate_metrics_v5.py --real ./imbalanced_datasets/min/pima_min.csv --imbalanced ./imbalanced_datasets/pima.csv --augmented_dir ./augmented_datasets --rfd_dir ./discovered_rfds_processed --output_dir ./evaluation_metrics
