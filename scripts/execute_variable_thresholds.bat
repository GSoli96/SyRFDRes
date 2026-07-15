@echo off
echo Avvio esecuzione su tutti i dataset...
echo.

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/abalone9-18_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_abalone9-18_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/cleveland-0_vs_4_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_cleveland-0_vs_4_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/dermatology-6_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_dermatology-6_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/ecoli-0_vs_1_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_ecoli-0_vs_1_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/ecoli1_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_ecoli1_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/iris0_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_iris0_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/kddcup-guess_passwd_vs_satan_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_kddcup-guess_passwd_vs_satan_red_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Migraine_onevsrest_0_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Migraine_onevsrest_0_red_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Migraine_onevsrest_1_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Migraine_onevsrest_1_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Migraine_onevsrest_2_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Migraine_onevsrest_2_red_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Migraine_onevsrest_3_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Migraine_onevsrest_3_red_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Migraine_onevsrest_4_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Migraine_onevsrest_4_red_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Migraine_onevsrest_5_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Migraine_onevsrest_5_red_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/new-thyroid1_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_new-thyroid1_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/newthyroid2_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_newthyroid2_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_0_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_0_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_1_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_1_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_2_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_2_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_3_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_3_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_4_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_4_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_5_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_5_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/Obesity_onevsrest_6_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_Obesity_onevsrest_6_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/page-blocks-1-3_vs_4_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_page-blocks-1-3_vs_4_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/pima_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_pima_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/transfusion_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_transfusion_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/vowel0_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_vowel0_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/yeast1_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_yeast1_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

py src/syrfd_learning_v7_resnet_variable.py --data ./imbalanced_datasets/min/yeast3_min.csv --rfd ./discovered_rfds_processed/Domino/output_false_0_yeast3_min.txt --oversampling 2000 --epochs 3000 --rfd_mode relaxed

echo.
echo Tutti i dataset sono stati elaborati.
pause