import os
import re
import time
import argparse
import pandas as pd
import torch
# Importiamo il Learner direttamente dal tuo script v7 per mantenere le stesse identiche logiche di denormalizzazione e filtraggio
from syrfd_learning_v7_resnet import SyRFDLearner

def run_grid_generation():
    parser = argparse.ArgumentParser(description="Generazione massiva di dati sintetici dai checkpoint della Grid Search (v7)")
    parser.add_argument("--data",         type=str, default="imbalanced_datasets/min/ecoli-0_vs_1_min.csv", help="Percorso del dataset reale")
    parser.add_argument("--rfd",          type=str, default="discovered_rfds_processed/RFD4_E0.0_ecoli-0_vs_1_min.txt", help="Percorso del file RFD")
    parser.add_argument("--oversampling", type=int, default=2000, help="Numero di tuple valide da generare per ciascun checkpoint")
    parser.add_argument("--rfd_mode",     type=str, default="strict", choices=["strict", "relaxed"], help="Modalità di filtraggio del validatore")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Cartella contenente i file .pt")
    args = parser.parse_args()

    # 1. Inizializzazione del Learner v7 (carica le strutture dati, calcola i range min/max e parsa le RFD)
    print("=========================================================================")
    print("   INITIALIZING SyRFD v7 BASE ENVIRONMENT")
    print("=========================================================================")
    learner = SyRFDLearner(
        data_path=args.data,
        rfd_path=args.rfd,
        latent_dim=32,
        rfd_mode=args.rfd_mode
    )

    # 2. Definizione esplicita della lista dei 18 checkpoint forniti
    checkpoint_files = [
        "ecoli-0_vs_1_min_grid_01_9533.pt",
        "ecoli-0_vs_1_min_grid_02_9112.pt",
        "ecoli-0_vs_1_min_grid_03_122e.pt",
        "ecoli-0_vs_1_min_grid_04_5512.pt",
        "ecoli-0_vs_1_min_grid_05_1ad4.pt",
        "ecoli-0_vs_1_min_grid_06_24c9.pt",
        "ecoli-0_vs_1_min_grid_07_5a0a.pt",
        "ecoli-0_vs_1_min_grid_08_e959.pt",
        "ecoli-0_vs_1_min_grid_09_75a3.pt",
        "ecoli-0_vs_1_min_grid_10_c885.pt",
        "ecoli-0_vs_1_min_grid_11_faad.pt",
        "ecoli-0_vs_1_min_grid_12_fd39.pt",
        "ecoli-0_vs_1_min_grid_13_2932.pt",
        "ecoli-0_vs_1_min_grid_14_5c0c.pt",
        "ecoli-0_vs_1_min_grid_15_845a.pt",
        "ecoli-0_vs_1_min_grid_16_bba9.pt",
        "ecoli-0_vs_1_min_grid_17_7142.pt",
        "ecoli-0_vs_1_min_grid_18_d66e.pt"
    ]

    output_dir = "augmentation_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTrovati {len(checkpoint_files)} checkpoint impostati per la generazione.")
    print(f"Target di campionamento: {args.oversampling} tuple per modello in modalità [{args.rfd_mode.upper()}].\n")

    # 3. Iterazione e generazione su ogni singolo checkpoint
    for idx, ckpt_name in enumerate(checkpoint_files, start=1):
        full_ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        
        print("-------------------------------------------------------------------------")
        print(f"[{idx}/{len(checkpoint_files)}] Processing: {ckpt_name}")
        print("-------------------------------------------------------------------------")
        
        if not os.path.exists(full_ckpt_path):
            print(f"[ERRORE] File di checkpoint non trovato in: {full_ckpt_path}. Salto questo modello.")
            continue

        # Estraiamo l'identificativo della griglia (es. "grid_01_9533") per preservarlo nel nome del file generato
        # Questo ti permetterà di mappare a quale riga della griglia corrisponde il file CSV di output
        grid_match = re.search(r"grid_\d+_[a-f0-9]+", ckpt_name)
        grid_id = grid_match.group(0) if grid_match else f"model_{idx}"

        # Caricamento dei pesi sinaptici nel generatore TabResNet
        try:
            learner.generator.load_state_dict(torch.load(full_ckpt_path, map_location=torch.device('cpu')))
            print(f"[OK] Pesi caricati con successo da {ckpt_name}")
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare il checkpoint {ckpt_name}. Errore: {e}")
            continue

        # Esecuzione del processo di campionamento e filtraggio logico vettoriale (con Hinge thresholds normalizzate)
        start_time = time.time()
        df_gen = learner.generate(num_samples=args.oversampling)
        elapsed = time.time() - start_time
        print(f"[INFO] Generazione completata in {elapsed:.2f} secondi.")

        # Costruzione dinamica del nome del file per l'esportazione
        dataset_name = os.path.splitext(os.path.basename(args.data))[0]
        threshold_int = int(learner.file_threshold) if learner.file_threshold.is_integer() else learner.file_threshold
        
        # Mantiene la struttura coerente inserendo il tag identificativo del checkpoint specifico della griglia
        output_filename = f"{dataset_name}_thr{threshold_int}_{grid_id}_{args.rfd_mode}_resnet.csv"
        full_output_path = os.path.join(output_dir, output_filename)

        # Salvataggio su disco
        df_gen.to_csv(full_output_path, index=False)
        print(f"[SALVATO] Dataset sintetico esportato in: {full_output_path}\n")

    print("=========================================================================")
    print("   GRID GENERATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=========================================================================")

if __name__ == "__main__":
    run_grid_generation()