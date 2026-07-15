import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- CONFIGURAZIONE --------------------------------
INPUT_DIR = "./evaluation_metrics/RFDcompliance"
OUT_DIR = "./evaluation_metrics/aggregated"
EXCLUDE_LEGACY_SYRFD = True

# metriche da visualizzare (una heatmap per ciascuna, per ciascuna soglia)
METRICS = ["RFD_tuple_compliance_relaxed", "RFD_tuple_compliance_strict"]

# righe = configurazioni (Label), colonne = dataset; se per una stessa
# (Label, Dataset, RFD_threshold) esistono piu' righe (es. run ripetute) si
# aggregano con questa statistica
CELL_STAT = "mean"


df = pd.read_csv("./evaluation_metrics/aggregated/aggregated_rfdcompliance_full_mean.csv")
label = df['Label']
print(set(label))
df_filtered = df[["Label","RFD_threshold","RFD_tuple_compliance_relaxed_mean", "RFD_tuple_compliance_strict_mean"]]
df_filtered_2 = df_filtered[df_filtered['RFD_threshold'] == 2]
df_filtered_4 = df_filtered[df_filtered['RFD_threshold'] == 4]
df_filtered_8 = df_filtered[df_filtered['RFD_threshold'] == 8]

df_filtered_2.to_csv("tmp2.csv",index=False)
df_filtered_4.to_csv("tmp4.csv",index=False)
df_filtered_8.to_csv("tmp8.csv",index=False)