import re
import os
import sys
import time
import json
import uuid
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import silhouette_score, davies_bouldin_score


def set_seed(seed: int) -> None:
    """Fix random/np.random/torch (+CUDA) seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================================
# 0. ROLLING BUFFER
# =========================================================================

class RollingBuffer:
    def __init__(self, capacity: int = 512, min_fill: int = 128):
        self.capacity = capacity
        self.min_fill = min_fill
        self._buf: list[np.ndarray] = []
        self._ptr: int = 0

    def push(self, x_np: np.ndarray):
        for row in x_np:
            if len(self._buf) < self.capacity:
                self._buf.append(row)
            else:
                self._buf[self._ptr] = row
                self._ptr = (self._ptr + 1) % self.capacity

    def get(self) -> np.ndarray | None:
        if len(self._buf) < self.min_fill:
            return None
        return np.stack(self._buf)

    def reset(self):
        self._buf = []
        self._ptr = 0

# =========================================================================
# 1. LOSS FUNCTIONS
# =========================================================================

class SoftRFDLoss(nn.Module):
    def __init__(self, dependencies, attr_to_idx, threshold=4.0, tau=0.5):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.dependencies = dependencies
        self.attr_to_idx = attr_to_idx

    def forward(self, x_gen, x_real):
        B, D = x_gen.shape
        N, _ = x_real.shape
        if B == 0 or N == 0 or not self.dependencies:
            return torch.tensor(0.0, device=x_gen.device, requires_grad=True)
        
        # Sfrutta il broadcasting ma ottimizzato per non saturare la memoria
        diffs = torch.abs(x_gen.unsqueeze(1) - x_real.unsqueeze(0))
        total, active = 0.0, 0
        for lhs_pairs, (rhs_attr, rhs_thr) in self.dependencies:
            known = [(a, t) for a, t in lhs_pairs if a in self.attr_to_idx]
            if len(known) < len(lhs_pairs) or rhs_attr not in self.attr_to_idx:
                continue  # attributi sconosciuti: dipendenza non valutabile
            rhs_idx = self.attr_to_idx[rhs_attr]
            active += 1
            if known:
                lhs_idx = [self.attr_to_idx[a] for a, _ in known]
                lhs_thr = torch.tensor([t for _, t in known],
                                       device=x_gen.device, dtype=diffs.dtype)
                lhs_sim = torch.sigmoid((lhs_thr - diffs[..., lhs_idx]) / self.tau).prod(-1)
            else:
                # LHS vuoto (solo 'class' in origine): condizione sempre vera
                lhs_sim = torch.ones(B, N, device=x_gen.device, dtype=diffs.dtype)
            rhs_dis = torch.sigmoid((diffs[..., rhs_idx] - rhs_thr) / self.tau)
            total = total + (lhs_sim * rhs_dis).mean()
        if active == 0:
            return torch.tensor(0.0, device=x_gen.device, requires_grad=True)
        return total / active


class MMDLoss(nn.Module):
    def __init__(self, scales=[0.05, 0.2, 0.5, 1.0, 2.0]):
        super().__init__()
        self.scales = scales

    def rbf_kernel(self, x, y):
        xs, ys, dim = x.size(0), y.size(0), x.size(1)
        dist_sq = torch.pow(
            x.unsqueeze(1).expand(xs, ys, dim) - y.unsqueeze(0).expand(xs, ys, dim), 2
        ).sum(2)
        return sum(torch.exp(-dist_sq / (s * dim + 1e-10)) for s in self.scales)

    def forward(self, x_gen, x_real, mins: torch.Tensor | None = None, ranges: torch.Tensor | None = None):
        if mins is not None and ranges is not None:
            x_gen  = torch.clamp((x_gen  - mins) / (ranges + 1e-8), 0.0, 1.0)
            x_real = torch.clamp((x_real - mins) / (ranges + 1e-8), 0.0, 1.0)
        return torch.clamp(
            self.rbf_kernel(x_gen, x_gen).mean()
            - 2.0 * self.rbf_kernel(x_gen, x_real).mean()
            + self.rbf_kernel(x_real, x_real).mean(),
            min=0.0
        )

# =========================================================================
# 2. COMPOSITE METRIC LOSS (v6 - Snella e non ridondante)
# =========================================================================

class CompositeMetricLoss(nn.Module):
    """
    Loss composita v6.1 — Fidelity + Diversity senza conflitti:

    Fidelity:
      - SlicedWasserstein (SWD): sostituisce JS marginale. Differenziabile,
        cattura le code meglio di JS (risolve kurt_diff elevato), non satura.

    Diversity:
      - CoverageLoss: minimizza la distanza media al nearest real neighbor.
        Direttamente correlata alla metrica Recall/Coverage finale.
        SOSTITUISCE _intra_compactness (che minimizzava la varianza
        spingendo verso il mode collapse).

    Logging non-differenziabile (solo numpy, nessun gradiente):
      - Silhouette e DBI sklearn per monitoraggio.
    """
    def __init__(self, n_projections: int = 50, w_swd: float = 1.0, w_coverage: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.n_proj    = n_projections
        self.w_swd     = w_swd
        self.w_cov     = w_coverage
        self.eps       = eps

    def _sliced_wasserstein(self, x_gen: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """
        Sliced Wasserstein Distance — O(n log n), differenziabile.
        Proietta su n_proj direzioni casuali e calcola la media delle WD 1D.
        Cattura le code della distribuzione (risolve kurt_diff elevato).
        """
        D = x_gen.shape[1]
        # Direzioni di proiezione casuali normalizzati
        projs = torch.randn(D, self.n_proj, device=x_gen.device)
        projs = projs / (projs.norm(dim=0, keepdim=True) + self.eps)   # (D, n_proj)

        x_proj = x_gen  @ projs   # (B, n_proj)
        r_proj = x_real @ projs   # (N, n_proj)

        # Sort lungo l'asse dei campioni per ogni proiezione
        x_sorted, _ = x_proj.sort(dim=0)
        r_sorted, _ = r_proj.sort(dim=0)

        # Allineamento dimensionale: se B != N, interpola il più corto
        if x_sorted.shape[0] != r_sorted.shape[0]:
            target = max(x_sorted.shape[0], r_sorted.shape[0])
            if x_sorted.shape[0] < target:
                idx = torch.linspace(0, x_sorted.shape[0] - 1, target, device=x_gen.device).long()
                x_sorted = x_sorted[idx]
            else:
                idx = torch.linspace(0, r_sorted.shape[0] - 1, target, device=x_gen.device).long()
                r_sorted = r_sorted[idx]

        return (x_sorted - r_sorted).pow(2).mean()

    def _coverage_loss(self, x_gen: torch.Tensor, x_real: torch.Tensor, w_div: float = 0.3) -> torch.Tensor:
        """
        Loss di coverage bilanciata: promuove copertura senza collasso al centroide.

        Termine 1 — nn_coverage: minimizza la distanza al nearest real neighbor
          → spinge ogni campione generato vicino a un reale diverso (alta recall).

        Termine 2 — diversity_penalty: penalizza la bassa varianza interna dei generati
          → impedisce il collasso su un unico punto/centroide.
          Costruita come max(0, target_var - var_attuale): attiva solo se la varianza
          scende sotto target_var (= varianza dei reali), quindi non comprime mai
          più del necessario. È l'opposto di _intra_compactness che minimizzava var.

        w_div=0.3: peso leggero sulla diversity — la copertura resta il segnale primario.
        """
        dists = torch.cdist(x_gen, x_real, p=2)                     # (B, N)
        nn_coverage = dists.min(dim=1).values.mean()

        # Varianza target = varianza dei reali; penalizza solo se gen è più compatto
        target_var   = x_real.var(dim=0).mean().detach()             # nessun gradiente sui reali
        current_var  = x_gen.var(dim=0).mean()
        diversity_penalty = torch.clamp(target_var - current_var, min=0.0)

        return nn_coverage + w_div * diversity_penalty

    @staticmethod
    def _numpy_silhouette(gen_np: np.ndarray, real_np: np.ndarray) -> float:
        ng, nr = len(gen_np), len(real_np)
        if ng < 2 or nr < 2: return 0.0
        X, labels = np.vstack([gen_np, real_np]), np.array([0]*ng + [1]*nr)
        try: return float(silhouette_score(X, labels))
        except: return 0.0

    @staticmethod
    def _numpy_dbi(gen_np: np.ndarray, real_np: np.ndarray) -> float:
        ng, nr = len(gen_np), len(real_np)
        if ng < 2 or nr < 2: return 1.0
        X, labels = np.vstack([gen_np, real_np]), np.array([0]*ng + [1]*nr)
        try: return float(davies_bouldin_score(X, labels))
        except: return 1.0

    def forward(self, x_gen_n: torch.Tensor, x_real_n: torch.Tensor, mins: torch.Tensor = None, ranges: torch.Tensor = None, gen_buffer: np.ndarray | None = None, real_buffer: np.ndarray | None = None) -> tuple[torch.Tensor, dict]:
        # x_gen_n e x_real_n sono già in [0,1]: nessuna ri-normalizzazione necessaria.
        # mins e ranges mantenuti come parametri opzionali per compatibilità firma.

        # --- Fidelity: Sliced Wasserstein (sostituisce JS marginale) ---
        swd = self._sliced_wasserstein(x_gen_n, x_real_n)

        # --- Diversity: Coverage Loss con anti-collapse (sostituisce _intra_compactness) ---
        cov_loss = self._coverage_loss(x_gen_n, x_real_n)

        # Tracciamento NumPy (solo log, nessun gradiente inquinato)
        gp = gen_buffer  if gen_buffer  is not None else x_gen_n.detach().cpu().numpy()
        rp = real_buffer if real_buffer is not None else x_real_n.detach().cpu().numpy()
        sil_val = self._numpy_silhouette(gp, rp)
        dbi_raw = self._numpy_dbi(gp, rp)

        loss = self.w_swd * swd + self.w_cov * cov_loss

        metrics = {
            "swd":            swd.item(),
            "coverage_loss":  cov_loss.item(),
            "dbi_sklearn":    dbi_raw,
            "silhouette":     sil_val,
            "loss_composite": loss.item(),
            "buffer_size":    len(gp),
            # Alias per compatibilità con _extract_final_metrics e logging
            "js":             swd.item(),       # SWD prende il ruolo di js nel logging
            "compactness":    cov_loss.item(),  # coverage_loss prende il ruolo di compactness
            "dbi_local":      0.0,              # rimosso, non più calcolato
        }
        return loss, metrics

# =========================================================================
# 3. GENERATORE AVANZATO (TabResNet - Conserva la diversity)
# =========================================================================

class ResidualBlock(nn.Module):
    """
    Blocco residuale con LayerNorm (stabile su batch piccoli) e Dropout leggero.
    LayerNorm normalizza per sample invece che per batch: nessuna instabilita'
    con batch_size piccoli o dataset ridotti come ecoli.
    Dropout=0.1 regolarizza senza soffocare il segnale.
    """
    def __init__(self, dim, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabResNetGenerator(nn.Module):
    """
    Generatore basato su blocchi residuali per dati tabulari.
    num_blocks=3: piu' profondita' per apprendere il manifold RFD-vincolato.
    Nessun sigmoid nell'output: il training usa valori liberi, il sigmoid
    e' applicato solo in generate() prima della denormalizzazione.
    """
    def __init__(self, latent_dim, data_dim, hidden_dim=256, num_blocks=3, dropout: float = 0.1):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.last_layer = nn.Linear(hidden_dim, data_dim)
        # Nessuna attivazione finale: output libero durante training,
        # sigmoid applicato esternamente solo in generate()

    def forward(self, z):
        x = self.first_layer(z)
        for block in self.res_blocks:
            x = block(x)
        return self.last_layer(x)

# =========================================================================
# 4. RFD VALIDATOR (Vettorizzato in NumPy - Velocità e Sicurezza Assoluta)
# =========================================================================

class RFDValidator:
    """
    Garantisce matematicamente e in modo invalicabile la validità delle RFD.
    Ottimizzato tramite vettorizzazione nativa NumPy in fette 3D per evitare cicli iterrows.
    """
    def __init__(self, dependencies, attributes, threshold: float, mode: str = "strict"):
        assert mode in ("strict", "relaxed")
        self.dependencies = dependencies
        self.attributes   = attributes
        self.threshold    = threshold
        self.mode         = mode
        self._generated_matrix = None  # Storage np.ndarray veloce per lo strict storico

    def reset_generated(self):
        self._generated_matrix = None

    def _check_valid_vectorized(self, candidates: np.ndarray, reference: np.ndarray, attr_idx: dict) -> np.ndarray:
        """
        Controlla in parallelo tutti i candidati rispetto alla matrice di riferimento.
        Ritorna un array booleano di size (Len(candidates),) dove True significa 'Valido'.
        """
        num_candidates = candidates.shape[0]
        is_valid = np.ones(num_candidates, dtype=bool)
        
        if num_candidates == 0 or reference.shape[0] == 0 or not self.dependencies:
            return is_valid

        # Calcolo distanze assolute cross-prodotte: (Candidates, Reference, Attributes)
        diffs = np.abs(candidates[:, np.newaxis, :] - reference[np.newaxis, :, :])

        for lhs_pairs, (rhs_attr, rhs_thr) in self.dependencies:
            known = [(a, t) for a, t in lhs_pairs if a in attr_idx]
            rhs_col = attr_idx.get(rhs_attr, -1)
            if len(known) < len(lhs_pairs) or rhs_col == -1:
                continue
            lhs_cols = [attr_idx[a] for a, _ in known]
            lhs_thrs = np.array([t for _, t in known], dtype=float)

            # Condizione di violazione RFD: ogni LHS <= propria soglia E RHS > propria soglia
            # (LHS vuoto -> np.all su asse di larghezza 0 -> sempre True)
            lhs_condition = np.all(diffs[:, :, lhs_cols] <= lhs_thrs, axis=2)
            rhs_condition = diffs[:, :, rhs_col] > rhs_thr
            violation_matrix = lhs_condition & rhs_condition

            # Se un candidato viola la regola con ALMENO una riga del reference, è invalido
            has_violation = np.any(violation_matrix, axis=1)
            is_valid = is_valid & (~has_violation)

        return is_valid

    def filter_valid(self, generated_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
        attr_idx = {a: i for i, a in enumerate(self.attributes)}
        candidates = generated_df[self.attributes].values.astype(float)
        reference  = reference_df[self.attributes].values.astype(float)

        # 1. Validazione rispetto agli originali
        valid_mask = self._check_valid_vectorized(candidates, reference, attr_idx)
        filtered_candidates = candidates[valid_mask]
        original_indices = np.where(valid_mask)[0]

        if self.mode == "strict" and filtered_candidates.shape[0] > 0:
            strict_mask = np.ones(filtered_candidates.shape[0], dtype=bool)
            
            # Se esiste già uno storico di tuple generate valide, controlla l'inter-validità totale
            if self._generated_matrix is not None:
                passed_history = self._check_valid_vectorized(filtered_candidates, self._generated_matrix, attr_idx)
                strict_mask = strict_mask & passed_history

            # Controllo sequenziale interno tra i candidati stessi del batch corrente per preservare l'ordine
            final_accepted = []
            current_history = list(self._generated_matrix) if self._generated_matrix is not None else []
            
            for idx, candidate in enumerate(filtered_candidates):
                if not strict_mask[idx]:
                    continue
                if len(current_history) > 0:
                    history_arr = np.array(current_history)
                    # Controllo singolo candidato vs storia accumulata finora
                    diffs_single = np.abs(history_arr - candidate)
                    violates = False
                    for lhs_pairs, (rhs_attr, rhs_thr) in self.dependencies:
                        known = [(a, t) for a, t in lhs_pairs if a in attr_idx]
                        if len(known) < len(lhs_pairs) or rhs_attr not in attr_idx:
                            continue
                        l_cols = [attr_idx[a] for a, _ in known]
                        l_thrs = np.array([t for _, t in known], dtype=float)
                        r_col = attr_idx[rhs_attr]
                        if np.any(np.all(diffs_single[:, l_cols] <= l_thrs, axis=1) & (diffs_single[:, r_col] > rhs_thr)):
                            violates = True
                            break
                    if violates:
                        strict_mask[idx] = False
                        continue
                
                current_history.append(candidate)
            
            # Aggiorna lo storage dello storico globale del validatore
            if len(current_history) > 0:
                self._generated_matrix = np.array(current_history)
            
            # Rimappa la maschera finale sul DataFrame originale
            final_mask = np.zeros(len(generated_df), dtype=bool)
            final_mask[original_indices[strict_mask]] = True
            return generated_df[final_mask].reset_index(drop=True)
        else:
            return generated_df[valid_mask].reset_index(drop=True)

# =========================================================================
# 5. CHECKPOINT MANAGER
# =========================================================================

class CheckpointManager:
    CKPT_DIR = "checkpoint"

    def __init__(self, base_name: str):
        self.base_name = base_name
        os.makedirs(self.CKPT_DIR, exist_ok=True)
        self.config_csv = os.path.join(self.CKPT_DIR, "configuration_parameters.csv")

    def save(self, generator: nn.Module, run_id: str, train_params: dict, final_metrics: dict) -> str:
        model_path = os.path.join(self.CKPT_DIR, f"{self.base_name}_{run_id}.pt")
        torch.save(generator.state_dict(), model_path)
        print(f"Checkpoint salvato: {model_path}")

        row = {
            "run_id":     run_id,
            "base_name":  self.base_name,
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": os.path.basename(model_path),
            **{f"param_{k}": v for k, v in train_params.items()},
            **{f"metric_{k}": v for k, v in final_metrics.items()},
        }
        df_row = pd.DataFrame([row])
        if os.path.exists(self.config_csv):
            existing = pd.read_csv(self.config_csv)
            updated  = pd.concat([existing, df_row], ignore_index=True)
        else:
            updated = df_row
        updated.to_csv(self.config_csv, index=False)
        return model_path

    def load(self, generator: nn.Module, run_id: str) -> nn.Module:
        model_path = os.path.join(self.CKPT_DIR, f"{self.base_name}_{run_id}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {model_path}")
        generator.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Checkpoint caricato: {model_path}")
        return generator

# =========================================================================
# 6. LEARNER PRINCIPALE
# =========================================================================

class SyRFDLearner:
    def __init__(self, data_path=None, rfd_path=None, threshold=4.0, latent_dim=64, rfd_mode="strict", device=None):
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold  = threshold
        self.latent_dim = latent_dim
        self.rfd_mode   = rfd_mode

        if data_path and rfd_path:
            self._load_real_data(data_path)
            self._load_rfds(rfd_path)
            self.base_name = os.path.splitext(os.path.basename(data_path))[0]
        else:
            self._load_mock_data()
            self.base_name = "mock"

        self.attr_to_idx = {a: i for i, a in enumerate(self.attributes)}
        self.data_dim    = len(self.attributes)

        self.mins   = torch.tensor(self.df_min[self.attributes].min().values, dtype=torch.float32, device=self.device)
        self.maxs   = torch.tensor(self.df_min[self.attributes].max().values, dtype=torch.float32, device=self.device)
        self.ranges = torch.clamp(self.maxs - self.mins, min=1e-5)

        self.real_tensor = torch.tensor(self.df_min[self.attributes].values, dtype=torch.float32, device=self.device)

        # Inizializzazione del nuovo generatore TabResNet
        self.generator = TabResNetGenerator(self.latent_dim, self.data_dim).to(self.device)
        self.validator = RFDValidator(self.dependencies, self.attributes, self.file_threshold, mode=self.rfd_mode)
        self.ckpt_mgr  = CheckpointManager(self.base_name)

    def _load_real_data(self, path):
        print(f"Caricamento dataset: {path}")
        df = pd.read_csv(path)
        counts = df['class'].value_counts()
        self.min_class  = counts.idxmin()
        self.df_min     = df[df['class'] == self.min_class].copy()
        self.attributes = [c for c in self.df_min.columns if c != 'class']

    def _load_rfds(self, path):
        print(f"Caricamento RFD dal file: {path}")
        
        # Estraiamo automaticamente la soglia dal nome del file (es. RFD2 -> soglia = 2.0)
        filename = os.path.basename(path)
        match = re.search(r"RFD(\d+(?:\.\d+)?)", filename)
        if match:
            self.file_threshold = float(match.group(1))
            print(f"[INFO] Soglia epsilon estratta automaticamente dal nome del file: {self.file_threshold}")
        else:
            self.file_threshold = self.threshold # Fallback sulla soglia passata come argomento
            print(f"[ATTENZIONE] Nessuna soglia trovata nel nome file. Uso il default: {self.file_threshold}")
            
        self.dependencies = []
        skipped_class_rhs = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if '->' not in line: continue
                lhs, rhs = line.split('->')
                # Ogni token e' 'Attr@soglia': la soglia e' per-attributo.
                # Se manca, fallback sulla soglia globale del file.
                lhs_pairs = []
                for tok in re.split('[, ]+', lhs):
                    m = re.match(r"(\w+)(?:@([0-9.]+))?", tok.strip())
                    if not m: continue
                    attr, thr = m.group(1), m.group(2)
                    thr = float(thr) if thr is not None else self.file_threshold
                    if attr == 'class':
                        # entro la partizione minoritaria la classe e' costante:
                        # la condizione LHS su class e' sempre vera -> si omette
                        continue
                    lhs_pairs.append((attr, thr))
                m = re.match(r"(\w+)(?:@([0-9.]+))?", rhs.strip())
                if not m: continue
                rhs_attr, rhs_thr = m.group(1), m.group(2)
                rhs_thr = float(rhs_thr) if rhs_thr is not None else self.file_threshold
                if rhs_attr == 'class':
                    # RHS su class: mai violabile entro la minoranza (diff = 0)
                    skipped_class_rhs += 1
                    continue
                self.dependencies.append((lhs_pairs, (rhs_attr, rhs_thr)))
        if skipped_class_rhs:
            print(f"[INFO] {skipped_class_rhs} RFD con RHS 'class' ignorate (mai violabili nella minoranza)")
        print(f"[INFO] {len(self.dependencies)} RFD caricate con soglie per-attributo")

    def _load_mock_data(self):
        self.min_class  = 1
        self.attributes = [f"Attr{i}" for i in range(1, 11)]
        data = {a: np.random.uniform(10, 100, 100) for a in self.attributes}
        data['class'] = self.min_class
        self.df_min = pd.DataFrame(data)
        self.dependencies = [(['Attr1', 'Attr2'], 'Attr3')]

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalizzazione con clamp: usata in generate() dopo sigmoid."""
        return torch.clamp(x_norm, 0.0, 1.0) * self.ranges + self.mins


    def train(self, epochs: int = 2000, batch_size: int = 64, lr: float = 0.001, lambda_fid: float = 1.0, lambda_rfd: float = 10.0, lambda_composite: float = 1.0, w_swd: float = 1.0, w_coverage: float = 0.5, n_projections: int = 50, tau: float = 0.5, warmup_epochs: int = 400, grad_clip: float = 1.0, buffer_capacity: int = 512, buffer_min_fill: int = 128, patience: int = 300, min_delta: float = 1e-4, log_every: int = 200, save_checkpoint: bool = True, run_id: str | None = None) -> tuple[list, dict]:
        if run_id is None:
            run_id = uuid.uuid4().hex[:8]

        train_params = dict(epochs=epochs, batch_size=batch_size, lr=lr, lambda_fid=lambda_fid, lambda_rfd=lambda_rfd, lambda_composite=lambda_composite, w_swd=w_swd, w_coverage=w_coverage, n_projections=n_projections, tau=tau, warmup_epochs=warmup_epochs, grad_clip=grad_clip, buffer_capacity=buffer_capacity, buffer_min_fill=buffer_min_fill, patience=patience, rfd_mode=self.rfd_mode)

        optimizer = optim.Adam(self.generator.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(warmup_epochs, 200), T_mult=2, eta_min=lr * 0.01)

        rfd_crit  = SoftRFDLoss(self.dependencies, self.attr_to_idx, threshold=self.file_threshold, tau=tau)
        fid_crit  = MMDLoss()
        comp_crit = CompositeMetricLoss(n_projections=n_projections, w_swd=w_swd, w_coverage=w_coverage)

        gen_buf  = RollingBuffer(capacity=buffer_capacity, min_fill=buffer_min_fill)
        real_buf_np = torch.clamp((self.real_tensor - self.mins) / (self.ranges + 1e-8), 0.0, 1.0).cpu().numpy()

        print(f"\nInizio Addestramento TabResNet — run_id={run_id}")
        
        best_comp, patience_ctr = float('inf'), 0
        history = []
        self.generator.train()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            z          = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Sigmoid applicato UNA SOLA VOLTA: output in [0,1] normalizzato.
            # x_gen_norm viene passato direttamente alle loss che lavorano già
            # in spazio normalizzato → nessuna ri-normalizzazione interna,
            # nessun double-sigmoid, nessun collasso da output libero non vincolato.
            # x_gen_den (denormalizzato) è usato solo per RFDLoss e generazione finale.
            x_gen_raw  = self.generator(z)
            x_gen_norm = torch.sigmoid(x_gen_raw)                    # [0,1], una volta sola
            x_gen_den  = self._denormalize(x_gen_norm)               # scala su [min, max]

            idx          = torch.randint(0, len(self.real_tensor), (batch_size,), device=self.device)
            x_real_batch = self.real_tensor[idx]
            # Versione normalizzata dei reali per le loss composite (già in [0,1])
            x_real_norm  = torch.clamp((x_real_batch - self.mins) / (self.ranges + 1e-8), 0.0, 1.0)

            gen_buf.push(x_gen_norm.detach().cpu().numpy())

            # MMDLoss riceve i denormalizzati + mins/ranges e ri-normalizza internamente: ok
            loss_fid = fid_crit(x_gen_den, x_real_batch, mins=self.mins, ranges=self.ranges)
            # RFDLoss lavora sui valori denormalizzati (confronto con soglia epsilon)
            loss_rfd = rfd_crit(x_gen_den, self.real_tensor)

            if epoch <= warmup_epochs:
                loss_total = lambda_fid * loss_fid + lambda_rfd * loss_rfd
                metrics = {k: 0.0 for k in ["swd", "coverage_loss", "js", "compactness", "dbi_local", "dbi_sklearn", "silhouette", "loss_composite"]}
                metrics["buffer_size"] = len(gen_buf._buf)
            else:
                buf_np = gen_buf.get()
                loss_comp, metrics = comp_crit(x_gen_norm, x_real_norm, gen_buffer=buf_np, real_buffer=real_buf_np)
                loss_total = (lambda_fid * loss_fid + lambda_rfd * loss_rfd + lambda_composite * loss_comp)

            loss_total.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            history.append({
                "epoch": epoch, "phase": 1 if epoch <= warmup_epochs else 2,
                **metrics, "mmd": loss_fid.item(), "rfd": loss_rfd.item(), "loss_total": loss_total.item(),
            })

            if epoch % log_every == 0 or epoch == 1 or epoch == warmup_epochs + 1:
                tag = "WU" if epoch <= warmup_epochs else "P2"
                print(f"[{tag}] Ep {epoch:5d} | Tot:{loss_total.item():.4f} | MMD:{loss_fid.item():.4f} | RFD:{loss_rfd.item():.4f} | SWD:{metrics['swd']:.4f} | CovLoss:{metrics['coverage_loss']:.4f} | Sil:{metrics['silhouette']:.4f}")

            if epoch > warmup_epochs and gen_buf.get() is not None:
                cv = metrics['loss_composite']
                if cv < best_comp - min_delta:
                    best_comp, patience_ctr = cv, 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        print(f"\nEarly stopping ep {epoch}")
                        break

        final_metrics = self._extract_final_metrics(history)
        if save_checkpoint:
            self.ckpt_mgr.save(self.generator, run_id, train_params, final_metrics)
        return history, final_metrics

    def _extract_final_metrics(self, history: list) -> dict:
        phase2 = [h for h in history if h.get("phase", 2) == 2]
        last   = phase2[-1] if phase2 else history[-1]
        return {
            "swd":            last.get("swd", 0.0),
            "coverage_loss":  last.get("coverage_loss", 0.0),
            "js":             last.get("js", 0.0),          # alias swd
            "compactness":    last.get("compactness", 0.0), # alias coverage_loss
            "dbi_local":      last.get("dbi_local", 0.0),
            "dbi_sklearn":    last.get("dbi_sklearn", 0.0),
            "silhouette":     last.get("silhouette", 0.0),
            "loss_composite": last.get("loss_composite", 0.0),
            "loss_total":     last["loss_total"],
            "final_epoch":    last["epoch"],
        }

    def generate(self, num_samples: int = 100, rfd_max_attempts: int = 10) -> pd.DataFrame:
        self.generator.eval()
        self.validator.reset_generated()
        valid_rows, attempt = [], 0
        total_generated_so_far = 0

        while total_generated_so_far < num_samples and attempt < rfd_max_attempts:
            attempt += 1
            # Campionamento dinamico: genera più campioni se l'accettazione è bassa
            bs = max(num_samples * 3, 512)
            with torch.no_grad():
                x_gen_den = self._denormalize(torch.sigmoid(self.generator(torch.randn(bs, self.latent_dim, device=self.device))))
            
            df_gen = pd.DataFrame(x_gen_den.cpu().numpy(), columns=self.attributes)
            for attr in self.attributes:
                if (self.df_min[attr].dtype in ['int64','int32'] or np.issubdtype(self.df_min[attr].dtype, np.integer)):
                    df_gen[attr] = df_gen[attr].round().astype(int)

            # Il Validator filtrerà istantaneamente via NumPy escludendo qualsiasi violazione
            df_valid = self.validator.filter_valid(df_gen, self.df_min)
            if len(df_valid) > 0:
                valid_rows.append(df_valid)
                total_generated_so_far = sum(len(d) for d in valid_rows)
            
            print(f"  Tentativo {attempt}: {len(df_valid)}/{bs} tuple valide superate. Totale: {total_generated_so_far}/{num_samples}")

        if not valid_rows:
            return pd.DataFrame(columns=self.attributes + ['class'])
            
        result_df = pd.concat(valid_rows, ignore_index=True)
        if len(result_df) >= num_samples:
            result_df = result_df.head(num_samples)
        result_df['class'] = self.min_class
        return result_df

    def load_checkpoint(self, run_id: str):
        self.ckpt_mgr.load(self.generator, run_id)

# =========================================================================
# 7. MAIN E INVOCAZIONE
# =========================================================================

def run_gridsearch(data_path, rfd_path, rfd_mode="relaxed", epochs=1000, warmup_epochs=300):
    """
    Esegue una Grid Search sistematica esplorando le combinazioni di iperparametri
    e salvando i risultati nel file CSV dei checkpoint per il confronto.
    """
    # Definiamo lo spazio di ricerca (puoi modificarlo a piacimento)
    grid = {
        "lr": [0.001],
        "lambda_fid": [1.0],               # Peso MMD (Fidelity di base)
        "lambda_rfd": [5.0, 10.0, 20.0],    # Peso Vincoli RFD (Soglia logica)
        "lambda_composite": [1.0, 2.0],    # Peso Geometria + Diversity
        "w_swd": [1.0],                     # Peso Sliced Wasserstein Distance
        "w_coverage": [0.2, 0.5, 1.0]      # Peso Coverage Loss (Diversity)
    }

    # Genera tutte le combinazioni possibili
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_runs = len(experiments)
    print(f"\n========================================================")
    
    # Estraiamo temporaneamente la soglia per stamparla
    filename = os.path.basename(rfd_path)
    match = re.search(r"RFD(\d+(?:\.\d+)?)", filename)
    file_eps = float(match.group(1)) if match else 4.0
    
    print(f"AVVIO GRID SEARCH: {total_runs} combinazioni totali.")
    print(f"Dataset: {data_path} | RFD: {rfd_path} (Epsilon={file_eps})")
    print(f"========================================================\n")

    for i, params in enumerate(experiments, 1):
        print(f"\n---> [Esperimento {i}/{total_runs}] Parametri attuali:")
        print(f"     lambda_rfd: {params['lambda_rfd']} | lambda_composite: {params['lambda_composite']} | w_coverage: {params['w_coverage']}")
        
        # Inizializziamo un learner fresco ad ogni iterazione per resettare la rete
        learner = SyRFDLearner(
            data_path=data_path, 
            rfd_path=rfd_path, 
            rfd_mode=rfd_mode
        )
        
        # Generiamo un run_id parlante o univoco per distinguerlo nel CSV
        run_id = f"grid_{i:02d}_{uuid.uuid4().hex[:4]}"
        
        try:
            # Eseguiamo il training passando i parametri della combinazione corrente
            _, final_metrics = learner.train(
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                batch_size=64,
                lr=params["lr"],
                lambda_fid=params["lambda_fid"],
                lambda_rfd=params["lambda_rfd"],
                lambda_composite=params["lambda_composite"],
                w_swd=params["w_swd"],
                w_coverage=params["w_coverage"],
                run_id=run_id,
                log_every=500,     # Meno log a schermo per non intasare il terminale
                save_checkpoint=True
            )
            
            print(f"     Risultati -> SWD (Fidelity): {final_metrics['swd']:.4f} | CovLoss (Diversity): {final_metrics['coverage_loss']:.4f} | Silhouette: {final_metrics['silhouette']:.4f}")
            
        except Exception as e:
            print(f"     [ERRORE] L'esperimento {i} è fallito: {e}")
            continue

    print(f"\n[OK] Grid Search Completata! Controlla 'checkpoint/configuration_parameters.csv' per analizzare i risultati.")

def main():
    parser = argparse.ArgumentParser(description="SyRFD v6 — TabResNet + Vectorized Validations")
    parser.add_argument("--data",         type=str,   default=None, required=True)
    parser.add_argument("--rfd",          type=str,   default=None, required=True)
    parser.add_argument("--oversampling", type=int,   default=100)
    parser.add_argument("--epochs",       type=int,   default=2000)
    parser.add_argument("--rfd_mode",     type=str,   default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--generate_only",type=str,   default=None)

    # NUOVO ARGOMENTO PER ABILITARE LA GRID SEARCH
    parser.add_argument("--gridsearch",   action="store_true", help="Abilita la ricerca a griglia sugli iperparametri")
    parser.add_argument("--seed",         type=int,   default=42, help="Seed per random/numpy/torch (riproducibilita')")

    args = parser.parse_args()
    set_seed(args.seed)

    # Se l'utente richiede la grid search, esegui il loop ed esci
    if args.gridsearch:
        run_gridsearch(
            data_path=args.data, 
            rfd_path=args.rfd, 
            rfd_mode=args.rfd_mode, 
            epochs=args.epochs,        # Es. 1000 o 2000
            warmup_epochs=int(args.epochs * 0.2)  # 20% di warmup proporzionale
        )
        return

    # --- Flusso Standard di Singolo Train (se --gridsearch non è attivo) ---
    # Nota: la threshold reale viene estratta e sovrascritta internamente da _load_rfds
    learner = SyRFDLearner(data_path=args.data, rfd_path=args.rfd, threshold=4.0, latent_dim=64, rfd_mode=args.rfd_mode)

    if args.generate_only:
        learner.load_checkpoint(args.generate_only)
    else:
        # Warmup proporzionale: 20% delle epoche totali, minimo 300.
        # Evita che con --epochs bassi il warmup occupi l'intera sessione.
        warmup = max(300, int(args.epochs * 0.20))
        learner.train(epochs=args.epochs, log_every=200, warmup_epochs=warmup)

    print(f"\nGenerazione finale di {args.oversampling} tuple...")
    df_gen = learner.generate(num_samples=args.oversampling)
    print("\nAnteprima dati generati:")
    print(df_gen.head())

    # --- LOGICA DI COSTRUZIONE DEL NOME FILE COMPORTAMENTALE ---
    # 1. Estrae il nome base del dataset (es. kddcup-guess_passwd_vs_satan_min)
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    
    # 2. Recupera la vera soglia utilizzata nel file (salvata nel learner dopo _load_rfds)
    # Se per qualsiasi motivo non è definita, prova ad estrarla al volo dal nome del file RFD
    file_threshold = getattr(learner, 'file_threshold', None)
    if file_threshold is None:
        filename_rfd = os.path.basename(args.rfd)
        match = re.search(r"RFD(\d+(?:\.\d+)?)", filename_rfd)
        file_threshold = float(match.group(1)) if match else 4.0

    # Converte in intero se non ha decimali significativi (es. 2.0 -> 2)
    threshold_int = int(file_threshold) if file_threshold.is_integer() else file_threshold

    # 3. Genera il timestamp in millisecondi
    timestamp_ms = int(time.time() * 1000)

    # 4. Compone il nome del file definitivo secondo le specifiche
    output_filename = f"{dataset_name}_thr{threshold_int}_{args.rfd_mode}_{args.oversampling}_resnet_{timestamp_ms}.csv"
    
    # Assicura che la cartella augmentation_results esista
    output_dir = "augmentation_results"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)

    # 5. Salvataggio finale
    df_gen.to_csv(full_output_path, index=False)
    print(f"\n[SALVATO] Dati generati salvati con successo in: {full_output_path}")
    
if __name__ == "__main__":
    main()