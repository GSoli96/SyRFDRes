"""
Dataset Augmentation with Advanced CoT Strategies + Ollama Cloud
==============================================================
Implemented strategies:
  - Explicit Schema Constraints
  - Distribution Guidance
  - Hierarchical CoT (Multi-level)          [3]
  - CoT with Verification (Self-Consistency) [4]
  - CoT with ReAct Pattern                  [5]
  - CoT Ensemble (Multi-Perspective)        [6]
  - CoT with Decision Tree                  [7]
  - CoT with Confidence Scoring             [8]

- Sequential execution (no threading)
- Auto-restart on hang via SIGALRM
- Checkpointing for automatic resume
- Only the minority class is augmented
"""

import os
import re
import sys
import json
import time
import signal
import logging
import hashlib
from io import StringIO
from enum import Enum
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from ollama import Client

# Global configuration
API_KEY = "<insert_ollama_api_key>"

MODELS    = ["gemma3:12b"]
PRIMARY_MODEL  = MODELS[0]
#FALLBACK_MODEL = MODELS[1]

LABEL_COLUMN   = "class"
N_FEW_SHOT     = 5
N_GEN_AT_TIME  = 10

DATASET_DIR    = "./imbalanced_datasets/"   # input dataset folder
OUTPUT_DIR     = "./LLM_raw/"               # raw LLM responses per strategy
AUGMENTED_DIR  = "./augmented_datasets/"    # final augmented datasets
CHECKPOINT_DIR = "./checkpoints/"
LOG_FILE       = "augmentation.log"

# Retry
MAX_RETRIES    = 3
RETRY_DELAY    = 5      # seconds between retries
# Timeout for a single LLM call (SIGALRM, Linux/macOS)
REQUEST_TIMEOUT = 120

# Logging
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Strategy definitions
class AugStrategy(str, Enum):
    SCHEMA_CONSTRAINTS   = "schema_constraints"
    DISTRIBUTION_GUIDE   = "distribution_guidance"
    COT_HIERARCHICAL     = "cot_hierarchical"
    COT_SELF_CONSISTENCY = "cot_self_consistency"
    COT_REACT            = "cot_react"
    COT_ENSEMBLE         = "cot_ensemble"
    COT_DECISION_TREE    = "cot_decision_tree"
    COT_CONFIDENCE       = "cot_confidence"

# Ollama Cloud client
ollama_client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + API_KEY},
)


def ollama_generate(prompt: str, model: str = PRIMARY_MODEL, temperature: float = 0.7) -> str:
    """
    Calls Ollama Cloud sequentially.
    - Tries PRIMARY_MODEL, then FALLBACK_MODEL.
    - Retries MAX_RETRIES times with backoff.
    - Uses SIGALRM for single-call timeout (Linux/macOS).
    """
    models_to_try = [model]# if model != FALLBACK_MODEL else [model]

    for attempt in range(1, MAX_RETRIES + 1):
        for mdl in models_to_try:
            try:
                log.info(f"  [LLM] model={mdl}  attempt={attempt}/{MAX_RETRIES}")

                # timeout via SIGALRM
                if hasattr(signal, "SIGALRM"):
                    def _handler(signum, frame):
                        raise TimeoutError(f"Timeout of {REQUEST_TIMEOUT}s exceeded")
                    signal.signal(signal.SIGALRM, _handler)
                    signal.alarm(REQUEST_TIMEOUT)

                #print(prompt)
                response = ollama_client.chat(
                    model=mdl,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={"temperature": temperature},
                )

                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

                content = response.message.content
                if content and content.strip():
                    return content.strip()

                log.warning(f"  [LLM] empty response from {mdl}")

            except TimeoutError as e:
                log.error(f"  [LLM] {e.with_traceback}")
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
            except Exception as e:
                log.error(f"  [LLM] error ({mdl}): {e}")
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

        # backoff before the next attempt
        if attempt < MAX_RETRIES:
            wait = RETRY_DELAY * attempt
            log.info(f"  [LLM] waiting {wait}s...")
            time.sleep(wait)

    log.critical("  [LLM] all attempts failed")
    return ""

# Dataset analysis (pandas)
class DatasetProfile:
    """Full statistical profile, computed once per dataset."""

    def __init__(self, df: pd.DataFrame):
        self.n_rows = len(df)
        self.n_cols = len(df.columns)
        self.columns = df.columns.tolist()

        self.label_distribution = df[LABEL_COLUMN].value_counts().to_dict()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categ_cols   = [c for c in df.select_dtypes(exclude=[np.number]).columns
                        if c != LABEL_COLUMN]

        # statistiche numeriche
        self.numeric_stats: dict = {}
        for c in numeric_cols:
            s = df[c].dropna()
            self.numeric_stats[c] = dict(
                min=float(s.min()), max=float(s.max()),
                mean=round(float(s.mean()), 4), std=round(float(s.std()), 4),
                q25=float(s.quantile(0.25)), q50=float(s.quantile(0.50)),
                q75=float(s.quantile(0.75)),
            )

        # statistiche categoriche
        self.categorical_stats: dict = {}
        for c in categ_cols:
            vc = df[c].value_counts()
            self.categorical_stats[c] = dict(
                n_unique=int(df[c].nunique()),
                top_values=vc.head(10).to_dict(),
            )

        # text ready for the prompts
        sc_lines = []
        for c in numeric_cols:
            st = self.numeric_stats[c]
            sc_lines.append(
                f"  - '{c}' (numeric): range [{st['min']}, {st['max']}], "
                f"mean={st['mean']}, std={st['std']}"
            )
        for c in categ_cols:
            top = list(self.categorical_stats[c]["top_values"].keys())[:5]
            sc_lines.append(f"  - '{c}' (categorical): allowed values -> {top}")
        self.schema_constraints = "SCHEMA CONSTRAINTS:\n" + "\n".join(sc_lines)

        dg_lines = []
        for c in numeric_cols:
            st = self.numeric_stats[c]
            if len(numeric_cols) < 15:
                dg_lines.append(f"  - '{c}': Q25={st['q25']}, median={st['q50']}, Q75={st['q75']}")
        self.distribution_summary = "DISTRIBUTION GUIDANCE:\n" + "\n".join(dg_lines)

    def log_analysis(self, minority_class: str):
        log.info("-" * 60)
        log.info(f"DATASET ANALYSIS  ({self.n_rows} rows x {self.n_cols} columns)")
        log.info(f"Class distribution : {self.label_distribution}")
        log.info(f"Minority class     : '{minority_class}'")
        for c, st in self.numeric_stats.items():
            log.info(f"  {c}: min={st['min']} max={st['max']} "
                     f"mean={st['mean']} std={st['std']}")
        for c, st in self.categorical_stats.items():
            log.info(f"  {c}: {st['n_unique']} unique, "
                     f"top={list(st['top_values'].keys())[:3]}")
        log.info("-" * 60)

# Checkpoint utilities
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _sanitize_model_name(model: str) -> str:
    return re.sub(r"[:/]", "_", model)


def get_strategy_output_path(model: str, dataset_name: str, strategy: AugStrategy) -> str:
    model_dir = os.path.join(AUGMENTED_DIR, _sanitize_model_name(model))
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{dataset_name}_{strategy.value}.csv")


def _ckpt_path(dataset_name: str, model: str, strategy: str, batch_idx: int) -> str:
    key = hashlib.md5(f"{dataset_name}_{model}_{strategy}_{batch_idx}".encode()).hexdigest()[:10]
    return os.path.join(CHECKPOINT_DIR, f"{key}.json")


def save_checkpoint(dataset_name: str, model: str, strategy: str, batch_idx: int, raw: str):
    with open(_ckpt_path(dataset_name, model, strategy, batch_idx), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "model": model, "strategy": strategy,
                   "batch": batch_idx, "ts": datetime.now().isoformat(),
                   "response": raw}, f)


def load_checkpoint(dataset_name: str, model: str, strategy: str, batch_idx: int) -> Optional[str]:
    p = _ckpt_path(dataset_name, model, strategy, batch_idx)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["response"]
    return None

# CSV parser
def parse_csv_from_response(response: str, expected_cols: list) -> pd.DataFrame:
    """Extracts a CSV DataFrame from the LLM response, tolerant of formatting."""
    clean = re.sub(r"```[a-z]*\n?", "", response).strip()
    lines = clean.splitlines()

    # find the header row
    start = 0
    for i, ln in enumerate(lines):
        if any(c in ln for c in expected_cols):
            start = i
            break

    csv_text = "\n".join(lines[start:])
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(StringIO(csv_text), sep=sep)
            #print(df.head())

            res = len(set(df.columns) & set(expected_cols)) >= max(1, len(expected_cols) // 2)
            #print("\n (set(df.columns)", set(df.columns),  res, "\n")
            #if len(set(df.columns) & set(expected_cols)) >= max(1, len(expected_cols) // 2):
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            continue
    return pd.DataFrame()

# Prompt builder
def _header(profile: DatasetProfile, examples_csv: str, n: int, minority_class: str) -> str:
    return (
        f"You are an expert in data augmentation for tabular datasets.\n"
        f"You must generate exactly {n} new synthetic rows\n"
        f"belonging ONLY to the minority class: '{minority_class}'.\n\n"
        f"{profile.schema_constraints}\n\n"
        f"{profile.distribution_summary}\n\n"
        f"Real examples of the minority class (few-shot):\n{examples_csv}\n"
    )


def _footer(cols: list) -> str:
    return (
        f"\nRespond ONLY with a valid CSV (comma-separated).\n"
        f"Exact header on the first row: {','.join(cols)}\n"
        f"No additional text, no markdown block, no empty lines.\n"
    )


# 1. Schema Constraints
def prompt_schema_constraints(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Strictly respect the schema constraints:
- No numeric value outside the indicated ranges.
- Only allowed values for categorical columns.
- No NULL or missing fields.
- Generate exactly {n} rows.
""" + _footer(p.columns)


# 2. Distribution Guidance
def prompt_distribution_guidance(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Follow the distribution guidance to produce realistic data:
- Numeric values must reflect the observed distribution (Q25-Q75).
- Introduce realistic variability: do not copy the examples identically.
- Preserve the correlations between columns present in the examples.
Generate {n} varied but statistically coherent rows.
""" + _footer(p.columns)


# 3. Hierarchical CoT
def prompt_cot_hierarchical(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Follow these reasoning levels BEFORE writing the CSV:

LEVEL 1 - STRUCTURE
  Which columns are numeric? Which are categorical?
  Are there obvious correlations between columns in the examples?

LEVEL 2 - DISTRIBUTION
  For each numeric column, what is the plausible range?
  For categorical columns, what are the dominant values?

LEVEL 3 - INTERNAL CONSISTENCY
  Do columns influence each other? (e.g. age<->income, hours<->output)
  The new rows must respect these relationships.

LEVEL 4 - GENERATION
  Now generate the {n} examples applying all constraints identified above.
""" + _footer(p.columns)


# 4. CoT Self-Consistency
def prompt_cot_self_consistency(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Use this self-verification process before producing the final CSV:

STEP 1 - First draft
  Generate {n} synthetic rows respecting schema and distribution.

STEP 2 - Statistical check
  Are the means and variances of the generated rows consistent with the examples?
  If not, correct the anomalous values.

STEP 3 - Semantic check
  Does each row make sense? Are there impossible or contradictory combinations?
  If so, fix them.

STEP 4 - Final output
  Write the definitive CSV only after applying the corrections.
""" + _footer(p.columns)


# 5. CoT ReAct
def prompt_cot_react(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Apply the ReAct pattern (Reasoning + Acting):

[Thought 1] What are the key statistical characteristics of class '{mc}'?
[Action 1]  Compute the mean and range for each numeric column from the examples.
[Obs 1]     Note the values.

[Thought 2] Are there important categorical constraints?
[Action 2]  List the allowed values for each categorical column.
[Obs 2]     Note the constraints.

[Thought 3] How do the columns correlate with each other?
[Action 3]  Identify the 2-3 strongest correlations from the examples.
[Obs 3]     Note the correlations.

[Thought 4] Generate {n} rows respecting everything above.
[Action 4]  Write the final CSV.
""" + _footer(p.columns)


# 6. CoT Ensemble
_PERSPECTIVES = [
    ("STATISTICAL", "Priority: statistical fidelity - mean, std and distribution must match."),
    ("SEMANTIC",    "Priority: semantic plausibility - every row must be realistic within the domain."),
    ("DIVERSITY",   "Priority: diversity - cover edge cases, rare cases, extreme but plausible values."),
]

def prompt_cot_ensemble(p, ex, n, mc, perspective: int = 0, **_):
    name, focus = _PERSPECTIVES[perspective % len(_PERSPECTIVES)]
    return _header(p, ex, n, mc) + f"""
Active perspective: {name}
{focus}

Reason from this perspective:
1. What must you guarantee as a priority from this perspective?
2. Which columns require more attention?
3. Generate {n} rows optimized according to this perspective.
""" + _footer(p.columns)


# 7. CoT Decision Tree
def prompt_cot_decision_tree(p, ex, n, mc, **_):
    pivot = list(p.numeric_stats.keys())[0] if p.numeric_stats else "main_feature"
    return _header(p, ex, n, mc) + f"""
Mentally build a decision tree and use it to generate:

ROOT -> class = '{mc}'

NODE [{pivot}]
  Branch A: {pivot} < median
    -> which other features tend to be low? which categorical values dominate?
  Branch B: {pivot} >= median
    -> which other features tend to be high? which categorical values dominate?

LEAVES -> generate examples following the paths.
          Distribute evenly between Branch A and Branch B to maximize variety.

Generate {n} rows in total (balanced between the branches).
""" + _footer(p.columns)


# 8. CoT Confidence
def prompt_cot_confidence(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
For each row, internally apply this confidence filter:

  Score = 0.4 x (numeric range respected)
        + 0.3 x (categorical consistency)
        + 0.3 x (semantic plausibility)

  If score < 0.7 -> discard and regenerate the row.
  Include in the final CSV ONLY rows with score >= 0.7.

Generate at least {n} valid rows (the number of discarded rows does not count).
The score column must NOT appear in the CSV - it is only an internal filter.
""" + _footer(p.columns)


# Dispatcher
_PROMPT_FN = {
    AugStrategy.SCHEMA_CONSTRAINTS:   prompt_schema_constraints,
    AugStrategy.DISTRIBUTION_GUIDE:   prompt_distribution_guidance,
    AugStrategy.COT_HIERARCHICAL:     prompt_cot_hierarchical,
    AugStrategy.COT_SELF_CONSISTENCY: prompt_cot_self_consistency,
    AugStrategy.COT_REACT:            prompt_cot_react,
    AugStrategy.COT_ENSEMBLE:         prompt_cot_ensemble,
    AugStrategy.COT_DECISION_TREE:    prompt_cot_decision_tree,
    AugStrategy.COT_CONFIDENCE:       prompt_cot_confidence,
}


def build_prompt(strategy: AugStrategy, profile: DatasetProfile,
                 examples_csv: str, n: int,
                 minority_class: str, perspective: int = 0) -> str:
    return _PROMPT_FN[strategy](
        p=profile, ex=examples_csv, n=n,
        mc=minority_class, perspective=perspective,
    )

# Batch generation (sequential)
def generate_batch(
    dataset_name: str,
    few_shot: pd.DataFrame,
    profile: DatasetProfile,
    minority_class: str,
    batch_idx: int,
    strategy: AugStrategy,
    model: str = PRIMARY_MODEL,
    n: int = N_GEN_AT_TIME,
    perspective: int = 0,
) -> pd.DataFrame:
    """Generates a batch of n rows with the given strategy, sequentially."""

    # use cache if it exists
    cached = load_checkpoint(dataset_name, model, strategy.value, batch_idx)
    if cached:
        log.info(f"    [cache] batch {batch_idx} already present")
        raw = cached
    else:
        examples_csv = few_shot.to_csv(index=False)
        prompt = build_prompt(strategy, profile, examples_csv, n,
                              minority_class, perspective)

        log.info(f"    [LLM] model={model}  strategy={strategy.value}  batch={batch_idx}  n={n}")
        raw = ollama_generate(prompt, model=model)

        if not raw:
            log.warning(f"    [LLM] empty response - batch {batch_idx} skipped")
            return pd.DataFrame()

        # save the raw response
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = f"{dataset_name}_{_sanitize_model_name(model)}_{strategy.value}_{batch_idx}_{batch_idx + n}.txt"
        with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(raw)
        save_checkpoint(dataset_name, model, strategy.value, batch_idx, raw)

    #print("raw\n\n", raw)
    #print("profile.columns\n\n", profile.columns)
    df_new = parse_csv_from_response(raw, profile.columns)
    if df_new.empty:
        log.warning(f"    [parse] could not extract a CSV from batch {batch_idx}")
    else:
        # force the correct class on all generated rows
        df_new[LABEL_COLUMN] = minority_class
        log.info(f"    [parse] {len(df_new)} rows extracted")
    return df_new

# Dataset augmentation
def augment_dataset(file_path: str, strategies: list, model: str) -> None:
    """
    Reads the CSV, augments ONLY the minority class until it matches
    the majority class. For each strategy saves a separate file in
    AUGMENTED_DIR/{model}/{dataset}_{strategy}.csv, updated after every batch.
    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    log.info(f"\n{'=' * 60}")
    log.info(f"Dataset: {dataset_name}  |  Model: {model}")

    df = pd.read_csv(file_path)

    if LABEL_COLUMN not in df.columns:
        log.error(f"  Column '{LABEL_COLUMN}' not found - skip.")
        return

    profile = DatasetProfile(df)
    counts  = df[LABEL_COLUMN].value_counts()

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    diff = int(counts[majority_class] - counts[minority_class])

    profile.log_analysis(minority_class)
    log.info(f"  To generate (diff): {diff} rows for '{minority_class}'")

    if diff <= 0:
        log.info("  Dataset already balanced - skip.")
        return

    few_shot = df[df[LABEL_COLUMN] == minority_class].sample(
        min(N_FEW_SHOT, int(counts[minority_class])), random_state=42
    )

    # loop over strategies (sequential)
    for strategy in strategies:
        out_path = get_strategy_output_path(model, dataset_name, strategy)
        log.info(f"\n  -- Strategy: {strategy.value}  |  Model: {model} --")

        # resume: count rows already generated for this strategy
        if os.path.exists(out_path):
            existing_aug = pd.read_csv(out_path)
            already_generated = max(0, len(existing_aug) - len(df))
            generated_rows = existing_aug.iloc[len(df):].reset_index(drop=True)
            if already_generated >= diff:
                log.info(f"    Strategy already completed ({already_generated} rows) - skip")
                continue
            #continue #added by stefano
            log.info(f"    Resume: {already_generated} rows already present")
        else:
            already_generated = 0
            generated_rows = pd.DataFrame()

        perspective = 0
        generated_count = already_generated

        for batch_start in range(already_generated, diff + N_GEN_AT_TIME, N_GEN_AT_TIME):
            if generated_count >= diff:
                break

            batch_df = generate_batch(
                dataset_name, few_shot, profile, minority_class,
                batch_start, strategy, model,
                n=N_GEN_AT_TIME, perspective=perspective,
            )

            if not batch_df.empty:
                generated_rows = pd.concat([generated_rows, batch_df], ignore_index=True)
                generated_count += len(batch_df)

                # save incrementally after every batch with data
                rows_to_save = generated_rows.head(diff)
                aug_df = pd.concat([df, rows_to_save], ignore_index=True)
                aug_df.to_csv(out_path, index=False)
                new_counts = aug_df[LABEL_COLUMN].value_counts().to_dict()
                log.info(f"    [save] {out_path} - {len(aug_df)} rows  |  dist: {new_counts}")

            if strategy == AugStrategy.COT_ENSEMBLE:
                perspective = (perspective + 1) % len(_PERSPECTIVES)

        log.info(f"  Strategy {strategy.value}: {generated_count} rows generated in total.")

# Main
def main():
    for d in [OUTPUT_DIR, AUGMENTED_DIR, CHECKPOINT_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok=True)

    csv_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".csv") and ("Migraine_onevsrest_5" in f)
    ])


    if not csv_files:
        log.warning(f"No CSV found in '{DATASET_DIR}'.")
        return

    log.info(f"Files found ({len(csv_files)}): {csv_files}")

    selected_strategies = [
        AugStrategy.SCHEMA_CONSTRAINTS,
        AugStrategy.DISTRIBUTION_GUIDE,
        AugStrategy.COT_HIERARCHICAL,
        AugStrategy.COT_SELF_CONSISTENCY,
        AugStrategy.COT_REACT,
        AugStrategy.COT_ENSEMBLE,
        AugStrategy.COT_DECISION_TREE,
        AugStrategy.COT_CONFIDENCE,
    ]

    for model in MODELS:
        log.info(f"\n{'#' * 60}")
        log.info(f"MODEL: {model}")
        for fname in tqdm(csv_files, desc=f"Dataset ({model})"):
            file_path = os.path.join(DATASET_DIR, fname)

            # retry with auto-restart on error
            for attempt in range(1, MAX_RETRIES + 1):
                log.info(f"\n[MAIN] {fname} | {model} - attempt {attempt}/{MAX_RETRIES}")
                try:
                    augment_dataset(file_path, selected_strategies, model)
                    break  # success -> next file
                except Exception as e:
                    log.error(f"[MAIN] error during {fname} ({model}): {e}")
                    if attempt < MAX_RETRIES:
                        wait = RETRY_DELAY * attempt
                        log.info(f"[MAIN] restarting in {wait}s...")
                        time.sleep(wait)
                    else:
                        log.critical(f"[MAIN] {fname} ({model}) failed after {MAX_RETRIES} attempts")

    log.info("\nPipeline completed.")


if __name__ == "__main__":
    main()