# LLM Prompting Strategies — Verbatim Templates

This document reports the verbatim prompt templates used to query the LLM baselines
(Devstral-Small-24B, Gemma-3-12B, Gemma-4-31B, GPT-OSS-20B, GPT-OSS-120B, and the
zero-shot DeepSeek3/Llama3 baselines). Each template is built by concatenating a
shared **header**, a **strategy-specific body**, and a shared **footer**.

The executable source of truth for these templates is
[`src/llm_generation_advanced2.py`](../src/llm_generation_advanced2.py) (functions
`prompt_*`); this file is a human-readable transcription for reviewers who want to
read the exact wording without running the code.

Placeholders: `{n}` = number of rows requested in the batch, `{minority_class}` /
`{mc}` = label of the minority class, `{profile.schema_constraints}` /
`{profile.distribution_summary}` = per-attribute ranges/admissible values and
per-attribute distribution statistics (min/max/quartiles) computed on the real
minority partition, `{examples_csv}` = five real minority tuples (few-shot
examples), `{cols}` = the exact output column header.

## Shared header (prepended to every strategy)

```
You are an expert in data augmentation for tabular datasets.
You must generate exactly {n} new synthetic rows
belonging ONLY to the minority class: '{minority_class}'.

{profile.schema_constraints}

{profile.distribution_summary}

Real examples of the minority class (few-shot):
{examples_csv}
```

## Shared footer (appended to every strategy)

```
Respond ONLY with a valid CSV (comma-separated).
Exact header on the first row: {cols}
No additional text, no markdown block, no empty lines.
```

---

## 1. Schema Constraints

```
Strictly respect the schema constraints:
- No numeric value outside the indicated ranges.
- Only allowed values for categorical columns.
- No NULL or missing fields.
- Generate exactly {n} rows.
```

## 2. Distribution Guidance

```
Follow the distribution guidance to produce realistic data:
- Numeric values must reflect the observed distribution (Q25-Q75).
- Introduce realistic variability: do not copy the examples identically.
- Preserve the correlations between columns present in the examples.
Generate {n} varied but statistically coherent rows.
```

## 3. Hierarchical Chain-of-Thought

```
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
```

## 4. Self-Consistency Chain-of-Thought

```
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
```

## 5. ReAct Chain-of-Thought

```
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
```

## 6. Ensemble Chain-of-Thought (multi-perspective)

Rotates across three perspectives (statistical fidelity, semantic plausibility,
diversity), one per generation round:

| Perspective | Focus |
|---|---|
| STATISTICAL | Priority: statistical fidelity - mean, std and distribution must match. |
| SEMANTIC    | Priority: semantic plausibility - every row must be realistic within the domain. |
| DIVERSITY   | Priority: diversity - cover edge cases, rare cases, extreme but plausible values. |

```
Active perspective: {perspective_name}
{perspective_focus}

Reason from this perspective:
1. What must you guarantee as a priority from this perspective?
2. Which columns require more attention?
3. Generate {n} rows optimized according to this perspective.
```

## 7. Decision-Tree Chain-of-Thought

```
Mentally build a decision tree and use it to generate:

ROOT -> class = '{mc}'

NODE [{pivot_attribute}]
  Branch A: {pivot_attribute} < median
    -> which other features tend to be low? which categorical values dominate?
  Branch B: {pivot_attribute} >= median
    -> which other features tend to be high? which categorical values dominate?

LEAVES -> generate examples following the paths.
          Distribute evenly between Branch A and Branch B to maximize variety.

Generate {n} rows in total (balanced between the branches).
```

`{pivot_attribute}` is the first numeric attribute in the dataset schema.

## 8. Confidence-Scoring Chain-of-Thought

```
For each row, internally apply this confidence filter:

  Score = 0.4 x (numeric range respected)
        + 0.3 x (categorical consistency)
        + 0.3 x (semantic plausibility)

  If score < 0.7 -> discard and regenerate the row.
  Include in the final CSV ONLY rows with score >= 0.7.

Generate at least {n} valid rows (the number of discarded rows does not count).
The score column must NOT appear in the CSV - it is only an internal filter.
```

---

## Zero-shot baselines (DeepSeek3, Llama3)

These two models receive only the shared header and footer above (schema,
distribution summary, few-shot examples, output format instructions), with no
strategy-specific reasoning body — i.e., the plain zero-shot setting described in
the paper.
