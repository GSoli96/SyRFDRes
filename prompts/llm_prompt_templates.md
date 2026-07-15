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
Sei un esperto di data augmentation per dataset tabulari.
Devi generare esattamente {n} nuove righe sintetiche
appartenenti SOLO alla classe minoritaria: '{minority_class}'.

{profile.schema_constraints}

{profile.distribution_summary}

Esempi reali della classe minoritaria (few-shot):
{examples_csv}
```

## Shared footer (appended to every strategy)

```
Rispondi SOLO con un CSV valido (separatore virgola).
Intestazione esatta prima riga: {cols}
Nessun testo aggiuntivo, nessun blocco markdown, nessuna riga vuota.
```

---

## 1. Schema Constraints

```
Rispetta RIGOROSAMENTE i vincoli di schema:
- Nessun valore numerico fuori dai range indicati.
- Solo valori ammessi per le colonne categoriche.
- Nessun campo NULL o mancante.
- Genera esattamente {n} righe.
```

## 2. Distribution Guidance

```
Segui la distribution guidance per produrre dati realistici:
- I valori numerici devono riflettere la distribuzione osservata (Q25-Q75).
- Introduci variabilita' realistica: non copiare gli esempi identici.
- Mantieni le correlazioni tra colonne presenti negli esempi.
Genera {n} righe variegate ma statisticamente coerenti.
```

## 3. Hierarchical Chain-of-Thought

```
Segui questi livelli di ragionamento PRIMA di scrivere il CSV:

LIVELLO 1 - STRUTTURA
  Quali colonne sono numeriche? Quali categoriche?
  Ci sono correlazioni evidenti tra le colonne negli esempi?

LIVELLO 2 - DISTRIBUZIONE
  Per ogni colonna numerica, qual e' l'intervallo plausibile?
  Per le categoriche, quali sono i valori dominanti?

LIVELLO 3 - COERENZA INTERNA
  Le colonne si influenzano a vicenda? (es. eta'<->reddito, ore<->produzione)
  Le nuove righe devono rispettare queste relazioni.

LIVELLO 4 - GENERAZIONE
  Ora genera i {n} esempi applicando tutti i vincoli identificati sopra.
```

## 4. Self-Consistency Chain-of-Thought

```
Usa questo processo di auto-verifica prima di produrre il CSV finale:

PASSO 1 - Prima bozza
  Genera {n} righe sintetiche rispettando schema e distribuzione.

PASSO 2 - Verifica statistica
  Le medie e varianze delle righe generate sono coerenti con gli esempi?
  Se no, correggi i valori anomali.

PASSO 3 - Verifica semantica
  Ogni riga ha senso? Ci sono combinazioni impossibili o contraddittorie?
  Se si, correggile.

PASSO 4 - Output finale
  Scrivi il CSV definitivo solo dopo aver applicato le correzioni.
```

## 5. ReAct Chain-of-Thought

```
Applica il pattern ReAct (Reasoning + Acting):

[Thought 1] Quali caratteristiche statistiche chiave ha la classe '{mc}'?
[Action 1]  Calcola media e range per ogni colonna numerica dagli esempi.
[Obs 1]     Annota i valori.

[Thought 2] Ci sono vincoli categorici importanti?
[Action 2]  Elenca i valori ammessi per ogni colonna categorica.
[Obs 2]     Annota i vincoli.

[Thought 3] Come si correlano le colonne tra loro?
[Action 3]  Identifica le 2-3 correlazioni piu' forti dagli esempi.
[Obs 3]     Annota le correlazioni.

[Thought 4] Genero {n} righe rispettando tutto quanto sopra.
[Action 4]  Scrivi il CSV finale.
```

## 6. Ensemble Chain-of-Thought (multi-perspective)

Rotates across three perspectives (statistical fidelity, semantic plausibility,
diversity), one per generation round:

| Perspective | Focus |
|---|---|
| STATISTICA  | Priorita': fedelta' statistica - media, std e distribuzione devono corrispondere. |
| SEMANTICA   | Priorita': plausibilita' semantica - ogni riga deve essere realistica nel dominio. |
| DIVERSITA'  | Priorita': diversita' - copri bordi, casi rari, valori estremi ma plausibili. |

```
Prospettiva attiva: {perspective_name}
{perspective_focus}

Ragiona da questa prospettiva:
1. Cosa devi garantire in modo prioritario con questa prospettiva?
2. Quali colonne richiedono piu' attenzione?
3. Genera {n} righe ottimizzate secondo questa prospettiva.
```

## 7. Decision-Tree Chain-of-Thought

```
Costruisci mentalmente un albero decisionale e usalo per generare:

RADICE -> classe = '{mc}'

NODO [{pivot_attribute}]
  Ramo A: {pivot_attribute} < mediana
    -> quali altre feature tendono ad essere basse? quali valori categorici dominano?
  Ramo B: {pivot_attribute} >= mediana
    -> quali altre feature tendono ad essere alte? quali valori categorici dominano?

FOGLIE -> genera esempi seguendo i percorsi.
         Distribuisci equamente tra Ramo A e Ramo B per massimizzare la varieta'.

Genera {n} righe in totale (bilanciato tra i rami).
```

`{pivot_attribute}` is the first numeric attribute in the dataset schema.

## 8. Confidence-Scoring Chain-of-Thought

```
Per ogni riga, applica internamente questo filtro di confidenza:

  Score = 0.4 x (range numerico rispettato)
        + 0.3 x (coerenza categorica)
        + 0.3 x (plausibilita' semantica)

  Se score < 0.7 -> scarta e rigenera la riga.
  Includi nel CSV finale SOLO righe con score >= 0.7.

Genera almeno {n} righe valide (il numero di righe scartate non conta).
La colonna score NON deve apparire nel CSV - e' solo un filtro interno.
```

---

## Zero-shot baselines (DeepSeek3, Llama3)

These two models receive only the shared header and footer above (schema,
distribution summary, few-shot examples, output format instructions), with no
strategy-specific reasoning body — i.e., the plain zero-shot setting described in
the paper.
