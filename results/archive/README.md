# Archived results (pre-restructure)

These files are kept as a record of the development process. **Do not feed them
into the Stage 5 Pareto analysis without re-running them first.**

## Why they are archived rather than migrated

They were produced before the `RunRecord` schema and the seeding fix, so they
carry two limitations that cannot be repaired after the fact:

1. **Incomplete parameter record.** Each file recorded only the parameters its
   script happened to list. `eps_retrieval`, `delta`, `alpha`, `omega` and
   `corpus_seed` were in play but never written down. Their values are
   recoverable from the `experiment_params.py` of the matching commit, but that
   is reconstruction, not measurement.
2. **Unseeded retrieval.** `pup_retrieve` drew its documents from the global RNG,
   so these runs cannot be reproduced exactly, and runs cannot be compared
   document-for-document against each other.

## What is here

| File | Produced by | Notes |
|---|---|---|
| `stage1_consistency_10000x200.json` | Stage 1.2, 200 queries, Llama-3.1-8B | Source of the ~87% greedy consistency rate |
| `stage2_temperature_sweep_50q.json` | Stage 2.3, 50 queries | 5 temperatures incl. T=1.0 |
| `smoke_10000x20_dp.json` / `_nodp.json` | Stage 1.1a/b | DP vs non-DP quality comparison |
| `evaluation.json`, `5000_evaluation.json` | Upstream sarus-tech/dp-rag | Synthetic-data results, unrelated to this project |

## Plan

Stage 1.2 and Stage 2.3 are to be re-run under the new schema once seeding is in
place, so the headline numbers are backed by reproducible runs. These archived
files stay as the untouched original record.
