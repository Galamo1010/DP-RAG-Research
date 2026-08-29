"""Single source of truth for experiment parameters.

Finalized 2026-07-02 for the project "基於 Top-K Logit 相似度之差分隱私生成改良研究".
Cross-referenced against dp_model.py / pup_vector_store.py and the proposal (專題計畫書).

Design rule for the DP generation side:
    clipping (= exponential-mechanism sensitivity Δ) is DERIVED, never set by hand:
        Δ = token_epsilon * temperature / 2          (proposal Eq4 == dp_model.py:113)
        token_epsilon = binary-search s.t. PLD-compose over max_new_tokens == generation budget
    So the only free DP knobs are: gen_epsilon, temperature, max_new_tokens, delta.

WHY A DATACLASS RATHER THAN MODULE CONSTANTS
--------------------------------------------
This file used to be a flat namespace of constants that every experiment then
overrode with its own locals. It claimed MAX_RETRIEVE=128 while all four
experiments used 10, and shipped a lowercase OpenRouter model id that cannot be
loaded locally -- neither value was ever used by anything. Worse, each experiment
hand-listed which parameters to record, so eps_retrieval / alpha / omega /
corpus_seed never reached the result files at all.

An ExperimentConfig instance fixes both problems: an experiment states only what
it changes, and `to_dict()` puts *every* parameter into the run record, so a
number can always be traced back to the settings that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

# ---------------------------------------------------------------------------
# Experiment-design constants: these describe the grid that runs sweep over,
# rather than the settings of any one run, so they are not part of the record.
# ---------------------------------------------------------------------------
EPS_TOTAL_GRID = [5, 10, 20, 40]   # cross-model control variable (proposal)

# ε_generation_budget is reverse-solved from eps_total via two-layer PLD (Eq3):
#   PLD_compose(eps_retrieval, gen_budget; T(q), delta) == eps_total
# Precomputed token_epsilon / clipping at max_new_tokens=128, temperature=1.0
# (retrieval layer included; computed 2026-07-02):
TOKEN_EPSILON_BY_TOTAL = {5: 0.12770, 10: 0.21920, 20: 0.36440, 40: 0.58940}
CLIPPING_BY_TOTAL      = {5: 0.06385, 10: 0.10960, 20: 0.18220, 40: 0.29470}

# Pre-filter strategies (this project's contribution), swept in Stage 3.2
# Strategy A: argmax(logit_RAG) == argmax(logit_NoRAG)  -> skip ε
# Strategy B: Jaccard(top-k S_RAG, S_NoRAG) >= tau       -> skip ε
STRATEGY_B_K_GRID = [10, 20, 50]
STRATEGY_B_TAU_GRID = [0.5, 0.7, 0.9, 1.0]   # 3 x 4 = 12 configs

# Cross-model comparison set. Every model listed here must support a `system`
# chat role, because dprag.prompts deliberately emits one format and does not
# branch per model.
#
# The third slot has moved twice. The proposal names Gemma-2-9B-IT; ADR 0001
# dropped it because Gemma 2's template rejects `system`; ADR 0003 named
# Mistral-7B-Instruct-v0.3 on an unverified claim; ADR 0004 settles on Gemma 4,
# whose template gives `system` its own turn -- verified against the repository's
# chat_template.jinja before adoption. Gemma 4 also brings the proposal's own
# model family back, so the report explains a version change rather than a
# substitution.
#
# NOTE the vocabulary spread, which matters to Strategy B: k tokens are a share
# of the vocabulary, and that share is what sets the candidate pool's relative
# width. 128,256 / 152,064 / 262,144 spans 2.0x. Mistral's 32,768 would have made
# it 4.6x. See ADR 0004.
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",   # Meta,    8.03B, vocab 128,256
    "Qwen/Qwen2.5-14B-Instruct",          # Alibaba, 14.8B, vocab 152,064
    "google/gemma-4-12B-it",              # Google,  ~12B,  vocab 262,144  (ADR 0004)
]

# On-disk data (located by dprag.paths), downloaded 2026-07-02 from the ChatDoctor
# README:
#   HealthCareMagic-100k.json : 112,165 rows -> 110,513 unique "output" corpus docs
#   iCliniq-10k.json          :   7,321 rows; "input" = query, "answer_icliniq" = reference
CORPUS = "HealthCareMagic-100k"    # doctor replies -> private corpus
QUERY_SET = "iCliniq-10k"          # patient questions -> queries + reference answers


@dataclass(frozen=True)
class ExperimentConfig:
    """Every parameter a run depends on; recorded verbatim into its RunRecord."""

    # -- Differential-privacy budget ----------------------------------------
    delta: float = 1e-3            # δ, matches dp_model default & proposal
    eps_retrieval: float = 0.2     # ε_retrieval, Grislain setting (PUP default is 0.1)
    gen_epsilon: float = 10.0      # generation budget per full answer

    # -- Generation (dp_model.DPGenerationConfig) ---------------------------
    temperature: float = 1.0       # fixed; makes Δ = token_epsilon/2
    max_new_tokens: int = 128      # cap; ε accounting still uses actual T(q)
    alpha: float = 1.0             # exp-mechanism score concentration
    omega: float = 0.01            # public-prior weight
    gen_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # -- Retrieval (pup_vector_store.PUPVectorStoreConfig) ------------------
    embed_model: str = "Snowflake/snowflake-arctic-embed-m-v1.5"
    retrieval_top_p: float | None = 0.02   # set retrieval_top_k instead if switching
    retrieval_top_k: int | None = None
    min_score: float = -0.5
    max_score: float = 0.8
    # 40, raised from 10 (ADR 0008). The DP threshold admits ~83 documents for the
    # median query and `pup_retrieve` then draws max_retrieve of them AT RANDOM, so
    # a cap of 10 discarded 88% of the qualifying evidence and delivered only 1.5
    # of the true top-10. At 40 that becomes ~4.9. Not higher: measured peak VRAM
    # is ~0.71 GB per document, so 69 -- the OOM ceiling -- sits at 93% of an 80 GB
    # A100 with no room for a longer prompt, while 40 lands near 70%.
    max_retrieve: int = 40
    embed_batch_size: int = 32

    # -- Quality scoring (Stage 3.2) ----------------------------------------
    # A BERTScore number means nothing without the model that produced it: the
    # same pair of texts scores differently under a different encoder, so this
    # belongs in the record for the same reason vocab_min_count does. The value
    # is what the BERTScore authors recommend over the package default
    # (roberta-large) for correlation with human judgement; it costs nothing
    # extra here because scoring runs offline, after generation.
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    # Raw BERTScore sits in a narrow band -- two unrelated English medical
    # passages still score above 0.8 -- so rescaling against the package's
    # random-pairing baseline is what makes differences between configurations
    # readable. Rescaled and raw scores are not comparable with each other.
    bertscore_rescale: bool = True

    # -- Medical entity detection (Stage 2.5) -------------------------------
    # A word counts as clinical only if it occurs at least this often in the
    # corpus, which is what separates a real term from a DP-noise artifact
    # ("Totosis" occurs 0 times). It belongs here rather than in the experiment
    # because the reported medical rates move with it: at 1 a one-off typo is
    # admitted, at 10 real conditions start dropping out ("orchitis" occurs 7).
    vocab_min_count: int = 3

    # -- Data sampling ------------------------------------------------------
    n_docs: int = 10000            # corpus docs embedded into the store
    n_queries: int = 200           # proposal requires >= 200 for stable rates
    corpus_seed: int = 7           # reproducible corpus sample
    query_seed: int = 42           # reproducible query sample

    # -- Reproducibility ----------------------------------------------------
    # Seeds retrieval (exponential-mechanism threshold + max_retrieve truncation)
    # AND generation sampling, so a run can be reproduced, or extended one config
    # at a time instead of re-running a whole sweep.
    # This is for experimental reproducibility ONLY. The DP guarantee is a
    # property of the mechanism's output distribution, not of any single draw, so
    # a fixed-seed output must never be described as "DP-protected".
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Full parameter snapshot, for the run record."""
        return asdict(self)

    def with_(self, **overrides: Any) -> "ExperimentConfig":
        """A copy with some fields changed (e.g. sweeping temperature)."""
        return replace(self, **overrides)
