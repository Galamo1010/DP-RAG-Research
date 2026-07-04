"""Single source of truth for all experiment parameters.

Finalized 2026-07-02 for the project "基於 Top-K Logit 相似度之差分隱私生成改良研究".
Cross-referenced against dp_model.py / pup_vector_store.py and the proposal (專題計畫書).

Design rule for the DP generation side:
    clipping (= exponential-mechanism sensitivity Δ) is DERIVED, never set by hand:
        Δ = token_epsilon * temperature / 2          (proposal Eq4 == dp_model.py:113)
        token_epsilon = binary-search s.t. PLD-compose over max_new_tokens == generation budget
    So the only free DP knobs are: EPS_TOTAL, TEMPERATURE, MAX_NEW_TOKENS, DELTA.
"""

# ---------------------------------------------------------------------------
# Differential-privacy budget
# ---------------------------------------------------------------------------
DELTA = 1e-3                       # δ, matches dp_model.py default & proposal
EPS_RETRIEVAL = 0.2                # ε_retrieval, Grislain setting (PUP default is 0.1)
EPS_TOTAL_GRID = [5, 10, 20, 40]   # cross-model control variable (proposal)

# ε_generation_budget is reverse-solved from EPS_TOTAL via two-layer PLD (Eq3):
#   PLD_compose(EPS_RETRIEVAL, gen_budget; T(q), DELTA) == eps_total
# Precomputed token_epsilon / clipping at MAX_NEW_TOKENS=128, TEMPERATURE=1.0
# (retrieval layer included; see experiment_params computation on 2026-07-02):
TOKEN_EPSILON_BY_TOTAL = {5: 0.12770, 10: 0.21920, 20: 0.36440, 40: 0.58940}
CLIPPING_BY_TOTAL       = {5: 0.06385, 10: 0.10960, 20: 0.18220, 40: 0.29470}

# ---------------------------------------------------------------------------
# Generation config (dp_model.py -> DPGenerationConfig)
# ---------------------------------------------------------------------------
TEMPERATURE = 1.0                  # fixed (no Stage-2 scan); makes Δ = token_epsilon/2
MAX_NEW_TOKENS = 128               # generation cap; ε accounting still uses actual T(q)
ALPHA = 1.0                        # exp-mechanism score concentration (test/main value)
OMEGA = 0.01                       # public-prior weight (test/main value)

# ---------------------------------------------------------------------------
# Retrieval config (pup_vector_store.py -> PUPVectorStoreConfig)
# ---------------------------------------------------------------------------
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"
RETRIEVAL_TOP_P = 0.02             # matches main() examples; set RETRIEVAL_TOP_K instead if switching
RETRIEVAL_TOP_K = None
MIN_SCORE = -0.5
MAX_SCORE = 0.8
MAX_RETRIEVE = 128
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Pre-filter strategies (this project's contribution)
# ---------------------------------------------------------------------------
# Strategy A: argmax(logit_RAG) == argmax(logit_NoRAG)  -> skip ε
# Strategy B: Jaccard(top-k S_RAG, S_NoRAG) >= tau       -> skip ε
STRATEGY_B_K_GRID = [10, 20, 50]
STRATEGY_B_TAU_GRID = [0.5, 0.7, 0.9, 1.0]   # 3 x 4 = 12 configs

# ---------------------------------------------------------------------------
# Models & data
# ---------------------------------------------------------------------------
MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-2-9b-it",
    "qwen/qwen-2.5-14b-instruct",
]
QUERY_SET = "iCliniq-10k"          # 200 sampled patient questions (chatdoctor_data.load_queries)
CORPUS = "HealthCareMagic-100k"    # doctor replies -> FAISS private corpus (chatdoctor_data.load_corpus)
N_QUERIES = 200
QUERY_SEED = 42                    # fixed seed for reproducible 200-query sample
# On-disk data (../ChatDoctor-main), downloaded 2026-07-02 from ChatDoctor README:
#   HealthCareMagic-100k.json : 112,165 rows -> 110,513 unique "output" corpus docs
#   iCliniq-10k.json          :   7,321 rows; "input" = query, "answer_icliniq" = reference
