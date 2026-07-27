"""Experiment setup, assembled once instead of in every experiment.

Every experiment opened with the same fourteen lines: build a PUPVectorStoreConfig
from six fields, build a DPRAGEngine, loop the corpus through `add()`, remember to
call `embeddings()` to force the embedding pass, and time it. Two consequences
followed. Changing one setting meant editing every experiment -- adding the seed
would have been a five-file change. And experiments that never generate (the
pre-filter ones) still had to hand DPRAGEngine a DPGenerationConfig it would not
use, paying for a PLD binary search to satisfy a constructor.

`Bench.build(cfg)` takes an ExperimentConfig and hands back a ready engine, so an
experiment states what it wants measured rather than how to assemble the parts.

`build()` also applies `cfg.seed` to both sources of randomness -- retrieval, via
PUPVectorStoreConfig, and generation sampling, via torch -- so an experiment
cannot half-seed a run by wiring one and forgetting the other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .chatdoctor import Query, load_corpus, load_queries
from .config import ExperimentConfig
from .dp_model import DPGenerationConfig
from .dp_rag_engine import DPRAGEngine
from .pup_vector_store import PUPVectorStoreConfig


@dataclass
class Bench:
    """A built engine plus the config and corpus it was built from."""

    config: ExperimentConfig
    engine: DPRAGEngine
    corpus: list[str]
    embed_seconds: float

    @property
    def dp_model(self):
        return self.engine.dp_model

    def queries(self, n: int | None = None) -> list[Query]:
        """The query sample this config specifies."""
        return load_queries(
            n=self.config.n_queries if n is None else n,
            seed=self.config.query_seed,
        )

    def retrieve(self, query: str) -> list[str]:
        return self.engine.pup_retrieve(query)

    def epsilon(self, delta: float | None = None) -> float:
        """End-to-end ε: retrieval PLD composed with generation PLD."""
        d = self.config.delta if delta is None else delta
        return self.engine.privacy_loss_distribution.get_epsilon_for_delta(d)

    @classmethod
    def build(
        cls,
        config: ExperimentConfig | None = None,
        *,
        differential_privacy: bool = True,
        verbose: bool = True,
    ) -> "Bench":
        """Build the engine and embed the corpus.

        differential_privacy=False gives the non-private baseline (no clipping,
        no noise) used to tell "the output is garbled because of DP" apart from
        "the output is garbled because something is broken".
        """
        cfg = config or ExperimentConfig()

        # Generation sampling reads torch's global RNG, so seed it here rather
        # than leaving each experiment to remember.
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        engine = DPRAGEngine(
            pup_vector_store_config=PUPVectorStoreConfig(
                model_id=cfg.embed_model,
                top_p=cfg.retrieval_top_p,
                top_k=cfg.retrieval_top_k,
                min_score=cfg.min_score,
                max_score=cfg.max_score,
                epsilon=cfg.eps_retrieval,
                max_retrieve=cfg.max_retrieve,
                batch_size=cfg.embed_batch_size,
                differential_pivacy=differential_privacy,
                seed=cfg.seed,
            ),
            model_id=cfg.gen_model,
            dp_generation_config=DPGenerationConfig(
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                alpha=cfg.alpha,
                omega=cfg.omega,
                epsilon=cfg.gen_epsilon,
                delta=cfg.delta,
                differential_pivacy=differential_privacy,
            ),
        )

        if verbose:
            print(f"Embedding {cfg.n_docs} corpus docs ...")
        started = time.time()
        corpus = load_corpus(limit=cfg.n_docs, sample_seed=cfg.corpus_seed)
        for doc in corpus:
            engine.add(doc)
        # Force the embedding pass now, so later timings measure generation only.
        engine.pup_vector_store.embeddings()
        elapsed = time.time() - started
        if verbose:
            print(f"Embedded {len(corpus)} docs in {elapsed:.1f}s")

        return cls(config=cfg, engine=engine, corpus=corpus, embed_seconds=elapsed)
