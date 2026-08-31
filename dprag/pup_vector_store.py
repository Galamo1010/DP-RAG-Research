from functools import cached_property
import hashlib
import random
from typing import Any
from dataclasses import dataclass
from termcolor import colored, cprint
import numpy as np
import torch
from torch import Tensor
# import huggingface_hub
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput
# DP accounting
from dp_accounting.pld.privacy_loss_distribution import from_privacy_parameters, identity
from dp_accounting.pld.common import DifferentialPrivacyParameters
# NOTE: synthetic-data helpers are imported inside main() below -- they pull in
# faker + datasets (~4.6s) and only the demo needs them.

class PUPVectorStoreConfig:
    def __init__(self, model_id: str = "Snowflake/snowflake-arctic-embed-m-v1.5", top_k: int | None = None, top_p: float | None = None, top_p_alpha: float = 5.0, min_score: float = -0.5, max_score: float = 0.8,  epsilon: float = 0.1, max_retrieve: int = 128, differential_pivacy: bool = True, batch_size: int = 32, seed: int | None = None):
        """
        alpha: the concentration of scores around top scores
        pi: the cumulated share of weight to select
        max_score: a level above wich the weight saturates
        batch_size: how many documents to embed per forward pass (caps peak memory)
        seed: makes retrieval reproducible. None keeps the original global-RNG
            behaviour. See PUPVectorStore for why this does not affect the DP
            guarantee.
        """
        self.model_id = model_id
        self.top_k = top_k
        self.top_p = top_p
        self.top_p_alpha = top_p_alpha
        self.min_score = min_score
        self.max_score = max_score
        self.epsilon = epsilon
        self.max_retrieve = max_retrieve
        self.differential_pivacy = differential_pivacy
        self.batch_size = batch_size
        self.seed = seed

class PUPVectorStore:
    def __init__(self, config: PUPVectorStoreConfig):
        """You can use models from https://sbert.net/ or https://huggingface.co/spaces/mteb/leaderboard
Possible choices are:
- Snowflake/snowflake-arctic-embed-m-v1.5
- sentence-transformers/multi-qa-MiniLM-L6-dot-v1
- sentence-transformers/all-MiniLM-L12-v1
- sentence-transformers/all-mpnet-base-v2

Retrieval draws twice from a random source: the exponential-mechanism threshold
and the truncation to max_retrieve. With `config.seed` set, both draws come from
a store-local generator, so a run can be reproduced or a single configuration
re-run without redrawing everyone else's documents.

That is a statement about experiments, not about privacy. The DP guarantee is a
property of the mechanism's output distribution over its randomness; seeding
does not change that distribution, but it does mean a seeded output must never
be presented as a DP-protected release. See docs/adr/0002-seeded-retrieval.md.
        """
        self.model_id = self.model_id = config.model_id
        self.store = []
        self.index = dict()
        self._embeddings = None
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.top_p_alpha = config.top_p_alpha
        self.min_score = config.min_score
        self.max_score = config.max_score
        self.epsilon = config.epsilon
        self.max_retrieve = config.max_retrieve
        self.privacy_loss_distribution = from_privacy_parameters(DifferentialPrivacyParameters(epsilon=self.epsilon))
        self.differential_pivacy = config.differential_pivacy
        self.batch_size = config.batch_size
        self.seed = config.seed
        # Seeded: store-local generators. Unseeded: the global RNGs, preserving
        # the original behaviour exactly.
        self._np_rng = np.random.default_rng(config.seed) if config.seed is not None else np.random
        self._py_rng = random.Random(config.seed) if config.seed is not None else random

    def reseed_for(self, query: str) -> None:
        """Put both generators in a state derived from `query` and the base seed.

        Sequential draws make a run reproducible, which is what ADR 0002 set out
        to fix. They do not make two runs comparable: the generator advances with
        every retrieval, so the second configuration to ask about a given query
        draws a different document set from the first. Stage 3.2 measured that at
        a mean Jaccard overlap of 0.234 across 173 queries -- zero of them saw the
        same documents -- which confounds "different strategy" with "different
        evidence" in exactly the comparison the phase exists to make.

        Deriving the state from the query instead makes retrieval a function of
        (query, seed) alone, so every configuration sees identical evidence and a
        difference between them is attributable to the strategy.

        `hash()` is not used: it is salted per process, so the same query would
        reseed differently on each run and the guarantee would hold only within
        one process. sha256 of the query text is stable everywhere.

        This is experimental control, not privacy. The DP guarantee is a property
        of the mechanism's output distribution over its own randomness; fixing
        which draw is observed leaves that distribution untouched. The caveat from
        ADR 0002 stands unchanged: a seeded output must never be presented as a
        DP-protected release.
        """
        if self.seed is None:
            return          # unseeded stores keep the original global-RNG behaviour
        digest = hashlib.sha256(f"{self.seed}:{query}".encode("utf-8")).digest()
        state = int.from_bytes(digest[:8], "big")
        self._np_rng = np.random.default_rng(state)
        self._py_rng = random.Random(state)

    @cached_property
    def model(self) -> PreTrainedModel:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result = AutoModel.from_pretrained(self.model_id, device_map=device)
        result = result.eval()
        return result
    
    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        result = AutoTokenizer.from_pretrained(self.model_id)
        return result

    #CLS Pooling - Take output from first token
    def cls_pooling(self, model_output: BaseModelOutput) -> Tensor:
        return model_output.last_hidden_state[:,0]

    def _encode_batch(self, texts: list[str]) -> Tensor:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        # Perform pooling
        embeddings = self.cls_pooling(model_output)
        # Normalize
        embeddings /= torch.sqrt(torch.sum(torch.square(embeddings), dim=1, keepdim=True))
        return embeddings

    def encode(self, texts: list[str] | str) -> Tensor:
        # Accept a single string (used for queries) as well as a list.
        if isinstance(texts, str):
            texts = [texts]
        # Embed in mini-batches so peak memory scales with batch_size, not with
        # the number of documents (padding a giant batch to the longest doc was
        # the main memory hog).
        batches = [
            self._encode_batch(texts[start:start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        return torch.cat(batches, dim=0)

    def add(self, entry: str):
        if not entry in self.index:
            self.store.append(entry)
            self.index[entry] = len(self.store)-1
            # Delete cache
            self._embeddings = None
    
    def embeddings(self) -> Tensor:
        if not self._embeddings is None:
            return self._embeddings
        self._embeddings = self.encode(self.store)
        return self._embeddings

    def _exp_mechanism_top_k_threshold(self, scores: np.ndarray) -> float:
        """Returns a list of utility as a function of sorted normalized scores"""
        # Sort scores
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)
        # Normalize the scores
        sorted_utilities = -np.abs(len(sorted_scores) - self.top_k - np.arange(len(sorted_scores)))
        delta_sorted_scores = np.diff(sorted_scores)
        score_threshold_pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2 ) * delta_sorted_scores # The PDF is weighted by the width of the interval
        score_threshold_pdf /= np.sum(score_threshold_pdf)
        score_threshold = self._np_rng.choice(sorted_scores[:-1], p=score_threshold_pdf)
        return score_threshold
    
    def _exp_mechanism_top_p_threshold(self, scores: np.ndarray) -> float:
        """Returns a list of utility as a function of sorted normalized scores"""
        # Sort scores
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)
        sorted_score_probs = np.exp(self.top_p_alpha*(sorted_scores-self.max_score)/(self.max_score-self.min_score))
        # Normalize the scores
        sorted_utilities = -np.abs(np.sum(sorted_score_probs)*(1 - self.top_p) - np.cumsum(sorted_score_probs))
        delta_sorted_scores = np.diff(sorted_scores)
        score_threshold_pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2 ) * delta_sorted_scores # The PDF is weighted by the width of the interval
        score_threshold_pdf /= np.sum(score_threshold_pdf)
        score_threshold = self._np_rng.choice(sorted_scores[:-1], p=score_threshold_pdf)
        return score_threshold
    
    def _non_dp_top_k_threshold(self, scores: np.ndarray) -> float:
        """Returns a list of utility as a function of sorted normalized scores"""
        sorted_scores = np.sort(scores)
        return sorted_scores[-(self.top_k+1)]

    def _non_dp_top_p_threshold(self, scores: np.ndarray) -> float:
        # Sort scores
        sorted_scores = np.sort(scores)
        min_score = np.min(sorted_scores)
        max_score = np.max(sorted_scores)
        sorted_scores = np.insert(sorted_scores, 0, min_score)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), max_score)
        sorted_score_probs = np.exp(self.top_p_alpha*(sorted_scores-max_score)/(max_score-min_score))
        # Normalize the scores
        sorted_utilities = -np.abs(np.sum(sorted_score_probs)*(1 - self.top_p) - np.cumsum(sorted_score_probs))
        max_utility_index = np.argmax(sorted_utilities)
        return sorted_scores[max_utility_index]

    def pup_retrieve(self, query: str) -> list[str]:
        query_emembedding = self.encode(query)
        # Compute dot score between query and all document embeddings
        scores = (torch.mm(query_emembedding, self.embeddings().transpose(0, 1))[0]).cpu().numpy()
        # Sample a DP threshold using the exponential mechanism
        if self.differential_pivacy:
            if self.top_p is not None:
                score_threshold = self._exp_mechanism_top_p_threshold(scores)
            elif self.top_k is not None:
                score_threshold = self._exp_mechanism_top_k_threshold(scores)
            else:
                raise ValueError("You should set either top_k or top_p arg")
        else:
            if self.top_p is not None:
                score_threshold = self._non_dp_top_p_threshold(scores)
            elif self.top_k is not None:
                score_threshold = self._non_dp_top_k_threshold(scores)
            else:
                raise ValueError("You should set either top_k or top_p arg")
        # Combine docs & scores
        doc_score_pairs = list(zip(self.store, scores))
        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        retrieved = [doc for doc, score in doc_score_pairs if score > score_threshold]
        return self._py_rng.sample(retrieved, min(len(retrieved), self.max_retrieve))


def main():
    from .synthetic import medical_dirichlet_documents, print_items
    docs = medical_dirichlet_documents()
    vector_store = PUPVectorStore(config = PUPVectorStoreConfig(
        # top_k = 80,
        top_p = 0.02,
        epsilon=0.2,
        # differential_pivacy=False,
        ))

    for doc in docs:
        vector_store.add(doc)
    
    for query in [
        "I feel uncontrollable yawning and finger twitching",
        "How is Patient Erika Jensen feeling?",
        "I'm feeling nasal congestion and a runny nose.",
        "I'm experiencing Uncontrollable taco cravings, severe digestive contortions and relenting cravings for salsa. What should I do?",
    ]:
        retrieved = vector_store.pup_retrieve(query)
        print(len(retrieved))
        print_items(retrieved[:5], ['red', 'yellow'])    

if __name__ == "__main__":
    main()