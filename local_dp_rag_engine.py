"""Local (GPU) DP-RAG engine -- the ORIGINAL DP-RAG design.

Unlike ``dp_rag_engine.DPRAGEngine`` (which routes generation through OpenRouter
and therefore drops token-level DP), this engine keeps BOTH differential-privacy
layers, exactly as the original sarus-tech/dp-rag does:

    retrieval : PUPVectorStore.pup_retrieve      (exponential-mechanism threshold)
    generation: DPModel.dp_chat + DPLogitsAggregator
                (k+1 parallel streams, full-vocab logit clipping + exp mechanism)

The total privacy loss is the PLD composition of the two layers, so
``privacy_loss_distribution.get_epsilon_for_delta(delta)`` gives the real
end-to-end epsilon (this is the two-layer accounting the proposal calls Eq3).

Requires a CUDA GPU (use load_in_4bit=True to fit 8B/9B/14B on 24GB / 8GB).
"""

from termcolor import cprint

from pup_vector_store import PUPVectorStore, PUPVectorStoreConfig
from dp_model import DPModel, DPGenerationConfig


class LocalDPRAGEngine:
    def __init__(
        self,
        pup_vector_store_config: PUPVectorStoreConfig = PUPVectorStoreConfig(),
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dp_generation_config: DPGenerationConfig = DPGenerationConfig(),
        load_in_4bit: bool = False,
    ):
        self.pup_vector_store = PUPVectorStore(config=pup_vector_store_config)
        self.dp_model = DPModel(model_id=model_id, load_in_4bit=load_in_4bit)
        self.dp_generation_config = dp_generation_config
        # Two-layer privacy accounting: retrieval PLD composed with generation PLD.
        self.privacy_loss_distribution = (
            self.pup_vector_store.privacy_loss_distribution.compose(
                dp_generation_config.privacy_loss_distribution
            )
        )

    def add(self, entry: str):
        self.pup_vector_store.add(entry)

    def pup_retrieve(self, query: str) -> list[str]:
        return self.pup_vector_store.pup_retrieve(query=query)

    def epsilon(self, delta: float | None = None) -> float:
        delta = delta if delta is not None else self.dp_generation_config.delta
        return self.privacy_loss_distribution.get_epsilon_for_delta(delta)

    def dp_chat(self, question: str) -> str:
        """Original DP-RAG: DP retrieval, then DP generation over k+1 streams."""
        retrieved_documents = self.pup_vector_store.pup_retrieve(question)
        return self.dp_model.dp_chat(
            retrieved_documents, question, self.dp_generation_config
        )


def main():
    """Smoke test on a tiny ChatDoctor slice with a small (cached) model."""
    from chatdoctor_data import load_corpus, load_queries
    import experiment_params as P

    engine = LocalDPRAGEngine(
        pup_vector_store_config=PUPVectorStoreConfig(
            model_id=P.EMBED_MODEL,
            top_p=P.RETRIEVAL_TOP_P,
            epsilon=P.EPS_RETRIEVAL,
            max_retrieve=5,          # small: keeps the k+1 batch tiny for a first run
            batch_size=P.EMBED_BATCH_SIZE,
        ),
        model_id="Qwen/Qwen2.5-1.5B-Instruct",   # cached, ungated, Qwen2.5 family
        dp_generation_config=DPGenerationConfig(
            temperature=P.TEMPERATURE,
            max_new_tokens=64,
            alpha=P.ALPHA,
            omega=P.OMEGA,
            epsilon=10.0,            # generation budget (per full answer)
        ),
        load_in_4bit=False,          # 1.5B fits in fp16; set True for 8B+
    )

    for doc in load_corpus(limit=300, sample_seed=7):
        engine.add(doc)
    engine.pup_vector_store.embeddings()  # build embeddings once

    print(f"End-to-end epsilon (retrieval + generation, delta={P.DELTA}): "
          f"{engine.epsilon(P.DELTA):.4f}\n")

    for q in load_queries(n=3, seed=P.QUERY_SEED):
        answer = engine.dp_chat(q.query)
        cprint("Q: " + q.query[:100].replace("\n", " "), "cyan")
        cprint("A(DP): " + answer[:200].replace("\n", " "), "green")
        print()


if __name__ == "__main__":
    main()
