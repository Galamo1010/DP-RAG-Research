"""OpenRouter-backed text generation.

This replaces the local, GPU-based generation of ``dp_model.DPModel`` with calls
to the OpenRouter chat-completions API (https://openrouter.ai).

IMPORTANT: OpenRouter only returns generated *text*. It does not expose the
per-token vocabulary logits, so the token-by-token differential-privacy
aggregation implemented in ``dp_model.DPLogitsAggregator`` CANNOT run through it.
Using this module therefore gives a *non-DP* RAG pipeline: the differential
privacy guarantee on generation is dropped. (The DP threshold used during
retrieval in ``pup_vector_store.PUPVectorStore`` is pure NumPy and still applies.)
"""

import os
from functools import cached_property

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv).

    Reads ``KEY=VALUE`` lines from ``path`` (next to this file, then the current
    working directory) into ``os.environ`` without overriding existing values.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
        path,
    ]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        break


class OpenRouterModel:
    """Chat/RAG generation through the OpenRouter API."""

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_URL,
        timeout: float = 120.0,
    ):
        _load_dotenv()
        self.model_id = model_id or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY in the "
                "environment or in the .env file."
            )
        self.base_url = base_url
        self.timeout = timeout

    @cached_property
    def _session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Optional attribution headers recommended by OpenRouter.
                "HTTP-Referer": "https://github.com/sarus-tech/dp-rag",
                "X-Title": "DP-RAG (OpenRouter)",
            }
        )
        return session

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 100,
    ) -> str:
        """Send a single conversation and return the assistant's text reply."""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._session.post(self.base_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def rag_chat(
        self,
        context_documents: list[str],
        question: str,
        temperature: float = 1.0,
        max_tokens: int = 100,
    ) -> str:
        """Standard (non-DP) RAG: answer ``question`` using the retrieved docs.

        The documents are concatenated into a single context block. This is the
        conventional RAG prompt; there is no differential-privacy aggregation.
        """
        if context_documents:
            context = "\n\n".join(
                f"[{i + 1}] {doc}" for i, doc in enumerate(context_documents)
            )
            system = (
                "You are a helpful assistant. Answer the user's question using only "
                "the information in the following documents. Give a short answer. "
                "If the answer is not contained in the documents, say you don't know.\n\n"
                f"Documents:\n{context}"
            )
        else:
            system = "You give a short answer to the user's question."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        return self.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    model = OpenRouterModel()
    print(f"Using model: {model.model_id}")
    answer = model.rag_chat(
        context_documents=[
            "Alexandre Nielsen has been diagnosed with Snurflaxitis. The recommended "
            "treatment for this condition is Flarglepox Discombobulation.",
        ],
        question="What is Alexandre Nielsen diagnosed with?",
        temperature=0.1,
        max_tokens=50,
    )
    print(answer)
