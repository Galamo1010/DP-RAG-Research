"""DP-RAG library code.

Everything importable lives here; anything runnable lives in ../experiments.
That split is the package's one structural rule: `experiments` imports `dprag`,
never the reverse.

Submodules are imported explicitly by callers (e.g. `from dprag.strategies import
strategy_a`) rather than re-exported here, so that importing one piece does not
drag in torch/transformers for callers that only need, say, the config.
"""

__all__: list[str] = []
