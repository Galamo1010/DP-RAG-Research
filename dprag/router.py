"""Stage 3.1 -- the RAG/NoRAG pre-filter router.

Everything measured before this module was hypothetical: Stage 1.2 and Stage 2.3/2.4
ran generation and recorded what a pre-filter *would* have decided, while the system
still spent epsilon at all 128 positions. This is where the branch actually happens.

At each step the router asks the strategy whether the RAG instance and the NoRAG
instance agree. On agreement it emits the NoRAG token and charges nothing. On
disagreement it runs the unmodified DPRAG aggregation over the k+1 streams and
charges token_epsilon (proposal 3.1).

WHY TWO STREAMS ADVANCE AT DIFFERENT RATES
------------------------------------------
The pre-filter batch (2 rows) must run every step -- without it there is no
decision. The DPRAG batch (k+1 rows) only runs when an aggregation is actually
needed. Skipping it is the whole point: measurements show batch width is not the
bottleneck (an 11-row batch costs about the same per query as a 2-row one), so the
saving comes from not taking those steps at all.

That leaves the DPRAG batch's KV cache behind the emitted sequence. Rather than
replaying the gap one token at a time, the router keeps a backlog and feeds it in a
single multi-token forward the next time an aggregation is required. The invariant
is simple: `backlog` is exactly the tokens the DPRAG stream has not seen yet.

Getting that synchronisation wrong does not raise -- it silently conditions the
aggregation on the wrong prefix and produces a plausible, wrong answer. So the
router takes its model as a constructor argument and is verified on CPU against a
fake, using two equivalence properties: a strategy that never agrees must reproduce
plain DPRAG token for token, and one that always agrees must reproduce the pure
NoRAG greedy sequence at zero cost.

EPSILON ACCOUNTING, AND WHAT IT ASSUMES
---------------------------------------
token_epsilon is unchanged -- still the value binary-searched so that composing
max_new_tokens steps reaches the generation budget. The router composes it over the
paid positions only, so epsilon_savings = budget - usage (proposal Eq5). Re-solving
token_epsilon for the paid count is not possible: that count is not known until
generation has finished.

This assumes the routing decisions are themselves free. They are not obviously so:
the RAG instance holds the retrieved documents, so *which* positions get skipped is
a function of private data. The proposal places this outside its formal scope and
tests it empirically in Stage 4.3. Every result carries EPSILON_ACCOUNTING_NOTE so
the assumption travels with the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch
from torch import Tensor

from . import prompts
from .dual_instance import NORAG_ROW, RAG_ROW
from .strategies import PrefilterDecision, Strategy

EPSILON_ACCOUNTING_NOTE = (
    "paid positions only; assumes the routing decisions are themselves free -- "
    "tested empirically in Stage 4.3"
)


@lru_cache(maxsize=None)
def _composed_epsilon(token_epsilon: float, steps: int, delta: float) -> float:
    """Epsilon after composing `steps` copies of token_epsilon, via PLD.

    Cached because a 200-query run composes the same handful of step counts over
    and over, and each composition is a loop over PLD objects.
    """
    if steps <= 0:
        return 0.0
    from dp_accounting.pld.common import DifferentialPrivacyParameters
    from dp_accounting.pld.privacy_loss_distribution import (
        from_privacy_parameters,
        identity,
    )

    pld = identity()
    single = from_privacy_parameters(DifferentialPrivacyParameters(epsilon=token_epsilon))
    for _ in range(steps):
        pld = pld.compose(single)
    return pld.get_epsilon_for_delta(delta)


class _Stream:
    """One batch of conversations plus the KV cache it has built up.

    Kept separate per batch because the two batches advance at different rates;
    sharing a cache between them would be exactly the bug this class exists to
    make impossible.
    """

    def __init__(self, model, input_ids: Tensor, attention_mask: Tensor):
        self.model = model
        self._prompt_ids = input_ids
        self._mask = attention_mask
        self._cache = None
        self.primed = False

    def _forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=self._cache,
                use_cache=True,
            )
        self._cache = out.past_key_values
        self._mask = mask
        return out.logits[:, -1, :]

    def advance(self, new_tokens: list[int]) -> Tensor:
        """Feed `new_tokens` and return the logits predicting the next position.

        The first call feeds the prompt. When past_key_values is in play the
        attention mask has to span past+new, not just the new tokens.
        """
        if not self.primed:
            logits = self._forward(self._prompt_ids, self._mask)
            self.primed = True
            if not new_tokens:
                return logits
        batch = self._mask.shape[0]
        ids = torch.tensor(
            [new_tokens] * batch, dtype=torch.long, device=self._mask.device
        )
        mask = torch.cat(
            [self._mask, torch.ones_like(ids, dtype=self._mask.dtype)], dim=1
        )
        return self._forward(ids, mask)


@dataclass
class RoutedResult:
    """One routed generation: what came out, and what it cost."""

    question: str
    n_documents: int
    emitted: list[int]
    text: str
    decisions: list[PrefilterDecision]
    paid_positions: list[int]
    epsilon_usage: float
    epsilon_budget: float
    token_epsilon: float
    epsilon_accounting: str = EPSILON_ACCOUNTING_NOTE
    norag_argmax: list[int] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.emitted)

    @property
    def n_paid(self) -> int:
        return len(self.paid_positions)

    @property
    def trigger_rate(self) -> float:
        """Share of positions routed around DP (the free path)."""
        if not self.emitted:
            return 0.0
        return 1.0 - self.n_paid / len(self.emitted)

    @property
    def epsilon_savings(self) -> float:
        return self.epsilon_budget - self.epsilon_usage

    def is_free(self, position: int) -> bool:
        return position not in set(self.paid_positions)


class Router:
    """Runs DPRAG with a pre-filter deciding, per position, whether to spend epsilon.

    The model arrives as an argument rather than being constructed here, so the
    loop can be driven against a fake on CPU. Swapping strategies is swapping this
    object's `strategy`; the routing logic itself does not change.
    """

    def __init__(self, dp_model, strategy: Strategy, generation_config):
        self.dp_model = dp_model
        self.strategy = strategy
        self.config = generation_config

    def _tokenize(self, conversations) -> tuple[Tensor, Tensor]:
        encoded = self.dp_model.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        # Move the tensors individually rather than calling .to() on the result:
        # tokenizers return a BatchEncoding, but nothing here needs that type.
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        device = getattr(self.dp_model.model, "device", None)
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask

    def generate(self, documents: list[str], question: str) -> RoutedResult:
        cfg = self.config
        tokenizer = self.dp_model.tokenizer
        aggregator = self.dp_model.dp_logits_aggregator(cfg)
        token_epsilon = cfg.token_epsilon()
        eos = tokenizer.eos_token_id

        prefilter = _Stream(self.dp_model.model, *self._tokenize(
            prompts.dual_instance_batch(documents, question)
        ))
        dprag = _Stream(self.dp_model.model, *self._tokenize(
            prompts.dprag_chat_batch(documents, question)
        ))

        emitted: list[int] = []
        norag_argmax: list[int] = []
        decisions: list[PrefilterDecision] = []
        paid_positions: list[int] = []
        # Exactly the tokens the DPRAG stream has not been shown yet.
        backlog: list[int] = []
        # What the pre-filter stream has not been shown yet (always 0 or 1 tokens).
        pending: list[int] = []

        for position in range(cfg.max_new_tokens):
            scores = prefilter.advance(pending)
            decision = self.strategy(scores[RAG_ROW], scores[NORAG_ROW])
            decisions.append(decision)
            norag_argmax.append(int(scores[NORAG_ROW].argmax()))

            if decision.consistent:
                token = decision.token_id
            else:
                # Bring the DPRAG stream up to date in one forward, then aggregate.
                dp_scores = dprag.advance(backlog)
                backlog = []
                aggregated = aggregator(None, dp_scores)
                # Same order generate() uses: processor first, then temperature,
                # then sample. Reversing it would break the exponential mechanism.
                probs = torch.softmax(aggregated[0] / cfg.temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
                paid_positions.append(position)

            emitted.append(token)
            backlog.append(token)
            pending = [token]

            if eos is not None and token == eos:
                emitted.pop()
                decisions.pop()
                norag_argmax.pop()
                if paid_positions and paid_positions[-1] == position:
                    paid_positions.pop()
                break

        epsilon_budget = _composed_epsilon(token_epsilon, cfg.max_new_tokens, cfg.delta)
        epsilon_usage = _composed_epsilon(token_epsilon, len(paid_positions), cfg.delta)

        return RoutedResult(
            question=question,
            n_documents=len(documents),
            emitted=emitted,
            text=tokenizer.decode(emitted, skip_special_tokens=True),
            decisions=decisions,
            paid_positions=paid_positions,
            epsilon_usage=epsilon_usage,
            epsilon_budget=epsilon_budget,
            token_epsilon=token_epsilon,
            norag_argmax=norag_argmax,
        )
