"""Stage 2.2 -- dual-instance (RAG / NoRAG) logit extraction.

Builds the two instances the proposal's pre-filter layer needs (計畫書 3.1):

    row 0  NoRAG instance : the query only, no documents
    row 1  RAG   instance : ALL retrieved documents concatenated + the query

Both run as ONE batch that shares a single emitted token stream, so at every
position the two logit vectors describe the SAME context. A driver processor
collapses the batch down to the driving row -- exactly the way
``DPLogitsAggregator`` collapses DPRAG's k+1 streams to one -- which makes
``generate`` broadcast the sampled token back to both rows so they advance in
lock-step.

This module deliberately does NOT touch ``dp_model.DPLogitsAggregator`` or any
DP aggregation. Stage 2 only measures what the pre-filter would decide.

Rather than buffering full-vocabulary logits (128k floats x 128 steps x 2 rows is
~130 MB per query), the recorder evaluates the strategy callables in-line and
keeps only the compact per-step decisions.

TWO LIMITATIONS TO CARRY INTO THE WRITE-UP
------------------------------------------
1. TRAJECTORY. The driver is the NoRAG instance, so generation walks NoRAG's
   trajectory -- not the mixed trajectory the Stage 3 router will actually walk
   (NoRAG token at consistent positions, DP-aggregated token elsewhere). Trigger
   rates from this module are an INITIAL INDICATOR; Stage 3.1 produces the
   authoritative numbers.
2. TEMPERATURE. Both strategies compare an argmax or a top-k index set, and
   dividing logits by T > 0 preserves both. Decisions here are therefore
   pointwise temperature-invariant; temperature only changes which trajectory is
   sampled. See prefilter_strategies for the full note.

Chat templates: the instances use a `system` role, matching dp_model.dp_chat.
Gemma-2's template rejects `system`, so this module works with Llama/Qwen as-is
and needs the prompt merge discussed for Stage 3 before it can run Gemma.

Requires a CUDA GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from prefilter_strategies import PrefilterDecision, Strategy

NORAG_ROW = 0
RAG_ROW = 1

# Byte-for-byte the system prompt dp_model.dp_chat gives its public-prior stream,
# so Stage 2's NoRAG instance stays comparable with the Stage 1 baseline.
_NORAG_SYSTEM = "You give a short response based on a predefined set documents."
_RAG_SYSTEM_HEAD = (
    "You give a short responses based on these documents or a predefined set of "
    "similar documents."
)


def build_norag_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _NORAG_SYSTEM},
        {"role": "user", "content": f"{question}"},
    ]


def build_rag_messages(documents: list[str], question: str) -> list[dict[str, str]]:
    """One instance holding every retrieved document (計畫書 3.1)."""
    if not documents:
        # DP retrieval can legitimately return nothing; with no documents the RAG
        # instance has nothing to condition on and collapses onto NoRAG.
        return build_norag_messages(question)
    joined = "\n\n".join(
        f'Document {i + 1}:\n"{doc}"' for i, doc in enumerate(documents)
    )
    return [
        {"role": "system", "content": f"{_RAG_SYSTEM_HEAD}\n{joined}"},
        {"role": "user", "content": f"{question}"},
    ]


class _DualInstanceRecorder(LogitsProcessor):
    """Pass-through: reads both rows and evaluates the strategies per step."""

    def __init__(self, strategies: dict[str, Strategy]):
        self.strategies = strategies
        self.norag_argmax: list[int] = []
        self.rag_argmax: list[int] = []
        self.decisions: dict[str, list[PrefilterDecision]] = {
            name: [] for name in strategies
        }

    def __call__(self, input_ids, scores):
        logit_norag = scores[NORAG_ROW]
        logit_rag = scores[RAG_ROW]
        self.norag_argmax.append(int(torch.argmax(logit_norag)))
        self.rag_argmax.append(int(torch.argmax(logit_rag)))
        for name, strategy in self.strategies.items():
            self.decisions[name].append(strategy(logit_rag, logit_norag))
        return scores


class _DriverProcessor(LogitsProcessor):
    """Collapse the batch to one row so both instances share the emitted token.

    Mirrors DPLogitsAggregator, which returns a [1, vocab] tensor from its
    [k+1, vocab] input; ``generate`` then broadcasts the sampled token to every
    row, keeping the instances on a single shared continuation.
    """

    def __init__(self, row: int):
        self.row = row

    def __call__(self, input_ids, scores):
        return scores[self.row : self.row + 1]


@dataclass
class DualRunResult:
    question: str
    n_documents: int
    n_steps: int
    emitted: list[int]
    norag_argmax: list[int]
    rag_argmax: list[int]
    decisions: dict[str, list[PrefilterDecision]] = field(default_factory=dict)
    text: str = ""

    def trigger_rate(self, strategy_name: str) -> float:
        """Fraction of positions the strategy would route around DP (zero epsilon)."""
        decisions = self.decisions[strategy_name]
        if not decisions:
            return 0.0
        return sum(1 for d in decisions if d.consistent) / len(decisions)

    def mean_score(self, strategy_name: str) -> float:
        decisions = self.decisions[strategy_name]
        if not decisions:
            return 0.0
        return sum(d.score for d in decisions) / len(decisions)


def make_generation_config(
    temperature: float, max_new_tokens: int, pad_token_id: int | None = None
) -> GenerationConfig:
    """Plain sampling config -- Stage 2 measures the pre-filter, so no DP machinery.

    Building a DPGenerationConfig here would pay for a PLD binary search that
    nothing in this module consumes.
    """
    return GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )


def run_dual_instance(
    dp_model,
    documents: list[str],
    question: str,
    generation_config: GenerationConfig,
    strategies: dict[str, Strategy],
    driver: str = "norag",
) -> DualRunResult:
    """Generate once and collect what each strategy would have decided per step.

    dp_model: a dp_model.DPModel (used only for its model + tokenizer).
    strategies: name -> callable with the (logit_rag, logit_norag) signature.
    driver: which instance drives sampling, "norag" (default) or "rag".
    """
    if driver not in ("norag", "rag"):
        raise ValueError(f"driver must be 'norag' or 'rag', got {driver!r}")
    driver_row = NORAG_ROW if driver == "norag" else RAG_ROW

    tokenizer = dp_model.tokenizer
    messages = [None, None]
    messages[NORAG_ROW] = build_norag_messages(question)
    messages[RAG_ROW] = build_rag_messages(documents, question)

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        continue_final_message=False,
    ).to("cuda")
    input_len = model_inputs["input_ids"].shape[-1]

    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id

    recorder = _DualInstanceRecorder(strategies)
    processors = LogitsProcessorList([recorder, _DriverProcessor(driver_row)])

    with torch.no_grad():
        out = dp_model.model.generate(
            **model_inputs,
            generation_config=generation_config,
            logits_processor=processors,
        )
    emitted = out[0, input_len:].tolist()
    if emitted and emitted[-1] == tokenizer.eos_token_id:
        emitted = emitted[:-1]

    # The recorder fires once per generation step; trim the bookkeeping to the
    # tokens actually kept so every list describes the same positions.
    n_steps = min(len(emitted), len(recorder.norag_argmax))
    return DualRunResult(
        question=question,
        n_documents=len(documents),
        n_steps=n_steps,
        emitted=emitted[:n_steps],
        norag_argmax=recorder.norag_argmax[:n_steps],
        rag_argmax=recorder.rag_argmax[:n_steps],
        decisions={
            name: values[:n_steps] for name, values in recorder.decisions.items()
        },
        text=tokenizer.decode(emitted[:n_steps], skip_special_tokens=True),
    )
