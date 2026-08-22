# ADR 0004 — Third comparison model: Gemma 4 12B

Date: 2026-08-20
Status: Accepted. Supersedes the model choice in [ADR 0003](0003-stage3-scope-and-routing.md) §4.

## Context

The proposal names Gemma-2-9B-IT as the third comparison model. [ADR 0001](0001-replace-gemma.md)
dropped it because Gemma 2's chat template rejects the `system` role that
`dprag.prompts` emits, and [ADR 0003](0003-stage3-scope-and-routing.md) §4 named
Mistral-7B-Instruct-v0.3 in its place, on the grounds that the resulting set gave
"7B/8B/14B across three different architectures, all accepting a system role".

**That last clause was never verified.** It is the same claim that turned out to be
false for Gemma 2, and nothing in the repository checks it: the only guard,
`test_prompts_use_a_system_role_as_the_model_set_assumes`, asserts that the string
`"gemma"` does not appear in `MODELS`. A name blacklist cannot catch a second model
whose template happens to reject the role, so Mistral would have passed every test
and failed in Stage 3.2 Phase 3.

Re-opening the question surfaced two facts that settle it differently:

**Mistral-7B-Instruct-v0.3 is 7.25B, below the 8B floor this project now sets.**
A capable model is wanted for two reasons: the setting is clinical, and Stage 3.2
has to establish that quality does *not* drop, which is undetectable if quality had
no headroom to begin with. Whether Mistral accepts a system role became moot.

**Gemma 4 (released 2026-04-02, Apache 2.0) added native system-role support.**
Its chat template gives `system` its own turn rather than merging or dropping it,
which is exactly the defect that removed Gemma 2 from the set.

## Decision

The third model is **`google/gemma-4-12B-it`**.

Verified before adoption — the step ADR 0003 skipped — by reading
`chat_template.jinja` from the model repository:

```jinja
{%- if messages and messages[0]['role'] in ['system', 'developer'] -%}
    {{- '<|turn>system\n' -}}
```

The system message gets dedicated turn markers, is not merged into the user turn,
is not dropped, and raises nothing. It must be the first message, which
`dprag.prompts` always makes it and a test pins.

### Rejected

**Mistral-7B-Instruct-v0.3** — 7.25B, below the 8B floor. Its system-role support
was never verified either; the template was changed in July 2024 to add the role,
having previously raised `Only user and assistant roles are supported!`, so the
answer depends on which snapshot is downloaded.

**Phi-3.5-mini-instruct** — 3.8B. Verified locally to accept a system role with its
content preserved, so it remains the fallback of last resort, but it is too small
for a clinical setting and would put the size spread at 3.8/8/14.

**gemma-4-31B and gemma-4-26B-A4B** — do not fit. ADR 0003 already records why a
dense ~32B is impractical here: it leaves too little of an 80 GB A100 for the KV
cache of eleven long-prompt rows, and runs 3–4× slower. The 26B-A4B variant is MoE,
which saves compute rather than VRAM.

### Not evaluated

**Ministral-3-8B** (Mistral AI, dense 8B, Apache 2.0) meets the size floor and would
have kept the Mistral architecture. It surfaced too late in the discussion to be
assessed and was **not weighed against Gemma 4**. Recorded so a later reader does
not mistake silence for rejection.

## Consequences

**The proposal deviation shrinks.** The report no longer has to explain replacing
the proposal's model family; it explains a version change — Gemma 2 could not take
the prompt format this project uses, Gemma 4 can.

**Size spread narrows to 8B/12B/14B**, from the 7B/8B/14B ADR 0003 planned. Still
three sizes across three architectures, but Gemma and Qwen now sit close together.

**12B is slower and more expensive to run than 7B.** Accepted deliberately.

**Vocabulary size cuts both ways, and this is the consequence most likely to be
missed.** Measured from each model's `config.json`:

| model | params | vocab | k=50 as share of vocab |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 8.03B | 128,256 | 0.0390% |
| Qwen2.5-14B-Instruct | 14.77B | 152,064 | 0.0329% |
| **gemma-4-12B-it** | ~12B | **262,144** | **0.0191%** |
| *(rejected)* Mistral-7B-v0.3 | 7.25B | 32,768 | 0.1526% |

*Against* the choice: Gemma has the largest vocabulary in the set, so the aggregated
score range of roughly ±(k × clipping) — a couple of nats — spreads over twice as
many tokens as on Llama and concentrates even less probability. Whatever the
`top_k=50` warper fixed on Llama matters more here, not less.

*For* the choice, and neither party anticipated this: Mistral's 32,768-token
vocabulary would have been the outlier. Its `k=50` covers 3.9× the share of the
vocabulary that Llama's does, so the same Strategy B configuration would mean a
materially wider candidate pool on one model than the others. Adopting Gemma
narrows the set's vocabulary spread from **4.6× to 2.0×**, making the cross-model
Strategy B comparison more nearly like-for-like.

**16.8% of Gemma's parameters are embeddings** (262,144 × 3,840, tied or not),
against 13.1% for Llama and 10.5% for Qwen. On non-embedding parameters — the part
that computes — 12B Gemma is closer to 10B. Still above Llama, but not by the
margin the headline number suggests.

**Gemma 4 is multimodal.** This project uses text only. Every call site goes through
`tokenizer.apply_chat_template`, not a processor, so that path must be confirmed to
pick up the template on first use.

**`transformers` must be new enough to know `gemma4_unified`.** `pyproject.toml`
pins only `transformers~=4.0`, which constrains nothing useful.

**A chat template is a mutable file in a model repository, not a specification.**
Mistral's changed in July 2024. Nothing here pins a revision, so the template that
arrives is whichever is on `main` the day it is downloaded. Verification therefore
belongs at download time, against the snapshot actually obtained, not in a document.
