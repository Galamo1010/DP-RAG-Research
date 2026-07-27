# ADR 0001 — Drop Gemma-2-9B-IT from the model set

Date: 2026-07-25
Status: Accepted (replacement model still to be chosen)

## Context

The proposal (專題計畫書) fixes three models for the cross-model comparison in
Stages 3 and 5:

    Llama-3.1-8B-Instruct · Gemma-2-9B-IT · Qwen-2.5-14B-Instruct

Running Gemma fails immediately:

    jinja2.exceptions.TemplateError: System role not supported

`DPModel.dp_chat` builds each of its k+1 streams as a `system` + `user` pair, and
Gemma-2's chat template accepts only `user` and `model` roles. This is a
documented property of the Gemma family, not a bug in this project. Llama-3.1 and
Qwen-2.5 both accept a system role, which is why only Gemma fails.

Two ways out:

1. **Merge `system` into `user`** for every model, so one prompt shape serves all
   three.
2. **Drop Gemma** and pick a third model that accepts a system role.

## Decision

Drop Gemma. `dprag.prompts` emits exactly one prompt format, built around a
`system` role, and does not branch per model. Any model added to
`dprag.config.MODELS` must accept that format.

The replacement is **not yet chosen**. Candidates, both of which accept a system
role:

- `mistralai/Mistral-7B-Instruct-v0.3` — close in size to Llama-3.1-8B, different
  architecture, so the comparison stays meaningful.
- `microsoft/Phi-3.5-mini-instruct` — the model upstream sarus-tech/dp-rag
  defaults to, which carries its own justification, but is much smaller (3.8B).

## Consequences

**The prompt stays uniform across models.** Stage 3 compares three models on ε
savings and quality; prompt structure is a controlled variable rather than
something that differs per model. Merging `system` into `user` would also have
achieved this, but at the cost of making every model's prompt differ from upstream
dp-rag's, weakening the "faithful reproduction" claim that Stage 1 rests on.

**This deviates from the proposal and has to be defended.** It is the second such
deviation, after moving from Together AI to a local GPU. The report needs to say
that Gemma's chat template is incompatible with DPRAG's prompt structure, and
that using each model's native format instead would have introduced prompt shape
as a confounder in the cross-model comparison.

**A per-model branch stays possible.** If a model that cannot take a system role
becomes necessary later, the merge belongs inside `dprag.prompts` — one place,
already the single owner of prompt construction — not spread across call sites.

**A test guards this.** `tests/test_prompts_and_seeding.py` asserts that no entry
in `MODELS` is a Gemma variant, so re-adding one fails loudly rather than at the
first GPU run.
