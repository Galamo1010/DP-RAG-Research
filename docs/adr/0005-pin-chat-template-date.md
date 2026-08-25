# ADR 0005 — Pin the chat template's date, and what that says about ADR 0002

Date: 2026-08-25
Status: Accepted. Qualifies [ADR 0002](0002-seeded-retrieval.md).

## Context

[ADR 0002](0002-seeded-retrieval.md) seeded retrieval and generation so that
"single configurations become re-runnable". That claim is false on Llama, and was
false for every run this project has made.

Llama's chat template supplies today's date when the caller does not:

```jinja
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
...
{{- "Today Date: " + date_string + <newline><newline> }}
```

The system message therefore carries the day the experiment ran. Measured on
Llama-3.2-1B with `prompts.norag_chat`:

- token **count** is unaffected (51 either way), so nothing downstream shifts;
- three token **values** change;
- generation diverges after ~178 characters under greedy decoding and ~64 under
  seeded sampling. The same date twice reproduces exactly.

So the seed was doing its job and the prompt was not. Stage 1.2, 2.3 and 2.4 ran
on 2026-07-30; Stage 2.5 ran on 2026-08-14. **Those two groups were never running
the same system prompt**, which is worth knowing before their trigger rates are
compared with each other.

Checked across the model set: Qwen and Phi templates contain no dynamic date and
ignore the keyword. Gemma 4 is unverified — it is not in the local cache, so this
goes into the preflight when the model is first downloaded.

## Decision

`prompts.DATE_STRING = "30 Jul 2026"`, spread into every `apply_chat_template`
call through `prompts.TEMPLATE_KWARGS`.

**Why that date.** It is the day Stage 1.2 ran, so the consistency ceiling of
0.874 — the number the whole contribution is measured against — stays
reproducible. Stage 2.5's 2026-08-14 was the alternative; only one group can be
preserved, and the ceiling is the more load-bearing of the two.

**Why the keyword rather than removing the line.** The template hard-codes
`"Today Date: " + date_string`, so an empty string renders `Today Date: ` — input
no model saw in training, trading a known small variable for an unknown one.
Replacing the template outright would abandon the official format that Stage 1.1's
reproduction claim rests on, and would need maintaining once per model.

**Why in `prompts.py`.** That module already exists to stop one piece of prompt
knowledge living in several places. A pin applied at four call sites out of five
fails silently, which is this project's characteristic failure mode.

## Consequences

**Past results are not recovered.** Pinning fixes reproducibility going forward.
Every run before 2026-08-25 used whatever date it ran on, and Stage 1/2.3/2.4 and
Stage 2.5 are on different prompts. The report should say so where their numbers
appear together.

**ADR 0002's claim needs the qualifier.** Seeding makes a configuration
re-runnable *given a fixed prompt*. It never controlled the prompt itself. The
privacy argument in ADR 0002 is untouched — this is about experimental control,
not about the DP guarantee.

**Two tests pin the behaviour**: that `TEMPLATE_KWARGS` carries a non-empty
`date_string`, and that the router's tokenisation path actually passes it through.
The second matters more: the constant existing is not the same as it arriving.

**A general lesson worth carrying.** A chat template is a mutable file shipped with
a model, and it can inject state the caller never asked for. [ADR
0004](0004-third-model-gemma-4.md) already records that templates change between
versions. This one records that they can also change between *days*. Anything a
template can vary is an experimental variable until pinned.
