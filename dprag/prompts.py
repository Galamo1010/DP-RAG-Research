"""The single owner of prompt construction.

Two callers need to agree on exactly what "the NoRAG instance" means:

  * dp_model.dp_chat        builds the k+1 DPRAG streams (row 0 = NoRAG)
  * dual_instance           builds the 2-row pre-filter batch (row 0 = NoRAG)

They used to hold separate copies of the same strings, kept in step by a comment
and a test asserting they matched byte-for-byte. That test passing was the tell:
if a copy is only correct because something checks it against the other copy, the
knowledge belongs in one place. Silent divergence there would not crash anything
-- it would just quietly invalidate the Stage 2 comparison, because the two
"NoRAG"s would no longer be the same thing.

Stage 3's router will be the third caller.

ONE FORMAT, NO PER-MODEL BRANCHING
----------------------------------
Every message here uses a `system` role. Gemma-2's chat template rejects that
("System role not supported"), which is why Gemma was dropped from the model set
rather than special-cased -- see dprag.config.MODELS and
docs/adr/0001-replace-gemma.md. Any model added to the comparison must support a
system role. If one ever has to be supported that does not, the merge belongs
here, in one place, not at each call site.
"""

from __future__ import annotations

Message = dict[str, str]
Conversation = list[Message]

# --- chat (question answering over retrieved documents) --------------------

NORAG_SYSTEM = "You give a short response based on a predefined set documents."

# Singular "this document": used by the k+1 streams, where each stream carries
# exactly one document. Wording preserved verbatim from upstream dp-rag so the
# reproduction stays faithful.
PER_DOCUMENT_SYSTEM = (
    "You give a short responses based on this document or a predefined set of "
    "similar documents.\nDocument:\n"
)

# Plural: used by the pre-filter's single RAG instance, which concatenates every
# retrieved document into one prompt (proposal 3.1).
ALL_DOCUMENTS_SYSTEM = (
    "You give a short responses based on these documents or a predefined set of "
    "similar documents."
)


def norag_chat(question: str) -> Conversation:
    """The instance that sees the question and no documents."""
    return [
        {"role": "system", "content": NORAG_SYSTEM},
        {"role": "user", "content": f"{question}"},
    ]


def single_document_chat(document: str, question: str) -> Conversation:
    """One DPRAG stream: the question plus exactly one document."""
    return [
        {"role": "system", "content": f'{PER_DOCUMENT_SYSTEM}"{document}"'},
        {"role": "user", "content": f"{question}"},
    ]


def all_documents_chat(documents: list[str], question: str) -> Conversation:
    """The pre-filter's RAG instance: the question plus every retrieved document.

    With no documents this collapses to `norag_chat`, which is the honest
    representation: DP retrieval returning nothing leaves the RAG instance with
    nothing to condition on.
    """
    if not documents:
        return norag_chat(question)
    joined = "\n\n".join(
        f'Document {i + 1}:\n"{doc}"' for i, doc in enumerate(documents)
    )
    return [
        {"role": "system", "content": f"{ALL_DOCUMENTS_SYSTEM}\n{joined}"},
        {"role": "user", "content": f"{question}"},
    ]


def dprag_chat_batch(documents: list[str], question: str) -> list[Conversation]:
    """The k+1 DPRAG streams: NoRAG first, then one per document.

    Row 0 being NoRAG is load-bearing -- DPLogitsAggregator reads `scores[0]` as
    the public prior, and stage1_consistency reads the same row as the NoRAG
    argmax.
    """
    return [norag_chat(question)] + [
        single_document_chat(doc, question) for doc in documents
    ]


def dual_instance_batch(documents: list[str], question: str) -> list[Conversation]:
    """The 2-row pre-filter batch: [NoRAG, RAG-with-all-documents].

    Row order matches dual_instance.NORAG_ROW / RAG_ROW.
    """
    return [norag_chat(question), all_documents_chat(documents, question)]


# --- summary (the upstream dp_summary demo) --------------------------------

SUMMARY_SYSTEM = "You are a rephrasing writer."


def summary_batch(documents: list[str], topic: str) -> list[Conversation]:
    """The k+1 streams for dp_summary, mirroring dprag_chat_batch's shape."""
    return [
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f'Can you write a short text about the following topics:\n"{topic}"?',
            },
        ]
    ] + [
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f'Can you rephrase this document:\n"{doc}"?\nJust output the text.',
            },
        ]
        for doc in documents
    ]
