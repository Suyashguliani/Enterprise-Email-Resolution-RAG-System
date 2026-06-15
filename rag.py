from __future__ import annotations

from typing import Any, Dict, List

from llm import generate_answer
from query import get_llm_context, retrieve_chunks

# Separates independent retrieved passages so the model does not read them as one run-on paragraph.
CONTEXT_SEPARATOR = "\n\n---\n\n"


def rag_pipeline(question: str) -> str:
    """
    Backwards-compatible entry point: return assistant answer text only.
    """
    return run_rag(question)["answer"]


def run_rag(question: str) -> Dict[str, Any]:
    """
    Full RAG pass: retrieve chunks once, build context, call LLM.

    Returns answer plus retrieval records for source attribution in UIs.
    """
    chunks: List[Dict[str, Any]] = retrieve_chunks(question)
    context = get_llm_context(chunks, separator=CONTEXT_SEPARATOR)
    answer = generate_answer(question, context)

    sources_ordered: List[str] = []
    seen = set()
    for c in chunks:
        name = c.get("source_filename") or "unknown"
        if name not in seen:
            seen.add(name)
            sources_ordered.append(name)

    return {
        "answer": answer,
        "chunks": chunks,
        "sources": sources_ordered,
        "context": context,
    }


if __name__ == "__main__":
    question = input("Ask your question: ")
    out = run_rag(question)
    print("\nFINAL ANSWER:\n")
    print(out["answer"])
