"""
Text chunking for dense-vector RAG.

Why chunking improves RAG:
    A single embedding for an entire email or long document averages many
    unrelated ideas into one vector, so similarity search often returns the
    wrong document or the right file but the wrong passage. Smaller chunks
    align embeddings with local semantics, so kNN returns the specific span
    that answers the question and the LLM receives focused context.

Why overlap is used:
    Facts often sit on boundaries (e.g. a sentence ends at one chunk and its
    resolution starts in the next). Sliding windows with overlap duplicate
    that boundary region in two chunks, so at least one chunk is likely to
    match the query embedding well and nothing critical is stranded on a hard cut.
"""

from __future__ import annotations

import re
from typing import List

# Tunable defaults (~500 chars, ~100 overlap per requirements)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
# Do not shrink a chunk below this when snapping to a natural break (keeps overlap meaningful)
_MIN_CHUNK_BEFORE_SOFT_BREAK = 200
# How far back from the tentative end to look for paragraph/sentence boundaries
_LOOKBACK_WINDOW = 160


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping segments of roughly *chunk_size* characters.

    Break priority (within the lookback window): blank line, single newline,
    sentence-like punctuation, then a hard cut at *chunk_size*.

    Empty or whitespace-only input yields an empty list.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0

    while start < len(normalized):
        min_end = min(start + _MIN_CHUNK_BEFORE_SOFT_BREAK, len(normalized))
        tentative_end = min(start + chunk_size, len(normalized))

        if tentative_end < len(normalized):
            split_at = _choose_split_end(
                normalized, start, tentative_end, min_end=min_end
            )
        else:
            split_at = len(normalized)

        piece = normalized[start:split_at].strip()
        if piece:
            chunks.append(piece)

        if split_at >= len(normalized):
            break

        next_start = split_at - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _choose_split_end(text: str, start: int, tentative_end: int, min_end: int) -> int:
    """
    Move the split point earlier (within lookback) to land on a semantic boundary.
    """
    look_from = max(min_end, tentative_end - _LOOKBACK_WINDOW)
    window = text[look_from:tentative_end]

    for separator in ("\n\n", "\n", ". ", "! ", "? ", "; "):
        rel = window.rfind(separator)
        if rel == -1:
            continue
        candidate = look_from + rel + len(separator)
        if candidate >= min_end:
            return candidate

    return tentative_end


def sanitize_index_id_component(name: str, max_len: int = 120) -> str:
    """Produce a safe Elasticsearch _id prefix from a filename."""
    safe = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    safe = safe.strip("_") or "doc"
    return safe[:max_len]
