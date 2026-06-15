"""
Parent-aware and neighbor-expanded context assembly for chunk-level vector RAG.

Why chunking alone hurts quality (especially email / support threads):
    Each chunk is embedded and retrieved in isolation. The model then sees only
    disconnected spans, so resolutions that span multiple chunks, late-reply
    clarifications, or "issue -> triage -> fix" narratives appear fragmented.
    Thread continuity (who said what, and the final outcome) is easy to lose.

Why parent-aware retrieval helps:
    When several top-k hits point to the same underlying document, that is strong
    evidence the user's answer lives in that document's global narrative—not only
    in one local window. Promoting the parent document (or a bounded expansion)
    restores cross-chunk dependencies while keeping prompts within token limits.

Why neighbor expansion helps:
    Even when no single parent dominates, the highest-scoring chunk is often
    anchored near a semantic boundary. Pulling the previous and next chunk
    reattaches local continuity (e.g. problem statement + resolution split across
    cuts) without shipping entire mailboxes into the prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.chunking import sanitize_index_id_component


def es_chunk_document_id(parent_id: str, chunk_id: int) -> str:
    """Elasticsearch _id used at index time: ``{sanitized_parent}_{chunk_id}``."""
    return f"{sanitize_index_id_component(parent_id)}_{int(chunk_id)}"


def parse_es_chunk_id(es_id: str) -> Tuple[str, int]:
    """Split ``es_id`` into (id_prefix, chunk_index). The prefix may contain ``_``."""
    prefix, last = es_id.rsplit("_", 1)
    return prefix, int(last)


def neighbor_chunk_ids(chunk_id: int, total_chunks: Optional[int]) -> List[int]:
    """
    Indices for previous, current, and next chunk (0-based).

    If *total_chunks* is unknown, the successor id is still requested; Elasticsearch
    ``mget`` simply omits missing documents so we do not guess document length.
    """
    cid = int(chunk_id)
    out = {cid}
    if cid - 1 >= 0:
        out.add(cid - 1)
    if total_chunks is None:
        out.add(cid + 1)
    else:
        hi = int(total_chunks) - 1
        if hi >= 0 and cid + 1 <= hi:
            out.add(cid + 1)
    return sorted(out)


def dominant_parent_id(hits: Sequence[Dict[str, Any]]) -> Optional[str]:
    """
    Return *parent_id* if a strict majority of hits share it; else ``None``.

    Example: 5 hits -> need >= 3 sharing the same parent.
    """
    if not hits:
        return None
    parents: List[str] = []
    for h in hits:
        src = h.get("_source") or {}
        pid = src.get("parent_id") or src.get("source_filename") or src.get("source")
        if pid:
            parents.append(str(pid))
    if not parents:
        return None
    counts = Counter(parents)
    winner, count = counts.most_common(1)[0]
    if count * 2 > len(hits):
        return winner
    return None


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def truncate_with_window(
    full_text: str,
    anchor_substring: str,
    max_chars: int,
) -> str:
    """
    If *full_text* fits *max_chars*, return it.

    Otherwise prefer a window centered on the first raw-text occurrence of a short
    literal prefix of *anchor_substring* (the best-matching chunk). If no anchor is
    found, bias toward the document start (many tickets lead with the problem statement).
    """
    if max_chars <= 0:
        return ""
    hay = full_text or ""
    if len(hay) <= max_chars:
        return hay

    probe = (anchor_substring or "").strip()[:120]
    raw_pos = -1
    for n in (120, 80, 48):
        p = probe[:n] if len(probe) >= n else probe
        if len(p) < 16:
            break
        raw_pos = hay.find(p)
        if raw_pos != -1:
            break
    if raw_pos == -1:
        raw_pos = 0

    half = max_chars // 2
    start = max(0, raw_pos - half)
    end = min(len(hay), start + max_chars)
    start = max(0, end - max_chars)
    window = hay[start:end].strip()
    if start > 0:
        window = "[... truncated above ...]\n\n" + window
    if end < len(hay):
        window = window + "\n\n[... truncated below ...]"
    return window


def apply_char_budget(blocks: List[str], max_chars: int, separator: str) -> str:
    """
    Concatenate *blocks* in order until *max_chars* is reached.

    Drops from the end first (lowest priority if callers pass best-first).
    """
    if max_chars <= 0:
        return ""
    sep_len = len(separator)
    out: List[str] = []
    used = 0
    for b in blocks:
        piece = (b or "").strip()
        if not piece:
            continue
        add = len(piece) if not out else sep_len + len(piece)
        if used + add <= max_chars:
            out.append(piece)
            used += add
            continue
        remaining = max_chars - used - (sep_len if out else 0)
        if remaining > 200:
            clipped = piece[:remaining].rstrip() + "\n[...truncated]"
            out.append(clipped)
        break
    return separator.join(out)


def collect_neighbor_es_ids_for_hits(hits: Sequence[Dict[str, Any]]) -> List[str]:
    """Unique Elasticsearch document ids for neighbor-expanded windows."""
    seen: set[str] = set()
    ordered: List[str] = []
    for h in hits:
        es_id = h.get("_id")
        src = h.get("_source") or {}
        if not es_id:
            pid = src.get("parent_id") or src.get("source_filename") or src.get("source")
            cid = src.get("chunk_id")
            if pid is None or cid is None:
                continue
            es_id = es_chunk_document_id(str(pid), int(cid))
        try:
            prefix, cid_int = parse_es_chunk_id(str(es_id))
        except (ValueError, AttributeError):
            continue
        total = src.get("total_chunks")
        if total is not None:
            try:
                total_int: Optional[int] = int(total)
            except (TypeError, ValueError):
                total_int = None
        else:
            total_int = None
        for n in neighbor_chunk_ids(cid_int, total_int):
            nid = f"{prefix}_{n}"
            if nid not in seen:
                seen.add(nid)
                ordered.append(nid)
    return ordered
