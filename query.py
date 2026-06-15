from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import Elasticsearch

from utils.helpers import get_embedding
from utils.retrieval_context import (
    apply_char_budget,
    collect_neighbor_es_ids_for_hits,
    dominant_parent_id,
    es_chunk_document_id,
    neighbor_chunk_ids,
    parse_es_chunk_id,
    truncate_with_window,
)

es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "rag_index"

# Retrieve a small set of the best matching chunks (dense kNN is chunk-level).
DEFAULT_TOP_K = 5
# Hard cap on combined context characters to avoid oversized prompts / token blowups
# (Groq and similar chat models are sensitive to prompt size).
MAX_CONTEXT_CHARS = 10_000

# Same visual separator rag.py historically used when stitching chunk blocks.
CONTEXT_SEPARATOR = "\n\n---\n\n"


def _format_chunk_block(source: dict) -> str:
    """One retrieved chunk with light provenance for the LLM."""
    filename = source.get("source_filename") or source.get("source") or "unknown"
    parent = source.get("parent_id") or filename
    chunk_id = source.get("chunk_id")
    total = source.get("total_chunks")
    doc_type = source.get("doc_type") or ""

    meta_parts = [f"file={filename}", f"parent={parent}"]
    if chunk_id is not None and total is not None:
        meta_parts.append(f"chunk={int(chunk_id) + 1}/{int(total)}")
    if doc_type:
        meta_parts.append(f"type={doc_type}")

    subject = source.get("email_subject")
    sender = source.get("email_sender")
    if subject:
        meta_parts.append(f"subject={subject}")
    if sender:
        meta_parts.append(f"from={sender}")

    header = "[" + " | ".join(meta_parts) + "]"
    text = source.get("text") or ""
    return f"{header}\n{text}".strip()


def _format_full_document_block(source: dict, body: str) -> str:
    """Single high-signal block when a parent dominates retrieval."""
    filename = source.get("source_filename") or source.get("source") or "unknown"
    parent = source.get("parent_id") or filename
    doc_type = source.get("doc_type") or ""
    meta_parts = [f"file={filename}", f"parent={parent}", "mode=full_parent_context"]
    if doc_type:
        meta_parts.append(f"type={doc_type}")
    subject = source.get("email_subject")
    sender = source.get("email_sender")
    if subject:
        meta_parts.append(f"subject={subject}")
    if sender:
        meta_parts.append(f"from={sender}")
    header = "[" + " | ".join(meta_parts) + "]"
    return f"{header}\n{body}".strip()


def _knn_search(query: str, top_k: int) -> Dict[str, Any]:
    """Run kNN search; top_k is clamped to [3, 5]."""
    top_k = max(3, min(top_k, 5))
    vector = get_embedding(query)
    return es.search(
        index=INDEX_NAME,
        size=top_k,
        query={
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": max(50, top_k * 20),
            }
        },
    )


def _mget_sources_by_ids(es_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch-fetch ``_source`` payloads for chunk document ids."""
    out: Dict[str, Dict[str, Any]] = {}
    if not es_ids:
        return out
    resp = es.mget(index=INDEX_NAME, ids=es_ids)
    for doc in resp.get("docs", []):
        if doc.get("found") and doc.get("_source") is not None:
            out[str(doc.get("_id"))] = doc["_source"]
    return out


def _reconstruct_full_text_from_chunks(prefix: str, total_chunks: int, by_id: Dict[str, Dict[str, Any]]) -> str:
    """Fallback when older indices omit ``full_text`` (concatenate chunk bodies in order)."""
    parts: List[str] = []
    for i in range(int(total_chunks)):
        sid = f"{prefix}_{i}"
        src = by_id.get(sid) or {}
        t = (src.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def _assemble_neighbor_context(
    knn_hits: List[Dict[str, Any]],
    by_id: Dict[str, Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Expand each kNN hit with previous/current/next chunks, de-duplicate, and
    concatenate in kNN score order (best passages first, stable sub-order by chunk id).
    """
    ordered_blocks: List[str] = []
    seen_pair: set[Tuple[str, int]] = set()

    for hit in knn_hits:
        es_id = str(hit.get("_id") or "")
        src0 = hit.get("_source") or {}
        if not es_id:
            pid = src0.get("parent_id") or src0.get("source_filename") or src0.get("source")
            cid = src0.get("chunk_id")
            if pid is None or cid is None:
                continue
            es_id = es_chunk_document_id(str(pid), int(cid))
        try:
            prefix, cid_int = parse_es_chunk_id(es_id)
        except (ValueError, AttributeError):
            continue

        total = src0.get("total_chunks")
        try:
            total_int = int(total) if total is not None else None
        except (TypeError, ValueError):
            total_int = None

        for n in neighbor_chunk_ids(cid_int, total_int):
            pair = (prefix, int(n))
            if pair in seen_pair:
                continue
            seen_pair.add(pair)
            sid = f"{prefix}_{n}"
            src = by_id.get(sid) or {}
            if not src:
                continue
            ordered_blocks.append(_format_chunk_block(src))

    text = apply_char_budget(ordered_blocks, MAX_CONTEXT_CHARS, CONTEXT_SEPARATOR)
    return text, ordered_blocks


def get_llm_context(chunks: List[Dict[str, Any]], separator: str = CONTEXT_SEPARATOR) -> str:
    """
    Return the string passed to the LLM.

    Parent-aware modes attach a single pre-assembled ``shared_llm_context`` on the
    first chunk record; otherwise we join per-hit ``llm_block`` values (legacy path).
    """
    if chunks and str(chunks[0].get("shared_llm_context") or "").strip():
        return str(chunks[0]["shared_llm_context"])
    return separator.join(c["llm_block"] for c in chunks if c.get("llm_block"))


def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Return retrieved chunks with metadata for UI attribution and LLM context.

    Vector search still ranks **chunks**. Post-processing then chooses how to expand
    context for the LLM:

    1. If a **strict majority** of top hits share the same ``parent_id``, we promote
       a bounded **full parent** view (truncated around the best matching chunk) so
       resolutions split across chunks are reunited.

    2. Otherwise we apply **neighbor expansion** (previous / current / next chunk)
       for each hit, de-duplicated, then concatenated under ``MAX_CONTEXT_CHARS``.

    Each item includes Elasticsearch fields plus score, ``llm_block`` (the chunk's
    own formatted span for backwards-compatible callers), ``context_strategy`` on
    the first row, and optional ``shared_llm_context`` on the first row for
    ``get_llm_context()``.
    """
    response = _knn_search(query, top_k)
    knn_hits: List[Dict[str, Any]] = list(response.get("hits", {}).get("hits", []))

    results: List[Dict[str, Any]] = []
    for hit in knn_hits:
        src = hit.get("_source") or {}
        block = _format_chunk_block(src)
        results.append(_chunk_record(hit, src, block))

    if not results:
        return results

    maj_parent = dominant_parent_id(knn_hits)
    strategy = "chunks_only"
    shared: Optional[str] = None

    if maj_parent:
        # Pick the highest-scoring hit among those that belong to the dominant parent.
        best_hit: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        for hit in knn_hits:
            src = hit.get("_source") or {}
            pid = src.get("parent_id") or src.get("source_filename") or src.get("source")
            if str(pid) != str(maj_parent):
                continue
            sc = float(hit.get("_score") or 0.0)
            if sc >= best_score:
                best_score = sc
                best_hit = hit

        if best_hit:
            src = best_hit.get("_source") or {}
            full_text = (src.get("full_text") or "").strip()
            anchor = (src.get("text") or "").strip()

            if not full_text:
                es_id = str(best_hit.get("_id") or "")
                total = src.get("total_chunks")
                try:
                    total_int = int(total) if total is not None else 0
                except (TypeError, ValueError):
                    total_int = 0
                if es_id and total_int > 0:
                    prefix, _ = parse_es_chunk_id(es_id)
                    need = [f"{prefix}_{i}" for i in range(total_int)]
                    by_all = _mget_sources_by_ids(need)
                    full_text = _reconstruct_full_text_from_chunks(prefix, total_int, by_all)

            windowed = truncate_with_window(full_text, anchor, MAX_CONTEXT_CHARS)
            shared = _format_full_document_block(src, windowed)
            strategy = "full_document"

    if shared is None:
        neighbor_ids = collect_neighbor_es_ids_for_hits(knn_hits)
        by_id = _mget_sources_by_ids(neighbor_ids)
        shared, _blocks = _assemble_neighbor_context(knn_hits, by_id)
        strategy = "neighbor_expansion"

    if not str(shared or "").strip():
        shared = CONTEXT_SEPARATOR.join(c["llm_block"] for c in results)
        strategy = "chunks_only"
        maj_parent = None

    results[0]["context_strategy"] = strategy
    results[0]["dominant_parent_id"] = maj_parent if strategy == "full_document" else None
    results[0]["shared_llm_context"] = shared

    return results


def _chunk_record(hit: Dict[str, Any], src: Dict[str, Any], llm_block: str) -> Dict[str, Any]:
    return {
        "score": hit.get("_score"),
        "source_filename": src.get("source_filename") or src.get("source") or "unknown",
        "parent_id": src.get("parent_id") or src.get("source_filename") or src.get("source"),
        "chunk_id": src.get("chunk_id"),
        "total_chunks": src.get("total_chunks"),
        "doc_type": src.get("doc_type") or "",
        "text": src.get("text") or "",
        "email_subject": src.get("email_subject"),
        "email_sender": src.get("email_sender"),
        "llm_block": llm_block,
    }


def search(query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
    """
    kNN over chunk embeddings. Returns formatted strings for the LLM (parent-aware
    assembly when applicable).
    """
    chunks = retrieve_chunks(query, top_k)
    return [get_llm_context(chunks)] if chunks else []


def index_exists() -> bool:
    """Whether the configured index is present (Elasticsearch reachable and index created)."""
    try:
        return bool(es.indices.exists(index=INDEX_NAME))
    except Exception:
        return False
