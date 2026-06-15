"""
Streamlit frontend for the enterprise RAG support assistant.

Architecture:
    - User messages and assistant replies live in ``st.session_state["messages"]``.
    - Retrieval + LLM run once per question via ``rag.run_rag()`` (same pipeline as
      ``rag_pipeline()`` but returns sources/chunks for attribution).
    - Re-indexing calls ``index.rebuild_index()`` (equivalent to ``python index.py``).
    - Uploads write only ``.txt`` / ``.eml`` files into the project ``data/`` folder.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime import exists as streamlit_runtime_exists
from elasticsearch import ApiError, ConnectionError as ESConnectionError, NotFoundError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import RateLimitError as OpenAIRateLimitError

import index as index_module
from query import index_exists
from rag import run_rag

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
ALLOWED_EXTENSIONS = {".txt", ".eml"}

SK_MESSAGES = "messages"
SK_PENDING = "pending_generation"
SK_LAST_ERROR = "last_error"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _init_session_state() -> None:
    # setdefault avoids KeyError on first access when keys were never written.
    st.session_state.setdefault(SK_MESSAGES, [])
    st.session_state.setdefault(SK_PENDING, False)
    st.session_state.setdefault(SK_LAST_ERROR, None)


def _clear_chat() -> None:
    st.session_state[SK_MESSAGES] = []
    st.session_state[SK_PENDING] = False
    st.session_state[SK_LAST_ERROR] = None


def _scroll_chat_to_bottom() -> None:
    """Best-effort scroll; Streamlit iframes may restrict parent access."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const main = doc.querySelector('section.main');
        if (main) {
            main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
        }
        </script>
        """,
        height=0,
        width=0,
    )


def _friendly_rag_error(exc: Exception) -> str:
    """Map technical failures to support-friendly copy."""
    if isinstance(exc, (ESConnectionError, ConnectionRefusedError, TimeoutError)):
        return (
            "Could not reach Elasticsearch. Confirm the cluster is running "
            "(for example `http://localhost:9200`) and try again."
        )
    if isinstance(exc, NotFoundError):
        return (
            "The search index is missing. Upload documents and choose "
            "**Re-index documents** in the sidebar."
        )
    if isinstance(exc, ApiError) and getattr(exc, "meta", None) is not None:
        status = getattr(exc.meta, "status", None)
        if status == 404:
            return (
                "The search index is missing. Upload documents and choose "
                "**Re-index documents** in the sidebar."
            )

    if isinstance(exc, (OpenAIAPIStatusError, OpenAIAPIConnectionError, OpenAIRateLimitError)):
        code = getattr(exc, "status_code", None)
        if code in (401, 403):
            return "The LLM API rejected the request (auth). Check API keys and permissions."
        if code == 429:
            return "The LLM API rate limit was hit. Wait a moment and try again."
        if isinstance(exc, OpenAIAPIConnectionError):
            return "Could not reach the LLM API. Check network, base URL, and firewall settings."

    body = str(exc).lower()
    if "connection" in body or "timeout" in body or "refused" in body:
        return "A network error occurred. Check Elasticsearch and LLM API connectivity."

    return (
        "Something went wrong while generating a response. "
        "If this persists, contact your platform administrator."
    )


def _save_uploaded_files(files: List[Any], reindex_after: bool) -> int:
    """Persist Streamlit UploadedFile objects to ``data/``. Returns count saved."""
    _ensure_data_dir()
    count = 0
    for uploaded in files:
        name = Path(uploaded.name).name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            continue
        dest = DATA_DIR / name
        dest.write_bytes(uploaded.getbuffer())
        count += 1
    if reindex_after and count:
        index_module.rebuild_index()
    return count


def _render_sidebar() -> None:
    st.sidebar.markdown("### Support Insights")
    st.sidebar.caption("Enterprise knowledge assistant · email & document RAG")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "Ask questions about historical **.txt** notes and **.eml** threads. "
        "Answers combine retrieved chunks with parent-aware context when one thread dominates."
    )
    st.sidebar.markdown("---")

    if not index_exists():
        st.sidebar.warning("Search index not found. Add files to **data/** and re-index.")

    st.sidebar.markdown("**Supported types**")
    st.sidebar.markdown("- `.txt` plain text\n- `.eml` email archives")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Upload knowledge**")
    uploads = st.sidebar.file_uploader(
        "Drop files here",
        type=["txt", "eml"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    reindex_after_upload = st.sidebar.checkbox(
        "Re-index immediately after upload",
        value=True,
        help="Recreates the index and re-embeds all files in data/.",
    )

    if st.sidebar.button("Save uploads to library", use_container_width=True):
        if not uploads:
            st.sidebar.error("Choose one or more files first.")
        else:
            try:
                n = _save_uploaded_files(uploads, reindex_after=reindex_after_upload)
                if n:
                    st.sidebar.success(f"Saved {n} file(s) to `{DATA_DIR.name}/`.")
                    st.toast("Upload complete", icon="✅")
                else:
                    st.sidebar.error("No valid .txt or .eml files in the selection.")
            except Exception as exc:  # noqa: BLE001 — surface any IO/ES failure
                st.sidebar.error(_friendly_rag_error(exc))
                st.session_state[SK_LAST_ERROR] = traceback.format_exc()

    st.sidebar.markdown("---")
    if st.sidebar.button("Re-index documents", type="primary", use_container_width=True):
        with st.spinner("Rebuilding Elasticsearch index and embeddings…"):
            try:
                index_module.rebuild_index()
                st.sidebar.success("Index rebuilt successfully.")
                st.toast("Indexing complete", icon="✅")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(_friendly_rag_error(exc))
                st.session_state[SK_LAST_ERROR] = traceback.format_exc()

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear conversation", use_container_width=True):
        _clear_chat()
        st.toast("Chat cleared", icon="🧹")
        st.rerun()


def _render_welcome() -> None:
    st.markdown("# Support Insights")
    st.markdown(
        "Welcome. Ask about past incidents, resolutions, and ownership using your "
        "indexed **emails** and **text** documents. Responses cite retrieved sources below each answer."
    )
    st.info("Tip: upload `.txt` / `.eml` files in the sidebar, then re-index if you skipped auto re-index.")


def _render_sources_and_chunks(
    sources: List[str],
    chunks: List[Dict[str, Any]],
    key_prefix: str,
) -> None:
    if sources:
        st.caption("Sources: " + ", ".join(f"`{s}`" for s in sources))
    if chunks:
        mode = chunks[0].get("context_strategy")
        if mode:
            st.caption(f"LLM context assembly: **{mode}**")
        with st.expander("Retrieved chunks (for transparency)", expanded=False):
            for i, c in enumerate(chunks, start=1):
                title = (
                    f"{i}. {c.get('source_filename', 'unknown')} "
                    f"(chunk {int(c.get('chunk_id', 0)) + 1}/{c.get('total_chunks', '?')})"
                )
                st.markdown(f"**{title}**")
                st.text_area(
                    "Chunk text",
                    value=c.get("text") or "",
                    height=160,
                    key=f"{key_prefix}_chunk_{i}",
                    label_visibility="collapsed",
                )


def _render_message(m: Dict[str, Any]) -> None:
    role = m["role"]
    with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🛟"):
        st.markdown(m.get("content") or "")
        if role == "assistant":
            _render_sources_and_chunks(
                m.get("sources") or [],
                m.get("chunks") or [],
                key_prefix=f"hist_{m.get('_id', 'm')}",
            )


def _append_assistant_from_rag(user_text: str) -> None:
    """Run retrieval + LLM and push an assistant message (or error) onto session."""
    with st.chat_message("assistant", avatar="🛟"):
        try:
            with st.spinner("Retrieving relevant passages and drafting an answer…"):
                out = run_rag(user_text)
            answer = out.get("answer") or ""
            st.markdown(answer)
            sources = out.get("sources") or []
            chunks = out.get("chunks") or []
            _render_sources_and_chunks(sources, chunks, key_prefix="pending")
            st.session_state[SK_MESSAGES].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks": chunks,
                    "_id": len(st.session_state[SK_MESSAGES]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            msg = _friendly_rag_error(exc)
            st.error(msg)
            st.session_state[SK_MESSAGES].append(
                {
                    "role": "assistant",
                    "content": msg,
                    "sources": [],
                    "chunks": [],
                    "_id": len(st.session_state[SK_MESSAGES]),
                }
            )
            st.session_state[SK_LAST_ERROR] = traceback.format_exc()


def main() -> None:
    # ``python app.py`` does not start Streamlit's Runtime; session_state and widgets
    # only work under ``streamlit run`` (see Streamlit docs on "raw mode").
    if not streamlit_runtime_exists():
        import sys

        sys.stderr.write(
            "\nThis app must be started with Streamlit, not plain Python:\n\n"
            "    streamlit run app.py\n\n"
            "Session state and the chat UI require the Streamlit server.\n\n"
        )
        raise SystemExit(1)

    st.set_page_config(
        page_title="Support Insights",
        page_icon="🛟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.25rem; max-width: 1200px; }
            div[data-testid="stSidebarUserContent"] { padding-top: 1rem; }
            h1 { font-weight: 600; letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _ensure_data_dir()
    _init_session_state()
    _render_sidebar()

    messages: List[Dict[str, Any]] = st.session_state[SK_MESSAGES]

    if st.session_state.get(SK_PENDING):
        st.session_state[SK_PENDING] = False
        last_user = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        if last_user:
            _append_assistant_from_rag(last_user)
            _scroll_chat_to_bottom()
            st.rerun()

    if not messages:
        _render_welcome()

    for m in messages:
        _render_message(m)

    if prompt := st.chat_input("Describe the issue or question for support…"):
        st.session_state[SK_MESSAGES].append({"role": "user", "content": prompt})
        st.session_state[SK_PENDING] = True
        st.rerun()

    _scroll_chat_to_bottom()

    if st.session_state.get(SK_LAST_ERROR) and st.checkbox("Show technical details (admin)", value=False):
        st.code(st.session_state[SK_LAST_ERROR])


if __name__ == "__main__":
    main()
