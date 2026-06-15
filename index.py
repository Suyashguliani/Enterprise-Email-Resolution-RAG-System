"""
Elasticsearch indexing for chunk-level dense retrieval with **parent-aware** fields.

Each chunk document stores both the local ``text`` (for precise kNN) and
``full_text`` (the entire source email or .txt file). At query time, when many
top hits share the same ``parent_id``, the retriever can promote the whole parent
body so thread-level context is not lost to fragmentation across chunks.
"""

from email import policy
from email.parser import BytesParser
import os

from elasticsearch import Elasticsearch

from utils.chunking import chunk_text, sanitize_index_id_component
from utils.helpers import get_embedding

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "rag_index"


def load_documents(folder_path="data"):
    """
    Load .txt and .eml files as logical documents (full text + metadata).
    Chunking and embedding happen in index_data().
    """
    docs = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            docs.append(
                {
                    "source_filename": file,
                    "text": text,
                    "doc_type": ".txt",
                    "email_subject": None,
                    "email_sender": None,
                }
            )

        elif file.endswith(".eml"):
            with open(path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)

            subject = msg["subject"] or ""
            sender = msg["from"] or ""

            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()

                    if content_type == "text/plain":
                        body += part.get_content()
            else:
                body = msg.get_content()

            full_text = f"""
Subject: {subject}

From: {sender}

Body:
{body}
"""

            docs.append(
                {
                    "source_filename": file,
                    "text": full_text.strip(),
                    "doc_type": ".eml",
                    "email_subject": subject or None,
                    "email_sender": sender or None,
                }
            )

    return docs


def create_index():
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "text": {"type": "text"},
                # Back-compat: original field kept as the display filename
                "source": {"type": "keyword"},
                "source_filename": {"type": "keyword"},
                # Logical parent key for grouping (typically the source filename).
                "parent_id": {"type": "keyword"},
                # Entire original document/email body (stored for parent-aware retrieval).
                # Not indexed as text to avoid bloating the inverted index; kNN uses chunk vectors.
                "full_text": {"type": "text", "index": False},
                "chunk_id": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "doc_type": {"type": "keyword"},
                "email_subject": {"type": "text"},
                "email_sender": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )


def index_data():
    docs = load_documents("data")
    indexed_chunks = 0

    for doc in docs:
        text = doc["text"]
        if not text:
            continue

        pieces = chunk_text(text)
        total = len(pieces)
        id_base = sanitize_index_id_component(doc["source_filename"])
        parent_id = doc["source_filename"]

        for chunk_id, chunk_body in enumerate(pieces):
            vector = get_embedding(chunk_body)
            es_id = f"{id_base}_{chunk_id}"

            body = {
                "source": doc["source_filename"],
                "source_filename": doc["source_filename"],
                "parent_id": parent_id,
                "chunk_id": chunk_id,
                "total_chunks": total,
                "doc_type": doc["doc_type"],
                "text": chunk_body,
                "full_text": text,
                "embedding": vector,
            }

            if doc["email_subject"] is not None:
                body["email_subject"] = doc["email_subject"]
            if doc["email_sender"] is not None:
                body["email_sender"] = doc["email_sender"]

            es.index(index=INDEX_NAME, id=es_id, document=body)
            indexed_chunks += 1

    print(f"Indexed {indexed_chunks} chunks from {len(docs)} source files")


def rebuild_index():
    """
    Recreate the Elasticsearch index and ingest all documents from the data folder.
    Used by the Streamlit admin action and matches `python index.py` behavior.
    """
    create_index()
    index_data()


if __name__ == "__main__":
    rebuild_index()
    print("Indexing done!")
