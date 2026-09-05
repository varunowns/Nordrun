"""
Embedding Provider — Phase 1
------------------------------
Concrete implementation of AbstractEmbeddingProvider that wraps the
existing EmbeddingIndex (TF-IDF + SQLite) introduced in Phase 0.

Why not add sentence-transformers or OpenAI embeddings yet?
  The existing TF-IDF index already works, is zero-dependency, and
  passes 163 tests.  The AbstractEmbeddingProvider interface means
  swapping to a real semantic model later is a one-line change in
  MemoryService — no other code needs to change.

Usage:
    from services.memory.embedding import TfIdfEmbeddingProvider
    provider = TfIdfEmbeddingProvider(conn=some_sqlite_conn)
    vec = provider.embed("machine learning projects")
    provider.embed_and_store("mem-uuid", "machine learning projects")
    provider.save()
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

from services.memory.base import AbstractEmbeddingProvider

log = logging.getLogger(__name__)

# Separate config key so the memory vectorizer state doesn't collide
# with the note EmbeddingIndex stored under "tfidf_vectorizer".
_MEMORY_CONFIG_KEY = "memory_tfidf_vectorizer"


class TfIdfEmbeddingProvider(AbstractEmbeddingProvider):
    """Embedding provider backed by the project's existing TF-IDF vectorizer.

    Uses a dedicated set of tables (memory_embeddings, memory_embedding_config,
    memory_doc_tokens) so memory embeddings never collide with note embeddings
    that live in the embeddings / embedding_config / doc_tokens tables.

    Thread safety: this object owns a SQLite connection that must be the
    same connection used by the calling MemoryStore.  Do not share a single
    TfIdfEmbeddingProvider across threads; create one per thread (matching
    the thread-local get_db() pattern used everywhere in Nordrun).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._init_schema()
        # Import here to avoid a circular import at module load time.
        from services.embedding_service import _TfIdfVectorizer
        self._vectorizer = self._load_vectorizer()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                doc_id     TEXT PRIMARY KEY,
                vector     BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embedding_config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_doc_tokens (
                doc_id TEXT PRIMARY KEY,
                tokens TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Vectorizer persistence (mirrors EmbeddingIndex._save/_load)
    # ------------------------------------------------------------------

    def _load_vectorizer(self):
        from services.embedding_service import _TfIdfVectorizer
        row = self._conn.execute(
            "SELECT value FROM memory_embedding_config WHERE key = ?",
            (_MEMORY_CONFIG_KEY,),
        ).fetchone()
        vectorizer = _TfIdfVectorizer.from_json(row[0]) if row else _TfIdfVectorizer()
        docs = {
            doc_id: set(tokens.split(",")) if tokens else set()
            for doc_id, tokens in self._conn.execute(
                "SELECT doc_id, tokens FROM memory_doc_tokens"
            ).fetchall()
        }
        vectorizer.set_corpus(docs)
        return vectorizer

    def _save_vectorizer(self) -> None:
        from services.embedding_service import _TfIdfVectorizer
        blob = self._vectorizer.to_json()
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_embedding_config (key, value) VALUES (?, ?)",
            (_MEMORY_CONFIG_KEY, blob),
        )
        self._conn.execute("DELETE FROM memory_doc_tokens")
        self._conn.executemany(
            "INSERT INTO memory_doc_tokens (doc_id, tokens) VALUES (?, ?)",
            [
                (doc_id, ",".join(sorted(tokens)))
                for doc_id, tokens in self._vectorizer.all_docs().items()
            ],
        )
        self._conn.commit()

    def _rebuild_vectors(self) -> None:
        """Recompute all stored embedding vectors using the current vocabulary.

        Called by save() so that every vector in memory_embeddings has the
        same dimension as the current vocabulary.  This prevents the
        shape-mismatch skip in similarity_search() that hides documents
        indexed before the vocabulary finished growing.
        """
        all_docs = self._vectorizer.all_docs()
        if not all_docs:
            return
        updates = []
        for doc_id, tokens in all_docs.items():
            vector = self._vectorizer.vector_for_tokens(tokens)
            updates.append((vector.astype(np.float32).tobytes(), doc_id))
        self._conn.executemany(
            "UPDATE memory_embeddings SET vector=? WHERE doc_id=?", updates
        )
        self._conn.commit()
        log.debug("Rebuilt %d memory embedding vectors after vocab expansion", len(updates))

    # ------------------------------------------------------------------
    # AbstractEmbeddingProvider implementation
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Return a TF-IDF vector for `text` without updating the corpus."""
        return self._vectorizer.transform(text)

    def embed_and_store(self, doc_id: str, text: str) -> np.ndarray:
        """Embed `text`, associate it with `doc_id`, persist the vector."""
        vector = self._vectorizer.add_document(doc_id, text)
        vector_bytes = vector.astype(np.float32).tobytes()
        tokens = self._vectorizer.all_docs().get(doc_id, set())
        self._conn.execute(
            """
            INSERT INTO memory_embeddings (doc_id, vector, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(doc_id) DO UPDATE SET
                vector=excluded.vector,
                updated_at=excluded.updated_at
            """,
            (doc_id, vector_bytes),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_doc_tokens (doc_id, tokens) VALUES (?, ?)",
            (doc_id, ",".join(sorted(tokens))),
        )
        self._conn.commit()
        log.debug("Embedded memory doc_id=%s", doc_id)
        return vector

    def remove(self, doc_id: str) -> None:
        """Remove `doc_id` from the embedding store."""
        self._vectorizer.remove_document(doc_id)
        self._conn.execute("DELETE FROM memory_embeddings WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM memory_doc_tokens WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        log.debug("Removed memory embedding doc_id=%s", doc_id)

    def save(self) -> None:
        """Flush the vectorizer vocabulary to the DB and rebuild all stored
        vectors so every embedding is in the current (final) vocabulary space.

        TF-IDF vocabularies grow monotonically as documents are added, so
        a vector stored after the first document is indexed has fewer
        dimensions than one stored after the tenth.  similarity_search
        skips dimension-mismatched vectors, causing early-indexed documents
        to disappear from results.

        Rebuilding all vectors on save() is cheap (in-memory numpy ops on
        the already-computed token sets) and matches the behaviour of
        EmbeddingIndex.rebuild_from_tokens() in services/embedding_service.py.
        """
        self._save_vectorizer()
        self._rebuild_vectors()

    # ------------------------------------------------------------------
    # Search helper (used by SqliteMemoryStore)
    # ------------------------------------------------------------------

    def similarity_search(
        self, query_text: str, top_k: int = 10, min_score: float = 0.0
    ) -> list[tuple[str, float]]:
        """Return (doc_id, cosine_score) pairs, sorted by score descending.

        Returns an empty list when the query has no vocabulary overlap
        with the corpus (zero vector).
        """
        query_vec = self._vectorizer.transform(query_text)
        if not np.any(query_vec):
            return []

        rows = self._conn.execute(
            "SELECT doc_id, vector FROM memory_embeddings"
        ).fetchall()

        scores: list[tuple[str, float]] = []
        for doc_id, blob in rows:
            if not blob:
                continue
            doc_vec = np.frombuffer(blob, dtype=np.float32)
            if doc_vec.shape != query_vec.shape:
                continue
            score = float(np.dot(query_vec, doc_vec))
            if score >= min_score:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
