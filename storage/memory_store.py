"""
SQLite-backed Memory Store — Phase 1
--------------------------------------
Concrete implementation of AbstractMemoryStore.

All memory records live in the `memories` table created by
storage.db._init_memory_schema().  Semantic search is delegated to a
TfIdfEmbeddingProvider instance that owns the memory_embeddings/*
tables in the same DB connection.

Thread safety:
  Each SqliteMemoryStore instance owns its SQLite connection.  The
  correct pattern (matching Phase 0 get_db()) is to create one store
  per thread via MemoryService.get_memory() → SqliteMemoryStore(get_db()).

Idempotency:
  put() is an upsert on memory.id — storing the same memory twice is
  safe and replaces the prior record.  The embedding provider follows
  the same idempotency guarantee.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from services.memory.base import AbstractMemoryStore, AbstractEmbeddingProvider
from services.memory.models import (
    Memory,
    MemoryMetadata,
    MemoryQuery,
    MemoryResult,
    MemorySource,
    MemoryType,
)
from storage.db import _init_memory_schema

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tags_to_str(tags: list[str]) -> str:
    return ",".join(t.strip() for t in tags if t.strip())


def _str_to_tags(s: str) -> list[str]:
    return [t for t in s.split(",") if t] if s else []


def _row_to_memory(row: tuple) -> Memory:
    """Convert a memories table row to a Memory object."""
    (
        mem_id, content, memory_type, source, importance, confidence,
        tags_str, plugin_source, vault_path, related_ids_str,
        created_at, updated_at,
    ) = row
    meta = MemoryMetadata(
        memory_type=MemoryType(memory_type),
        source=MemorySource(source),
        importance=float(importance),
        confidence=float(confidence),
        tags=_str_to_tags(tags_str),
        plugin_source=plugin_source or "",
        vault_path=vault_path or "",
        related_entity_ids=_str_to_tags(related_ids_str),
    )
    return Memory(
        id=mem_id,
        content=content,
        metadata=meta,
        created_at=created_at,
        updated_at=updated_at,
    )


class SqliteMemoryStore(AbstractMemoryStore):
    """SQLite-backed implementation of AbstractMemoryStore.

    Parameters
    ----------
    conn:
        An open SQLite connection.  Pass the same connection used by
        the rest of the DB layer (get_db()) so WAL-mode atomicity applies
        across notes + memories in the same transaction boundary.
    embedding_provider:
        An AbstractEmbeddingProvider instance used for semantic search.
        If None, text-based queries fall back to SQLite LIKE matching.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_provider: AbstractEmbeddingProvider | None = None,
    ) -> None:
        self._conn = conn
        self._embeddings = embedding_provider
        _init_memory_schema(conn)
        log.debug("SqliteMemoryStore initialised")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def put(self, memory: Memory) -> None:
        """Upsert a Memory record.  Also embeds the content if a provider
        is present.
        """
        meta = memory.metadata
        now = _now_iso()
        created_at = memory.created_at or now
        updated_at = memory.updated_at or now

        self._conn.execute(
            """
            INSERT INTO memories
                (id, content, memory_type, source, importance, confidence,
                 tags, plugin_source, vault_path, related_ids,
                 created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,
                memory_type=excluded.memory_type,
                source=excluded.source,
                importance=excluded.importance,
                confidence=excluded.confidence,
                tags=excluded.tags,
                plugin_source=excluded.plugin_source,
                vault_path=excluded.vault_path,
                related_ids=excluded.related_ids,
                updated_at=excluded.updated_at
            """,
            (
                memory.id,
                memory.content,
                meta.memory_type.value,
                meta.source.value,
                meta.importance,
                meta.confidence,
                _tags_to_str(meta.tags),
                meta.plugin_source,
                meta.vault_path,
                _tags_to_str(meta.related_entity_ids),
                created_at,
                updated_at,
            ),
        )
        self._conn.commit()

        if self._embeddings and memory.content.strip():
            self._embeddings.embed_and_store(memory.id, memory.content)
            self._embeddings.save()

        log.debug("Stored memory id=%s type=%s", memory.id, meta.memory_type.value)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update a memory's content and/or metadata fields.

        Only the fields explicitly provided in metadata_updates are changed.
        Recognised keys: importance, confidence, tags, memory_type, source,
        plugin_source, vault_path.
        """
        existing = self.get(memory_id)
        if existing is None:
            return None

        updates: dict[str, Any] = {}
        if content is not None:
            updates["content"] = content
        if metadata_updates:
            field_map = {
                "importance": "importance",
                "confidence": "confidence",
                "tags": None,       # handled specially
                "memory_type": "memory_type",
                "source": "source",
                "plugin_source": "plugin_source",
                "vault_path": "vault_path",
            }
            for key, col in field_map.items():
                if key in metadata_updates:
                    if key == "tags":
                        updates["tags"] = _tags_to_str(metadata_updates["tags"])
                    elif key == "memory_type":
                        updates["memory_type"] = MemoryType(metadata_updates[key]).value
                    elif key == "source":
                        updates["source"] = MemorySource(metadata_updates[key]).value
                    else:
                        updates[col] = metadata_updates[key]

        updates["updated_at"] = _now_iso()

        if updates:
            cols = ", ".join(f"{k}=?" for k in updates)
            vals = list(updates.values()) + [memory_id]
            self._conn.execute(
                f"UPDATE memories SET {cols} WHERE id=?", vals  # noqa: S608
            )
            self._conn.commit()

        # Re-embed if content changed
        new_content = updates.get("content", existing.content)
        if "content" in updates and self._embeddings and new_content.strip():
            self._embeddings.embed_and_store(memory_id, new_content)
            self._embeddings.save()

        return self.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.  Returns True if it existed."""
        row = self._conn.execute(
            "SELECT id FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.commit()
        if self._embeddings:
            self._embeddings.remove(memory_id)
        log.debug("Deleted memory id=%s", memory_id)
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Memory | None:
        """Fetch a single memory by ID."""
        row = self._conn.execute(
            "SELECT id, content, memory_type, source, importance, confidence, "
            "tags, plugin_source, vault_path, related_ids, created_at, updated_at "
            "FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        return _row_to_memory(row) if row else None

    def query(self, query: MemoryQuery) -> list[MemoryResult]:
        """Return memories matching the structured query.

        Order of operations:
          1. Apply SQL filters (type, source, importance, tags).
          2. If query.text is non-empty and an embedding provider is
             present, rank by cosine similarity and apply min_score.
          3. Limit to top_k.
        """
        # --- 1. Build SQL WHERE clause -----------------------------------
        conditions: list[str] = []
        params: list[Any] = []

        if query.memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(query.memory_type.value)

        if query.source is not None:
            conditions.append("source = ?")
            params.append(query.source.value)

        if query.min_importance > 0.0:
            conditions.append("importance >= ?")
            params.append(query.min_importance)

        for tag in query.tags:
            conditions.append("',' || tags || ',' LIKE ?")
            params.append(f"%,{tag},%")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            "SELECT id, content, memory_type, source, importance, confidence, "
            "tags, plugin_source, vault_path, related_ids, created_at, updated_at "
            f"FROM memories {where} ORDER BY importance DESC"  # noqa: S608
        )
        rows = self._conn.execute(sql, params).fetchall()
        candidates = [_row_to_memory(r) for r in rows]

        # --- 2. Semantic ranking -----------------------------------------
        if query.text.strip() and self._embeddings:
            scored = self._embeddings.similarity_search(
                query.text,
                top_k=len(candidates) or query.top_k,
                min_score=query.min_score,
            )
            score_map = {doc_id: score for doc_id, score in scored}
            results = [
                MemoryResult(memory=m, score=score_map.get(m.id, 0.0))
                for m in candidates
                if m.id in score_map or query.min_score == 0.0
            ]
            # Re-sort by score descending; fall back to importance
            results.sort(
                key=lambda r: (r.score, r.memory.metadata.importance),
                reverse=True,
            )
        else:
            results = [MemoryResult(memory=m, score=1.0) for m in candidates]

        return results[: query.top_k]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def make_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.NOTE,
        source: MemorySource = MemorySource.USER,
        importance: float = 0.5,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        plugin_source: str = "",
        vault_path: str = "",
    ) -> Memory:
        """Factory: create and store a new Memory, returning it."""
        now = _now_iso()
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            metadata=MemoryMetadata(
                memory_type=memory_type,
                source=source,
                importance=importance,
                confidence=confidence,
                tags=tags or [],
                plugin_source=plugin_source,
                vault_path=vault_path,
            ),
            created_at=now,
            updated_at=now,
        )
        self.put(mem)
        return mem
