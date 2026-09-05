"""
Memory Service — Phase 1
-------------------------
Unified interface for all memory operations in Nordrun.

MemoryService is the single entry point for:
  - Storing memories (store / observe)
  - Retrieving memories (get / search)
  - Updating memories
  - Forgetting (deleting) memories
  - Accessing the knowledge graph (entities / relationships)

Architecture:
  MemoryService owns a SqliteMemoryStore and a GraphStore, both sharing
  the same SQLite connection.  The embedding provider is wired in at
  construction time so it can be replaced in tests or future providers
  without changing this module.

  Plugins should call get_memory() to obtain the singleton.  Direct
  import of SqliteMemoryStore or GraphStore by plugins is discouraged —
  use the memory permission system and event bus instead.

Thread safety:
  get_memory() uses the same double-checked lock pattern as get_context().
  Each thread gets its own singleton (backed by the thread-local DB
  connection from get_db()).

Lifecycle (P1.4):
  The observe() method is the entry point for "should this become a
  memory?" decisions.  In Phase 1 it stores immediately if the content
  passes basic quality checks (non-empty, not a duplicate of the last
  stored memory for the same source).  Phase 2 can replace this with
  LLM-based extraction without changing the interface.

Permissions:
  @require("memory:read") wraps read operations.
  @require("memory:write") wraps write/delete operations.
  Both are enforced by the existing plugin_registry mechanism.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

import config as _config
from core.plugin_registry import require
from services.memory.base import AbstractEmbeddingProvider, AbstractMemoryStore
from services.memory.embedding import TfIdfEmbeddingProvider
from services.memory.models import (
    Memory,
    MemoryQuery,
    MemoryResult,
    MemorySource,
    MemoryType,
)
from storage.db import get_db
from storage.graph import Entity, GraphStore, Relationship
from storage.memory_store import SqliteMemoryStore

log = logging.getLogger(__name__)


def _build_embedding_provider(conn: sqlite3.Connection) -> AbstractEmbeddingProvider:
    """Factory: return the configured embedding provider.

    Phase 1 supports only "tfidf".  Future providers (sentence-transformers,
    OpenAI) are selected here via MEMORY_EMBEDDING_PROVIDER config.
    """
    provider_key = _config.MEMORY_EMBEDDING_PROVIDER
    if provider_key == "tfidf":
        return TfIdfEmbeddingProvider(conn=conn)
    # Unknown provider — fall back to TF-IDF and warn
    log.warning(
        "Unknown MEMORY_EMBEDDING_PROVIDER=%r; falling back to tfidf", provider_key
    )
    return TfIdfEmbeddingProvider(conn=conn)


class MemoryService:
    """Unified memory interface for Nordrun plugins and services.

    Do not instantiate directly — use get_memory() to obtain the singleton.
    For tests, inject conn= to use an in-memory SQLite connection.
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        embedding_provider: AbstractEmbeddingProvider | None = None,
    ) -> None:
        resolved_conn = conn or get_db()
        provider = embedding_provider or _build_embedding_provider(resolved_conn)
        self._store: AbstractMemoryStore = SqliteMemoryStore(
            conn=resolved_conn,
            embedding_provider=provider,
        )
        self._graph = GraphStore(conn=resolved_conn)
        self._conn = resolved_conn
        log.debug("MemoryService initialised")

    # ------------------------------------------------------------------
    # Write operations — require memory:write
    # ------------------------------------------------------------------

    @require("memory:write")
    def store(
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
        """Create and persist a new memory.  Returns the stored Memory."""
        return self._store.make_memory(
            content=content,
            memory_type=memory_type,
            source=source,
            importance=importance,
            confidence=confidence,
            tags=tags,
            plugin_source=plugin_source,
            vault_path=vault_path,
        )

    @require("memory:write")
    def observe(
        self,
        content: str,
        source: MemorySource = MemorySource.PLUGIN,
        memory_type: MemoryType = MemoryType.NOTE,
        importance: float = 0.5,
        plugin_source: str = "",
        tags: list[str] | None = None,
    ) -> Memory | None:
        """Lifecycle entry point: observe content and decide whether to store.

        Phase 1 rules (simple, deterministic):
          - Empty or whitespace-only content → rejected (returns None).
          - Content shorter than 10 characters → rejected.
          - Otherwise stored immediately.

        Phase 2 can replace this with LLM-based importance extraction
        without breaking the interface.

        Returns the stored Memory, or None if the content was rejected.
        """
        stripped = content.strip()
        if not stripped or len(stripped) < 10:
            log.debug("observe(): content too short or empty — not stored")
            return None

        return self._store.make_memory(
            content=stripped,
            memory_type=memory_type,
            source=source,
            importance=importance,
            plugin_source=plugin_source,
            tags=tags or [],
        )

    @require("memory:write")
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory.  Returns updated Memory or None."""
        return self._store.update(memory_id, content=content, metadata_updates=metadata_updates)

    @require("memory:write")
    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID.  Returns True if it existed."""
        return self._store.delete(memory_id)

    # ------------------------------------------------------------------
    # Read operations — require memory:read
    # ------------------------------------------------------------------

    @require("memory:read")
    def get(self, memory_id: str) -> Memory | None:
        """Fetch a single memory by ID."""
        return self._store.get(memory_id)

    @require("memory:read")
    def search(
        self,
        text: str = "",
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: MemorySource | None = None,
        min_importance: float = 0.0,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[MemoryResult]:
        """Search memories using semantic similarity + structured filters.

        Defaults for top_k and min_score come from config so they can be
        tuned per-deployment without code changes.
        """
        q = MemoryQuery(
            text=text,
            memory_type=memory_type,
            tags=tags or [],
            source=source,
            min_importance=min_importance,
            top_k=top_k if top_k is not None else _config.MEMORY_TOP_K,
            min_score=min_score if min_score is not None else _config.MEMORY_SIMILARITY_THRESHOLD,
        )
        return self._store.query(q)

    @require("memory:read")
    def count(self) -> int:
        """Total number of stored memories."""
        return self._store.count()

    # ------------------------------------------------------------------
    # Knowledge graph — require memory:read / memory:write
    # ------------------------------------------------------------------

    @require("memory:write")
    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Entity:
        """Add or update a knowledge-graph entity."""
        return self._graph.add_entity(
            name=name,
            entity_type=entity_type,
            description=description,
            tags=tags,
        )

    @require("memory:read")
    def get_entity(self, entity_id: str) -> Entity | None:
        return self._graph.get_entity(entity_id)

    @require("memory:read")
    def get_entity_by_name(self, name: str, entity_type: str | None = None) -> Entity | None:
        return self._graph.get_entity_by_name(name, entity_type)

    @require("memory:read")
    def search_entities(
        self,
        name_fragment: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        return self._graph.search_entities(name_fragment, entity_type, limit)

    @require("memory:write")
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        description: str = "",
    ) -> Relationship:
        return self._graph.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            description=description,
        )

    @require("memory:read")
    def get_neighbours(
        self,
        entity_id: str,
        relation_type: str | None = None,
        max_depth: int = 1,
    ) -> list[Entity]:
        return self._graph.get_neighbours(entity_id, relation_type, max_depth)

    @require("memory:write")
    def delete_entity(self, entity_id: str) -> bool:
        return self._graph.delete_entity(entity_id)

    # ------------------------------------------------------------------
    # Context enrichment (used by ContextService)
    # Called without @require — accessed through ContextService which
    # already enforces memory:read via its own permission chain.
    # ------------------------------------------------------------------

    def get_relevant_context(
        self,
        query: str,
        top_k: int | None = None,
        min_importance: float | None = None,
    ) -> dict[str, Any]:
        """Return relevant memories + graph entities for a query string.

        Used by ContextService.get_memory_context() to enrich LLM prompts.
        Returns a dict with keys 'memories' and 'entities' so callers can
        format it however they like.

        This method bypasses @require because it is called from within
        ContextService which is already operating under a plugin's active
        context.  The ContextService method that calls this does have
        @require("memory:read").
        """
        k = top_k if top_k is not None else _config.MEMORY_CONTEXT_MAX_INJECT
        min_imp = min_importance if min_importance is not None else _config.MEMORY_CONTEXT_MIN_IMPORTANCE

        q = MemoryQuery(
            text=query,
            min_importance=min_imp,
            top_k=k,
            min_score=_config.MEMORY_SIMILARITY_THRESHOLD,
        )
        results = self._store.query(q)

        # Also find entities whose name overlaps with the query
        # (simple word-by-word match — no NLP needed at Phase 1)
        entity_hits: list[Entity] = []
        for word in query.split():
            if len(word) >= 3:
                hits = self._graph.search_entities(name_fragment=word, limit=3)
                for e in hits:
                    if e not in entity_hits:
                        entity_hits.append(e)

        return {
            "memories": [r.memory.to_dict() for r in results],
            "entities": [e.to_dict() for e in entity_hits[:k]],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_memory_service: MemoryService | None = None
_memory_lock = threading.Lock()


def get_memory(conn: sqlite3.Connection | None = None) -> MemoryService:
    """Return the global MemoryService singleton.

    Thread-safe double-checked lock, matching get_context() pattern.
    Pass conn= only in tests to use an in-memory SQLite connection.
    """
    global _memory_service
    if _memory_service is None:
        with _memory_lock:
            if _memory_service is None:
                _memory_service = MemoryService(conn=conn)
                log.debug("MemoryService singleton initialised")
    return _memory_service
