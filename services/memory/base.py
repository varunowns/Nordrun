"""
Memory Abstractions — Phase 1
------------------------------
Abstract base classes that define the contracts for the two core
components of the memory stack:

  AbstractEmbeddingProvider
    Generates float vectors from text. Concrete implementations:
      - TfIdfEmbeddingProvider  (wraps existing EmbeddingIndex, default)
      - Future: SentenceTransformerProvider, OpenAIEmbeddingProvider, …

  AbstractMemoryStore
    Persists, retrieves, and searches Memory records. Concrete implementations:
      - SqliteMemoryStore  (SQLite-backed, default, shares the DB)
      - Future: any other store that satisfies the contract

The rest of Nordrun (MemoryService, ContextService, plugins) depends only
on these interfaces, never on concrete classes — so swapping the backing
store or embedding model is a single-line change in MemoryService.

Conventions:
  - All methods are synchronous (matches the rest of Nordrun).
  - No global state in implementations — everything injected or late-bound.
  - Concrete classes must be deterministic in tests (no random seeds).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from services.memory.models import Memory, MemoryQuery, MemoryResult


class AbstractEmbeddingProvider(ABC):
    """Contract for anything that can convert text to a float vector."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a normalised float32 vector for `text`.

        The dimensionality is implementation-defined but must be
        consistent within a single store's lifetime.  Returns a
        zero-vector for text that produces no tokens.
        """
        ...

    @abstractmethod
    def embed_and_store(self, doc_id: str, text: str) -> np.ndarray:
        """Embed `text`, associate it with `doc_id`, and persist.

        Re-calling with the same doc_id replaces the prior embedding
        without inflating corpus statistics (idempotent update).
        Returns the vector.
        """
        ...

    @abstractmethod
    def remove(self, doc_id: str) -> None:
        """Remove `doc_id` from the embedding store."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Flush any in-memory state to the backing store."""
        ...


class AbstractMemoryStore(ABC):
    """Contract for anything that can persist and query Memory records."""

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def put(self, memory: Memory) -> None:
        """Insert or replace a Memory record (upsert on memory.id)."""
        ...

    @abstractmethod
    def update(self, memory_id: str, content: str | None = None,
               metadata_updates: dict[str, Any] | None = None) -> Memory | None:
        """Update fields of an existing memory.  Returns the updated
        Memory, or None if memory_id is not found.
        """
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.  Returns True if it existed."""
        ...

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Fetch a single memory by ID, or None if not found."""
        ...

    @abstractmethod
    def query(self, query: MemoryQuery) -> list[MemoryResult]:
        """Return memories matching the structured query.

        Implementations must honour:
          - query.memory_type filter (exact match)
          - query.tags filter (all tags must be present)
          - query.source filter
          - query.min_importance threshold
          - query.top_k limit
          - query.text for semantic similarity (when non-empty)
          - query.min_score cosine similarity threshold
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored memories."""
        ...
