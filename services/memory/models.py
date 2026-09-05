"""
Memory Models — Phase 1
------------------------
Data models for Nordrun's structured memory system.

These are plain Python dataclasses with no external dependencies.
They define the canonical shapes that flow through every layer of the
memory stack: MemoryService → MemoryStore → GraphStore → ContextService.

Design notes:
  - All IDs are plain strings (UUID4 at creation time — no UUID library
    needed for the type; callers generate with str(uuid.uuid4())).
  - Timestamps are ISO-8601 strings (timezone-aware UTC) so they survive
    SQLite round-trips without a datetime object.
  - importance is [0.0, 1.0] — 0.5 is neutral, 1.0 is critical.
  - confidence is [0.0, 1.0] — used when memory was inferred, not
    explicitly stated.
  - tags is a plain list[str] — stored comma-separated in SQLite (same
    pattern as the existing NoteIndex).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Category of a stored memory.

    Using str mixin so values serialize cleanly to/from SQLite TEXT.
    """
    FACT = "fact"            # Objective facts: "I work at Acme Corp"
    PREFERENCE = "preference"  # Preferences: "I prefer dark mode"
    PERSON = "person"        # A person Nordrun knows about
    PROJECT = "project"      # A project the user is working on
    GOAL = "goal"            # A stated goal or objective
    DECISION = "decision"    # A decision that was made
    EXPERIENCE = "experience"  # Something that happened / was learned
    SKILL = "skill"          # A skill the user has or is learning
    NOTE = "note"            # General note — catch-all for untyped content


class MemorySource(str, Enum):
    """Origin of a memory — who or what created it."""
    USER = "user"            # Explicitly stored by the user
    PLUGIN = "plugin"        # Written by a plugin (career, github, etc.)
    INFERRED = "inferred"    # Derived from context by Nordrun
    VAULT = "vault"          # Sourced from an Obsidian vault note


@dataclass
class MemoryMetadata:
    """Metadata associated with every Memory record.

    Stored as individual columns in the memories table so they are
    filterable without JSON parsing.
    """
    memory_type: MemoryType = MemoryType.NOTE
    source: MemorySource = MemorySource.USER
    importance: float = 0.5       # [0.0, 1.0]
    confidence: float = 1.0       # [0.0, 1.0]
    tags: list[str] = field(default_factory=list)
    plugin_source: str = ""       # plugin name if source == PLUGIN
    vault_path: str = ""          # vault note path if source == VAULT
    related_entity_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """A single unit of persistent memory in Nordrun.

    id         — UUID4 string, generated at store time.
    content    — The raw text of the memory.
    metadata   — Typed metadata (type, source, importance, tags, …).
    created_at — ISO-8601 UTC timestamp.
    updated_at — ISO-8601 UTC timestamp (equals created_at on first store).
    embedding  — Optional float vector cached from the embedding provider.
                 Not persisted in the memories table — stored separately
                 in the memory_embeddings table so the core record stays
                 lightweight.
    """
    id: str
    content: str
    metadata: MemoryMetadata
    created_at: str
    updated_at: str
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for event payloads and JSON."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.metadata.memory_type.value,
            "source": self.metadata.source.value,
            "importance": self.metadata.importance,
            "confidence": self.metadata.confidence,
            "tags": self.metadata.tags,
            "plugin_source": self.metadata.plugin_source,
            "vault_path": self.metadata.vault_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Memory":
        """Reconstruct from a plain dict (e.g. from an event payload)."""
        meta = MemoryMetadata(
            memory_type=MemoryType(d.get("type", MemoryType.NOTE.value)),
            source=MemorySource(d.get("source", MemorySource.USER.value)),
            importance=float(d.get("importance", 0.5)),
            confidence=float(d.get("confidence", 1.0)),
            tags=d.get("tags", []),
            plugin_source=d.get("plugin_source", ""),
            vault_path=d.get("vault_path", ""),
        )
        return cls(
            id=d["id"],
            content=d["content"],
            metadata=meta,
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class MemoryQuery:
    """Parameters for a memory retrieval operation.

    text          — Natural-language query for semantic search (optional).
    memory_type   — Filter to a specific type (optional).
    tags          — All listed tags must be present (optional).
    source        — Filter by source (optional).
    min_importance — Minimum importance score (default 0.0 = no filter).
    top_k         — Maximum number of results to return.
    min_score     — Minimum cosine similarity (0.0 = return all).
    """
    text: str = ""
    memory_type: MemoryType | None = None
    tags: list[str] = field(default_factory=list)
    source: MemorySource | None = None
    min_importance: float = 0.0
    top_k: int = 10
    min_score: float = 0.0


@dataclass
class MemoryResult:
    """A single result from a memory retrieval or search operation."""
    memory: Memory
    score: float = 1.0    # Cosine similarity [0.0, 1.0]; 1.0 for exact/filter-only results
