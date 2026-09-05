"""
Context Service
---------------
Unified interface for plugins to access vault context without knowing
the underlying services (obsidian_service, storage.db, embedding_service).

This is the single point of access for:
- Reading/writing notes with metadata
- Tag-based queries
- Semantic search
- Note indexing

Plugins should use this instead of calling services directly.

Thread safety (Phase 0 hardening):
  get_context() uses a module-level lock so concurrent callers on
  different threads always get the same singleton instance and never
  race on the None check.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from config import VAULT_PATH
from core.plugin_registry import require
from services import obsidian_service
from services.embedding_service import EmbeddingIndex
from storage.db import NoteIndex, get_db, _row_to_note

log = logging.getLogger(__name__)


class ContextService:
    """
    High-level context access for plugins.

    Usage:
        ctx = ContextService()
        content = ctx.read_note("Career/README.md")
        notes = ctx.find_by_tag("career")
        results = ctx.search("machine learning")
        ctx.write_note("New/Note.md", "Content", tags=["tag1"], plugin_source="my_plugin")
    """

    def __init__(self, conn: sqlite3.Connection | None = None,
                 vault_path: Path | None = None):
        self._note_index: NoteIndex | None = None
        self._embedding_index: EmbeddingIndex | None = None
        self._conn: sqlite3.Connection | None = conn
        self._vault_path: Path | None = vault_path

    @property
    def _notes(self) -> NoteIndex:
        if self._note_index is None:
            conn = self._conn or get_db()
            self._note_index = NoteIndex(conn)
        return self._note_index

    @property
    def _embeddings(self) -> EmbeddingIndex:
        if self._embedding_index is None:
            self._embedding_index = EmbeddingIndex(conn=self._conn)
        return self._embedding_index

    # -------------------------------------------------------------------------
    # Note read/write
    # -------------------------------------------------------------------------

    def read_note(self, relative_path: str) -> str:
        """Read a note's raw markdown content."""
        return obsidian_service.read_note(relative_path)

    def write_note(
        self,
        relative_path: str,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        plugin_source: str = "",
    ) -> Path:
        """
        Write a note and index it in metadata + embeddings.
        Returns the full path written to.
        """
        # Write to vault
        note_path = obsidian_service.write_note(
            relative_path=relative_path,
            content=content,
            title=title,
            tags=tags,
            plugin_source=plugin_source,
        )

        # Also index in embeddings for semantic search
        self._embeddings.index_note(relative_path, content)
        self._embeddings.save_state()

        return note_path

    def note_exists(self, relative_path: str) -> bool:
        """Check if a note exists in the vault."""
        root = self._vault_path or VAULT_PATH
        try:
            target = obsidian_service._resolve_vault_path(relative_path, root)
        except ValueError:
            return False
        return target.exists()

    def delete_note(self, relative_path: str) -> bool:
        """Delete a note from vault and indexes."""
        root = self._vault_path or VAULT_PATH
        note_path = obsidian_service._resolve_vault_path(relative_path, root)
        if not note_path.exists():
            return False

        note_path.unlink()
        self._notes.delete_note(relative_path)
        self._embeddings.remove_note(relative_path)
        self._embeddings.save_state()
        return True

    # -------------------------------------------------------------------------
    # Tag-based queries
    # -------------------------------------------------------------------------

    def find_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """
        Find all notes with a given tag.
        Returns list of dicts with: path, title, tags, last_modified, plugin_source
        """
        return self._notes.get_notes_by_tag(tag)

    def get_note_metadata(self, relative_path: str) -> dict[str, Any] | None:
        """Get metadata for a single note (title, tags, last_modified, plugin_source)."""
        return self._notes.get_note(relative_path)

    def get_all_tags(self) -> list[str]:
        """Get all unique tags across all indexed notes."""
        conn = self._conn or get_db()
        rows = conn.execute("SELECT tags FROM notes WHERE tags != ''").fetchall()
        tags = set()
        for row in rows:
            if row[0]:
                tags.update(t.strip() for t in row[0].split(",") if t.strip())
        return sorted(tags)

    def get_notes_by_plugin(self, plugin_source: str) -> list[dict[str, Any]]:
        """Get all notes written by a specific plugin."""
        conn = self._conn or get_db()
        cursor = conn.execute(
            "SELECT path, title, tags, last_modified, plugin_source FROM notes WHERE plugin_source = ?",
            (plugin_source,),
        )
        return [_row_to_note(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Semantic search
    # -------------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Semantic search over vault notes.
        Returns list of dicts with: path, score, title, tags, plugin_source
        """
        return self._embeddings.search(query, top_k=top_k)

    def reindex_all(self, scan_vault: bool = False) -> dict[str, Any]:
        """Re-index all vault notes.

        When scan_vault is True, first discovers all .md files in the
        vault and indexes them in SQLite metadata + embeddings, including
        notes not created by Nordrun. When False, only re-indexes notes
        already present in the SQLite notes table.

        Stale rows are self-healing: notes that no longer exist on disk
        are pruned from both the metadata and embedding indexes, and the
        corpus statistics are reconciled so a reindex never double-counts.
        """
        if scan_vault:
            vault_notes = obsidian_service.scan_vault()
            for note in vault_notes:
                # Preserve provenance for notes already written by a
                # plugin; a plain re-scan otherwise resets it to "".
                existing_meta = self._notes.get_note(note["path"])
                plugin_source = (
                    existing_meta.get("plugin_source", "")
                    if existing_meta
                    else ""
                )
                self._notes.index_note(
                    path=note["path"],
                    title=note["title"],
                    tags=note["tags"],
                    plugin_source=plugin_source,
                )

        paths = self._notes.get_all_paths()
        indexed = 0
        errors = []
        for path in paths:
            try:
                content = self.read_note(path)
                self._embeddings.index_note(path, content)
                indexed += 1
            except Exception as exc:
                errors.append({"path": path, "error": str(exc)})

        # Prune index rows for notes that no longer exist on disk.
        stale = [p for p in paths if not self.note_exists(p)]
        for path in stale:
            self._notes.delete_note(path)
            self._embeddings.remove_note(path)
        if stale:
            errors.append({
                "path": ", ".join(stale),
                "error": "removed from index (note no longer on disk)",
            })

        self._embeddings.save_state()

        return {
            "action": "reindex",
            "indexed": indexed,
            "errors": errors,
            "total_requested": len(paths),
            "pruned": stale,
        }

    def reindex_note(self, relative_path: str) -> bool:
        """Re-index a single note in the embedding store."""
        try:
            content = self.read_note(relative_path)
            self._embeddings.index_note(relative_path, content)
            self._embeddings.save_state()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Convenience methods for common plugin patterns
    # -------------------------------------------------------------------------

    def read_and_index(self, relative_path: str) -> str:
        """Read a note and ensure it's indexed in embeddings."""
        content = self.read_note(relative_path)
        self._embeddings.index_note(relative_path, content)
        self._embeddings.save_state()
        return content

    def append_to_note(self, relative_path: str, addition: str) -> Path:
        """Append content to an existing note and re-index."""
        existing = self.read_note(relative_path)
        updated = existing.rstrip() + "\n\n" + addition
        meta = self._notes.get_note(relative_path)
        plugin_source = meta.get("plugin_source", "") if meta else ""
        return self.write_note(relative_path, updated, plugin_source=plugin_source)

    def add_section(self, relative_path: str, heading: str, content: str) -> Path:
        """Add a new section (heading + content) to a note."""
        addition = f"## {heading}\n\n{content}"
        return self.append_to_note(relative_path, addition)

    def get_recent_notes(self, limit: int = 10, plugin_source: str | None = None) -> list[dict[str, Any]]:
        """Get recently modified notes, optionally filtered by plugin."""
        conn = self._conn or get_db()
        if plugin_source:
            cursor = conn.execute(
                "SELECT path, title, tags, last_modified, plugin_source FROM notes WHERE plugin_source = ? ORDER BY last_modified DESC LIMIT ?",
                (plugin_source, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT path, title, tags, last_modified, plugin_source FROM notes ORDER BY last_modified DESC LIMIT ?",
                (limit,),
            )
        return [_row_to_note(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Memory context enrichment (Phase 1)
    # -------------------------------------------------------------------------

    @require("memory:read")
    def get_memory_context(
        self,
        query: str,
        top_k: int | None = None,
        min_importance: float | None = None,
    ) -> dict:
        """Retrieve relevant memories and knowledge-graph entities for a query.

        Returns a dict with keys:
          'memories'  — list of memory dicts (id, content, type, importance, …)
          'entities'  — list of entity dicts (id, name, entity_type, …)

        This is the integration point between the existing ContextService
        (vault notes + embeddings) and the new MemoryService (structured
        long-term memory + knowledge graph).

        The result can be formatted into an LLM system prompt so the model
        is aware of relevant personal context before responding.

        Failure handling: if the MemoryService is unavailable or the
        memory DB has not been initialised, returns empty lists rather
        than propagating an exception — normal plugin behaviour must not
        break because memory is missing.
        """
        try:
            from services.memory_service import get_memory
            svc = get_memory(conn=self._conn)
            return svc.get_relevant_context(query=query, top_k=top_k, min_importance=min_importance)
        except Exception as exc:
            log.warning("get_memory_context() failed (query=%r): %s", query, exc)
            return {"memories": [], "entities": []}


# Singleton instance for easy import.
# _context_lock guards the double-checked initialisation so concurrent
# callers on different threads never race on the None check.
_context_service: ContextService | None = None
_context_lock = threading.Lock()


def get_context() -> ContextService:
    """Return the global ContextService singleton.

    Thread-safe: uses a double-checked lock so the cost of acquiring the
    lock is only paid once (subsequent calls read _context_service without
    locking after the first initialisation).
    """
    global _context_service
    if _context_service is None:
        with _context_lock:
            if _context_service is None:
                _context_service = ContextService()
                log.debug("ContextService singleton initialised")
    return _context_service