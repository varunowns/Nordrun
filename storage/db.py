"""
SQLite Metadata Layer
---------------------
Stores note metadata (path, title, tags, last_modified, plugin_source)
alongside the vault files. The vault's markdown files remain the source
of truth — this is a searchable index, not a replacement.

Late-binding (Phase 0 hardening):
  _DB_PATH is resolved inside get_db() rather than at module import time.
  This ensures that tests (and anything else that patches config.VAULT_PATH
  or obsidian_service.VAULT_PATH after import) always get the correct
  database path — an import-time binding would capture the un-patched
  value before any monkeypatching takes effect.

Usage:
    from storage.db import get_db, NoteIndex
    idx = NoteIndex(get_db())
    idx.index_note("Career/README.md", "Career Overview", ["career"], "career")
    results = idx.get_notes_by_tag("career")
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as _config

log = logging.getLogger(__name__)

_local = threading.local()


def _db_path() -> Path:
    """Resolve the database path from the current VAULT_PATH at call time.

    Called inside get_db() so tests that monkeypatch config.VAULT_PATH
    (or the bound name on obsidian_service) always get the right path.
    """
    return _config.VAULT_PATH / ".nordrun" / "metadata.db"


def get_db() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating the DB + schema
    on first access.

    The DB path is resolved from config.VAULT_PATH at call time (not at
    import time) so tests that monkeypatch VAULT_PATH always land in the
    right database.
    """
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        db_path = _db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(conn)
        _local.conn = conn
        log.debug("Opened SQLite connection at %s", db_path)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            path          TEXT PRIMARY KEY,
            title         TEXT NOT NULL DEFAULT '',
            tags          TEXT NOT NULL DEFAULT '',
            last_modified TEXT NOT NULL DEFAULT '',
            plugin_source TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _row_to_note(row: tuple) -> dict[str, Any]:
    """Convert a SQLite row from the notes table to a dict."""
    return {
        "path": row[0],
        "title": row[1],
        "tags": row[2].split(",") if row[2] else [],
        "last_modified": row[3],
        "plugin_source": row[4],
    }


class NoteIndex:
    """High-level interface for indexing and querying vault notes."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def index_note(
        self,
        path: str,
        title: str = "",
        tags: list[str] | None = None,
        plugin_source: str = "",
    ) -> None:
        """Insert or update metadata for a vault note."""
        now = datetime.now(timezone.utc).isoformat()
        tags_str = ",".join(tags) if tags else ""
        self._conn.execute(
            """
            INSERT INTO notes (path, title, tags, last_modified, plugin_source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title=excluded.title,
                tags=excluded.tags,
                last_modified=excluded.last_modified,
                plugin_source=excluded.plugin_source
            """,
            (path, title, tags_str, now, plugin_source),
        )
        self._conn.commit()

    def get_notes_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Return all notes whose tags field contains `tag` as a whole tag.

        Tags are stored comma-separated (e.g. "career,summary"), so the
        match is on comma-delimited boundaries — never a substring. This
        keeps `learning` from matching `machine-learning` and `career`
        from matching `non-career`.
        """
        cursor = self._conn.execute(
            "SELECT path, title, tags, last_modified, plugin_source FROM notes "
            "WHERE ',' || tags || ',' LIKE '%,' || ? || ',%'",
            (tag,),
        )
        return [_row_to_note(r) for r in cursor.fetchall()]

    def get_all_paths(self) -> list[str]:
        """Return all indexed note paths."""
        cursor = self._conn.execute("SELECT path FROM notes")
        return [r[0] for r in cursor.fetchall()]

    def delete_note(self, path: str) -> None:
        """Remove a note from the index by path."""
        self._conn.execute("DELETE FROM notes WHERE path = ?", (path,))
        self._conn.commit()

    def get_note(self, path: str) -> dict[str, Any] | None:
        """Look up a single note by path."""
        cursor = self._conn.execute(
            "SELECT path, title, tags, last_modified, plugin_source FROM notes WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        return _row_to_note(row) if row else None


# ---------------------------------------------------------------------------
# Phase 1 — Memory schema helpers
# ---------------------------------------------------------------------------

def _init_memory_schema(conn: sqlite3.Connection) -> None:
    """Create the memories, entities, and relationships tables.

    Called explicitly by SqliteMemoryStore and GraphStore so the tables
    are only created when the memory system is actually used — not on
    every startup.  All three tables share the same DB file so foreign-key
    integrity is possible without cross-file joins.
    """
    # Core memory records
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id             TEXT PRIMARY KEY,
            content        TEXT NOT NULL DEFAULT '',
            memory_type    TEXT NOT NULL DEFAULT 'note',
            source         TEXT NOT NULL DEFAULT 'user',
            importance     REAL NOT NULL DEFAULT 0.5,
            confidence     REAL NOT NULL DEFAULT 1.0,
            tags           TEXT NOT NULL DEFAULT '',
            plugin_source  TEXT NOT NULL DEFAULT '',
            vault_path     TEXT NOT NULL DEFAULT '',
            related_ids    TEXT NOT NULL DEFAULT '',
            created_at     TEXT NOT NULL DEFAULT '',
            updated_at     TEXT NOT NULL DEFAULT ''
        )
        """
    )
    # Knowledge-graph entities
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            entity_type TEXT NOT NULL DEFAULT 'unknown',
            description TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT '',
            updated_at  TEXT NOT NULL DEFAULT ''
        )
        """
    )
    # Knowledge-graph relationships
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relationships (
            id              TEXT PRIMARY KEY,
            source_id       TEXT NOT NULL,
            target_id       TEXT NOT NULL,
            relation_type   TEXT NOT NULL DEFAULT 'related_to',
            weight          REAL NOT NULL DEFAULT 1.0,
            description     TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)"
    )
    conn.commit()
