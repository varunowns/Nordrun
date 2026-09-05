"""
Shared test fixtures.

Usage in tests:
    def test_something(memory_db, test_plugin, note_index):
        note_index.index_note("Test/a.md", "Title", ["tag1"], "plugin")
        results = note_index.get_notes_by_tag("tag1")
        assert len(results) == 1

    def test_with_vault(test_vault, test_db):
        ctx = ContextService(conn=test_db, vault_path=test_vault)
        ctx.write_note("Career/New.md", "# Hello", plugin_source="test_plugin")

Available fixtures:
    - memory_db: in-memory SQLite connection with the notes schema
    - note_index: NoteIndex backed by memory_db
    - embedding_index: EmbeddingIndex backed by memory_db
    - test_db: in-memory SQLite connection with notes + embeddings schema
    - test_vault: seeded tmp_path standing in for the real Obsidian vault
    - isolated_env (autouse): redirects vault/DB access to tmp_path + in-memory
      SQLite, registers the test plugin, and resets singleton state between tests
"""

import sqlite3

import pytest

from core.plugin_registry import _reset_registry, register_plugin, set_active_plugin
from services import context_service, embedding_service, obsidian_service
from services.embedding_service import EmbeddingIndex
from storage.db import NoteIndex, _init_schema

_TEST_PLUGIN_PERMISSIONS = [
    "vault:read", "vault:write", "llm:call",
    # Phase 1: memory permissions so tests exercising MemoryService /
    # ContextService.get_memory_context() pass the @require checks.
    "memory:read", "memory:write",
]


@pytest.fixture
def memory_db() -> sqlite3.Connection:
    """Create a clean in-memory SQLite database with the notes schema."""
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    return conn


@pytest.fixture
def note_index(memory_db: sqlite3.Connection) -> NoteIndex:
    """NoteIndex backed by an in-memory SQLite database."""
    return NoteIndex(memory_db)


@pytest.fixture
def embedding_index(memory_db: sqlite3.Connection) -> EmbeddingIndex:
    """EmbeddingIndex backed by an in-memory SQLite database."""
    return EmbeddingIndex(conn=memory_db)


@pytest.fixture
def test_db() -> sqlite3.Connection:
    """In-memory SQLite with the full notes + embeddings schema.

    Both schema init steps are needed: _init_schema creates the notes
    table, EmbeddingIndex._init_schema adds the embeddings tables used
    by ContextService.search / reindex_note.
    """
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    EmbeddingIndex(conn=conn)._init_schema()
    return conn


@pytest.fixture
def test_vault(tmp_path):
    """A small Obsidian-style vault standing in for the real one.

    Seeded with a few markdown notes so scan_vault() and plugin reads
    have content to work with. No .nordrun folder is needed — tests use
    the in-memory test_db instead of a file-backed metadata DB.
    """
    (tmp_path / "Career").mkdir()
    (tmp_path / "Learning").mkdir()
    (tmp_path / "Career" / "README.md").write_text(
        "---\ntitle: Career README\ntags: career\n---\n\n"
        "# Career\n\nMachine learning engineer.\n",
        encoding="utf-8",
    )
    (tmp_path / "Career" / "notes-on-ml.md").write_text(
        "---\ntitle: Notes on ML\ntags: ai learning\n---\n\n"
        "# Notes on ML\n\nDeep learning basics.\n",
        encoding="utf-8",
    )
    (tmp_path / "Learning" / "study-notes.md").write_text(
        "# Study Notes\n\nTransformer architecture.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, test_vault, test_db) -> None:
    """Redirect all vault I/O to a temp vault and SQLite to memory.

    Replaces the old test_context fixture. Production services bind
    VAULT_PATH (from config) at import time, so the fixture patches the
    bound name on each module that uses it. get_db() is never patched:
    ContextService methods fall back to get_db() only when no conn is
    injected, and tests always pass test_db.
    """
    # Reset singleton/cached state so each test starts clean
    context_service._context_service = None
    obsidian_service._index = None

    # Reset the Phase 1 MemoryService singleton too, so a memory service
    # built against one test's DB never leaks into the next test.
    import services.memory_service as memory_service
    memory_service._memory_service = None

    # Redirect vault path (bound at import in each of these modules)
    monkeypatch.setattr(context_service, "VAULT_PATH", test_vault)
    monkeypatch.setattr(obsidian_service, "VAULT_PATH", test_vault)
    monkeypatch.setattr(embedding_service, "VAULT_PATH", test_vault)

    # Redirect the DB connection. obsidian_service.write_note indexes
    # metadata through its own get_db() -> _index, and ContextService's
    # methods fall back to get_db() when no conn is injected, so every
    # module that bound get_db at import must resolve to the in-memory DB.
    monkeypatch.setattr(obsidian_service, "get_db", lambda: test_db)
    monkeypatch.setattr(context_service, "get_db", lambda: test_db)
    monkeypatch.setattr(embedding_service, "get_db", lambda: test_db)

    # Reset singleton/cached state AFTER redirects so the obsidian_service
    # index and context singleton are rebuilt against the isolated paths
    context_service._context_service = None
    obsidian_service._index = None

    # Register the test plugin and make it active for permission checks.
    # The registry is reset first so load_and_register tests start clean.
    _reset_registry()
    register_plugin("test_plugin", _TEST_PLUGIN_PERMISSIONS)
    set_active_plugin("test_plugin")
    yield
    set_active_plugin(None)
    memory_service._memory_service = None
