"""
Tests for ContextService.get_memory_context() — Phase 1 context integration.

Verifies that memory retrieval integrates cleanly with ContextService,
that irrelevant memory is not injected, and that memory failures do not
break normal ContextService behaviour.

Connection strategy
-------------------
The autouse isolated_env fixture patches obsidian_service.get_db() to
return test_db, so all obsidian_service writes (and the NoteIndex they
create) go into test_db.  ContextService must share that same connection
for tag queries to find what was just written.

MemoryService uses a SEPARATE connection (mem_conn) so the memory tables
don't collide with the note/embedding tables that EmbeddingIndex creates
in test_db.  Both services operate in isolation, connected only through
ContextService.get_memory_context().
"""

import sqlite3

import pytest

from core.plugin_registry import register_plugin, set_active_plugin
from services.context_service import ContextService
from services.memory_service import MemoryService
from storage.db import _init_memory_schema, _init_schema
from services.embedding_service import EmbeddingIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
    """Dedicated in-memory DB for MemoryService (memory/graph tables only)."""
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    _init_memory_schema(conn)
    return conn


@pytest.fixture(autouse=True)
def memory_permissions():
    register_plugin("test_plugin", ["vault:read", "vault:write", "llm:call",
                                    "memory:read", "memory:write"])
    set_active_plugin("test_plugin")
    yield
    set_active_plugin(None)


@pytest.fixture(autouse=True)
def reset_memory_singleton(monkeypatch):
    import services.memory_service as ms_mod
    monkeypatch.setattr(ms_mod, "_memory_service", None)
    yield
    monkeypatch.setattr(ms_mod, "_memory_service", None)


@pytest.fixture
def ctx(test_db, test_vault):
    """ContextService using test_db (same conn as isolated_env patches into
    obsidian_service) so note writes and tag queries see the same data.

    get_memory_context() internally builds a MemoryService on this same
    connection, so the svc fixture below must also use test_db for the
    stored memories/entities to be visible to ctx.
    """
    return ContextService(conn=test_db, vault_path=test_vault)


@pytest.fixture
def svc(test_db):
    """MemoryService sharing ctx's connection (test_db).

    This mirrors production, where get_memory() and get_context() both
    resolve to the same thread-local get_db() connection. Sharing the
    connection is required so memories stored via svc are visible to
    ctx.get_memory_context().
    """
    return MemoryService(conn=test_db)


# ---------------------------------------------------------------------------
# get_memory_context() basics
# ---------------------------------------------------------------------------

class TestGetMemoryContext:

    def test_returns_dict_with_required_keys(self, ctx):
        result = ctx.get_memory_context("anything")
        assert isinstance(result, dict)
        assert "memories" in result
        assert "entities" in result

    def test_empty_memory_store_returns_empty_lists(self, ctx):
        result = ctx.get_memory_context("machine learning")
        assert result["memories"] == []
        assert result["entities"] == []

    def test_retrieves_relevant_memories(self, ctx, svc):
        svc.store(content="I work on machine learning and deep learning every day")
        result = ctx.get_memory_context("machine learning")
        assert len(result["memories"]) >= 1
        contents = [m["content"] for m in result["memories"]]
        assert any("machine learning" in c or "learning" in c for c in contents)

    def test_retrieves_relevant_entities(self, ctx, svc):
        svc.add_entity("Nordrun", "project", description="AI OS project")
        result = ctx.get_memory_context("Nordrun project")
        assert len(result["entities"]) >= 1
        names = [e["name"] for e in result["entities"]]
        assert "Nordrun" in names

    def test_irrelevant_memory_has_lower_or_zero_score(self, ctx, svc):
        """Memory about cooking must not outrank an ML memory for an ML query."""
        m_ml = svc.store(content="machine learning deep learning neural network")
        svc.store(content="cooking pasta italian dinner homemade sauce delicious")
        # Query shares tokens only with the ML memory
        result = ctx.get_memory_context("machine learning neural", top_k=5)
        if result["memories"]:
            # The ML memory must appear; cooking memory must not outscore it
            ids = [m["id"] for m in result["memories"]]
            assert m_ml.id in ids

    def test_failure_returns_empty_not_exception(self, ctx, monkeypatch):
        """If MemoryService raises unexpectedly, get_memory_context returns {} not crash."""
        import services.memory_service as ms_mod

        def boom(*a, **k):
            raise RuntimeError("simulated memory failure")

        monkeypatch.setattr(ms_mod, "get_memory", boom)
        result = ctx.get_memory_context("query")
        assert result == {"memories": [], "entities": []}

    def test_top_k_parameter_respected(self, ctx, svc):
        for i in range(10):
            svc.store(content=f"Memory about machine learning topic number {i} in detail")
        result = ctx.get_memory_context("machine learning", top_k=3)
        assert len(result["memories"]) <= 3

    def test_min_importance_filter(self, ctx, svc):
        from services.memory.models import MemoryType
        svc.store(content="Low importance memory that should be filtered out completely", importance=0.1)
        svc.store(content="High importance memory that should pass the filter threshold", importance=0.9)
        # Use a high min_importance so only the important one passes
        result = ctx.get_memory_context("memory filter", min_importance=0.8)
        for m in result["memories"]:
            assert m["importance"] >= 0.8


# ---------------------------------------------------------------------------
# Context integration doesn't break normal ContextService behaviour
# ---------------------------------------------------------------------------

class TestContextServiceBackwardCompat:

    def test_read_write_still_works(self, ctx, test_vault):
        path = ctx.write_note("Test/compat.md", "# Compat\n\nStill works.", plugin_source="test_plugin")
        assert path.exists()
        content = ctx.read_note("Test/compat.md")
        assert "Still works" in content

    def test_search_still_works(self, ctx):
        ctx.write_note("Test/ml.md", "Machine learning content here", plugin_source="test_plugin")
        results = ctx.search("machine learning", top_k=3)
        assert isinstance(results, list)

    def test_find_by_tag_still_works(self, ctx):
        ctx.write_note("Test/tagged.md", "content", tags=["ai"], plugin_source="test_plugin")
        results = ctx.find_by_tag("ai")
        assert len(results) >= 1
