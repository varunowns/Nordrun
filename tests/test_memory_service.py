"""
Tests for MemoryService (services/memory_service.py).

MemoryService wraps SqliteMemoryStore + GraphStore + TfIdfEmbeddingProvider.
All tests inject a fresh in-memory conn so the real vault/DB is never touched.
The autouse isolated_env grants vault permissions; memory permissions are
added per-fixture.
"""

import sqlite3
import threading

import pytest

from core.plugin_registry import register_plugin, set_active_plugin
from services.memory.models import MemorySource, MemoryType
from services.memory_service import MemoryService, get_memory
from storage.db import _init_memory_schema, _init_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
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


@pytest.fixture
def svc(mem_conn):
    """Fresh MemoryService with an isolated in-memory DB."""
    return MemoryService(conn=mem_conn)


@pytest.fixture(autouse=True)
def reset_memory_singleton(monkeypatch):
    """Reset the module-level _memory_service singleton between tests."""
    import services.memory_service as ms_mod
    monkeypatch.setattr(ms_mod, "_memory_service", None)
    yield
    monkeypatch.setattr(ms_mod, "_memory_service", None)


# ---------------------------------------------------------------------------
# store() and get()
# ---------------------------------------------------------------------------

class TestMemoryServiceStore:

    def test_store_returns_memory(self, svc):
        m = svc.store(content="I am a machine learning engineer working on AI projects")
        assert m.id
        assert m.content == "I am a machine learning engineer working on AI projects"

    def test_store_persists(self, svc):
        m = svc.store(content="Persistent fact that should survive retrieval")
        retrieved = svc.get(m.id)
        assert retrieved is not None
        assert retrieved.id == m.id

    def test_store_with_metadata(self, svc):
        m = svc.store(
            content="My primary goal is to build Nordrun into a Jarvis-like AI",
            memory_type=MemoryType.GOAL,
            source=MemorySource.USER,
            importance=0.9,
            tags=["goal", "ai"],
        )
        assert m.metadata.memory_type == MemoryType.GOAL
        assert m.metadata.importance == 0.9
        assert "goal" in m.metadata.tags

    def test_get_nonexistent_returns_none(self, svc):
        assert svc.get("does-not-exist-id") is None

    def test_count_increments(self, svc):
        assert svc.count() == 0
        svc.store(content="First stored memory content for counting test")
        svc.store(content="Second stored memory content for counting test")
        assert svc.count() == 2


# ---------------------------------------------------------------------------
# observe() — lifecycle
# ---------------------------------------------------------------------------

class TestMemoryServiceObserve:

    def test_observe_stores_valid_content(self, svc):
        m = svc.observe(content="I prefer to use Python for all my projects")
        assert m is not None
        assert m.id

    def test_observe_rejects_empty(self, svc):
        assert svc.observe(content="") is None

    def test_observe_rejects_whitespace(self, svc):
        assert svc.observe(content="   ") is None

    def test_observe_rejects_too_short(self, svc):
        # < 10 chars
        assert svc.observe(content="short") is None

    def test_observe_stores_10_plus_chars(self, svc):
        m = svc.observe(content="exactly ten")  # 11 chars with space
        assert m is not None

    def test_observe_strips_whitespace_before_check(self, svc):
        # Content that is too short after stripping
        m = svc.observe(content="   hi   ")  # stripped = "hi" (2 chars)
        assert m is None

    def test_observe_default_source_is_plugin(self, svc):
        m = svc.observe(content="Something meaningful observed from a plugin")
        assert m.metadata.source == MemorySource.PLUGIN


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestMemoryServiceUpdate:

    def test_update_content(self, svc):
        m = svc.store(content="Original content that will be updated shortly")
        updated = svc.update(m.id, content="Updated content replacing the original text")
        assert updated.content == "Updated content replacing the original text"

    def test_update_importance(self, svc):
        m = svc.store(content="Memory whose importance will be raised higher")
        updated = svc.update(m.id, metadata_updates={"importance": 0.99})
        assert updated.metadata.importance == 0.99

    def test_update_nonexistent(self, svc):
        assert svc.update("ghost", content="x") is None


# ---------------------------------------------------------------------------
# forget()
# ---------------------------------------------------------------------------

class TestMemoryServiceForget:

    def test_forget_deletes(self, svc):
        m = svc.store(content="Memory that should be forgotten and deleted")
        assert svc.forget(m.id) is True
        assert svc.get(m.id) is None

    def test_forget_nonexistent(self, svc):
        assert svc.forget("ghost-id") is False

    def test_forget_decrements_count(self, svc):
        m = svc.store(content="Memory to forget and check count decrements")
        assert svc.count() == 1
        svc.forget(m.id)
        assert svc.count() == 0


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestMemoryServiceSearch:

    def test_search_semantic(self, svc):
        m1 = svc.store(content="machine learning deep learning neural network transformer architecture")
        svc.store(content="cooking pasta italian dinner recipes homemade sauce")
        # Query shares tokens only with m1 — it must appear with score > 0
        results = svc.search(text="machine learning neural", top_k=5)
        assert len(results) >= 1
        ids = [r.memory.id for r in results]
        assert m1.id in ids
        score = next(r.score for r in results if r.memory.id == m1.id)
        assert score > 0

    def test_search_filter_by_type(self, svc):
        svc.store(content="Goal to learn transformer architecture thoroughly", memory_type=MemoryType.GOAL)
        svc.store(content="Fact about transformer attention mechanism details", memory_type=MemoryType.FACT)
        results = svc.search(memory_type=MemoryType.GOAL, top_k=10)
        assert all(r.memory.metadata.memory_type == MemoryType.GOAL for r in results)

    def test_search_filter_by_importance(self, svc):
        svc.store(content="Low importance memory to test importance filtering", importance=0.1)
        svc.store(content="High importance memory to test importance filtering", importance=0.9)
        results = svc.search(min_importance=0.5, top_k=10)
        assert all(r.memory.metadata.importance >= 0.5 for r in results)

    def test_search_empty_results(self, svc):
        results = svc.search(text="machine learning")
        assert results == []

    def test_search_respects_top_k(self, svc):
        for i in range(10):
            svc.store(content=f"Memory number {i} for top k limit testing")
        results = svc.search(top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Knowledge graph via MemoryService
# ---------------------------------------------------------------------------

class TestMemoryServiceGraph:

    def test_add_and_get_entity(self, svc):
        e = svc.add_entity("Varun", "person", description="ML engineer")
        retrieved = svc.get_entity(e.id)
        assert retrieved is not None
        assert retrieved.name == "Varun"

    def test_get_entity_by_name(self, svc):
        svc.add_entity("EchoSign", "project")
        e = svc.get_entity_by_name("EchoSign")
        assert e is not None

    def test_search_entities(self, svc):
        svc.add_entity("Machine Learning", "skill")
        svc.add_entity("Deep Learning", "skill")
        results = svc.search_entities(name_fragment="Learning")
        assert len(results) == 2

    def test_add_relationship(self, svc):
        e1 = svc.add_entity("Varun", "person")
        e2 = svc.add_entity("Nordrun", "project")
        rel = svc.add_relationship(e1.id, e2.id, "works_on")
        assert rel.source_id == e1.id
        assert rel.target_id == e2.id

    def test_get_neighbours(self, svc):
        e1 = svc.add_entity("Varun", "person")
        e2 = svc.add_entity("Nordrun", "project")
        svc.add_relationship(e1.id, e2.id, "works_on")
        neighbours = svc.get_neighbours(e1.id)
        assert any(n.id == e2.id for n in neighbours)

    def test_delete_entity(self, svc):
        e = svc.add_entity("ToDelete", "project")
        assert svc.delete_entity(e.id) is True
        assert svc.get_entity(e.id) is None


# ---------------------------------------------------------------------------
# get_relevant_context()
# ---------------------------------------------------------------------------

class TestMemoryServiceContext:

    def test_returns_memories_and_entities_keys(self, svc):
        result = svc.get_relevant_context("machine learning")
        assert "memories" in result
        assert "entities" in result

    def test_retrieves_relevant_memories(self, svc):
        svc.store(content="I work on machine learning and deep learning projects")
        result = svc.get_relevant_context("machine learning")
        assert len(result["memories"]) >= 1

    def test_retrieves_relevant_entities(self, svc):
        svc.add_entity("Nordrun", "project", description="AI OS project")
        result = svc.get_relevant_context("Nordrun project")
        assert len(result["entities"]) >= 1

    def test_empty_store_returns_empty_lists(self, svc):
        result = svc.get_relevant_context("anything")
        assert result["memories"] == []
        assert result["entities"] == []


# ---------------------------------------------------------------------------
# Singleton thread safety
# ---------------------------------------------------------------------------

class TestGetMemorySingleton:

    def test_same_instance_across_calls(self, mem_conn):
        import services.memory_service as ms_mod
        ms_mod._memory_service = None
        svc1 = get_memory(conn=mem_conn)
        svc2 = get_memory(conn=mem_conn)
        assert svc1 is svc2

    def test_concurrent_calls_return_same_instance(self, mem_conn):
        """get_memory() double-checked lock must initialise exactly once.

        SQLite in-memory connections cannot cross thread boundaries, so we
        test the lock guarantee by pre-seeding the singleton on the main
        thread and verifying all threads retrieve the exact same object.
        """
        import services.memory_service as ms_mod
        ms_mod._memory_service = None
        # Initialise on the main thread first
        main_svc = get_memory(conn=mem_conn)

        results = []
        errors = []
        barrier = threading.Barrier(8)

        def call():
            try:
                barrier.wait()
                # Singleton is already set — no new conn needed cross-thread
                results.append(get_memory())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8
        assert all(r is main_svc for r in results)


# ---------------------------------------------------------------------------
# Permission enforcement
# ---------------------------------------------------------------------------

class TestMemoryServicePermissions:

    def test_write_requires_memory_write(self, mem_conn, monkeypatch):
        """A plugin without memory:write cannot call store()."""
        from core.plugin_registry import _reset_registry
        _reset_registry()
        register_plugin("limited_plugin", ["vault:read"])
        set_active_plugin("limited_plugin")

        svc = MemoryService(conn=mem_conn)
        with pytest.raises(RuntimeError, match="missing required permissions"):
            svc.store(content="This should not be stored by limited plugin")
        set_active_plugin(None)

    def test_read_requires_memory_read(self, mem_conn, monkeypatch):
        """A plugin without memory:read cannot call get()."""
        from core.plugin_registry import _reset_registry
        _reset_registry()
        register_plugin("limited_plugin", ["vault:read"])
        set_active_plugin("limited_plugin")

        svc = MemoryService(conn=mem_conn)
        with pytest.raises(RuntimeError, match="missing required permissions"):
            svc.get("any-id")
        set_active_plugin(None)
