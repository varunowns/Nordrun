"""
Tests for the Memory plugin (plugins/memory/plugin.py).

Verifies the plugin loads, declares correct permissions, subscribes to all
memory events, and that its handlers correctly delegate to MemoryService
and return well-formed dicts.

Handlers call get_memory() (the singleton), so tests reset the singleton
and point it at an in-memory connection via the reset_memory_singleton
fixture. Memory permissions are granted so @require passes.
"""

import sqlite3

import pytest

from core.event_bus import EventBus
from core.plugin_loader import discover_plugins, load_and_register, validate_manifest
from core.plugin_registry import register_plugin, set_active_plugin
from plugins.memory import plugin as memory_plugin
from services.memory_service import MemoryService
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


@pytest.fixture(autouse=True)
def wired_memory_singleton(monkeypatch, mem_conn):
    """Point get_memory() at an isolated MemoryService for handler tests."""
    import services.memory_service as ms_mod
    svc = MemoryService(conn=mem_conn)
    monkeypatch.setattr(ms_mod, "_memory_service", svc)
    yield svc
    monkeypatch.setattr(ms_mod, "_memory_service", None)


# ---------------------------------------------------------------------------
# Discovery / manifest / registration
# ---------------------------------------------------------------------------

class TestMemoryPluginLoading:

    def test_discover_finds_memory(self):
        names = [p["name"] for p in discover_plugins()]
        assert "memory" in names

    def test_memory_manifest_valid(self):
        mp = next(p for p in discover_plugins() if p["name"] == "memory")
        assert validate_manifest(mp) == []

    def test_memory_manifest_permissions(self):
        mp = next(p for p in discover_plugins() if p["name"] == "memory")
        perms = mp.get("permissions", "")
        assert "memory:read" in perms
        assert "memory:write" in perms

    def test_memory_registers_events(self):
        bus = EventBus()
        report = load_and_register(bus)
        assert "memory" in report.registered
        events = bus.registered_events()
        for ev in ("memory.store", "memory.search", "memory.get",
                   "memory.update", "memory.forget", "memory.observe",
                   "memory.entity.add", "memory.entity.search",
                   "memory.entity.get", "memory.relationship.add",
                   "memory.neighbours"):
            assert ev in events, f"missing event: {ev}"

    def test_all_plugins_still_load(self):
        """Adding the memory plugin must not break existing plugin loading."""
        bus = EventBus()
        report = load_and_register(bus)
        for name in ("career", "github", "resume", "search", "ctx2img", "learning", "memory"):
            assert name in report.registered


# ---------------------------------------------------------------------------
# Memory handlers
# ---------------------------------------------------------------------------

class TestMemoryHandlers:

    def test_handle_store(self):
        result = memory_plugin.handle_store({
            "content": "I am building Nordrun as a personal AI operating system",
            "importance": 0.8,
            "tags": ["ai", "project"],
        })
        assert result["stored"] is True
        assert "memory" in result
        assert result["memory"]["importance"] == 0.8

    def test_handle_store_requires_content(self):
        result = memory_plugin.handle_store({})
        assert "error" in result

    def test_handle_observe_stores_valid(self):
        result = memory_plugin.handle_observe({
            "content": "I prefer Python over JavaScript for backend work",
        })
        assert result["stored"] is True

    def test_handle_observe_rejects_short(self):
        result = memory_plugin.handle_observe({"content": "short"})
        assert result["stored"] is False
        assert "reason" in result

    def test_handle_get_roundtrip(self):
        stored = memory_plugin.handle_store({"content": "A memory to fetch back by its id"})
        mem_id = stored["memory"]["id"]
        result = memory_plugin.handle_get({"memory_id": mem_id})
        assert result["found"] is True
        assert result["memory"]["id"] == mem_id

    def test_handle_get_missing(self):
        result = memory_plugin.handle_get({"memory_id": "ghost-id"})
        assert result["found"] is False

    def test_handle_get_requires_id(self):
        result = memory_plugin.handle_get({})
        assert "error" in result

    def test_handle_search(self):
        memory_plugin.handle_store({"content": "machine learning deep learning neural networks"})
        result = memory_plugin.handle_search({"text": "machine learning neural"})
        assert "results" in result
        assert result["result_count"] >= 1

    def test_handle_update(self):
        stored = memory_plugin.handle_store({"content": "Original content before the update happens"})
        mem_id = stored["memory"]["id"]
        result = memory_plugin.handle_update({
            "memory_id": mem_id,
            "metadata": {"importance": 0.99},
        })
        assert result["updated"] is True
        assert result["memory"]["importance"] == 0.99

    def test_handle_update_missing(self):
        result = memory_plugin.handle_update({"memory_id": "ghost"})
        assert result["updated"] is False

    def test_handle_forget(self):
        stored = memory_plugin.handle_store({"content": "A memory that will soon be forgotten forever"})
        mem_id = stored["memory"]["id"]
        result = memory_plugin.handle_forget({"memory_id": mem_id})
        assert result["forgotten"] is True
        # confirm gone
        assert memory_plugin.handle_get({"memory_id": mem_id})["found"] is False

    def test_handle_forget_missing_id(self):
        result = memory_plugin.handle_forget({})
        assert "error" in result


# ---------------------------------------------------------------------------
# Knowledge graph handlers
# ---------------------------------------------------------------------------

class TestGraphHandlers:

    def test_handle_entity_add(self):
        result = memory_plugin.handle_entity_add({
            "name": "Nordrun",
            "entity_type": "project",
            "description": "AI OS",
        })
        assert "entity" in result
        assert result["entity"]["name"] == "Nordrun"

    def test_handle_entity_add_requires_fields(self):
        result = memory_plugin.handle_entity_add({"name": "OnlyName"})
        assert "error" in result

    def test_handle_entity_search(self):
        memory_plugin.handle_entity_add({"name": "Machine Learning", "entity_type": "skill"})
        result = memory_plugin.handle_entity_search({"name": "Learning"})
        assert result["count"] >= 1

    def test_handle_entity_get(self):
        added = memory_plugin.handle_entity_add({"name": "EchoSign", "entity_type": "project"})
        entity_id = added["entity"]["id"]
        result = memory_plugin.handle_entity_get({"entity_id": entity_id})
        assert result["found"] is True

    def test_handle_relationship_add(self):
        e1 = memory_plugin.handle_entity_add({"name": "Varun", "entity_type": "person"})
        e2 = memory_plugin.handle_entity_add({"name": "Nordrun", "entity_type": "project"})
        result = memory_plugin.handle_relationship_add({
            "source_id": e1["entity"]["id"],
            "target_id": e2["entity"]["id"],
            "relation_type": "works_on",
        })
        assert "relationship" in result
        assert result["relationship"]["relation_type"] == "works_on"

    def test_handle_relationship_add_requires_fields(self):
        result = memory_plugin.handle_relationship_add({"source_id": "a"})
        assert "error" in result

    def test_handle_neighbours(self):
        e1 = memory_plugin.handle_entity_add({"name": "Alice", "entity_type": "person"})
        e2 = memory_plugin.handle_entity_add({"name": "ProjectX", "entity_type": "project"})
        memory_plugin.handle_relationship_add({
            "source_id": e1["entity"]["id"],
            "target_id": e2["entity"]["id"],
            "relation_type": "works_on",
        })
        result = memory_plugin.handle_neighbours({"entity_id": e1["entity"]["id"]})
        names = [n["name"] for n in result["neighbours"]]
        assert "ProjectX" in names

    def test_handle_neighbours_requires_id(self):
        result = memory_plugin.handle_neighbours({})
        assert "error" in result


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------

class TestEventBusIntegration:

    def test_store_via_event_bus(self):
        bus = EventBus()
        memory_plugin.register(bus, plugin_name="memory")
        # Grant the memory plugin its permissions for dispatch
        register_plugin("memory", ["memory:read", "memory:write"])
        results = bus.publish("memory.store", {"content": "Stored through the event bus dispatch path"})
        assert results
        assert results[0]["stored"] is True

    def test_search_via_event_bus(self):
        bus = EventBus()
        memory_plugin.register(bus, plugin_name="memory")
        register_plugin("memory", ["memory:read", "memory:write"])
        bus.publish("memory.store", {"content": "neural networks and deep learning models"})
        results = bus.publish("memory.search", {"text": "neural networks deep"})
        assert results
        assert results[0]["result_count"] >= 1
