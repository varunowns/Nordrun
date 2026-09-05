"""
Tests for GraphStore (storage/graph.py).

All tests use in-memory SQLite. The autouse isolated_env keeps vault
access safe. Memory permissions are granted per-fixture (same pattern
as test_memory_store.py).
"""

import sqlite3

import pytest

from core.plugin_registry import register_plugin, set_active_plugin
from storage.db import _init_memory_schema, _init_schema
from storage.graph import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_conn():
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    _init_memory_schema(conn)
    return conn


@pytest.fixture
def graph(graph_conn):
    return GraphStore(conn=graph_conn)


@pytest.fixture(autouse=True)
def memory_permissions():
    register_plugin("test_plugin", ["vault:read", "vault:write", "llm:call",
                                    "memory:read", "memory:write"])
    set_active_plugin("test_plugin")
    yield
    set_active_plugin(None)


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------

class TestEntityCRUD:

    def test_add_entity_basic(self, graph):
        e = graph.add_entity("Varun", "person", description="ML engineer")
        assert e.id
        assert e.name == "Varun"
        assert e.entity_type == "person"
        assert e.description == "ML engineer"

    def test_add_entity_with_tags(self, graph):
        e = graph.add_entity("Nordrun", "project", tags=["ai", "python"])
        assert "ai" in e.tags
        assert "python" in e.tags

    def test_add_entity_upserts_on_name_type(self, graph):
        """Calling add_entity twice with same name+type updates, not duplicates."""
        e1 = graph.add_entity("Varun", "person", description="first")
        e2 = graph.add_entity("Varun", "person", description="second")
        assert e1.id == e2.id
        assert graph.count_entities() == 1
        assert e2.description == "second"

    def test_add_different_types_same_name(self, graph):
        """Same name with different types creates distinct entities."""
        e1 = graph.add_entity("Python", "skill")
        e2 = graph.add_entity("Python", "language")
        assert e1.id != e2.id
        assert graph.count_entities() == 2

    def test_get_entity(self, graph):
        e = graph.add_entity("Nordrun", "project")
        retrieved = graph.get_entity(e.id)
        assert retrieved is not None
        assert retrieved.name == "Nordrun"

    def test_get_entity_nonexistent(self, graph):
        assert graph.get_entity("ghost-id") is None

    def test_get_entity_by_name(self, graph):
        graph.add_entity("EchoSign", "project")
        e = graph.get_entity_by_name("EchoSign")
        assert e is not None
        assert e.name == "EchoSign"

    def test_get_entity_by_name_with_type_filter(self, graph):
        graph.add_entity("Python", "skill")
        graph.add_entity("Python", "language")
        e = graph.get_entity_by_name("Python", entity_type="skill")
        assert e is not None
        assert e.entity_type == "skill"

    def test_get_entity_by_name_nonexistent(self, graph):
        assert graph.get_entity_by_name("NotHere") is None

    def test_delete_entity(self, graph):
        e = graph.add_entity("ToDelete", "person")
        assert graph.delete_entity(e.id) is True
        assert graph.get_entity(e.id) is None

    def test_delete_nonexistent_entity(self, graph):
        assert graph.delete_entity("ghost-id") is False

    def test_delete_entity_removes_relationships(self, graph):
        """Deleting an entity must cascade-delete its relationships."""
        e1 = graph.add_entity("Alice", "person")
        e2 = graph.add_entity("ProjectX", "project")
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.delete_entity(e1.id)
        rels = graph.get_relationships(e2.id, direction="inbound")
        assert len(rels) == 0

    def test_count_entities(self, graph):
        assert graph.count_entities() == 0
        graph.add_entity("A", "person")
        graph.add_entity("B", "project")
        assert graph.count_entities() == 2


# ---------------------------------------------------------------------------
# Entity search
# ---------------------------------------------------------------------------

class TestEntitySearch:

    def test_search_by_name_fragment(self, graph):
        graph.add_entity("Machine Learning", "skill")
        graph.add_entity("Deep Learning", "skill")
        graph.add_entity("Cooking", "hobby")
        results = graph.search_entities(name_fragment="Learning")
        names = [e.name for e in results]
        assert "Machine Learning" in names
        assert "Deep Learning" in names
        assert "Cooking" not in names

    def test_search_by_entity_type(self, graph):
        graph.add_entity("Python", "skill")
        graph.add_entity("Nordrun", "project")
        results = graph.search_entities(entity_type="skill")
        assert all(e.entity_type == "skill" for e in results)

    def test_search_combined_filters(self, graph):
        graph.add_entity("Python Skill", "skill")
        graph.add_entity("Python Project", "project")
        results = graph.search_entities(name_fragment="Python", entity_type="skill")
        assert len(results) == 1
        assert results[0].entity_type == "skill"

    def test_search_limit(self, graph):
        for i in range(10):
            graph.add_entity(f"Entity {i}", "person")
        results = graph.search_entities(limit=3)
        assert len(results) <= 3

    def test_search_empty_returns_all_up_to_limit(self, graph):
        graph.add_entity("Alpha", "person")
        graph.add_entity("Beta", "project")
        results = graph.search_entities()
        assert len(results) >= 2

    def test_search_no_match_returns_empty(self, graph):
        graph.add_entity("Varun", "person")
        results = graph.search_entities(name_fragment="xyzzy")
        assert results == []


# ---------------------------------------------------------------------------
# Relationship CRUD
# ---------------------------------------------------------------------------

class TestRelationshipCRUD:

    @pytest.fixture
    def two_entities(self, graph):
        e1 = graph.add_entity("Varun", "person")
        e2 = graph.add_entity("Nordrun", "project")
        return e1, e2

    def test_add_relationship(self, graph, two_entities):
        e1, e2 = two_entities
        rel = graph.add_relationship(e1.id, e2.id, "works_on")
        assert rel.id
        assert rel.source_id == e1.id
        assert rel.target_id == e2.id
        assert rel.relation_type == "works_on"

    def test_add_relationship_upserts(self, graph, two_entities):
        e1, e2 = two_entities
        r1 = graph.add_relationship(e1.id, e2.id, "works_on", weight=0.5)
        r2 = graph.add_relationship(e1.id, e2.id, "works_on", weight=0.9)
        assert r1.id == r2.id
        assert graph.count_relationships() == 1
        assert r2.weight == 0.9

    def test_get_relationship(self, graph, two_entities):
        e1, e2 = two_entities
        rel = graph.add_relationship(e1.id, e2.id, "owns")
        retrieved = graph.get_relationship(rel.id)
        assert retrieved is not None
        assert retrieved.relation_type == "owns"

    def test_get_relationship_nonexistent(self, graph):
        assert graph.get_relationship("ghost-rel") is None

    def test_delete_relationship(self, graph, two_entities):
        e1, e2 = two_entities
        rel = graph.add_relationship(e1.id, e2.id, "knows")
        assert graph.delete_relationship(rel.id) is True
        assert graph.get_relationship(rel.id) is None

    def test_delete_nonexistent_relationship(self, graph):
        assert graph.delete_relationship("ghost") is False

    def test_get_outbound_relationships(self, graph, two_entities):
        e1, e2 = two_entities
        graph.add_relationship(e1.id, e2.id, "works_on")
        rels = graph.get_relationships(e1.id, direction="outbound")
        assert len(rels) == 1
        assert rels[0].source_id == e1.id

    def test_get_inbound_relationships(self, graph, two_entities):
        e1, e2 = two_entities
        graph.add_relationship(e1.id, e2.id, "works_on")
        rels = graph.get_relationships(e2.id, direction="inbound")
        assert len(rels) == 1
        assert rels[0].target_id == e2.id

    def test_get_both_directions(self, graph):
        e1 = graph.add_entity("A", "person")
        e2 = graph.add_entity("B", "project")
        e3 = graph.add_entity("C", "skill")
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.add_relationship(e3.id, e1.id, "used_by")
        rels = graph.get_relationships(e1.id, direction="both")
        assert len(rels) == 2

    def test_filter_relationships_by_type(self, graph, two_entities):
        e1, e2 = two_entities
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.add_relationship(e1.id, e2.id, "owns")  # same pair, different type
        rels = graph.get_relationships(e1.id, relation_type="works_on")
        assert all(r.relation_type == "works_on" for r in rels)

    def test_count_relationships(self, graph, two_entities):
        e1, e2 = two_entities
        assert graph.count_relationships() == 0
        graph.add_relationship(e1.id, e2.id, "works_on")
        assert graph.count_relationships() == 1

    def test_relationship_to_dict(self, graph, two_entities):
        e1, e2 = two_entities
        rel = graph.add_relationship(e1.id, e2.id, "knows", description="teammates")
        d = rel.to_dict()
        assert d["relation_type"] == "knows"
        assert d["description"] == "teammates"


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

class TestGraphTraversal:

    def test_get_neighbours_depth1(self, graph):
        e1 = graph.add_entity("Varun", "person")
        e2 = graph.add_entity("Nordrun", "project")
        e3 = graph.add_entity("Python", "skill")
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.add_relationship(e1.id, e3.id, "uses")

        neighbours = graph.get_neighbours(e1.id)
        ids = {e.id for e in neighbours}
        assert e2.id in ids
        assert e3.id in ids
        assert e1.id not in ids  # starting entity excluded

    def test_get_neighbours_depth2(self, graph):
        e1 = graph.add_entity("A", "person")
        e2 = graph.add_entity("B", "project")
        e3 = graph.add_entity("C", "skill")
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.add_relationship(e2.id, e3.id, "uses")

        neighbours = graph.get_neighbours(e1.id, max_depth=2)
        ids = {e.id for e in neighbours}
        assert e2.id in ids
        assert e3.id in ids

    def test_get_neighbours_isolated_entity(self, graph):
        e = graph.add_entity("Isolated", "person")
        assert graph.get_neighbours(e.id) == []

    def test_get_neighbours_filter_by_relation_type(self, graph):
        e1 = graph.add_entity("A", "person")
        e2 = graph.add_entity("B", "project")
        e3 = graph.add_entity("C", "skill")
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.add_relationship(e1.id, e3.id, "knows")

        neighbours = graph.get_neighbours(e1.id, relation_type="works_on")
        ids = {e.id for e in neighbours}
        assert e2.id in ids
        assert e3.id not in ids

    def test_get_neighbours_no_duplicates(self, graph):
        """A node reachable via multiple paths should appear only once."""
        e1 = graph.add_entity("Hub", "person")
        e2 = graph.add_entity("Spoke", "project")
        graph.add_relationship(e1.id, e2.id, "works_on")
        graph.add_relationship(e1.id, e2.id, "owns")

        neighbours = graph.get_neighbours(e1.id)
        ids = [e.id for e in neighbours]
        assert ids.count(e2.id) == 1

    def test_entity_to_dict(self, graph):
        e = graph.add_entity("Varun", "person", description="ML engineer", tags=["ai"])
        d = e.to_dict()
        assert d["name"] == "Varun"
        assert d["entity_type"] == "person"
        assert "ai" in d["tags"]
