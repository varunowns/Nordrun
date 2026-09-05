"""
Tests for Phase 1 memory models (services/memory/models.py).

Pure unit tests — no DB, no fixtures, no I/O.
Covers: Memory, MemoryType, MemorySource, MemoryMetadata, MemoryQuery, MemoryResult.
"""

import pytest

from services.memory.models import (
    Memory,
    MemoryMetadata,
    MemoryQuery,
    MemoryResult,
    MemorySource,
    MemoryType,
)


# ---------------------------------------------------------------------------
# MemoryType
# ---------------------------------------------------------------------------

class TestMemoryType:
    def test_all_values_are_strings(self):
        for mt in MemoryType:
            assert isinstance(mt.value, str)

    def test_round_trip(self):
        for mt in MemoryType:
            assert MemoryType(mt.value) == mt

    def test_known_types_exist(self):
        for name in ("fact", "preference", "person", "project", "goal",
                     "decision", "experience", "skill", "note"):
            assert MemoryType(name) is not None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            MemoryType("totally_unknown")


# ---------------------------------------------------------------------------
# MemorySource
# ---------------------------------------------------------------------------

class TestMemorySource:
    def test_all_values_are_strings(self):
        for ms in MemorySource:
            assert isinstance(ms.value, str)

    def test_round_trip(self):
        for ms in MemorySource:
            assert MemorySource(ms.value) == ms

    def test_known_sources_exist(self):
        for name in ("user", "plugin", "inferred", "vault"):
            assert MemorySource(name) is not None


# ---------------------------------------------------------------------------
# MemoryMetadata
# ---------------------------------------------------------------------------

class TestMemoryMetadata:
    def test_defaults(self):
        m = MemoryMetadata()
        assert m.memory_type == MemoryType.NOTE
        assert m.source == MemorySource.USER
        assert m.importance == 0.5
        assert m.confidence == 1.0
        assert m.tags == []
        assert m.plugin_source == ""
        assert m.vault_path == ""
        assert m.related_entity_ids == []

    def test_custom_values(self):
        m = MemoryMetadata(
            memory_type=MemoryType.GOAL,
            source=MemorySource.PLUGIN,
            importance=0.9,
            confidence=0.7,
            tags=["ai", "career"],
            plugin_source="career",
        )
        assert m.memory_type == MemoryType.GOAL
        assert m.source == MemorySource.PLUGIN
        assert m.importance == 0.9
        assert "ai" in m.tags

    def test_mutable_defaults_are_independent(self):
        a = MemoryMetadata()
        b = MemoryMetadata()
        a.tags.append("x")
        assert b.tags == []


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestMemory:
    def _make(self, **kwargs) -> Memory:
        defaults = dict(
            id="test-id-001",
            content="I am a machine learning engineer",
            metadata=MemoryMetadata(memory_type=MemoryType.FACT, importance=0.8),
            created_at="2026-09-05T00:00:00+00:00",
            updated_at="2026-09-05T00:00:00+00:00",
        )
        defaults.update(kwargs)
        return Memory(**defaults)

    def test_basic_construction(self):
        m = self._make()
        assert m.id == "test-id-001"
        assert "machine learning" in m.content
        assert m.metadata.importance == 0.8

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("id", "content", "type", "source", "importance",
                    "confidence", "tags", "created_at", "updated_at"):
            assert key in d, f"missing key: {key}"

    def test_to_dict_values(self):
        m = self._make()
        d = m.to_dict()
        assert d["id"] == "test-id-001"
        assert d["type"] == MemoryType.FACT.value
        assert d["importance"] == 0.8

    def test_from_dict_round_trip(self):
        original = self._make()
        d = original.to_dict()
        restored = Memory.from_dict(d)
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.metadata.memory_type == original.metadata.memory_type
        assert restored.metadata.importance == original.metadata.importance

    def test_from_dict_defaults_for_missing_keys(self):
        m = Memory.from_dict({
            "id": "x",
            "content": "some content",
        })
        assert m.metadata.memory_type == MemoryType.NOTE
        assert m.metadata.source == MemorySource.USER
        assert m.metadata.importance == 0.5

    def test_embedding_is_optional(self):
        m = self._make()
        assert m.embedding is None
        m2 = self._make(embedding=[0.1, 0.2, 0.3])
        assert m2.embedding == [0.1, 0.2, 0.3]
        # embedding is excluded from to_dict
        assert "embedding" not in m2.to_dict()


# ---------------------------------------------------------------------------
# MemoryQuery
# ---------------------------------------------------------------------------

class TestMemoryQuery:
    def test_defaults(self):
        q = MemoryQuery()
        assert q.text == ""
        assert q.memory_type is None
        assert q.tags == []
        assert q.source is None
        assert q.min_importance == 0.0
        assert q.top_k == 10
        assert q.min_score == 0.0

    def test_custom_query(self):
        q = MemoryQuery(
            text="machine learning",
            memory_type=MemoryType.SKILL,
            tags=["ai"],
            min_importance=0.3,
            top_k=5,
        )
        assert q.text == "machine learning"
        assert q.memory_type == MemoryType.SKILL
        assert q.top_k == 5

    def test_mutable_defaults_are_independent(self):
        q1 = MemoryQuery()
        q2 = MemoryQuery()
        q1.tags.append("x")
        assert q2.tags == []


# ---------------------------------------------------------------------------
# MemoryResult
# ---------------------------------------------------------------------------

class TestMemoryResult:
    def test_default_score(self):
        m = Memory(
            id="r1", content="test",
            metadata=MemoryMetadata(),
            created_at="", updated_at="",
        )
        r = MemoryResult(memory=m)
        assert r.score == 1.0

    def test_custom_score(self):
        m = Memory(id="r2", content="test", metadata=MemoryMetadata(), created_at="", updated_at="")
        r = MemoryResult(memory=m, score=0.73)
        assert r.score == 0.73
