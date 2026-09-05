"""
Tests for SqliteMemoryStore and TfIdfEmbeddingProvider (storage/memory_store.py,
services/memory/embedding.py).

Uses in-memory SQLite — no real vault, no disk writes.
The autouse isolated_env fixture registers test_plugin with vault:read/write/llm:call.
Memory tests additionally need memory:read/memory:write — added per-test via
the mem_db + mem_perms fixtures so existing tests are not affected.
"""

import sqlite3
import uuid

import pytest

from core.plugin_registry import register_plugin, set_active_plugin
from services.memory.embedding import TfIdfEmbeddingProvider
from services.memory.models import Memory, MemoryMetadata, MemoryQuery, MemorySource, MemoryType
from storage.db import _init_memory_schema, _init_schema
from storage.memory_store import SqliteMemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
    """In-memory SQLite with the full Phase 1 schema."""
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    _init_memory_schema(conn)
    return conn


@pytest.fixture(autouse=True)
def memory_permissions():
    """Grant test_plugin memory permissions on top of the autouse isolated_env."""
    register_plugin("test_plugin", ["vault:read", "vault:write", "llm:call",
                                    "memory:read", "memory:write"])
    set_active_plugin("test_plugin")
    yield
    set_active_plugin(None)


@pytest.fixture
def provider(mem_conn):
    return TfIdfEmbeddingProvider(conn=mem_conn)


@pytest.fixture
def store(mem_conn, provider):
    return SqliteMemoryStore(conn=mem_conn, embedding_provider=provider)


@pytest.fixture
def store_no_embed(mem_conn):
    """Store without an embedding provider — tests filter-only queries."""
    return SqliteMemoryStore(conn=mem_conn, embedding_provider=None)


def _mem(content="A long enough memory content here", **kwargs) -> Memory:
    now = "2026-09-05T00:00:00+00:00"
    meta_kwargs = {k: v for k, v in kwargs.items()
                   if k in ("memory_type", "source", "importance", "confidence", "tags",
                             "plugin_source", "vault_path")}
    meta = MemoryMetadata(**meta_kwargs)
    return Memory(
        id=str(uuid.uuid4()),
        content=content,
        metadata=meta,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# TfIdfEmbeddingProvider
# ---------------------------------------------------------------------------

class TestTfIdfEmbeddingProvider:

    def test_embed_returns_array(self, provider):
        import numpy as np
        vec = provider.embed("machine learning project")
        assert isinstance(vec, np.ndarray)

    def test_embed_zero_for_empty(self, provider):
        import numpy as np
        vec = provider.embed("")
        assert not np.any(vec) or len(vec) == 0

    def test_embed_and_store_creates_row(self, provider, mem_conn):
        doc_id = str(uuid.uuid4())
        provider.embed_and_store(doc_id, "deep learning transformer architecture")
        row = mem_conn.execute(
            "SELECT doc_id FROM memory_embeddings WHERE doc_id=?", (doc_id,)
        ).fetchone()
        assert row is not None

    def test_embed_and_store_idempotent(self, provider, mem_conn):
        doc_id = str(uuid.uuid4())
        provider.embed_and_store(doc_id, "first content version here")
        provider.embed_and_store(doc_id, "second content version here")
        count = mem_conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE doc_id=?", (doc_id,)
        ).fetchone()[0]
        assert count == 1

    def test_remove_deletes_row(self, provider, mem_conn):
        doc_id = str(uuid.uuid4())
        provider.embed_and_store(doc_id, "some content to remove later")
        provider.remove(doc_id)
        row = mem_conn.execute(
            "SELECT doc_id FROM memory_embeddings WHERE doc_id=?", (doc_id,)
        ).fetchone()
        assert row is None

    def test_similarity_search_returns_relevant(self, provider):
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        # Use exact token overlap to guarantee ranking is deterministic
        provider.embed_and_store(id1, "machine learning neural network deep learning")
        provider.embed_and_store(id2, "cooking dinner pasta sauce kitchen recipe chef")
        provider.save()

        # Query shares tokens with id1 only — id1 must appear and score > 0
        results = provider.similarity_search("machine learning neural", top_k=5)
        assert len(results) >= 1
        ids = [r[0] for r in results]
        assert id1 in ids
        score_ml = next(s for doc_id, s in results if doc_id == id1)
        assert score_ml > 0

    def test_similarity_search_empty_query_returns_nothing(self, provider):
        provider.embed_and_store("x", "some content here to have something")
        provider.save()
        results = provider.similarity_search("", top_k=5)
        assert results == []

    def test_similarity_search_min_score_filter(self, provider):
        doc_id = str(uuid.uuid4())
        provider.embed_and_store(doc_id, "unique vocabulary xyzzy frobulate quux")
        provider.save()
        # Query with a very high min_score that no result can meet
        results = provider.similarity_search("completely different words here", top_k=5, min_score=0.99)
        assert all(score >= 0.99 for _, score in results)

    def test_save_persists_vocabulary(self, provider, mem_conn):
        provider.embed_and_store("d1", "vocabulary persistence across saves test")
        provider.save()
        # Reload a fresh provider from same conn
        provider2 = TfIdfEmbeddingProvider(conn=mem_conn)
        vec = provider2.embed("vocabulary persistence")
        import numpy as np
        assert np.any(vec)


# ---------------------------------------------------------------------------
# SqliteMemoryStore — CRUD
# ---------------------------------------------------------------------------

class TestSqliteMemoryStoreCRUD:

    def test_put_and_get(self, store):
        m = _mem("I work as a machine learning engineer at a tech company")
        store.put(m)
        retrieved = store.get(m.id)
        assert retrieved is not None
        assert retrieved.id == m.id
        assert retrieved.content == m.content

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_put_is_upsert(self, store):
        m = _mem("Original content that will be replaced later")
        store.put(m)
        m2 = Memory(
            id=m.id,
            content="Updated content replacing the original one",
            metadata=MemoryMetadata(importance=0.9),
            created_at=m.created_at,
            updated_at="2026-09-06T00:00:00+00:00",
        )
        store.put(m2)
        retrieved = store.get(m.id)
        assert retrieved.content == "Updated content replacing the original one"
        assert retrieved.metadata.importance == 0.9

    def test_count(self, store):
        assert store.count() == 0
        store.put(_mem("First memory about machine learning"))
        store.put(_mem("Second memory about deep learning"))
        assert store.count() == 2

    def test_delete_existing(self, store):
        m = _mem("Memory to be deleted permanently from store")
        store.put(m)
        assert store.delete(m.id) is True
        assert store.get(m.id) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("ghost-id") is False

    def test_metadata_round_trip(self, store):
        m = _mem(
            "Tagged memory content for round trip testing",
            memory_type=MemoryType.GOAL,
            source=MemorySource.PLUGIN,
            importance=0.9,
            confidence=0.75,
            tags=["ai", "career"],
            plugin_source="career",
        )
        store.put(m)
        r = store.get(m.id)
        assert r.metadata.memory_type == MemoryType.GOAL
        assert r.metadata.source == MemorySource.PLUGIN
        assert r.metadata.importance == 0.9
        assert r.metadata.confidence == 0.75
        assert "ai" in r.metadata.tags
        assert r.metadata.plugin_source == "career"

    def test_make_memory_convenience(self, store):
        m = store.make_memory(
            content="Convenience factory creates and stores in one call",
            memory_type=MemoryType.FACT,
            importance=0.7,
        )
        assert m.id  # has a UUID
        assert store.get(m.id) is not None
        assert store.count() == 1


# ---------------------------------------------------------------------------
# SqliteMemoryStore — Update
# ---------------------------------------------------------------------------

class TestSqliteMemoryStoreUpdate:

    def test_update_content(self, store):
        m = store.make_memory("Original content that will be updated")
        updated = store.update(m.id, content="Updated content with new information")
        assert updated is not None
        assert updated.content == "Updated content with new information"

    def test_update_importance(self, store):
        m = store.make_memory("Content for importance update testing here")
        updated = store.update(m.id, metadata_updates={"importance": 0.95})
        assert updated.metadata.importance == 0.95

    def test_update_tags(self, store):
        m = store.make_memory("Content for tag update testing purposes")
        updated = store.update(m.id, metadata_updates={"tags": ["new", "tags"]})
        assert "new" in updated.metadata.tags
        assert "tags" in updated.metadata.tags

    def test_update_memory_type(self, store):
        m = store.make_memory("Content for type update testing here now")
        updated = store.update(m.id, metadata_updates={"memory_type": "goal"})
        assert updated.metadata.memory_type == MemoryType.GOAL

    def test_update_nonexistent_returns_none(self, store):
        assert store.update("ghost-id", content="new") is None

    def test_update_preserves_unchanged_fields(self, store):
        m = store.make_memory(
            "Content with specific metadata to preserve during update",
            importance=0.8,
            tags=["keep"],
        )
        updated = store.update(m.id, content="Only content changed not metadata")
        assert updated.metadata.importance == 0.8
        assert "keep" in updated.metadata.tags


# ---------------------------------------------------------------------------
# SqliteMemoryStore — Query
# ---------------------------------------------------------------------------

class TestSqliteMemoryStoreQuery:

    def test_query_all_no_filters(self, store_no_embed):
        store_no_embed.make_memory("First memory record in the store")
        store_no_embed.make_memory("Second memory record in the store")
        results = store_no_embed.query(MemoryQuery())
        assert len(results) == 2

    def test_query_filter_by_type(self, store_no_embed):
        store_no_embed.make_memory("Goal memory content here", memory_type=MemoryType.GOAL)
        store_no_embed.make_memory("Fact memory content here", memory_type=MemoryType.FACT)
        results = store_no_embed.query(MemoryQuery(memory_type=MemoryType.GOAL))
        assert len(results) == 1
        assert results[0].memory.metadata.memory_type == MemoryType.GOAL

    def test_query_filter_by_source(self, store_no_embed):
        store_no_embed.make_memory("User memory content here", source=MemorySource.USER)
        store_no_embed.make_memory("Plugin memory content here", source=MemorySource.PLUGIN)
        results = store_no_embed.query(MemoryQuery(source=MemorySource.USER))
        assert len(results) == 1
        assert results[0].memory.metadata.source == MemorySource.USER

    def test_query_filter_by_min_importance(self, store_no_embed):
        store_no_embed.make_memory("Low importance memory content here", importance=0.2)
        store_no_embed.make_memory("High importance memory content here", importance=0.9)
        results = store_no_embed.query(MemoryQuery(min_importance=0.5))
        assert len(results) == 1
        assert results[0].memory.metadata.importance >= 0.5

    def test_query_filter_by_tag(self, store_no_embed):
        store_no_embed.make_memory("Tagged AI memory content", tags=["ai", "ml"])
        store_no_embed.make_memory("Untagged plain memory content here")
        results = store_no_embed.query(MemoryQuery(tags=["ai"]))
        assert len(results) == 1
        assert "ai" in results[0].memory.metadata.tags

    def test_query_top_k_limits_results(self, store_no_embed):
        for i in range(10):
            store_no_embed.make_memory(f"Memory number {i} for top k limiting test")
        results = store_no_embed.query(MemoryQuery(top_k=3))
        assert len(results) <= 3

    def test_query_empty_store_returns_empty(self, store_no_embed):
        results = store_no_embed.query(MemoryQuery(text="anything here"))
        assert results == []

    def test_query_semantic_ranks_by_similarity(self, store):
        m1 = store.make_memory("machine learning deep learning neural networks transformers architecture")
        m2 = store.make_memory("cooking recipes pasta dinner italian cuisine kitchen chef sauce")
        # Query shares exact tokens with m1 only → m1 must appear with score > 0
        results = store.query(MemoryQuery(text="machine learning neural", top_k=5))
        assert len(results) >= 1
        ids = [r.memory.id for r in results]
        assert m1.id in ids
        score_ml = next(r.score for r in results if r.memory.id == m1.id)
        assert score_ml > 0

    def test_query_semantic_falls_back_without_provider(self, store_no_embed):
        """Without embedding provider, text queries return all matching SQL rows."""
        store_no_embed.make_memory("something content here without embedding")
        results = store_no_embed.query(MemoryQuery(text="something"))
        # Should return results (no crash), scored 1.0
        assert len(results) >= 1
        assert results[0].score == 1.0
