"""
Knowledge Graph Store — Phase 1
---------------------------------
Lightweight entity/relationship graph backed by SQLite.

Uses the `entities` and `relationships` tables created by
storage.db._init_memory_schema().  No graph database dependency — the
data volume at Phase 1 makes in-DB adjacency lists more than adequate.

Entity types (open-ended strings; common values listed below):
  person, project, repository, skill, goal, decision, organization, tool

Relationship types (open-ended strings; common values):
  works_on, owns, contributes_to, knows, uses, depends_on,
  demonstrates, decided_by, related_to

The graph is accessible through MemoryService so plugins never import
storage.graph directly.  Future phases can add richer traversal
(BFS, shortest path) without touching the service interface.

Thread safety: same rules as SqliteMemoryStore — one instance per thread,
sharing the same connection as the rest of the DB layer.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from storage.db import _init_memory_schema

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: str        # e.g. "person", "project", "skill"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Relationship:
    """A directed edge in the knowledge graph."""
    id: str
    source_id: str          # Entity.id
    target_id: str          # Entity.id
    relation_type: str      # e.g. "works_on", "knows"
    weight: float = 1.0     # Relationship strength [0.0, 1.0]
    description: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "description": self.description,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """SQLite-backed knowledge graph.

    All writes are idempotent by name+type for entities (upsert on
    natural key) and by id for relationships.  This means calling
    add_entity("Varun", "person") twice produces one row.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        _init_memory_schema(conn)
        log.debug("GraphStore initialised")

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        tags: list[str] | None = None,
        entity_id: str | None = None,
    ) -> Entity:
        """Create or update an entity.

        Upserts on (name, entity_type) — calling with the same name and
        type updates description/tags without creating a duplicate.
        Returns the Entity (existing or newly created).
        """
        now = _now_iso()
        tags_str = ",".join(tags or [])

        # Check for an existing entity with the same name + type
        row = self._conn.execute(
            "SELECT id FROM entities WHERE name=? AND entity_type=?",
            (name, entity_type),
        ).fetchone()

        if row:
            eid = row[0]
            self._conn.execute(
                "UPDATE entities SET description=?, tags=?, updated_at=? WHERE id=?",
                (description, tags_str, now, eid),
            )
            self._conn.commit()
        else:
            eid = entity_id or str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO entities
                    (id, name, entity_type, description, tags, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (eid, name, entity_type, description, tags_str, now, now),
            )
            self._conn.commit()

        log.debug("Entity upserted id=%s name=%s type=%s", eid, name, entity_type)
        return self.get_entity(eid)  # type: ignore[return-value]

    def get_entity(self, entity_id: str) -> Entity | None:
        """Fetch an entity by ID."""
        row = self._conn.execute(
            "SELECT id, name, entity_type, description, tags, created_at, updated_at "
            "FROM entities WHERE id=?",
            (entity_id,),
        ).fetchone()
        return _row_to_entity(row) if row else None

    def get_entity_by_name(self, name: str, entity_type: str | None = None) -> Entity | None:
        """Fetch an entity by name (and optionally type)."""
        if entity_type:
            row = self._conn.execute(
                "SELECT id, name, entity_type, description, tags, created_at, updated_at "
                "FROM entities WHERE name=? AND entity_type=?",
                (name, entity_type),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id, name, entity_type, description, tags, created_at, updated_at "
                "FROM entities WHERE name=? LIMIT 1",
                (name,),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def search_entities(
        self,
        name_fragment: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Search entities by partial name and/or type."""
        conditions: list[str] = []
        params: list[Any] = []
        if name_fragment:
            conditions.append("name LIKE ?")
            params.append(f"%{name_fragment}%")
        if entity_type:
            conditions.append("entity_type=?")
            params.append(entity_type)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._conn.execute(
            f"SELECT id, name, entity_type, description, tags, created_at, updated_at "  # noqa: S608
            f"FROM entities {where} ORDER BY name LIMIT ?",
            params + [limit],
        ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        row = self._conn.execute(
            "SELECT id FROM entities WHERE id=?", (entity_id,)
        ).fetchone()
        if not row:
            return False
        self._conn.execute(
            "DELETE FROM relationships WHERE source_id=? OR target_id=?",
            (entity_id, entity_id),
        )
        self._conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
        self._conn.commit()
        log.debug("Deleted entity id=%s and its relationships", entity_id)
        return True

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        description: str = "",
        rel_id: str | None = None,
    ) -> Relationship:
        """Create a directed relationship.

        Upserts on (source_id, target_id, relation_type) so calling the
        same triple twice updates weight/description rather than
        creating duplicates.
        """
        now = _now_iso()
        row = self._conn.execute(
            "SELECT id FROM relationships WHERE source_id=? AND target_id=? AND relation_type=?",
            (source_id, target_id, relation_type),
        ).fetchone()

        if row:
            rid = row[0]
            self._conn.execute(
                "UPDATE relationships SET weight=?, description=? WHERE id=?",
                (weight, description, rid),
            )
            self._conn.commit()
        else:
            rid = rel_id or str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO relationships
                    (id, source_id, target_id, relation_type, weight, description, created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (rid, source_id, target_id, relation_type, weight, description, now),
            )
            self._conn.commit()

        log.debug("Relationship upserted id=%s %s -[%s]-> %s", rid, source_id, relation_type, target_id)
        return self.get_relationship(rid)  # type: ignore[return-value]

    def get_relationship(self, rel_id: str) -> Relationship | None:
        row = self._conn.execute(
            "SELECT id, source_id, target_id, relation_type, weight, description, created_at "
            "FROM relationships WHERE id=?",
            (rel_id,),
        ).fetchone()
        return _row_to_relationship(row) if row else None

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "outbound",
        relation_type: str | None = None,
    ) -> list[Relationship]:
        """Return relationships for an entity.

        direction: "outbound" (entity is source), "inbound" (entity is
        target), or "both".
        """
        conditions: list[str] = []
        params: list[Any] = []

        if direction == "outbound":
            conditions.append("source_id=?")
            params.append(entity_id)
        elif direction == "inbound":
            conditions.append("target_id=?")
            params.append(entity_id)
        else:  # both
            conditions.append("(source_id=? OR target_id=?)")
            params.extend([entity_id, entity_id])

        if relation_type:
            conditions.append("relation_type=?")
            params.append(relation_type)

        where = "WHERE " + " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT id, source_id, target_id, relation_type, weight, description, created_at "  # noqa: S608
            f"FROM relationships {where} ORDER BY weight DESC",
            params,
        ).fetchall()
        return [_row_to_relationship(r) for r in rows]

    def delete_relationship(self, rel_id: str) -> bool:
        row = self._conn.execute(
            "SELECT id FROM relationships WHERE id=?", (rel_id,)
        ).fetchone()
        if not row:
            return False
        self._conn.execute("DELETE FROM relationships WHERE id=?", (rel_id,))
        self._conn.commit()
        return True

    def get_neighbours(
        self,
        entity_id: str,
        relation_type: str | None = None,
        max_depth: int = 1,
    ) -> list[Entity]:
        """Return entities reachable from entity_id within max_depth hops.

        Phase 1 implements depth=1 traversal (direct neighbours) and
        iterative BFS up to max_depth.  Returns a deduplicated list
        excluding the starting entity.
        """
        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}
        result: list[Entity] = []

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for eid in frontier:
                rels = self.get_relationships(eid, direction="both", relation_type=relation_type)
                for r in rels:
                    neighbour_id = r.target_id if r.source_id == eid else r.source_id
                    if neighbour_id not in visited:
                        visited.add(neighbour_id)
                        next_frontier.add(neighbour_id)
                        entity = self.get_entity(neighbour_id)
                        if entity:
                            result.append(entity)
            frontier = next_frontier
            if not frontier:
                break

        return result

    def count_entities(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def count_relationships(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def _row_to_entity(row: tuple) -> Entity:
    eid, name, entity_type, description, tags_str, created_at, updated_at = row
    return Entity(
        id=eid,
        name=name,
        entity_type=entity_type,
        description=description or "",
        tags=[t for t in (tags_str or "").split(",") if t],
        created_at=created_at or "",
        updated_at=updated_at or "",
    )


def _row_to_relationship(row: tuple) -> Relationship:
    rid, source_id, target_id, relation_type, weight, description, created_at = row
    return Relationship(
        id=rid,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=float(weight),
        description=description or "",
        created_at=created_at or "",
    )
