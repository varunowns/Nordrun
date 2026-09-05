"""
Memory Plugin — Phase 1
------------------------
Exposes Nordrun's MemoryService through the event bus so other plugins
and the scheduler can store, search, update, and delete memories without
importing service modules directly.

Event catalog
-------------
  memory.store            — store a new memory
  memory.observe          — lifecycle entry-point (evaluates before storing)
  memory.search           — search memories (semantic + filters)
  memory.get              — fetch a single memory by ID
  memory.update           — update an existing memory
  memory.forget           — delete a memory by ID
  memory.entity.add       — add/update a knowledge-graph entity
  memory.entity.search    — search entities by name fragment / type
  memory.entity.get       — fetch entity by ID
  memory.relationship.add — add a directed relationship between entities
  memory.neighbours       — get neighbour entities from the graph

All handlers return plain dicts safe to forward through event bus results.
"""

import logging

from core.event_bus import EventBus
from services.memory.models import MemorySource, MemoryType
from services.memory_service import get_memory

log = logging.getLogger(__name__)

_DEFAULT_IMPORTANCE: float = 0.5
_DEFAULT_TOP_K: int = 10


def _svc():
    return get_memory()


# ---------------------------------------------------------------------------
# Memory handlers
# ---------------------------------------------------------------------------

def handle_store(payload: dict) -> dict:
    """Store a new memory explicitly.

    payload keys:
      content: str          (required)
      memory_type: str      (default "note")
      source: str           (default "user")
      importance: float     (default 0.5)
      confidence: float     (default 1.0)
      tags: list[str]
      plugin_source: str
      vault_path: str
    """
    content = payload.get("content", "")
    if not content:
        return {"error": "memory.store requires 'content'"}
    mem = _svc().store(
        content=content,
        memory_type=MemoryType(payload.get("memory_type", MemoryType.NOTE.value)),
        source=MemorySource(payload.get("source", MemorySource.USER.value)),
        importance=float(payload.get("importance", _DEFAULT_IMPORTANCE)),
        confidence=float(payload.get("confidence", 1.0)),
        tags=payload.get("tags"),
        plugin_source=payload.get("plugin_source", ""),
        vault_path=payload.get("vault_path", ""),
    )
    return {"stored": True, "memory": mem.to_dict()}


def handle_observe(payload: dict) -> dict:
    """Lifecycle entry-point — evaluates content quality before storing.

    payload keys:
      content: str
      source: str           (default "plugin")
      memory_type: str      (default "note")
      importance: float
      plugin_source: str
      tags: list[str]
    """
    result = _svc().observe(
        content=payload.get("content", ""),
        source=MemorySource(payload.get("source", MemorySource.PLUGIN.value)),
        memory_type=MemoryType(payload.get("memory_type", MemoryType.NOTE.value)),
        importance=float(payload.get("importance", _DEFAULT_IMPORTANCE)),
        plugin_source=payload.get("plugin_source", ""),
        tags=payload.get("tags"),
    )
    if result is None:
        return {"stored": False, "reason": "content rejected by lifecycle filter"}
    return {"stored": True, "memory": result.to_dict()}


def handle_search(payload: dict) -> dict:
    """Search memories with optional semantic query + structured filters.

    payload keys:
      text: str             — semantic query (optional)
      memory_type: str      — filter by type
      tags: list[str]       — all tags must be present
      source: str           — filter by source
      min_importance: float
      top_k: int
      min_score: float
    """
    results = _svc().search(
        text=payload.get("text", ""),
        memory_type=MemoryType(payload["memory_type"]) if "memory_type" in payload else None,
        tags=payload.get("tags"),
        source=MemorySource(payload["source"]) if "source" in payload else None,
        min_importance=float(payload.get("min_importance", 0.0)),
        top_k=int(payload.get("top_k", _DEFAULT_TOP_K)),
        min_score=float(payload.get("min_score", 0.0)),
    )
    return {
        "results": [{"memory": r.memory.to_dict(), "score": r.score} for r in results],
        "result_count": len(results),
        "query": payload.get("text", ""),
    }


def handle_get(payload: dict) -> dict:
    """Fetch a single memory by ID.

    payload keys:
      memory_id: str        (required)
    """
    mem_id = payload.get("memory_id", "")
    if not mem_id:
        return {"error": "memory.get requires 'memory_id'"}
    mem = _svc().get(mem_id)
    if mem is None:
        return {"found": False, "memory_id": mem_id}
    return {"found": True, "memory": mem.to_dict()}


def handle_update(payload: dict) -> dict:
    """Update an existing memory's content and/or metadata.

    payload keys:
      memory_id: str        (required)
      content: str          (optional — new content)
      metadata: dict        (optional — keys: importance, confidence, tags,
                             memory_type, source, plugin_source, vault_path)
    """
    mem_id = payload.get("memory_id", "")
    if not mem_id:
        return {"error": "memory.update requires 'memory_id'"}
    updated = _svc().update(
        memory_id=mem_id,
        content=payload.get("content"),
        metadata_updates=payload.get("metadata"),
    )
    if updated is None:
        return {"updated": False, "memory_id": mem_id}
    return {"updated": True, "memory": updated.to_dict()}


def handle_forget(payload: dict) -> dict:
    """Delete a memory by ID.

    payload keys:
      memory_id: str        (required)
    """
    mem_id = payload.get("memory_id", "")
    if not mem_id:
        return {"error": "memory.forget requires 'memory_id'"}
    deleted = _svc().forget(mem_id)
    return {"forgotten": deleted, "memory_id": mem_id}


# ---------------------------------------------------------------------------
# Knowledge graph handlers
# ---------------------------------------------------------------------------

def handle_entity_add(payload: dict) -> dict:
    """Add or update a knowledge-graph entity (upserts on name+type).

    payload keys:
      name: str             (required)
      entity_type: str      (required — e.g. "person", "project", "skill")
      description: str
      tags: list[str]
    """
    name = payload.get("name", "")
    entity_type = payload.get("entity_type", "")
    if not name or not entity_type:
        return {"error": "memory.entity.add requires 'name' and 'entity_type'"}
    entity = _svc().add_entity(
        name=name,
        entity_type=entity_type,
        description=payload.get("description", ""),
        tags=payload.get("tags"),
    )
    return {"entity": entity.to_dict()}


def handle_entity_search(payload: dict) -> dict:
    """Search entities by partial name and/or type.

    payload keys:
      name: str             — fragment to search (optional)
      entity_type: str      — filter by type (optional)
      limit: int            (default 20)
    """
    entities = _svc().search_entities(
        name_fragment=payload.get("name", ""),
        entity_type=payload.get("entity_type"),
        limit=int(payload.get("limit", 20)),
    )
    return {"entities": [e.to_dict() for e in entities], "count": len(entities)}


def handle_entity_get(payload: dict) -> dict:
    """Fetch an entity by ID.

    payload keys:
      entity_id: str        (required)
    """
    entity_id = payload.get("entity_id", "")
    if not entity_id:
        return {"error": "memory.entity.get requires 'entity_id'"}
    entity = _svc().get_entity(entity_id)
    if entity is None:
        return {"found": False, "entity_id": entity_id}
    return {"found": True, "entity": entity.to_dict()}


def handle_relationship_add(payload: dict) -> dict:
    """Add a directed relationship between two entities.

    payload keys:
      source_id: str        (required)
      target_id: str        (required)
      relation_type: str    (required — e.g. "works_on", "knows")
      weight: float         (default 1.0)
      description: str
    """
    source_id = payload.get("source_id", "")
    target_id = payload.get("target_id", "")
    relation_type = payload.get("relation_type", "")
    if not source_id or not target_id or not relation_type:
        return {"error": "memory.relationship.add requires source_id, target_id, relation_type"}
    rel = _svc().add_relationship(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=float(payload.get("weight", 1.0)),
        description=payload.get("description", ""),
    )
    return {"relationship": rel.to_dict()}


def handle_neighbours(payload: dict) -> dict:
    """Get entities reachable from an entity within max_depth hops.

    payload keys:
      entity_id: str        (required)
      relation_type: str    (optional — filter by type)
      max_depth: int        (default 1)
    """
    entity_id = payload.get("entity_id", "")
    if not entity_id:
        return {"error": "memory.neighbours requires 'entity_id'"}
    neighbours = _svc().get_neighbours(
        entity_id=entity_id,
        relation_type=payload.get("relation_type"),
        max_depth=int(payload.get("max_depth", 1)),
    )
    return {"entity_id": entity_id, "neighbours": [e.to_dict() for e in neighbours]}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(event_bus: EventBus, plugin_name: str = "", config: dict | None = None) -> None:
    """Called once at startup to wire this plugin into the event bus."""
    global _DEFAULT_IMPORTANCE, _DEFAULT_TOP_K
    cfg = config or {}
    if "default_importance" in cfg:
        _DEFAULT_IMPORTANCE = float(cfg["default_importance"])
    if "default_top_k" in cfg:
        _DEFAULT_TOP_K = int(cfg["default_top_k"])

    event_bus.subscribe("memory.store", handle_store, plugin_name=plugin_name)
    event_bus.subscribe("memory.observe", handle_observe, plugin_name=plugin_name)
    event_bus.subscribe("memory.search", handle_search, plugin_name=plugin_name)
    event_bus.subscribe("memory.get", handle_get, plugin_name=plugin_name)
    event_bus.subscribe("memory.update", handle_update, plugin_name=plugin_name)
    event_bus.subscribe("memory.forget", handle_forget, plugin_name=plugin_name)
    event_bus.subscribe("memory.entity.add", handle_entity_add, plugin_name=plugin_name)
    event_bus.subscribe("memory.entity.search", handle_entity_search, plugin_name=plugin_name)
    event_bus.subscribe("memory.entity.get", handle_entity_get, plugin_name=plugin_name)
    event_bus.subscribe("memory.relationship.add", handle_relationship_add, plugin_name=plugin_name)
    event_bus.subscribe("memory.neighbours", handle_neighbours, plugin_name=plugin_name)
    log.debug("Memory plugin registered (%d events)", 11)
