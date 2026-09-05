# Architecture

## Current

```
nordrun/
├── config.py                  # Environment-based settings
├── main.py                    # CLI entrypoint, auto-discovers plugins
├── core/
│   ├── event_bus.py           # In-memory pub/sub dispatcher
│   ├── plugin_loader.py       # Auto-discovery + registration
│   └── plugin_registry.py     # Permission declarations and enforcement
├── services/
│   ├── obsidian_service.py    # Vault read/write (enforces vault:read, vault:write)
│   ├── llm_service.py         # Anthropic API wrapper (enforces llm:call)
│   ├── embedding_service.py   # TF-IDF vectorizer + cosine similarity search
│   └── context_service.py     # Unified plugin-facing API (aggregates all services)
├── services/
│   └── memory/                # Phase 1 memory abstraction package
│       ├── models.py          # Memory, MemoryType, MemoryMetadata, MemoryQuery, ...
│       ├── base.py            # AbstractMemoryStore, AbstractEmbeddingProvider
│       └── embedding.py       # TfIdfEmbeddingProvider (wraps the TF-IDF vectorizer)
│   └── memory_service.py      # MemoryService: unified memory + graph API (get_memory())
├── storage/
│   ├── db.py                  # SQLite metadata index (notes + Phase 1 memory schema)
│   ├── memory_store.py        # SqliteMemoryStore (implements AbstractMemoryStore)
│   └── graph.py               # GraphStore: entities + relationships (knowledge graph)
├── automation/
│   └── scheduler.py           # Background job loop with YAML schedule config
└── plugins/
    ├── career/                # note.summarize
    ├── github/                # repo.commits.summarize
    ├── resume/                # resume.review
    ├── search/                # note.search, note.reindex
    ├── ctx2img/               # note.toimage
    ├── learning/              # learning.digest
    └── memory/                # memory.store, memory.search, memory.entity.add, ...
```

## Key relationships

- **Event bus**: in-memory pub/sub. Plugins subscribe to events; CLI commands
  or the scheduler publish events. Handlers run synchronously in registration order.
- **Plugin permissions**: Each `manifest.yaml` declares permissions
  (`vault:read`, `vault:write`, `llm:call`). Services check these at runtime
  via the `@require()` decorator. A plugin can only call a service if it
  declared the matching permission.
- **Vault containment**: every vault-relative path is resolved through
  `_resolve_vault_path` and rejected with `ValueError` if it escapes the
  vault root. Plugin event payloads are untrusted, so `../../` traversal
  can never read, write, or delete files outside the vault.
- **ContextService**: The unified interface plugins should use. Aggregates
  `obsidian_service` (read/write), `NoteIndex` (tag queries, metadata),
  and `EmbeddingIndex` (semantic search). New plugins should prefer
  `from services.context_service import get_context` over importing
  individual services.
- **Storage**: SQLite database lives at `VAULT_PATH/.nordrun/metadata.db`.
  The vault's markdown files remain the source of truth — SQLite is a
  searchable index, not a replacement.
- **Search index is idempotent**: `EmbeddingIndex` tracks each note's
  tokens in a `doc_tokens` table, so re-indexing an existing note replaces
  its terms instead of double-counting them (which previously inflated
  IDF statistics and degraded ranking over time). Corpus stats are
  reconciled from `doc_tokens` on load, so the in-memory vectorizer always
  matches the persisted corpus. `reindex` is self-healing: notes that no
  longer exist on disk are pruned from both the metadata and embedding
  tables.

## Event catalog

| Event | Publisher | Plugin handler(s) | Payload |
|-------|-----------|-------------------|---------|
| `note.summarize` | CLI | career | source_note, output_note |
| `repo.commits.summarize` | CLI, scheduler | github | repo, count, output_note |
| `resume.review` | CLI | resume | resume_note, output_note |
| `note.search` | CLI | search | query, top_k |
| `note.reindex` | CLI | search | (none) |
| `note.toimage` | CLI | ctx2img | source_note, style |
| `learning.digest` | CLI | learning | tag, output_note |
| `memory.store` | CLI, plugins | memory | content, memory_type, importance, tags |
| `memory.observe` | plugins | memory | content, source, importance |
| `memory.search` | CLI, plugins | memory | text, memory_type, tags, top_k |
| `memory.get` | CLI, plugins | memory | memory_id |
| `memory.update` | plugins | memory | memory_id, content, metadata |
| `memory.forget` | plugins | memory | memory_id |
| `memory.entity.add` | plugins | memory | name, entity_type, description |
| `memory.entity.search` | plugins | memory | name, entity_type, limit |
| `memory.entity.get` | plugins | memory | entity_id |
| `memory.relationship.add` | plugins | memory | source_id, target_id, relation_type |
| `memory.neighbours` | plugins | memory | entity_id, relation_type, max_depth |

## Memory & Knowledge (Phase 1)

Nordrun has persistent, semantic, structured personal memory exposed through
a `MemoryService` abstraction. The rest of the system depends on the memory
*interfaces*, never on a concrete vector DB or embedding model, so the
implementation can be swapped without rewriting callers.

```
Nordrun
  → MemoryService            (services/memory_service.py — get_memory() singleton)
      → AbstractMemoryStore  (services/memory/base.py)
          → SqliteMemoryStore (storage/memory_store.py)
      → AbstractEmbeddingProvider (services/memory/base.py)
          → TfIdfEmbeddingProvider (services/memory/embedding.py)
      → GraphStore           (storage/graph.py — entities + relationships)
```

**Data models** (`services/memory/models.py`): `Memory`, `MemoryType`
(fact, preference, person, project, goal, decision, experience, skill,
note), `MemorySource` (user, plugin, inferred, vault), `MemoryMetadata`
(type, source, importance, confidence, tags, provenance, related entities),
`MemoryQuery`, `MemoryResult`.

**Semantic memory**: The `TfIdfEmbeddingProvider` reuses the Phase 0 TF-IDF
vectorizer but stores memory vectors in dedicated tables
(`memory_embeddings`, `memory_embedding_config`, `memory_doc_tokens`) so
they never collide with note embeddings. Vectors are rebuilt to the current
vocabulary on `save()` so early-indexed memories are never lost to
vocabulary growth. The embedding provider is abstracted: a future
sentence-transformers or API-based provider is a config change, not a
rewrite.

**Long-term structured memory**: `SqliteMemoryStore` persists memories in
the `memories` table with full metadata as filterable columns. Queries
support semantic ranking + structured filters (type, source, tags,
min_importance) + top_k + similarity threshold.

**Knowledge graph**: `GraphStore` stores `entities` (person, project,
repository, skill, goal, decision, organization, tool) and directed
`relationships` (works_on, owns, contributes_to, knows, uses, depends_on,
demonstrates, decided_by, related_to). Upserts are idempotent on natural
keys. `get_neighbours()` does BFS traversal up to a configurable depth.
The graph is reached through `MemoryService`, never as an isolated subsystem.

**Memory lifecycle**: `MemoryService.observe()` is the controlled entry
point that decides whether content becomes a memory (Phase 1 rule: reject
empty / <10-char content, store otherwise). Memory is never an
uncontrolled transcript dump — creation goes through explicit rules.
LLM-based extraction and consolidation are deferred to Phase 2, but the
`observe → store` seam is in place.

**Context integration**: `ContextService.get_memory_context(query)` retrieves
relevant memories + graph entities to enrich LLM prompts. It fails soft —
if memory is unavailable it returns empty lists rather than breaking normal
plugin behaviour. The LLM service is never coupled to a concrete memory DB.

**Permissions**: Two new permissions — `memory:read` and `memory:write` —
are enforced via `@require()` on every `MemoryService` method. An LLM or
plugin cannot bypass the permission layer to touch memory.

**Storage location**: All memory tables share the same
`VAULT_PATH/.nordrun/metadata.db` as the note index, created lazily on
first use via `storage.db._init_memory_schema()`.

## Scheduler (Hermes)

The scheduler runs plugin events on a timer. Schedule config is stored in
`VAULT_PATH/.nordrun/schedules.yaml`. Run with `python main.py serve`.

Each schedule's `last_run` timestamp is persisted in `schedules.yaml`,
so restarting the daemon does not immediately re-fire every enabled
schedule — only schedules whose interval has elapsed since their last
run execute.

Default schedule: daily GitHub commits summary for `varunowns/Nordrun`.

## Plugin contract

A plugin is a folder under `plugins/` with `manifest.yaml` + `plugin.py`
that exports `register(event_bus, plugin_name="", config=None)`. The
loader passes the plugin's manifest `config` dict to `register()` so a
plugin's defaults live in its manifest, not hardcoded in code. The
manifest declares the plugin's contract, enforced at load time by
`validate_manifest()` in `core/plugin_loader.py`:

| Field | Required | Shape |
|-------|----------|-------|
| `name` | yes | kebab-case string, matches the plugin dir |
| `version` | yes | semver `x.y.z` |
| `description` | yes | non-empty string |
| `subscribes` / `publishes` | no | list of non-empty event names |
| `permissions` | no | string or list of known permissions (`vault:read`, `vault:write`, `llm:call`, `network:call`, `memory:read`, `memory:write`) |
| `commands` | no | `cmd:event[:help]` entries (semicolon- or list-separated) |
| `config` | no | free-form plugin config |

Invalid or unparseable manifests are skipped loudly at load — a plugin is
never half-loaded, and one bad plugin never blocks the others.
`load_and_register(bus, plugins_dir=...)` returns a `PluginLoadReport`
(registered / skipped / failed) so callers can surface *why* a plugin is
missing; `discover_plugins(plugins_dir=...)` returns unvalidated metadata.
Both accept a custom directory for testing.

## Design decisions

- **One milestone at a time**: No speculative architecture. Each piece is
  built only when a real plugin needs it.
- **Plugin contract over convention**: manifests are validated at load time
  (see `Projects/Nordrun/Decisions/ADR-005` in the vault). Invalid contracts
  are skipped loudly, never half-loaded.
- **Python over TypeScript**: Best library fit for markdown/SQLite/LLM SDKs.
  See `Projects/Nordrun/Decisions/ADR-001` in the vault.
- **Vertical slice first**: One working plugin before full architecture.
  See ADR-002.
- **Semantic search before CLI polish**: Real content from multiple plugins
  needed a search layer before quality-of-life CLI improvements.
  See ADR-004.

## Deferred to Phase 2 (memory system)

- LLM-based memory extraction (Phase 1 uses a simple length/quality rule in
  `observe()`)
- Memory consolidation / forgetting policies (`MEMORY_MAX_RECORDS` config
  exists but no automatic consolidation runs yet)
- A real semantic embedding model (Phase 1 ships the abstracted TF-IDF
  provider; sentence-transformers / API providers slot in behind
  `AbstractEmbeddingProvider`)
- Deeper graph reasoning (Phase 1 has entity/relationship CRUD + BFS
  traversal; no inference or path-ranking yet)

## Not yet built

- Plugin sandboxing/isolation
- Multi-user
- Web UI
- Real image generation
- Additional plugins (linkedin, portfolio, calendar, email, etc.)
