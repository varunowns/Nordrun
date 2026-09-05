# Nordrun — Setup & Usage Guide

A complete, tested walkthrough: from a fresh clone to running every
feature through the CLI. Written against the code as it exists today
(all commands below were verified on a Windows machine).

---

## 1. What Nordrun is

Nordrun is a personal AI platform that treats an **Obsidian vault as
permanent memory** and exposes features through **small plugins** that
subscribe to events on a shared event bus.

- Your vault stays the source of truth — Nordrun only reads/writes
  markdown notes through a single service.
- One LLM provider wrapper — plugins never touch an API key or provider
  SDK directly.
- Six plugins ship out of the box (see the command table below).

## 2. Requirements

| Thing | Need | Notes |
|-------|------|-------|
| Python | 3.10+ | Verified on 3.14 (numpy prints a harmless MINGW warning) |
| An Obsidian vault | Any existing vault | Must exist on disk; nothing gets installed into it |
| A GitHub repo | Any public repo | Only if you want the `commits` plugin (public API, no token needed) |
| An LLM key OR proxy | `ANTHROPIC_API_KEY` or a 9router-style proxy | All plugin commands except `search`/`reindex` call the LLM |

> **No GitHub token required.** The `commits` plugin uses GitHub's public
> REST API (rate-limited to 60 requests/hour per IP).

## 3. Install & configure

```bash
# 1. Clone the repo
git clone https://github.com/<you>/Nordrun.git
cd Nordrun

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your config from the template
cp .env.example .env
```

Now open `.env` and set the critical values:

```ini
# The ROOT folder of your Obsidian vault — the folder that contains the
# actual markdown notes. Use forward slashes even on Windows.
NORDRUN_VAULT_PATH=V:/Obsidian/Obsidian Vault

# Option A: direct Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Option B (mutually exclusive with A): a 9router-style proxy
# NORDRUN_LLM_BASE_URL=http://localhost:20128/v1
# NORDRUN_LLM_API_KEY=sk-...
```

Three rules to avoid the classic mistakes:

1. `NORDRUN_VAULT_PATH` must point at the **vault root that holds your
   notes** — not the container folder above it. `V:/Obsidian/Obsidian Vault`,
   not `V:/Obsidian`. Pointing at the container silently routes notes into
   the wrong top-level folders.
2. Use **forward slashes** (`V:/Obsidian/...`), not backslashes.
3. If you use a proxy, leave `ANTHROPIC_API_KEY` blank and fill the
   two `NORDRUN_LLM_*` values instead. Setting `BASE_URL` without an API
   key is a startup error.

### Verify the install

```bash
python main.py --help
```

You should see the plugin-derived commands:

```
{commits,digest,review-resume,search,summarize,toimage,reindex,serve,list-plugins}
```

If instead you get a `Configuration errors:` block, the message tells
you exactly which `.env` value is missing or invalid.

## 4. All commands at a glance

```bash
python main.py --help              # see every command

# Plugins
python main.py summarize "Career/README.md"            # summarize a note (+ action items)
python main.py commits "varunowns/Nordrun"             # GitHub commit summary
python main.py commits "varunowns/Nordrun" --count 5  # ... limit to 5 commits
python main.py review-resume "Career/Resume.md"        # review resume vs recent career notes
python main.py search "machine learning"               # semantic search over the vault
python main.py search "machine learning" --top-k 10    # ... return 10 results
python main.py toimage "Career/README.md"              # generate an image from a note
python main.py digest                                  # weekly learning digest + review questions

# Built-ins
python main.py reindex                # re-index notes Nordrun already knows about
python main.py reindex --scan-vault   # scan the whole vault for notes (do this FIRST)
python main.py list-plugins           # loaded plugins, permissions, events
python main.py serve                  # run the background scheduler (Hermes)
```

## 5. First run — recommended order

The one thing that trips up first-time users: **run the scan once so
semantic search has content.**

```bash
# 1. Load everything into the search index
python main.py reindex --scan-vault
# e.g. "Re-indexed 37/37 notes"

# 2. Confirm the search layer works (does NOT call the LLM)
python main.py search "machine learning"
```

> Why `--scan-vault`? Plain `python main.py reindex` only processes
> notes Nordrun has already written. Your pre-existing vault notes are
> invisible to it until you scan.

### Then try an LLM command

```bash
# Summarize a note — writes <note name>-summary.md next to it
python main.py summarize "Career/README.md"
# Output: Wrote to: V:\...\Career\README-summary.md
#         plus the summary + action items printed to the terminal
```

If this succeeds, your LLM config is working end to end.

## 6. What each plugin does (and its output)

| Command | Plugin | What it does | Where output goes |
|---------|--------|--------------|-------------------|
| `summarize <note>` | career | Summarizes the note and extracts action items | `<vault>/<note without .md>-summary.md` |
| `commits <owner/repo>` | github | Fetches recent commits via the public GitHub API, groups them into a summary | `<vault>/Dev/<repo>-commits.md` |
| `review-resume <note>` | resume | Reviews a resume against recent career summaries in the index | `<vault>/<note without .md>-review.md` |
| `search "<query>"` | search | Semantic search (TF-IDF + cosine similarity), ranked by score | Terminal output only |
| `toimage <note>` | ctx2img | Summarizes the note, then generates a visual and links it back into the note | `<vault>/workspace/ctx2img/*.svg`, linked as `## Visual` in the source note |
| `digest` | learning | Reads notes tagged `#learning`, summarizes the week, writes a digest with review questions | `<vault>/Learning/Digests/weekly-<date>.md` |

### Optional `--out` flag

Most plugin commands accept `--out <path>` to control where the result
note is written:

```bash
python main.py commits "varunowns/Nordrun" --out "Dev/Nordrun-commits.md"
```

(`digest` accepts `--out <path>` and `--tag <tag>`; `toimage` defines its
own output location and doesn't accept `--out`.)

### The search command

```bash
python main.py search "machine learning"
```

```
Search results for: "machine learning" (5 found)

  1. Career/Resume.md  (score: 0.1749)
     Title: Varun's Resume
  2. Career/README.md  (score: 0.1362)
     Title: Career Overview
  ...
```

Results show the vault-relative path, similarity score (0–1), title,
tags, and which plugin produced the note.

### Honest caveat: `toimage`

`toimage` does **not** generate real images yet. It summarizes the note
via the LLM, then writes a branded SVG placeholder (purple gradient +
the summary's first words) to `workspace/ctx2img/` and embeds it in the
source note with `![[...]]`. That placeholder pipeline is listed in
"Not yet built" in README.md. The summarization + link-back machinery is
real; the image generation is stubbed.

## 7. Automation: the scheduler (Hermes)

`python main.py serve` runs a background loop that wakes every 60 seconds
and fires any schedule whose interval has elapsed.

```bash
python main.py serve
```

- On first run it creates `schedules.yaml` under `<vault>/.nordrun/`
  with a template schedule: **daily GitHub commits summary for
  `varunowns/Nordrun`** (24h interval).
- Each schedule stores `last_run`, so restarting the daemon does **not**
  immediately re-fire every schedule — only ones whose interval has
  elapsed since the last run.
- Stop with `Ctrl+C`.

### Editing schedules

Open `<vault>/.nordrun/schedules.yaml`:

```yaml
schedules:
  - id: daily-github-commits
    label: Daily GitHub commits summary
    event: repo.commits.summarize
    payload:
      repo: varunowns/Nordrun
      count: 5
    interval_hours: 24
    enabled: true
```

Any event from the catalog can be scheduled. For example, a weekly
learning digest every 7 days:

```yaml
  - id: weekly-learning-digest
    label: Weekly learning digest
    event: learning.digest
    payload:
      tag: learning
    interval_hours: 168
    enabled: true
```

> The scheduler is a lightweight loop, not a production job system. It
> only runs while `serve` is running in a terminal.

## 8. Configuration errors — troubleshooting table

Nordrun validates `.env` on every startup and lists issues before doing
anything:

| Error message | Fix |
|---------------|-----|
| `NORDRUN_VAULT_PATH is not set` | Add it to `.env` |
| `NORDRUN_VAULT_PATH=... does not exist or is not a directory` | Fix the path; remember forward slashes and the vault ROOT |
| `No LLM API key configured` | Set `ANTHROPIC_API_KEY` or `NORDRUN_LLM_API_KEY` |
| `NORDRUN_LLM_BASE_URL is set but NORDRUN_LLM_API_KEY is missing` | Fill in `NORDRUN_LLM_API_KEY` |
| Plugins: `0 loaded` | Check your vault path and that `plugins/` is intact |
| `Re-indexed 0/0 notes` | Use `reindex --scan-vault` to include pre-existing notes |
| Search returns `0 found` | Run `reindex --scan-vault` first, then search again |

Other things you may see but can ignore:

- `Warning: Numpy built with MINGW-W64 ... experimental` — noisy but
  harmless; a side effect of the experimental Windows numpy wheel.
- `RuntimeWarning: invalid value encountered in exp2` (numpy internals) —
  same source, no effect on results.
- `�` characters in terminal titles/output — Windows console encoding
  of em-dashes; the actual notes on disk are fine.

## 9. Project layout — where things live

```
Nordrun/
├── config.py                  # Environment settings (reads .env)
├── main.py                    # CLI entrypoint, auto-discovers plugins
├── core/
│   ├── event_bus.py           # Pub/sub dispatcher
│   ├── plugin_loader.py       # Auto-discovers & validates plugins
│   └── plugin_registry.py     # Permission declarations & enforcement
├── services/
│   ├── obsidian_service.py    # All vault read/write (enforces vault:read/write)
│   ├── llm_service.py         # The ONLY place that talks to the LLM
│   ├── embedding_service.py   # TF-IDF semantic search
│   └── context_service.py     # Unified API plugins should use
├── storage/
│   └── db.py                  # SQLite index: <vault>/.nordrun/metadata.db
├── automation/
│   └── scheduler.py           # Background scheduler (Hermes)
└── plugins/
    ├── career/                # note.summarize
    ├── github/                # repo.commits.summarize
    ├── resume/                # resume.review
    ├── search/                # note.search, note.reindex
    ├── ctx2img/               # note.toimage
    └── learning/              # learning.digest
```

Nordrun's runtime state lives **inside your vault**:

| Path (vault-relative) | Purpose |
|-----------------------|---------|
| `.nordrun/metadata.db` | SQLite search index |
| `.nordrun/schedules.yaml` | Scheduler config + last-run timestamps |
| `workspace/ctx2img/` | Generated images |

Your markdown notes remain the source of truth; SQLite is only a
searchable index and can be rebuilt with `reindex --scan-vault`.

## 10. Extending with a new plugin

Every plugin follows the same contract: a folder under `plugins/`
containing `manifest.yaml` + `plugin.py` with a `register(event_bus)`
function. `plugins/career/` is the reference implementation.

```bash
# 1. Make the folder
mkdir plugins/example

# 2. manifest.yaml
#    name: example
#    version: 0.1.0
#    description: What it does
#    subscribes: [example.event]
#    commands: ex:example.event:Short help text
#    permissions: vault:read, vault:write, llm:call

# 3. plugin.py
#    from core.event_bus import EventBus
#    def handle(payload): ...
#    def register(event_bus, plugin_name="", config=None):
#        event_bus.subscribe("example.event", handle, plugin_name=plugin_name)
```

Rules enforced at load time:

- The manifest `name` must match the plugin folder.
- `version` must be semver (`x.y.z`); `description` non-empty.
- `permissions` must be from the known set (`vault:read`, `vault:write`,
  `llm:call`) — services refuse calls the manifest didn't declare.
- Invalid manifests are **skipped loudly**, never half-loaded — one bad
  plugin doesn't block the others.

Run `python main.py list-plugins` after adding one to confirm it loaded.

## 11. Notes on this document

- Commands were verified against the current tree (all six plugins load,
  `reindex --scan-vault` indexed 37 notes, `search` and `summarize`
  completed successfully).
- The scheduler's default schedule references `varunowns/Nordrun`; if
  your repo lives elsewhere, edit `schedules.yaml` accordingly.
- `toimage` is a placeholder pipeline today — treat its output as a stub
  until real image generation lands.
