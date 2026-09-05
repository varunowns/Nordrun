"""
Obsidian Service
----------------
Treats your existing Obsidian vault as Nordrun's permanent memory,
per the constitution rule "Obsidian is source of truth."

Provides vault read/write, frontmatter parsing, and vault scanning.
"""

import logging
import re
from pathlib import Path

from config import VAULT_PATH
from core.plugin_registry import require
from storage.db import NoteIndex, get_db

log = logging.getLogger(__name__)

# Lazy-init index so we don't force SQLite setup at import time.
_index: NoteIndex | None = None


def _get_index() -> NoteIndex:
    global _index
    if _index is None:
        _index = NoteIndex(get_db())
    return _index


def _resolve_vault_path(relative_path: str, vault_root: Path | None = None) -> Path:
    """Resolve a vault-relative path, rejecting paths that escape the vault.

    A payload like '../../evil.md' must never resolve to a file outside
    the vault root — plugin event payloads are treated as untrusted.
    Raises ValueError when the resolved target is not inside vault_root.

    vault_root defaults to VAULT_PATH read at call time (not bound at
    import), so tests that monkeypatch obsidian_service.VAULT_PATH get
    the isolated root.
    """
    root = (vault_root or VAULT_PATH).resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(
            f"Path '{relative_path}' resolves outside the vault root ({root})"
        )
    return target


@require("vault:read")
def read_note(relative_path: str) -> str:
    """
    Read a note's raw markdown content.
    relative_path is relative to the vault root, e.g. "Career/README.md"
    """
    note_path = _resolve_vault_path(relative_path)
    if not note_path.exists():
        raise FileNotFoundError(f"No note found at {note_path}")
    log.debug("Reading note: %s", relative_path)
    return note_path.read_text(encoding="utf-8")


@require("vault:write")
def write_note(
    relative_path: str,
    content: str,
    title: str = "",
    tags: list[str] | None = None,
    plugin_source: str = "",
) -> Path:
    """
    Write (or overwrite) a note. Creates parent folders if needed.
    Also indexes the note in the SQLite metadata layer.
    Returns the full path written to.
    """
    note_path = _resolve_vault_path(relative_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(content, encoding="utf-8")
    log.debug("Wrote note: %s (source=%s)", relative_path, plugin_source)

    # Index metadata
    _get_index().index_note(
        path=relative_path,
        title=title or _extract_title(content),
        tags=tags or [],
        plugin_source=plugin_source,
    )

    return note_path


def _extract_title(content: str) -> str:
    """Extract the first H1 heading from markdown content."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return ""


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body) where body is everything after the
    closing '---'. Returns ({}, content) if no frontmatter is found.
    Only handles top-level string and list values — nested YAML is returned
    as raw strings.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break

    if end == -1:
        return {}, content

    body = "\n".join(lines[end + 1:])
    fm: dict[str, str | list[str]] = {}
    current_key = None
    current_list: list[str] = []

    for line in lines[1:end]:
        stripped = line.lstrip()

        # List item under a key
        if stripped.startswith("- ") and current_key:
            current_list.append(stripped[2:].strip())
            continue

        # Flush any list-in-progress
        if current_list:
            fm[current_key] = current_list
            current_list = []

        # New key: value
        if ":" in line:
            key, _, rest = line.partition(":")
            current_key = key.strip()
            value = rest.strip()
            if value:
                # Remove surrounding quotes
                fm[current_key] = value.strip("\"'")
            else:
                # Could be a list starting on the next line
                continue
        elif current_key and line.strip():
            # Continuation of a multi-line value
            if isinstance(fm.get(current_key), str):
                fm[current_key] = fm[current_key] + " " + line.strip()

    # Flush final list
    if current_list:
        fm[current_key] = current_list

    return fm, body


def extract_tags_from_frontmatter(frontmatter: dict) -> list[str]:
    """Extract tags from frontmatter dict (supports 'tags' key as string or list)."""
    tags_raw = frontmatter.get("tags", [])
    if isinstance(tags_raw, str):
        return [t.strip() for t in tags_raw.replace(",", " ").split() if t.strip()]
    elif isinstance(tags_raw, list):
        return tags_raw
    return []


# ---------------------------------------------------------------------------
# Vault scanning
# ---------------------------------------------------------------------------

def scan_vault() -> list[dict]:
    """Walk the vault and return metadata for every .md file found.

    Returns a list of dicts with: path, title, tags, last_modified.
    Skips hidden directories and the .nordrun metadata folder.
    """
    notes = []
    skip_dirs = {".nordrun", ".obsidian", ".git", "__pycache__", "node_modules", ".trash", ".DS_Store"}

    for md_file in sorted(VAULT_PATH.rglob("*.md")):
        rel = md_file.relative_to(VAULT_PATH)
        # Skip hidden directories
        if any(p.startswith(".") for p in rel.parts[:-1]):
            continue
        if any(p in skip_dirs for p in rel.parts[:-1]):
            continue

        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            fm, _ = parse_frontmatter(content)
            title = fm.get("title", _extract_title(content))
            tags = extract_tags_from_frontmatter(fm)
            notes.append({
                "path": str(rel.as_posix()),
                "title": title,
                "tags": tags,
                "last_modified": md_file.stat().st_mtime,
            })
        except Exception:
            continue

    return notes