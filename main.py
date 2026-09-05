"""
Nordrun entrypoint
------------------
Wires up the event bus via auto-discovery, and auto-generates CLI
subcommands from each plugin's CLI_COMMANDS metadata — no more
hardcoding commands in main.py.

Logging is configured here (once, at process startup) so every module
that calls logging.getLogger(__name__) automatically inherits the root
handler and level. Set NORDRUN_LOG_LEVEL in your .env to adjust verbosity:
  DEBUG    — trace every event dispatch, vault read/write, DB query
  INFO     — scheduled job starts/completions, plugin load summary
  WARNING  — transient LLM errors being retried (default)
  ERROR    — plugin failures, unretryable errors (always shown)

.env is loaded exclusively by config.py — main.py no longer double-loads
it so there is a single authoritative load point.

Usage:
    python main.py --help
    python main.py summarize "Career/README.md"
    python main.py commits "v4run/EchoSign"
"""

import argparse
import logging
import os
import sys

from core.event_bus import EventBus
from core.plugin_loader import PluginLoadReport, discover_plugins, load_and_register, validate_manifest
from core.plugin_registry import get_registered_plugins

# ---------------------------------------------------------------------------
# Logging setup — called once before any other nordrun module does work.
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Configure the root logger from NORDRUN_LOG_LEVEL (default: WARNING).

    WARNING is the production default: LLM retry warnings and plugin errors
    surface; debug/info chatter stays quiet. Set DEBUG to trace everything.
    """
    level_name = os.environ.get("NORDRUN_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

_configure_logging()


def validate_config() -> list[str]:
    """Check critical config values and return a list of issues."""
    issues = []
    vault = os.environ.get("NORDRUN_VAULT_PATH", "")
    if not vault:
        issues.append("NORDRUN_VAULT_PATH is not set. Add it to your .env file "
                       "(e.g. NORDRUN_VAULT_PATH=V:/Obsidian/Obsidian Vault)")
    else:
        from pathlib import Path
        if not Path(vault).is_dir():
            issues.append(f"NORDRUN_VAULT_PATH={vault} does not exist or is not a directory")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    proxy_key = os.environ.get("NORDRUN_LLM_API_KEY", "")
    proxy_url = os.environ.get("NORDRUN_LLM_BASE_URL", "")
    if not api_key and not proxy_key:
        issues.append("No LLM API key configured. Set ANTHROPIC_API_KEY or "
                       "NORDRUN_LLM_API_KEY in your .env file")
    if proxy_url and not proxy_key:
        issues.append("NORDRUN_LLM_BASE_URL is set but NORDRUN_LLM_API_KEY is missing")
    if not proxy_url and not api_key:
        # No proxy and no key — that's already caught above
        pass

    return issues


_load_report: "PluginLoadReport | None" = None

_log = logging.getLogger(__name__)


def build_event_bus() -> EventBus:
    """Build the event bus and register all valid plugins.

    The load result is stashed globally so list-plugins can annotate
    skipped/failed plugins. Summary is printed to stdout; plugin-level
    detail comes from the logging system.
    """
    global _load_report
    bus = EventBus()
    _load_report = load_and_register(bus)
    print(f"Plugins: {_load_report.summary()}.")
    if not _load_report.registered:
        _log.warning("No plugins were loaded.")
    return bus


def _build_command_map(plugins: list[dict]) -> dict[str, tuple[str, str, dict]]:
    """Map CLI command names to (event_name, plugin_name, cli_meta).

    Plugins whose manifest violates the contract are excluded from the
    command map — they are never reachable from the CLI.
    """
    command_map: dict[str, tuple[str, str, dict]] = {}
    for meta in plugins:
        plugin_name = meta["name"]
        if "_parse_error" in meta or validate_manifest(meta):
            continue
        commands_raw = meta.get("commands", "")
        if commands_raw:
            # Parse commands from manifest: "summarize:note.summarize"
            for entry in commands_raw.split(";"):
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split(":", 2)
                cmd_name = parts[0].strip()
                event_name = parts[1].strip() if len(parts) > 1 else ""
                cmd_help = parts[2].strip() if len(parts) > 2 else ""
                if cmd_name and event_name:
                    command_map[cmd_name] = (event_name, plugin_name, {"help": cmd_help})
    return command_map


def _build_parser() -> argparse.ArgumentParser:
    """Build the full argparse parser (commands + built-ins)."""
    plugins = discover_plugins()
    command_map = _build_command_map(plugins)

    parser = argparse.ArgumentParser(
        description="Nordrun — personal AI platform over your Obsidian vault"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd_name, (event_name, plugin_name, cmd_meta) in sorted(command_map.items()):
        help_text = cmd_meta.get("help") or f"Trigger {event_name} event"

        # Determine if this command takes a positional "note" arg
        sp = subparsers.add_parser(cmd_name, help=help_text)

        # Most commands take a note/repo path
        if event_name in ("note.summarize", "note.review", "resume.review"):
            sp.add_argument("note", help="Path to the note, relative to vault root")
        elif event_name == "repo.commits.summarize":
            sp.add_argument("repo", nargs="?", default="varunowns/Nordrun",
                            help="owner/repo (default: varunowns/Nordrun)")
            sp.add_argument("--count", type=int, default=10,
                            help="Number of commits to fetch (default: 10)")
        elif event_name == "learning.digest":
            sp.add_argument("--tag", default="learning", help="Tag to search for (default: learning)")
            sp.add_argument("--out", help="Where to write the digest (optional)")
        elif event_name == "note.toimage":
            sp.add_argument("note", help="Path to the note, relative to vault root")
            sp.add_argument("--style", default="vector illustration with flat colours",
                            help="Style hint for the generated image (optional)")
        elif event_name == "note.search":
            sp.add_argument("query", help="Search query")
        # Most commands support --out; toimage writes to the workspace
        # and learning.digest declares its own --out.
        if event_name not in ("learning.digest", "note.toimage"):
            sp.add_argument("--out", help="Where to write the result (optional)")

        # search also has --top-k
        if event_name == "note.search":
            sp.add_argument("--top-k", type=int, default=5,
                            help="Number of results (default: 5)")

    # Add built-in commands not driven by plugins
    reindex_parser = subparsers.add_parser("reindex", help="Re-index all vault notes for semantic search")
    reindex_parser.add_argument("--scan-vault", action="store_true", help="Also scan vault for new notes not created by Nordrun")
    subparsers.add_parser("serve", help="Start the background scheduler (Hermes)")
    subparsers.add_parser("list-plugins", help="List all loaded plugins, their events, and permissions")

    return parser


def main():
    issues = validate_config()
    if issues:
        print("Configuration errors:", file=sys.stderr)
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}", file=sys.stderr)
        print("\nFix these in your .env file and try again.", file=sys.stderr)
        sys.exit(1)

    parser = _build_parser()
    args = parser.parse_args()
    bus = build_event_bus()

    # --- Route to the right event ---
    command_map = _build_command_map(discover_plugins())
    if args.command in command_map:
        event_name, _, _ = command_map[args.command]
        payload = {}

        if event_name == "note.summarize":
            payload["source_note"] = args.note
            if args.out:
                payload["output_note"] = args.out
        elif event_name == "learning.digest":
            payload["tag"] = args.tag
            if args.out:
                payload["output_note"] = args.out
        elif event_name == "note.toimage":
            payload["source_note"] = args.note
            payload["style"] = args.style
        elif event_name == "repo.commits.summarize":
            payload["repo"] = args.repo
            payload["count"] = args.count
            if args.out:
                payload["output_note"] = args.out
        elif event_name == "resume.review":
            payload["resume_note"] = args.note
            if args.out:
                payload["output_note"] = args.out
        elif event_name == "note.search":
            payload["query"] = args.query
            payload["top_k"] = args.top_k
        else:
            # Generic fallback: pass args as payload
            payload = vars(args)

        results = bus.publish(event_name, payload)
        if not results:
            print(f"No plugin handled '{event_name}'.", file=sys.stderr)
            sys.exit(1)

        result = results[0]

        if event_name == "note.search":
            if result.get("error"):
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)
            print(f"Search results for: \"{result['query']}\" ({result['result_count']} found)\n")
            for i, r in enumerate(result["results"], 1):
                print(f"  {i}. {r['path']}  (score: {r['score']})")
                if r["title"]:
                    print(f"     Title: {r['title']}")
                if r["tags"]:
                    print(f"     Tags: {', '.join(r['tags'])}")
                if r["plugin_source"]:
                    print(f"     Source: {r['plugin_source']}")
                print()
        else:
            output_key = "output_note" if "output_note" in result else None
            if output_key:
                print(f"Wrote to: {result[output_key]}\n")
            summary = result.get("summary") or result.get("review") or ""
            if summary:
                print(summary)

    elif args.command == "reindex":
        result = bus.publish("note.reindex", {"scan_vault": getattr(args, "scan_vault", False)})
        if not result:
            print("No plugin handled 'note.reindex'.", file=sys.stderr)
            sys.exit(1)

        r = result[0]
        print(f"Re-indexed {r['indexed']}/{r['total_requested']} notes")
        if r["errors"]:
            for e in r["errors"]:
                print(f"  [X] {e['path']}: {e['error']}")

    elif args.command == "list-plugins":
        plugins = discover_plugins()
        registered = get_registered_plugins()
        subscribers = bus.get_subscribers()

        # Build a reverse map: plugin -> events
        plugin_events: dict[str, list[str]] = {}
        for event, names in subscribers.items():
            for name in names:
                plugin_events.setdefault(name, []).append(event)

        print("Loaded plugins:")
        print("=" * 60)
        for meta in plugins:
            name = meta["name"]
            perms = registered.get(name, set())
            events = plugin_events.get(name, [])
            status = "[*]" if name in registered else "[ ]"
            print(f"  {status} {name}  v{meta.get('version', '?')}")
            print(f"      Permissions: {', '.join(sorted(perms)) if perms else 'none'}")
            print(f"      Events:      {', '.join(sorted(events)) if events else 'none'}")
            desc = meta.get("description", "").strip()
            if desc:
                # Truncate to first 80 chars
                print(f"      Description: {desc[:80]}{'...' if len(desc) > 80 else ''}")
            if "_parse_error" in meta:
                print(f"      [X] Manifest failed to parse: {meta['_parse_error']}")
            elif _load_report is not None and name in _load_report.skipped:
                print(f"      [X] Skipped — invalid manifest:")
                for issue in _load_report.skipped[name]:
                    print(f"          - {issue}")
            elif meta.get("_validation_errors") or validate_manifest(meta):
                errors = meta.get("_validation_errors") or validate_manifest(meta)
                print(f"      [X] Invalid manifest:")
                for issue in errors:
                    print(f"          - {issue}")
            elif _load_report is not None and name in _load_report.failed:
                print(f"      [X] Failed to load: {_load_report.failed[name]}")
            print()

    elif args.command == "serve":
        import signal
        from automation.scheduler import start_scheduler

        stop = start_scheduler(bus, interval_seconds=60)
        print("[scheduler] Running. Press Ctrl+C to stop.")

        def _handle_sig(*_):
            print("\n[scheduler] Shutting down...")
            stop.set()

        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)

        try:
            stop.wait()
        except KeyboardInterrupt:
            stop.set()


if __name__ == "__main__":
    main()