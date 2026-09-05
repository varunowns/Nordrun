"""
Scheduler / Automation (Hermes)
--------------------------------
Runs plugin events on a schedule using a simple background loop (no
external scheduler dependency). Schedule config is stored in
schedules.yaml in the vault's .nordrun/ directory.

Usage:
    python main.py serve

The scheduler thread wakes every 60 seconds, checks which schedules
are due, and publishes the corresponding event on the event bus.

Error tracking (Phase 0 hardening):
  Per-schedule failures are now captured in the schedule's `last_error`
  and `error_count` fields and persisted to schedules.yaml, so you can
  inspect why a scheduled job kept failing without digging through logs.
  Errors are also emitted at ERROR level via the logging module so they
  surface in any log aggregator or file handler configured at startup.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import VAULT_PATH
from core.event_bus import EventBus

log = logging.getLogger(__name__)


def _schedules_path() -> Path:
    """Resolve the schedules file path at call time (not import time)."""
    return VAULT_PATH / ".nordrun" / "schedules.yaml"


# Default schedules — written as a template on first run.
_DEFAULT_SCHEDULES = {
    "schedules": [
        {
            "id": "daily-github-commits",
            "label": "Daily GitHub commits summary",
            "event": "repo.commits.summarize",
            "payload": {"repo": "v4run/Nordrun", "count": 5},
            "interval_hours": 24,
            "enabled": True,
        },
    ]
}


def load_schedules() -> dict[str, Any]:
    """Load schedules from schedules.yaml, creating defaults if missing."""
    path = _schedules_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(_DEFAULT_SCHEDULES, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        log.debug("Created default schedules.yaml at %s", path)
        # Return a fresh copy so callers never share the template's dict.
        return yaml.safe_load(
            yaml.dump(_DEFAULT_SCHEDULES, default_flow_style=False, sort_keys=False)
        )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"schedules": []}


def save_schedules(data: dict[str, Any]) -> None:
    """Write schedules back to schedules.yaml."""
    path = _schedules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _run_schedule(schedule: dict, bus: EventBus) -> None:
    """Publish the event for one schedule entry.

    On success, clears any prior error state.
    On failure (handler raised or returned an error dict), logs the error
    and updates last_error / error_count in the schedule dict (persisted
    by _run_cycle via save_schedules).

    The event bus catches handler exceptions and returns them as
    {"error": ..., "plugin": ...} dicts rather than re-raising, so we
    inspect the results list for error payloads in addition to catching
    any exception that escapes bus.publish() itself.
    """
    event = schedule["event"]
    payload = schedule.get("payload", {})
    label = schedule.get("label", event)
    log.info("Running scheduled job '%s' (event=%s)", label, event)
    try:
        results = bus.publish(event, payload)
        # The event bus catches handler exceptions and returns them as dicts.
        # Treat any result that carries an "error" key as a job failure.
        errors = [r for r in results if isinstance(r, dict) and r.get("error")]
        if errors:
            raise RuntimeError(errors[0]["error"])
        # Clear stale error state on success
        schedule.pop("last_error", None)
        schedule["error_count"] = 0
        log.info("Scheduled job '%s' completed successfully", label)
    except Exception as exc:
        error_count = schedule.get("error_count", 0) + 1
        schedule["last_error"] = str(exc)
        schedule["error_count"] = error_count
        log.error(
            "Scheduled job '%s' failed (attempt #%d): %s",
            label, error_count, exc,
            exc_info=True,
        )


def _run_cycle(data: dict, bus: EventBus, now: float) -> bool:
    """Run every due schedule in `data`, stamping last_run in place.

    Returns True when any schedule fired (and thus should be persisted).
    A schedule with no last_run has never run and fires on its first
    cycle. Per-schedule errors are handled inside _run_schedule and
    persisted here via the returned `changed` flag.
    """
    changed = False
    for schedule in data.get("schedules", []):
        if not schedule.get("enabled", True):
            continue

        interval_h = schedule.get("interval_hours", 24)
        interval_s = interval_h * 3600
        last = schedule.get("last_run", 0)
        if now - last >= interval_s:
            _run_schedule(schedule, bus)
            schedule["last_run"] = now
            changed = True
    return changed


def serve(bus: EventBus, interval_seconds: int = 60, stop_event: threading.Event | None = None) -> None:
    """
    Main scheduler loop. Runs in a background thread, waking every
    `interval_seconds` to check and fire due schedules.

    Each schedule's last-run time is persisted in schedules.yaml, so a
    restart does not re-fire every enabled schedule immediately. Error
    state (last_error, error_count) is also persisted so failures are
    visible without log files.
    """
    if stop_event is None:
        stop_event = threading.Event()

    log.info("Scheduler started (check interval: %ds)", interval_seconds)

    while not stop_event.is_set():
        now = time.time()
        try:
            data = load_schedules()
            if _run_cycle(data, bus, now):
                save_schedules(data)
        except Exception as exc:
            log.error("Scheduler check cycle error: %s", exc, exc_info=True)

        stop_event.wait(timeout=interval_seconds)

    log.info("Scheduler stopped.")


def start_scheduler(bus: EventBus, interval_seconds: int = 60) -> threading.Event:
    """Start the scheduler in a daemon thread. Returns the stop event."""
    stop_event = threading.Event()
    t = threading.Thread(
        target=serve,
        args=(bus, interval_seconds, stop_event),
        daemon=True,
    )
    t.start()
    return stop_event
