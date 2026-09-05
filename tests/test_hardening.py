"""
Phase 0 Foundation Hardening — tests for every new reliability feature.

Covers:
  1. Thread-local plugin registry  — concurrent dispatches on different
     threads never clobber each other's active-plugin context.
  2. LLM retry logic               — rate-limit, 5xx, and connection errors
     are retried; 4xx auth errors are NOT.
  3. network:call permission        — plugin_loader knows the permission,
     github manifest declares it, validator accepts it.
  4. Scheduler error tracking       — per-schedule last_error / error_count
     written on failure and cleared on success.
  5. Structured logging             — key code paths emit log records at the
     correct level instead of printing to stderr.
  6. get_context() thread safety    — concurrent callers always get the same
     singleton without a race on the None-check.
  7. Late-bound DB path             — storage.db resolves _DB_PATH at call
     time so a monkeypatched VAULT_PATH is respected.
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Thread-local plugin registry
# ---------------------------------------------------------------------------

class TestThreadLocalRegistry:
    """_active_plugin is per-thread — concurrent dispatches never interfere."""

    def test_set_and_get_on_same_thread(self):
        from core.plugin_registry import get_active_plugin, set_active_plugin
        set_active_plugin("alpha")
        assert get_active_plugin() == "alpha"
        set_active_plugin(None)
        assert get_active_plugin() is None

    def test_different_threads_see_independent_values(self):
        """Two threads set different active plugins and read back their own."""
        from core.plugin_registry import get_active_plugin, set_active_plugin

        seen: dict[str, str | None] = {}
        ready = threading.Barrier(2)

        def worker(name: str) -> None:
            set_active_plugin(name)
            ready.wait()          # both threads set before either reads
            seen[name] = get_active_plugin()
            set_active_plugin(None)

        t1 = threading.Thread(target=worker, args=("plugin-a",))
        t2 = threading.Thread(target=worker, args=("plugin-b",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert seen["plugin-a"] == "plugin-a"
        assert seen["plugin-b"] == "plugin-b"

    def test_clearing_on_one_thread_does_not_affect_another(self):
        """set_active_plugin(None) on one thread must not clear another's value."""
        from core.plugin_registry import get_active_plugin, set_active_plugin

        other_saw: list[str | None] = []
        step1 = threading.Event()
        step2 = threading.Event()

        def other_thread() -> None:
            set_active_plugin("persistent-plugin")
            step1.set()       # tell main thread we've set our value
            step2.wait()      # wait for main thread to clear its own
            other_saw.append(get_active_plugin())
            set_active_plugin(None)

        t = threading.Thread(target=other_thread)
        t.start()
        step1.wait()
        set_active_plugin("main-plugin")
        set_active_plugin(None)   # clear main thread
        step2.set()
        t.join()

        # other thread's value must be intact
        assert other_saw == ["persistent-plugin"]

    def test_require_enforces_permission_on_correct_thread(self):
        """@require works correctly against the thread-local active plugin."""
        from core.plugin_registry import (
            _reset_registry, register_plugin, require, set_active_plugin,
        )
        _reset_registry()
        register_plugin("allowed-plugin", ["vault:read"])

        @require("vault:read")
        def guarded(_):
            return "ok"

        set_active_plugin("allowed-plugin")
        try:
            assert guarded(None) == "ok"
        finally:
            set_active_plugin(None)

    def test_require_raises_for_missing_permission(self):
        from core.plugin_registry import (
            _reset_registry, register_plugin, require, set_active_plugin,
        )
        _reset_registry()
        register_plugin("limited-plugin", [])

        @require("vault:write")
        def guarded(_):
            return "ok"

        set_active_plugin("limited-plugin")
        try:
            with pytest.raises(RuntimeError, match="missing required permissions"):
                guarded(None)
        finally:
            set_active_plugin(None)


# ---------------------------------------------------------------------------
# 2. LLM retry logic
# ---------------------------------------------------------------------------

class TestLlmRetry:
    """ask() retries transient errors and raises non-retryable ones immediately."""

    # ------------------------------------------------------------------ helpers

    @pytest.fixture(autouse=True)
    def _patch_llm(self, monkeypatch):
        """Patch the llm_service module so tests don't need a real client.

        We also pin _MAX_RETRIES to 3 and _BASE_DELAY to 0 (no actual
        sleep in tests) and reset the cached _client between tests.
        """
        import services.llm_service as llm
        monkeypatch.setattr(llm, "_client", None)
        monkeypatch.setattr(llm, "_MAX_RETRIES", 3)
        monkeypatch.setattr(llm, "_BASE_DELAY", 0.0)
        monkeypatch.setattr(llm, "time", MagicMock(sleep=lambda _: None))

    def _make_client(self, responses):
        """Return a mock Anthropic client whose messages.create cycles through
        `responses`. Each item is either an exception class/instance (raised)
        or a string (returned as a mock message).
        """
        call_iter = iter(responses)

        def create(**kwargs):
            item = next(call_iter)
            if isinstance(item, type) and issubclass(item, Exception):
                raise item(MagicMock(), MagicMock())
            if isinstance(item, Exception):
                raise item
            msg = MagicMock()
            block = MagicMock()
            block.type = "text"
            block.text = item
            msg.content = [block]
            return msg

        client = MagicMock()
        client.messages.create.side_effect = create
        return client

    # ------------------------------------------------------------------ tests

    def test_success_on_first_attempt(self, monkeypatch):
        import services.llm_service as llm
        from core.plugin_registry import register_plugin, set_active_plugin, _reset_registry
        _reset_registry()
        register_plugin("t", ["llm:call"])
        set_active_plugin("t")

        monkeypatch.setattr(llm, "_client", self._make_client(["hello"]))
        result = llm.ask("prompt")
        assert result == "hello"
        set_active_plugin(None)

    def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        from anthropic import RateLimitError
        import services.llm_service as llm
        from core.plugin_registry import register_plugin, set_active_plugin, _reset_registry
        _reset_registry()
        register_plugin("t", ["llm:call"])
        set_active_plugin("t")

        call_count = 0

        def create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Build a minimal RateLimitError using only the required
                # positional arg (message string) for SDK compatibility.
                exc = RateLimitError.__new__(RateLimitError)
                exc.status_code = 429
                exc.message = "rate limited"
                exc.body = {}
                raise exc
            msg = MagicMock()
            block = MagicMock()
            block.type = "text"
            block.text = "final answer"
            msg.content = [block]
            return msg

        client = MagicMock()
        client.messages.create.side_effect = create
        monkeypatch.setattr(llm, "_client", client)
        result = llm.ask("prompt")
        assert result == "final answer"
        assert call_count == 3
        set_active_plugin(None)

    def test_raises_after_all_retries_exhausted(self, monkeypatch):
        from anthropic import APIConnectionError
        import services.llm_service as llm
        from core.plugin_registry import register_plugin, set_active_plugin, _reset_registry
        _reset_registry()
        register_plugin("t", ["llm:call"])
        set_active_plugin("t")

        # All 3 attempts fail with connection error
        responses = [
            APIConnectionError.__new__(APIConnectionError),
            APIConnectionError.__new__(APIConnectionError),
            APIConnectionError.__new__(APIConnectionError),
        ]
        monkeypatch.setattr(llm, "_client", self._make_client(responses))
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            llm.ask("prompt")
        set_active_plugin(None)

    def test_non_retryable_4xx_raises_immediately(self, monkeypatch):
        """A 401 auth error must not be retried — raise on first attempt."""
        from anthropic import APIStatusError
        import services.llm_service as llm
        from core.plugin_registry import register_plugin, set_active_plugin, _reset_registry
        _reset_registry()
        register_plugin("t", ["llm:call"])
        set_active_plugin("t")

        call_count = 0

        def create(**kwargs):
            nonlocal call_count
            call_count += 1
            exc = APIStatusError.__new__(APIStatusError)
            exc.status_code = 401
            exc.message = "Unauthorized"
            exc.body = {}
            raise exc

        client = MagicMock()
        client.messages.create.side_effect = create
        monkeypatch.setattr(llm, "_client", client)

        with pytest.raises(APIStatusError):
            llm.ask("prompt")

        assert call_count == 1, "A 401 must not trigger any retries"
        set_active_plugin(None)

    def test_retryable_500_retries_then_raises(self, monkeypatch):
        """A 500 server error should be retried up to _MAX_RETRIES times."""
        from anthropic import APIStatusError
        import services.llm_service as llm
        from core.plugin_registry import register_plugin, set_active_plugin, _reset_registry
        _reset_registry()
        register_plugin("t", ["llm:call"])
        set_active_plugin("t")

        call_count = 0

        def create(**kwargs):
            nonlocal call_count
            call_count += 1
            exc = APIStatusError.__new__(APIStatusError)
            exc.status_code = 500
            exc.message = "Internal Server Error"
            exc.body = {}
            raise exc

        client = MagicMock()
        client.messages.create.side_effect = create
        monkeypatch.setattr(llm, "_client", client)

        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            llm.ask("prompt")

        assert call_count == 3, f"Expected 3 attempts, got {call_count}"
        set_active_plugin(None)

    def test_jitter_delay_stays_within_bounds(self):
        import services.llm_service as llm
        for attempt in range(5):
            d = llm._jitter_delay(attempt)
            ceiling = min(llm._MAX_DELAY, llm._BASE_DELAY * (2 ** attempt))
            assert 0 <= d <= ceiling + 1e-9


# ---------------------------------------------------------------------------
# 3. network:call permission
# ---------------------------------------------------------------------------

class TestNetworkCallPermission:
    """network:call is a known permission; github declares it."""

    def test_network_call_is_known_permission(self):
        from core.plugin_loader import _KNOWN_PERMISSIONS
        assert "network:call" in _KNOWN_PERMISSIONS

    def test_network_call_does_not_fail_validation(self):
        from core.plugin_loader import validate_manifest
        manifest = {
            "name": "net-plugin",
            "version": "1.0.0",
            "description": "A plugin that calls the network",
            "permissions": ["vault:read", "network:call"],
        }
        assert validate_manifest(manifest) == []

    def test_github_manifest_declares_network_call(self):
        from core.plugin_loader import discover_plugins
        plugins = discover_plugins()
        gh = next(p for p in plugins if p["name"] == "github")
        perms = gh.get("permissions", "")
        assert "network:call" in perms

    def test_github_manifest_still_passes_validation(self):
        from core.plugin_loader import discover_plugins, validate_manifest
        plugins = discover_plugins()
        gh = next(p for p in plugins if p["name"] == "github")
        assert validate_manifest(gh) == []

    def test_unknown_permission_still_rejected(self):
        from core.plugin_loader import validate_manifest
        manifest = {
            "name": "bad-plugin",
            "version": "1.0.0",
            "description": "Uses an unknown permission",
            "permissions": ["vault:read", "sky:is:limit"],
        }
        issues = validate_manifest(manifest)
        assert any("sky:is:limit" in i for i in issues)


# ---------------------------------------------------------------------------
# 4. Scheduler error tracking
# ---------------------------------------------------------------------------

class TestSchedulerErrorTracking:
    """_run_schedule persists last_error / error_count on failure."""

    @pytest.fixture
    def isolated_schedules(self, tmp_path, monkeypatch):
        """Point the scheduler at a temp directory."""
        from automation import scheduler
        monkeypatch.setattr(scheduler, "VAULT_PATH", tmp_path)
        return tmp_path / ".nordrun" / "schedules.yaml"

    def _bus_with_failing_handler(self):
        """Return (bus, schedule_dict) where the handler always raises."""
        from core.event_bus import EventBus
        bus = EventBus()

        def boom(payload):
            raise RuntimeError("handler exploded")

        # Register without a plugin name so permission checks don't fire
        bus._subscribers["fail.event"].append(("", boom))
        return bus

    def _bus_with_success_handler(self):
        from core.event_bus import EventBus
        bus = EventBus()
        bus._subscribers["ok.event"].append(("", lambda p: "done"))
        return bus

    def test_error_count_increments_on_failure(self, isolated_schedules):
        from automation import scheduler

        bus = self._bus_with_failing_handler()
        schedule = {
            "id": "x",
            "label": "X",
            "event": "fail.event",
            "payload": {},
        }
        scheduler._run_schedule(schedule, bus)
        assert schedule["error_count"] == 1
        assert "handler exploded" in schedule["last_error"]

    def test_error_count_accumulates_across_calls(self, isolated_schedules):
        from automation import scheduler

        bus = self._bus_with_failing_handler()
        schedule = {"id": "x", "label": "X", "event": "fail.event", "payload": {}}
        scheduler._run_schedule(schedule, bus)
        scheduler._run_schedule(schedule, bus)
        assert schedule["error_count"] == 2

    def test_error_cleared_on_success(self, isolated_schedules):
        from automation import scheduler

        bus_ok = self._bus_with_success_handler()
        schedule = {
            "id": "x",
            "label": "X",
            "event": "ok.event",
            "payload": {},
            "error_count": 5,
            "last_error": "old error",
        }
        scheduler._run_schedule(schedule, bus_ok)
        assert schedule.get("last_error") is None
        assert schedule["error_count"] == 0

    def test_error_state_persisted_to_yaml(self, isolated_schedules, tmp_path, monkeypatch):
        """Error state written into the schedule dict survives a save/load round-trip."""
        from automation import scheduler
        monkeypatch.setattr(scheduler, "VAULT_PATH", tmp_path)

        bus = self._bus_with_failing_handler()
        data = {
            "schedules": [{
                "id": "fail-job",
                "label": "Fail Job",
                "event": "fail.event",
                "payload": {},
                "interval_hours": 1,
                "enabled": True,
            }]
        }
        scheduler._run_cycle(data, bus, now=time.time())
        scheduler.save_schedules(data)

        loaded = scheduler.load_schedules()
        s = loaded["schedules"][0]
        assert s.get("error_count", 0) == 1
        assert s.get("last_error") is not None

    def test_run_cycle_still_stamps_last_run_on_failure(self, isolated_schedules):
        """Even when a job fails, last_run is stamped so it isn't retried immediately."""
        from automation import scheduler

        bus = self._bus_with_failing_handler()
        now = 1_000_000.0
        data = {"schedules": [{
            "id": "f",
            "event": "fail.event",
            "interval_hours": 1,
        }]}
        changed = scheduler._run_cycle(data, bus, now=now)
        assert changed
        assert data["schedules"][0]["last_run"] == now


# ---------------------------------------------------------------------------
# 5. Structured logging — event bus and plugin loader emit log records
# ---------------------------------------------------------------------------

class TestStructuredLogging:
    """Key code paths emit log records at the right level, not print() calls."""

    def test_event_bus_logs_handler_error_at_error_level(self, caplog):
        from core.event_bus import EventBus
        from core.plugin_registry import register_plugin, set_active_plugin, _reset_registry
        _reset_registry()

        bus = EventBus()

        def bad_handler(payload):
            raise ValueError("intentional failure")

        bus._subscribers["test.event"].append(("", bad_handler))

        with caplog.at_level(logging.ERROR, logger="core.event_bus"):
            bus.publish("test.event", {})

        assert any("intentional failure" in r.message for r in caplog.records)
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_event_bus_logs_publish_at_debug(self, caplog):
        from core.event_bus import EventBus
        bus = EventBus()
        bus._subscribers["probe.event"].append(("probe", lambda p: "ok"))

        with caplog.at_level(logging.DEBUG, logger="core.event_bus"):
            bus.publish("probe.event", {})

        messages = [r.message for r in caplog.records]
        assert any("probe.event" in m for m in messages)

    def test_plugin_loader_logs_skipped_invalid_manifest(self, caplog, tmp_path):
        from core.plugin_loader import load_and_register
        from core.event_bus import EventBus

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "manifest.yaml").write_text(
            "name: bad\nversion: not-semver\ndescription: Broken\n",
            encoding="utf-8",
        )
        (bad / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None): pass\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.ERROR, logger="core.plugin_loader"):
            load_and_register(EventBus(), plugins_dir=tmp_path)

        assert any("bad" in r.message for r in caplog.records)
        assert any(r.levelno == logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# 6. get_context() thread safety
# ---------------------------------------------------------------------------

class TestGetContextThreadSafety:
    """Concurrent calls to get_context() always return the same singleton."""

    def test_concurrent_calls_return_same_instance(self, monkeypatch):
        from services import context_service
        # Start with a clean slate
        context_service._context_service = None

        results: list = []
        barrier = threading.Barrier(10)

        def call_get_context():
            barrier.wait()  # all threads hit get_context at the same moment
            results.append(context_service.get_context())

        threads = [threading.Thread(target=call_get_context) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        first = results[0]
        assert all(r is first for r in results), "get_context() returned different instances"


# ---------------------------------------------------------------------------
# 7. Late-bound DB path
# ---------------------------------------------------------------------------

class TestLateBoundDbPath:
    """storage.db._db_path() resolves from config.VAULT_PATH at call time."""

    def test_db_path_uses_current_vault_path(self, tmp_path, monkeypatch):
        import config
        import storage.db as db_mod

        monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
        path = db_mod._db_path()
        assert path == tmp_path / ".nordrun" / "metadata.db"

    def test_db_path_reflects_patched_value(self, tmp_path, monkeypatch):
        """Two different tmp_paths → two different resolved DB paths."""
        import config
        import storage.db as db_mod

        path_a = tmp_path / "vault_a"
        path_b = tmp_path / "vault_b"

        monkeypatch.setattr(config, "VAULT_PATH", path_a)
        resolved_a = db_mod._db_path()

        monkeypatch.setattr(config, "VAULT_PATH", path_b)
        resolved_b = db_mod._db_path()

        assert resolved_a != resolved_b
        assert resolved_a.parent.parent == path_a
        assert resolved_b.parent.parent == path_b
