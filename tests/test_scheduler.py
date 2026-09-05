"""
Tests for the scheduler (automation/scheduler.py).

The _run_cycle pure function is tested directly; persistence is tested
via the module-level load/save with a monkeypatched schedules path.
"""

import time

import pytest

from automation import scheduler
from core.event_bus import EventBus


@pytest.fixture
def isolated_schedules(tmp_path, monkeypatch):
    """Point the scheduler's VAULT_PATH at a temp location.

    The old code had a module-level _SCHEDULES_PATH; the hardened
    scheduler resolves the path at call time via _schedules_path()
    which reads VAULT_PATH. Patching VAULT_PATH on the scheduler module
    is the correct seam.
    """
    monkeypatch.setattr(scheduler, "VAULT_PATH", tmp_path)
    return tmp_path / ".nordrun" / "schedules.yaml"


def test_load_schedules_creates_defaults(isolated_schedules):
    data = scheduler.load_schedules()
    assert len(data["schedules"]) == 1
    assert data["schedules"][0]["id"] == "daily-github-commits"
    # Template is never shared by reference
    assert data is not scheduler._DEFAULT_SCHEDULES


def test_load_schedules_returns_fresh_copy(isolated_schedules):
    s1 = scheduler.load_schedules()
    s2 = scheduler.load_schedules()
    assert s1 is not s2


def test_save_then_load_roundtrip(isolated_schedules):
    data = {"schedules": [{"id": "x", "event": "e", "interval_hours": 1}]}
    scheduler.save_schedules(data)
    loaded = scheduler.load_schedules()
    assert loaded["schedules"][0]["id"] == "x"


class TestRunCycle:

    def _bus(self):
        bus = EventBus()
        fired = []
        bus.subscribe("test.event", lambda p: fired.append(p), plugin_name="probe")
        return bus, fired

    def test_fires_schedule_with_no_last_run(self):
        """A never-run schedule fires on its first cycle."""
        bus, fired = self._bus()
        data = {"schedules": [
            {"id": "a", "event": "test.event", "interval_hours": 24},
        ]}
        changed = scheduler._run_cycle(data, bus, now=time.time())
        assert changed
        assert len(fired) == 1

    def test_fires_when_interval_elapsed(self):
        bus, fired = self._bus()
        now = 1_000_000.0
        data = {"schedules": [
            {"id": "a", "event": "test.event", "interval_hours": 1, "last_run": now - 3600},
        ]}
        changed = scheduler._run_cycle(data, bus, now=now)
        assert changed
        assert len(fired) == 1
        assert data["schedules"][0]["last_run"] == now

    def test_does_not_fire_before_interval(self):
        """A schedule that ran recently must not re-fire."""
        bus, fired = self._bus()
        now = 1_000_000.0
        data = {"schedules": [
            {"id": "a", "event": "test.event", "interval_hours": 1, "last_run": now - 60},
        ]}
        changed = scheduler._run_cycle(data, bus, now=now)
        assert not changed
        assert len(fired) == 0

    def test_restart_does_not_refire_recent_schedule(self):
        """The persist-restart bug: a schedule that ran 30 minutes ago
        must not fire again after a restart."""
        bus, fired = self._bus()
        now = time.time()
        data = {"schedules": [
            {"id": "a", "event": "test.event", "interval_hours": 24,
             "last_run": now - 1800},  # ran 30 min ago
        ]}
        changed = scheduler._run_cycle(data, bus, now=now)
        assert not changed
        assert len(fired) == 0

    def test_disabled_schedule_never_fires(self):
        bus, fired = self._bus()
        data = {"schedules": [
            {"id": "a", "event": "test.event", "interval_hours": 1, "enabled": False},
        ]}
        changed = scheduler._run_cycle(data, bus, now=time.time())
        assert not changed
        assert len(fired) == 0

    def test_run_cycle_stamps_last_run_only_for_fired(self):
        bus, _ = self._bus()
        now = 1_000_000.0
        data = {"schedules": [
            {"id": "a", "event": "test.event", "interval_hours": 1, "last_run": now - 60},   # not due
            {"id": "b", "event": "test.event", "interval_hours": 1, "last_run": now - 7200},  # due
        ]}
        scheduler._run_cycle(data, bus, now=now)
        assert data["schedules"][0].get("last_run") == now - 60
        assert data["schedules"][1].get("last_run") == now
