"""
Event Bus
---------
In-memory pub/sub dispatcher. Plugins subscribe to named events and the
core (or other plugins) publish events with a payload dict.

Logging (Phase 0 hardening):
  Uses the standard `logging` module. Debug-level messages trace every
  publish/dispatch so you can follow exactly what fired and when.
  Errors in individual plugin handlers are logged at ERROR level and
  include the plugin name and event — the remaining handlers still run
  (one failing plugin never blocks others on the same event).
"""

import logging
from collections import defaultdict
from typing import Any, Callable

from core.plugin_registry import set_active_plugin

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[tuple[str, Callable[[dict], Any]]]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable[[dict], Any], plugin_name: str = "") -> None:
        """Register a handler to run when `event_name` is published."""
        self._subscribers[event_name].append((plugin_name, handler))
        log.debug("Plugin '%s' subscribed to '%s'", plugin_name, event_name)

    def publish(self, event_name: str, payload: dict | None = None) -> list[Any]:
        """Fire an event.

        Every subscribed handler runs synchronously in registration order.
        Returns the list of handler return values (useful for the CLI to
        print results).

        If a handler raises an exception it is logged at ERROR level and
        the remaining handlers still run — one failing plugin does not
        block others listening to the same event.
        """
        payload = payload or {}
        results = []
        handlers = self._subscribers.get(event_name, [])

        log.debug("Publishing '%s' to %d handler(s)", event_name, len(handlers))

        for plugin_name, handler in handlers:
            set_active_plugin(plugin_name)
            try:
                result = handler(payload)
                results.append(result)
                log.debug("Handler '%s' for '%s' completed successfully", plugin_name, event_name)
            except Exception as exc:
                log.error(
                    "Plugin '%s' raised an error handling '%s': %s",
                    plugin_name, event_name, exc,
                    exc_info=True,
                )
                results.append({"error": str(exc), "plugin": plugin_name})
            finally:
                set_active_plugin(None)

        return results

    def registered_events(self) -> list[str]:
        return list(self._subscribers.keys())

    def get_subscribers(self) -> dict[str, list[str]]:
        """Return a mapping of event_name -> list of plugin names subscribed."""
        return {
            event: [plugin_name for plugin_name, _ in handlers]
            for event, handlers in self._subscribers.items()
        }
