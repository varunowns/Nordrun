"""
Plugin Registry + Permissions
-------------------------------
Tracks which plugins are loaded and what permissions each declares
in its manifest.yaml. Services (obsidian_service, llm_service) call
check_permission() before acting; if the plugin hasn't declared the
required permission, the call is denied with a RuntimeError.

Permission naming convention:
  vault:read   — read a note from the vault
  vault:write  — write a note to the vault
  llm:call     — call the LLM service
  network:call — make outbound network requests (HTTP, GitHub API, etc.)

Thread safety:
  _active_plugin is stored in a threading.local() so concurrent event
  dispatches on different threads never clobber each other's active plugin
  context. The registry dict itself is write-once at load time (only
  register_plugin mutates it, only from the main thread during startup),
  so reads during dispatch are safe without a lock.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any

# Global registry: plugin_name -> set of permission strings.
# Written only during startup (load_and_register), read during dispatch.
_registry: dict[str, set[str]] = {}

# Per-thread active plugin name so concurrent dispatches don't interfere.
_local = threading.local()


def register_plugin(name: str, permissions: list[str]) -> None:
    """Register a plugin and its declared permissions."""
    _registry[name] = set(permissions)


def _reset_registry() -> None:
    """Clear the plugin registry. Test-only helper so each test starts
    from a clean registry (load_and_register accumulates across tests)."""
    _registry.clear()


def require(*permissions: str) -> None:
    """
    Decorator for service functions that require certain permissions.
    Usage:
        @require("vault:read")
        def read_note(path): ...
    """
    # If called as @require (no parens) — permissions is the function
    if len(permissions) == 1 and callable(permissions[0]):
        raise TypeError(
            "Missing parentheses — use @require('perm_name') not @require"
        )

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            plugin = _local.__dict__.get("active_plugin")
            if plugin is None:
                raise RuntimeError(
                    f"Service function '{func.__name__}' called outside a plugin handler. "
                    f"Required permissions: {permissions}"
                )
            declared = _registry.get(plugin, set())
            missing = [p for p in permissions if p not in declared]
            if missing:
                raise RuntimeError(
                    f"Plugin '{plugin}' missing required permissions: {missing}. "
                    f"Declared: {declared}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def set_active_plugin(name: str | None) -> None:
    """Called by the event bus before/after dispatching to a plugin's handler.

    Stored in threading.local() so concurrent dispatches on different
    threads each track their own active plugin without interference.
    """
    _local.active_plugin = name


def get_active_plugin() -> str | None:
    """Return the plugin currently being dispatched on this thread."""
    return _local.__dict__.get("active_plugin")


def get_registered_plugins() -> dict[str, set[str]]:
    """Return a copy of the registry for introspection."""
    return dict(_registry)
