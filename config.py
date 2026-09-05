"""
Nordrun Config
--------------
Central place for settings. For now this is just environment variables,
loaded via a .env file if present. As Nordrun grows, this can be replaced
by config/ files per the original plan — but a single config.py is enough
for the vertical slice.
"""

import os
from pathlib import Path

# Load a local .env file if python-dotenv is installed and a .env exists.
# Kept optional so the slice runs even without the dependency.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Path to your Obsidian vault root, e.g. "V:/Obsidian"
VAULT_PATH = Path(os.environ.get("NORDRUN_VAULT_PATH", ""))

# Anthropic API key — get one at https://console.anthropic.com/
# Leave blank when using a proxy like 9router.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Optional LLM proxy/base URL (e.g. 9router). When set, the Anthropic SDK
# targets this endpoint instead of the default api.anthropic.com.
LLM_BASE_URL = os.environ.get("NORDRUN_LLM_BASE_URL", "")

# API key for the proxy (used in place of ANTHROPIC_API_KEY when LLM_BASE_URL is set)
LLM_API_KEY = os.environ.get("NORDRUN_LLM_API_KEY", "")

# Which model to call for plugin LLM tasks
LLM_MODEL = os.environ.get("NORDRUN_LLM_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Phase 1 — Memory & Knowledge configuration
# ---------------------------------------------------------------------------

# Embedding provider for memory: "tfidf" (default, no extra deps) or
# a future provider key like "sentence-transformers".
MEMORY_EMBEDDING_PROVIDER = os.environ.get("NORDRUN_MEMORY_EMBEDDING_PROVIDER", "tfidf")

# Cosine similarity threshold for memory retrieval [0.0, 1.0].
# Memories with score below this are excluded from semantic results.
MEMORY_SIMILARITY_THRESHOLD = float(os.environ.get("NORDRUN_MEMORY_SIMILARITY_THRESHOLD", "0.0"))

# Default number of memories returned by a semantic search.
MEMORY_TOP_K = int(os.environ.get("NORDRUN_MEMORY_TOP_K", "10"))

# Maximum total memories stored before oldest low-importance memories
# are candidates for consolidation.  0 = no limit (Phase 1 default).
MEMORY_MAX_RECORDS = int(os.environ.get("NORDRUN_MEMORY_MAX_RECORDS", "0"))

# Minimum importance threshold for a memory to be injected into LLM context.
MEMORY_CONTEXT_MIN_IMPORTANCE = float(os.environ.get("NORDRUN_MEMORY_CONTEXT_MIN_IMPORTANCE", "0.3"))

# Maximum number of memories to inject into a single LLM context window.
MEMORY_CONTEXT_MAX_INJECT = int(os.environ.get("NORDRUN_MEMORY_CONTEXT_MAX_INJECT", "5"))
