"""
Tests for the CLI (main.py) command/event/payload routing.

The parser is exercised directly (it does not require config or a live
event bus), and the payload-building branch is covered via a unit-level
assert on the command map.
"""

import pytest

import main as main_mod
from core.plugin_loader import discover_plugins


def _build_parser():
    """Build the argparse parser the way main() does."""
    return main_mod._build_parser()


def _command_map():
    """Recompute the command map from discovered plugin manifests."""
    return main_mod._build_command_map(discover_plugins())


class TestCommandMap:

    def test_all_plugin_commands_present(self):
        # Phase 1 (Memory & Knowledge) intentionally adds the `memory` plugin,
        # which exposes three new CLI commands: memory-store, memory-search,
        # and memory-recall. This assertion was updated to reflect that
        # deliberate API expansion — no existing command was removed or renamed.
        commands = _command_map()
        assert set(commands) == {"summarize", "commits", "review-resume",
                                 "search", "toimage", "digest",
                                 "memory-store", "memory-search", "memory-recall"}

    def test_command_map_event_names(self):
        commands = _command_map()
        assert commands["summarize"][0] == "note.summarize"
        assert commands["commits"][0] == "repo.commits.summarize"
        assert commands["review-resume"][0] == "resume.review"
        assert commands["search"][0] == "note.search"
        assert commands["toimage"][0] == "note.toimage"
        assert commands["digest"][0] == "learning.digest"


class TestToimageFlags:

    def test_toimage_exposes_style_not_out(self):
        parser = _build_parser()
        args = parser.parse_args(["toimage", "Career/README.md", "--style", "line art"])
        assert args.command == "toimage"
        assert args.note == "Career/README.md"
        assert args.style == "line art"
        # --out must not exist on toimage
        with pytest.raises(SystemExit):
            parser.parse_args(["toimage", "Career/README.md", "--out", "x.md"])

    def test_toimage_default_style(self):
        parser = _build_parser()
        args = parser.parse_args(["toimage", "Career/README.md"])
        assert "vector illustration" in args.style


class TestOutFlagPresence:

    def test_summarize_has_out(self):
        parser = _build_parser()
        args = parser.parse_args(["summarize", "Career/README.md", "--out", "out.md"])
        assert args.out == "out.md"

    def test_commits_has_out(self):
        parser = _build_parser()
        args = parser.parse_args(["commits", "varunowns/Nordrun", "--count", "5", "--out", "out.md"])
        assert args.out == "out.md"
        assert args.count == 5

    def test_digest_has_out(self):
        parser = _build_parser()
        args = parser.parse_args(["digest", "--tag", "ai", "--out", "out.md"])
        assert args.out == "out.md"
        assert args.tag == "ai"

    def test_search_has_top_k(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "ml", "--top-k", "3"])
        assert args.top_k == 3


class TestBuiltinCommands:

    def test_reindex_scan_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["reindex", "--scan-vault"])
        assert args.command == "reindex"
        assert args.scan_vault is True

    def test_serve_and_list_plugins(self):
        parser = _build_parser()
        assert parser.parse_args(["serve"]).command == "serve"
        assert parser.parse_args(["list-plugins"]).command == "list-plugins"
