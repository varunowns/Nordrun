"""
Tests for the plugin manifest contract (core/plugin_loader.validate_manifest)
and loader behavior for invalid manifests.
"""

from pathlib import Path

import pytest

from core.event_bus import EventBus
from core.plugin_loader import (
    PluginLoadReport,
    discover_plugins,
    load_and_register,
    parse_permissions,
    validate_manifest,
)
from core.plugin_registry import get_registered_plugins


def _valid_manifest() -> dict:
    return {
        "name": "sample",
        "version": "1.2.3",
        "description": "A sample plugin.",
        "subscribes": ["sample.event"],
        "publishes": ["sample.done"],
        "permissions": ["vault:read"],
        "commands": "do-it:sample.event:Do the thing",
    }


class TestValidateManifest:

    def test_valid_manifest(self):
        assert validate_manifest(_valid_manifest()) == []

    def test_missing_name(self):
        m = _valid_manifest()
        del m["name"]
        issues = validate_manifest(m)
        assert any("name" in i for i in issues)

    def test_name_must_be_kebab_case(self):
        m = _valid_manifest()
        m["name"] = "Bad Name!"
        issues = validate_manifest(m)
        assert any("kebab-case" in i for i in issues)

    def test_name_must_match_dir(self):
        """The manifest name must match its plugin dir."""
        m = _valid_manifest()
        m["name"] = "different"
        m["dir"] = "plugins/sample"
        issues = validate_manifest(m)
        assert any("does not match the plugin dir" in i for i in issues)

    def test_name_matching_dir_is_valid(self):
        """A name that matches the dir passes validation."""
        m = _valid_manifest()
        m["name"] = "sample"
        m["dir"] = "plugins/sample"
        assert validate_manifest(m) == []

    def test_name_dir_match_ignores_absent_dir(self):
        """validate_manifest without a dir (bare metadata) still checks format."""
        m = _valid_manifest()
        assert validate_manifest(m) == []

    def test_version_not_semver(self):
        m = _valid_manifest()
        m["version"] = "abc"
        issues = validate_manifest(m)
        assert any("semver" in i for i in issues)

    def test_missing_version(self):
        m = _valid_manifest()
        del m["version"]
        issues = validate_manifest(m)
        assert any("version" in i for i in issues)

    def test_missing_description(self):
        m = _valid_manifest()
        del m["description"]
        issues = validate_manifest(m)
        assert any("description" in i for i in issues)

    def test_subscribes_must_be_list(self):
        m = _valid_manifest()
        m["subscribes"] = "note.summarize"
        issues = validate_manifest(m)
        assert any("subscribes" in i for i in issues)

    def test_publishes_entries_non_empty(self):
        m = _valid_manifest()
        m["publishes"] = ["valid.event", "   "]
        issues = validate_manifest(m)
        assert any("publishes" in i for i in issues)

    def test_unknown_permission(self):
        m = _valid_manifest()
        m["permissions"] = ["vault:read", "totally:bogus"]
        issues = validate_manifest(m)
        assert any("totally:bogus" in i for i in issues)

    def test_permissions_as_csv_string(self):
        m = _valid_manifest()
        m["permissions"] = "vault:read, vault:write"
        assert validate_manifest(m) == []

    def test_permissions_as_list(self):
        m = _valid_manifest()
        m["permissions"] = ["vault:read", "vault:write"]
        assert validate_manifest(m) == []

    def test_permissions_wrong_type(self):
        m = _valid_manifest()
        m["permissions"] = 42
        issues = validate_manifest(m)
        assert any("permissions" in i for i in issues)

    def test_permissions_list_with_unknown(self):
        m = _valid_manifest()
        m["permissions"] = ["vault:read", "totally:bogus"]
        issues = validate_manifest(m)
        assert any("totally:bogus" in i for i in issues)

    def test_parse_permissions_normalises_forms(self):
        assert parse_permissions("vault:read, vault:write") == ["vault:read", "vault:write"]
        assert parse_permissions(["vault:read", "vault:write"]) == ["vault:read", "vault:write"]
        assert parse_permissions([]) == []
        assert parse_permissions("") == []
        assert parse_permissions(42) == []

    def test_bad_command_format(self):
        m = _valid_manifest()
        m["commands"] = "noevent"
        issues = validate_manifest(m)
        assert any("commands" in i for i in issues)

    def test_commands_as_list(self):
        m = _valid_manifest()
        m["commands"] = ["a:x", "b:y:Help"]
        assert validate_manifest(m) == []

    def test_multiple_issues_reported_together(self):
        m = {"name": "x", "version": "nope", "permissions": ["bogus:perm"]}
        issues = validate_manifest(m)
        assert any("semver" in i for i in issues)
        assert any("bogus:perm" in i for i in issues)


class TestLoaderSkipsInvalid:

    @pytest.fixture
    def broken_plugins_dir(self, tmp_path: Path) -> Path:
        """A plugins dir with one valid and one invalid plugin."""
        good = tmp_path / "good"
        good.mkdir()
        (good / "manifest.yaml").write_text(
            "name: good\nversion: 1.0.0\ndescription: A good plugin\n"
            "subscribes:\n  - good.event\npermissions: vault:read\n",
            encoding="utf-8",
        )
        (good / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None):\n"
            "    event_bus.subscribe('good.event', lambda p: {'ok': True}, plugin_name=plugin_name)\n",
            encoding="utf-8",
        )

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "manifest.yaml").write_text(
            "name: bad\nversion: not-semver\ndescription: Broken plugin\npermissions: vault:read\n",
            encoding="utf-8",
        )
        (bad / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None):\n"
            "    raise AssertionError('should never be called')\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_invalid_plugin_is_skipped(self, broken_plugins_dir: Path, caplog):
        import logging
        bus = EventBus()
        with caplog.at_level(logging.ERROR, logger="core.plugin_loader"):
            report = load_and_register(bus, plugins_dir=broken_plugins_dir)
        assert report.registered == ["good"]
        assert report.skipped["bad"]  # non-empty reasons
        assert "bad" not in report.failed
        # The bad plugin's register() must never run — error goes to logging now
        messages = [r.message for r in caplog.records]
        assert any("bad" in m for m in messages)
        assert any("invalid manifest" in m or "Skipping" in m for m in messages)

    def test_report_ok_is_false_when_skipped(self, broken_plugins_dir: Path):
        report = load_and_register(EventBus(), plugins_dir=broken_plugins_dir)
        assert not report.ok

    def test_report_summary(self, broken_plugins_dir: Path):
        report = load_and_register(EventBus(), plugins_dir=broken_plugins_dir)
        s = report.summary()
        assert "1 loaded" in s
        assert "1 skipped" in s

    def test_discover_still_reports_both(self, broken_plugins_dir: Path):
        metas = discover_plugins(broken_plugins_dir)
        names = {m["name"] for m in metas}
        assert names == {"good", "bad"}

    def test_name_dir_mismatch_is_skipped_not_failed(self, tmp_path: Path):
        """A name that doesn't match the dir is a validation skip, not an
        opaque import failure."""
        real_dir = tmp_path / "real-dir"
        real_dir.mkdir()
        (real_dir / "manifest.yaml").write_text(
            "name: different\nversion: 1.0.0\ndescription: Mismatch\n",
            encoding="utf-8",
        )
        (real_dir / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None):\n"
            "    event_bus.subscribe('x.event', lambda p: {}, plugin_name=plugin_name)\n",
            encoding="utf-8",
        )
        report = load_and_register(EventBus(), plugins_dir=tmp_path)
        assert report.registered == []
        assert "different" in report.skipped
        assert "different" not in report.failed
        assert any("does not match" in s for s in report.skipped["different"])


class TestLoaderReportsRegisterFailure:

    @pytest.fixture
    def dir_with_register_failure(self, tmp_path: Path) -> Path:
        """A plugins dir whose plugin raises inside register()."""
        bad = tmp_path / "boom"
        bad.mkdir()
        (bad / "manifest.yaml").write_text(
            "name: boom\nversion: 1.0.0\ndescription: Fails on register\npermissions: vault:read\n",
            encoding="utf-8",
        )
        (bad / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None):\n"
            "    raise RuntimeError('register exploded')\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_register_failure_recorded(self, dir_with_register_failure: Path, caplog):
        import logging
        bus = EventBus()
        with caplog.at_level(logging.ERROR, logger="core.plugin_loader"):
            report = load_and_register(bus, plugins_dir=dir_with_register_failure)
        assert report.registered == []
        assert report.failed["boom"] == "register exploded"
        assert not report.ok
        # Error now goes to logging, not stderr
        messages = [r.message for r in caplog.records]
        assert any("boom" in m for m in messages)

    def test_one_failing_plugin_does_not_block_others(
        self, dir_with_register_failure: Path, tmp_path: Path
    ):
        # Add a good sibling next to the failing plugin
        good = tmp_path / "good"
        good.mkdir()
        (good / "manifest.yaml").write_text(
            "name: good\nversion: 1.0.0\ndescription: A good plugin\npermissions: vault:read\n",
            encoding="utf-8",
        )
        (good / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None):\n"
            "    event_bus.subscribe('good.event', lambda p: {}, plugin_name=plugin_name)\n",
            encoding="utf-8",
        )
        report = load_and_register(EventBus(), plugins_dir=dir_with_register_failure)
        assert report.failed["boom"]
        assert "good" in report.registered


class TestLoaderPassesConfig:
    """The loader must pass each plugin's manifest config to register()."""

    @staticmethod
    def _write_plugin(plugins_dir, name, config_yaml=""):
        """Write a plugin whose register() records config to a JSON file."""
        import json
        plugin_dir = plugins_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "manifest.yaml").write_text(
            f"name: {name}\nversion: 1.0.0\ndescription: Has config\n{config_yaml}\n",
            encoding="utf-8",
        )
        # register() dumps config to a file in the plugin dir, so the test
        # can assert what the loader passed without cross-module plumbing.
        (plugin_dir / "plugin.py").write_text(
            "import json, os\n"
            "def register(event_bus, plugin_name='', config=None):\n"
            f"    with open(os.path.join(os.path.dirname(__file__), 'captured.json'), 'w') as f:\n"
            f"        json.dump(config, f)\n",
            encoding="utf-8",
        )

    def test_config_passed_to_register(self, tmp_path: Path):
        """The manifest config dict must reach register()."""
        import json
        self._write_plugin(
            tmp_path, "good",
            "config:\n  greeting: hello\n  count: 3\n",
        )
        report = load_and_register(EventBus(), plugins_dir=tmp_path)
        assert report.registered == ["good"]
        captured = json.loads((tmp_path / "good" / "captured.json").read_text(encoding="utf-8"))
        assert captured == {"greeting": "hello", "count": 3}

    def test_absent_config_is_empty_dict(self, tmp_path: Path):
        """A plugin with no config gets an empty dict, not None."""
        import json
        self._write_plugin(tmp_path, "good", "")
        load_and_register(EventBus(), plugins_dir=tmp_path)
        captured = json.loads((tmp_path / "good" / "captured.json").read_text(encoding="utf-8"))
        assert captured == {}


class TestLoaderListPermissions:

    @pytest.fixture
    def list_permissions_plugin_dir(self, tmp_path: Path) -> Path:
        """A valid plugin whose manifest declares permissions as a YAML list."""
        good = tmp_path / "good"
        good.mkdir()
        (good / "manifest.yaml").write_text(
            "name: good\nversion: 1.0.0\ndescription: List-form permissions\n"
            "permissions:\n  - vault:read\n  - vault:write\n"
            "subscribes:\n  - good.event\n",
            encoding="utf-8",
        )
        (good / "plugin.py").write_text(
            "def register(event_bus, plugin_name='', config=None):\n"
            "    event_bus.subscribe('good.event', lambda p: {'ok': True}, plugin_name=plugin_name)\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_list_permissions_load_ok(self, list_permissions_plugin_dir: Path):
        """A plugin declaring permissions as a list must load (not crash)."""
        bus = EventBus()
        report = load_and_register(bus, plugins_dir=list_permissions_plugin_dir)
        assert report.registered == ["good"]
        assert report.failed == {}
        assert report.skipped == {}

    def test_list_permissions_registered(self, list_permissions_plugin_dir: Path):
        """The registry must hold the permissions declared as a list."""
        load_and_register(EventBus(), plugins_dir=list_permissions_plugin_dir)
        assert get_registered_plugins().get("good") == {"vault:read", "vault:write"}


class TestAllRealPluginsValid:

    def test_every_shipped_plugin_has_a_valid_manifest(self):
        metas = discover_plugins()
        assert len(metas) >= 6
        for meta in metas:
            assert "_parse_error" not in meta, f"{meta['name']} failed to parse"
            issues = validate_manifest(meta)
            assert issues == [], f"{meta['name']} has invalid manifest: {issues}"
