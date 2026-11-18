"""The hub manifest contract: schema, validation and the errors a bad release gets.

These tests are torch-free and network-free (AGENTS.md, "Testing"). The manifest is the
one file a release can get wrong that makes every later number meaningless, so each
rejection here asserts the message as well as the failure.
"""

from __future__ import annotations

import json

import pytest

from symtrs.hub.errors import ManifestError
from symtrs.hub.manifest import load_manifest_bytes, load_manifest_file


def _manifest_dict(hub_mirror) -> dict:
    return json.loads((hub_mirror.root / "metadata" / "symtrs-hub.json").read_text())


def _load(data: dict):
    return load_manifest_bytes(json.dumps(data).encode(), source="test manifest")


def test_the_fixture_release_loads(hub_mirror):
    manifest = load_manifest_file(hub_mirror.root / "metadata" / "symtrs-hub.json")
    assert manifest.schema == "symtrs-hub/2"
    assert manifest.release.revision == "v1"
    assert set(manifest.groups) >= {
        "metadata", "depth/2048", "segmentation/2048",
        "domain_adapt/day-night", "shadow_removal/2048", "superresolution/x2",
    }
    depth = manifest.groups["depth/2048"]
    assert depth.table == "tables/depth/{split}/2048/table.jsonl.gz"
    assert depth.requires == ("metadata",)
    assert depth.table_bytes["train"] > 0
    assert manifest.models["midas_small"].resolutions == (2048, 1024)


def test_unknown_schema_is_refused_with_the_supported_version(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["schema"] = "symtrs-hub/99"
    with pytest.raises(ManifestError, match="symtrs-hub/2"):
        _load(data)


def test_a_table_must_have_a_split_placeholder(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["groups"][1]["table"] = "tasks/depth/train/2048/table.jsonl.gz"
    with pytest.raises(ManifestError, match=r"\{split\}"):
        _load(data)


def test_a_require_must_name_a_published_group(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["groups"][1]["requires"] = ["metadata", "missing-group"]
    with pytest.raises(ManifestError, match="missing-group"):
        _load(data)


def test_a_group_must_declare_a_size_for_every_split_it_publishes(hub_mirror):
    data = _manifest_dict(hub_mirror)
    del data["groups"][1]["table_bytes"]["val"]
    with pytest.raises(ManifestError, match="val"):
        _load(data)


def test_duplicate_group_ids_are_refused(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["groups"].append(json.loads(json.dumps(data["groups"][1])))
    with pytest.raises(ManifestError, match="appears twice"):
        _load(data)


def test_a_preset_must_name_published_groups(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["presets"][0]["groups"].append("not-a-group")
    with pytest.raises(ManifestError, match="not-a-group"):
        _load(data)


def test_a_model_path_cannot_escape_the_destination(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["models"][0]["files"][0]["path"] = "../../.ssh/config"
    with pytest.raises(ManifestError, match=r"\.\."):
        _load(data)


def test_a_model_hash_must_be_hexadecimal(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["models"][0]["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ManifestError, match="sha256"):
        _load(data)


def test_an_unknown_layout_is_refused(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["release"]["layout"] = "nested"
    with pytest.raises(ManifestError, match="task-first"):
        _load(data)


def test_every_referenced_repo_must_be_declared(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["groups"][1]["repo"] = "not-a-repo"
    with pytest.raises(ManifestError, match="not-a-repo"):
        _load(data)


def test_a_model_must_live_in_a_model_repo(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["models"][0]["repo"] = "views"
    with pytest.raises(ManifestError, match="needs a model repo"):
        _load(data)


def test_a_file_route_must_name_a_declared_repo(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["file_repos"]["views/"] = "nowhere"
    with pytest.raises(ManifestError, match="nowhere"):
        _load(data)


def test_a_default_group_must_exist(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["default_groups"] = ["metadata", "nope"]
    with pytest.raises(ManifestError, match="nope"):
        _load(data)


def test_the_catalog_repo_must_be_declared(hub_mirror):
    data = _manifest_dict(hub_mirror)
    data["repos"] = [repo for repo in data["repos"] if repo["key"] != "metadata"]
    with pytest.raises(ManifestError, match="catalog repo"):
        _load(data)
