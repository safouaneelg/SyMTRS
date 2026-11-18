"""The planner: selection, dedupe, requirements and destination arithmetic.

Pure: a local mirror supplies the tables, nothing reaches the network. The assertions that
matter are the ones a launcher can get quietly wrong - counting a shared `rgb_day` twice,
dropping a required metadata group, or treating an empty resolution tuple as "none".
"""

from __future__ import annotations

import pytest

from symtrs.hub.errors import ManifestError, SelectionError
from symtrs.hub.plan import (
    Selection,
    inspect_destination,
    parse_model_spec,
    plan_as_dict,
    resolve,
    select_tasks,
)


def _backend(hub_mirror):
    return hub_mirror.backends


def _select(groups, splits=("train",), models=None) -> Selection:
    selection = Selection()
    for group_id in groups:
        selection.add_group(group_id, splits)
    for model_id, resolutions in (models or {}).items():
        selection.add_model(model_id, resolutions)
    return selection


def test_shared_rgb_day_is_counted_once(hub_mirror):
    plan = resolve(
        hub_mirror.manifest,
        _select(["depth/2048", "segmentation/2048"]),
        _backend(hub_mirror).table,
    )
    shared = [ref for ref in plan.dataset if "rgb_day_2048" in ref.path]
    assert len(shared) == len(hub_mirror.keys["train"])
    assert all(len(ref.groups) == 2 for ref in shared)
    assert plan.dataset_bytes == plan.referenced_dataset_bytes - plan.saved_bytes
    assert plan.saved_bytes == sum(ref.bytes for ref in shared)


def test_a_required_group_is_added_with_the_same_splits(hub_mirror):
    plan = resolve(hub_mirror.manifest, _select(["depth/2048"], splits=("train", "val")),
                   _backend(hub_mirror).table)
    by_id = {group.id: group for group in plan.groups}
    assert set(by_id) == {"depth/2048", "metadata"}
    assert by_id["metadata"].splits == ("train", "val")


def test_the_task_table_itself_is_part_of_the_plan(hub_mirror):
    plan = resolve(hub_mirror.manifest, _select(["depth/2048"]), _backend(hub_mirror).table)
    tables = [ref for ref in plan.dataset if ref.path.startswith("tables/")]
    assert {ref.path for ref in tables} == {
        "tables/metadata/train.jsonl.gz",
        "tables/depth/train/2048/table.jsonl.gz",
    }
    group = next(group for group in plan.groups if group.id == "depth/2048")
    depth_table = next(ref for ref in tables if ref.path.endswith("depth/train/2048/table.jsonl.gz"))
    assert depth_table.path in group.by_split["train"]
    assert depth_table.bytes > 0


def test_models_keep_only_the_requested_resolution(hub_mirror):
    plan = resolve(
        hub_mirror.manifest,
        _select([], models={"midas_small": (1024,)}),
        _backend(hub_mirror).table,
    )
    assert [ref.path for ref in plan.models] == [
        "depth/midas_small/1024/weights/best.pt"
    ]
    assert plan.model_plans[0].resolutions == (1024,)


def test_an_empty_resolution_tuple_means_every_published_resolution(hub_mirror):
    plan = resolve(
        hub_mirror.manifest,
        _select([], models={"midas_small": ()}),
        _backend(hub_mirror).table,
    )
    assert {ref.resolution for ref in plan.models} == {1024, 2048}


def test_an_unknown_group_names_the_list_command(hub_mirror):
    with pytest.raises(SelectionError, match="symtrs get --list"):
        resolve(hub_mirror.manifest, _select(["depth/9999"]), _backend(hub_mirror).table)


def test_an_unknown_model_lists_near_matches(hub_mirror):
    with pytest.raises(SelectionError, match="midas_small"):
        resolve(
            hub_mirror.manifest,
            _select([], models={"midas_smal": ()}),
            _backend(hub_mirror).table,
        )


def test_a_split_a_group_does_not_publish_is_refused(hub_mirror):
    selection = Selection()
    # Set the mapping directly: `add_group` filters to the package's split names, and this
    # test is about a manifest that advertises a split the group does not publish.
    selection.groups["depth/2048"] = ("train", "nope")
    with pytest.raises(SelectionError, match="nope"):
        resolve(hub_mirror.manifest, selection, _backend(hub_mirror).table)


def test_select_tasks_filters_by_resolution_and_variant(hub_mirror):
    selection = select_tasks(hub_mirror.manifest, ["depth"], resolutions=[2048],
                             splits=["train"])
    assert set(selection.groups) == {"depth/2048"}
    selection = select_tasks(hub_mirror.manifest, ["superresolution"], variants=["x2"])
    assert set(selection.groups) == {"superresolution/x2"}
    with pytest.raises(SelectionError, match="Published tasks"):
        select_tasks(hub_mirror.manifest, ["not_a_task"])


def test_parse_model_spec_accepts_several_resolutions_and_all(hub_mirror):
    assert parse_model_spec(hub_mirror.manifest, ["midas_small@2048+1024"]) == {
        "midas_small": (2048, 1024)
    }
    assert parse_model_spec(hub_mirror.manifest, ["midas_small@all"]) == {"midas_small": ()}
    with pytest.raises(SelectionError, match="not 999"):
        parse_model_spec(hub_mirror.manifest, ["midas_small@999"])


def test_a_table_that_disagrees_about_a_size_is_refused(hub_mirror):
    def bad_source(repo_key: str, path: str):
        yield {"key": "x", "files": [["views/train/loc000/a.png", 1]]}
        yield {"key": "y", "files": [["views/train/loc000/a.png", 2]]}

    with pytest.raises(ManifestError, match="disagrees with itself"):
        resolve(hub_mirror.manifest, _select(["depth/2048"]), bad_source)


def test_a_table_path_cannot_escape_the_destination(hub_mirror):
    def bad_source(repo_key: str, path: str):
        yield {"key": "x", "files": [["../../etc/passwd", 1]]}

    with pytest.raises(ManifestError, match=r"\.\."):
        resolve(hub_mirror.manifest, _select(["depth/2048"]), bad_source)


def test_destination_reports_present_missing_and_conflicts(hub_mirror, tmp_path):
    plan = resolve(hub_mirror.manifest, _select(["depth/2048"]), _backend(hub_mirror).table)
    root = tmp_path / "data"
    first = plan.dataset[0]
    (root / first.path).parent.mkdir(parents=True)
    (root / first.path).write_bytes(b"x" * first.bytes)
    second = plan.dataset[1]
    (root / second.path).parent.mkdir(parents=True, exist_ok=True)
    (root / second.path).write_bytes(b"short")

    report = inspect_destination(plan.dataset, root)
    assert first.path in report.present
    assert second.path in report.conflicts
    assert len(report.missing) == len(plan.dataset) - 2


def test_plan_as_dict_carries_the_arithmetic(hub_mirror):
    plan = resolve(hub_mirror.manifest, _select(["depth/2048"]), _backend(hub_mirror).table)
    data = plan_as_dict(plan)
    assert data["dataset"]["bytes"] == plan.dataset_bytes
    assert data["dataset"]["shared_bytes"] == plan.shared_bytes
    assert data["revision"] == "v1"


def test_each_file_carries_the_repo_that_holds_it(hub_mirror):
    plan = resolve(hub_mirror.manifest, _select(["depth/2048"]), _backend(hub_mirror).table)
    by_path = {ref.path: ref.repo for ref in plan.dataset}
    key = hub_mirror.keys["train"][0]
    assert by_path[f"train/rgb_day/rgb_day_2048_{key}.png"] == "views"
    assert by_path["metadata/views.csv"] == "metadata"
    assert by_path["tables/depth/train/2048/table.jsonl.gz"] == "depth"
    assert set(plan.used_repos) == {"metadata", "views", "depth"}


def test_a_default_group_rides_along_without_an_explicit_requires(hub_mirror):
    import dataclasses

    depth = hub_mirror.manifest.groups["depth/2048"]
    groups = dict(hub_mirror.manifest.groups)
    groups["depth/2048"] = dataclasses.replace(depth, requires=())
    manifest = dataclasses.replace(hub_mirror.manifest, groups=groups)

    plan = resolve(manifest, _select(["depth/2048"]), _backend(hub_mirror).table)
    assert {group.id for group in plan.groups} == {"metadata", "depth/2048"}


def test_a_models_only_selection_does_not_pull_the_default_dataset_group(hub_mirror):
    plan = resolve(
        hub_mirror.manifest,
        _select([], models={"midas_small": (1024,)}),
        _backend(hub_mirror).table,
    )
    assert plan.dataset == ()
    assert [ref.repo for ref in plan.models] == ["models-depth"]
