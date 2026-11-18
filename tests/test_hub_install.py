"""Installation end to end against a local mirror.

These are the promises the launcher makes to a user's disk: the published paths land
unchanged, a shared file is downloaded once, a group is marked installed only when complete,
and a verify notices a truncated file.
"""

from __future__ import annotations

import pytest

from symtrs.hub.errors import HubError
from symtrs.hub.install import install
from symtrs.hub.plan import Selection, resolve
from symtrs.hub.state import read_lock, repair_paths, verify_dataset, verify_models


def _backend(hub_mirror):
    return hub_mirror.backends


def _plan(hub_mirror, groups, splits=("train",), models=None):
    selection = Selection()
    for group_id in groups:
        selection.add_group(group_id, splits)
    for model_id, resolutions in (models or {}).items():
        selection.add_model(model_id, resolutions)
    return resolve(hub_mirror.manifest, selection, _backend(hub_mirror).table)


def _install(hub_mirror, plan, data_root=None, models_dir=None, **kwargs):
    return install(plan, hub_mirror.backends, data_root=data_root, models_dir=models_dir, **kwargs)


def test_dataset_install_writes_the_published_paths_and_lock(hub_mirror, tmp_path):
    data = tmp_path / "data"
    plan = _plan(hub_mirror, ["depth/2048"])
    report = _install(hub_mirror, plan, data_root=data)

    assert report.dataset_files == len(plan.dataset)
    for ref in plan.dataset:
        assert (data / ref.path).is_file()

    lock = read_lock(data)
    assert lock is not None and lock.kind == "dataset"
    assert lock.revision == "v1"
    assert lock.installed_splits("depth/2048") == ("train",)
    assert lock.installed_splits("metadata") == ("train",)

    key = hub_mirror.keys["train"][0]
    # The launcher copies; it does not link, move or rename dataset files.
    image = data / "train" / "rgb_day" / f"rgb_day_2048_{key}.png"
    depth = data / "train" / "depth" / f"depth_day_2048_{key}.exr"
    assert image.is_file() and not image.is_symlink()
    assert depth.is_file() and not depth.is_symlink()
    assert image.read_bytes() == (hub_mirror.root / "views" / "train" / "rgb_day" / image.name).read_bytes()
    # Nothing below native was selected, so no _lr folders may appear.
    assert not (data / "train" / "rgb_day_lr").exists()
    assert verify_dataset(data, lock).ok


def test_a_second_group_reuses_the_shared_image_instead_of_downloading_it_again(hub_mirror, tmp_path):
    data = tmp_path / "data"
    _install(hub_mirror, _plan(hub_mirror, ["depth/2048"]), data_root=data)
    report = _install(hub_mirror, _plan(hub_mirror, ["segmentation/2048"]), data_root=data)

    key = hub_mirror.keys["train"][0]
    images = list((data / "train" / "rgb_day").iterdir())
    assert len(images) == len(hub_mirror.keys["train"])
    assert report.skipped_files >= len(hub_mirror.keys["train"])
    assert (data / "train" / "semantic" / f"seg_day_2048_{key}.png").is_file()


def test_superresolution_lands_its_input_at_the_x8_role(hub_mirror, tmp_path):
    data = tmp_path / "data"
    _install(hub_mirror, _plan(hub_mirror, ["superresolution/x2"]), data_root=data)
    key = hub_mirror.keys["train"][0]
    assert (data / "train" / "rgb_day_lr" / "x8" / f"rgb_day_256_{key}.png").is_file()


def test_a_conflicting_file_is_refused_unless_forced(hub_mirror, tmp_path):
    data = tmp_path / "data"
    plan = _plan(hub_mirror, ["depth/2048"])
    ref = next(ref for ref in plan.dataset if ref.path.startswith(("train/", "val/", "test/")))
    (data / ref.path).parent.mkdir(parents=True)
    (data / ref.path).write_bytes(b"wrong size")

    with pytest.raises(HubError, match="different size"):
        _install(hub_mirror, plan, data_root=data)
    _install(hub_mirror, plan, data_root=data, force=True)
    assert (data / ref.path).stat().st_size == ref.bytes


def test_install_is_resumable(hub_mirror, tmp_path):
    data = tmp_path / "data"
    plan = _plan(hub_mirror, ["depth/2048"])
    _install(hub_mirror, plan, data_root=data)
    report = _install(hub_mirror, plan, data_root=data)
    assert report.downloaded_files == 0
    assert report.skipped_files == len(plan.dataset)


def test_verify_notices_a_truncated_dataset_file_and_a_rerun_repairs_it(hub_mirror, tmp_path):
    data = tmp_path / "data"
    plan = _plan(hub_mirror, ["depth/2048"])
    _install(hub_mirror, plan, data_root=data)
    lock = read_lock(data)
    ref = next(ref for ref in plan.dataset if ref.path.startswith(("train/", "val/", "test/")))
    (data / ref.path).write_bytes(b"x")

    report = verify_dataset(data, lock)
    assert not report.ok
    assert any(str(ref.path) in issue.path.as_posix() and issue.kind == "size"
               for issue in report.issues)

    _install(hub_mirror, plan, data_root=data, force=True)
    assert verify_dataset(data, read_lock(data)).ok


def test_verify_notices_a_truncated_table(hub_mirror, tmp_path):
    data = tmp_path / "data"
    _install(hub_mirror, _plan(hub_mirror, ["depth/2048"]), data_root=data)
    table = data / "tables" / "depth" / "train" / "2048" / "table.jsonl.gz"
    original = table.read_bytes()
    table.write_bytes(original[:-2])

    report = verify_dataset(data, read_lock(data))
    assert not report.ok
    assert any(issue.kind == "size" and issue.path == table for issue in report.issues)


def test_models_install_into_the_run_layout_and_verify_by_hash(hub_mirror, tmp_path):
    models = tmp_path / "runs"
    plan = _plan(hub_mirror, [], models={"midas_small": (1024,)})
    _install(hub_mirror, plan, models_dir=models)

    path = models / "depth" / "midas_small" / "1024" / "weights" / "best.pt"
    assert path.is_file()
    lock = read_lock(models)
    assert lock is not None and lock.kind == "models"
    assert lock.models["midas_small"]["resolutions"] == [1024]
    assert verify_models(models, lock, hub_mirror.manifest).ok

    # Same size, different bytes: only the hash can catch this.
    data = path.read_bytes()
    path.write_bytes(b"\0" * len(data))
    report = verify_models(models, lock, hub_mirror.manifest)
    assert not report.ok
    assert report.issues[0].kind == "sha256"
    assert repair_paths(report) == [path]


def test_each_file_is_downloaded_from_its_declared_repo(hub_mirror, tmp_path):
    models = tmp_path / "runs"
    plan = _plan(hub_mirror, [], models={"midas_small": (1024,)})
    ref = plan.models[0]
    assert ref.repo == "models-depth"
    assert (hub_mirror.root / "models-depth" / ref.path).is_file()
    _install(hub_mirror, plan, models_dir=models)
    assert (models / ref.path).is_file()



