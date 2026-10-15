"""The model registry: one table, and every entry in it has to be true.

The registry is the only place that knows which models exist. When an entry names a
constructor argument the class does not accept, or a task the class does not implement,
the CLI still accepts the model key and the failure shows up at build time - after the
weights have been downloaded. These tests check what can be checked statically, by
reading the adapter sources rather than importing them, so they run without torch and
without downloading anything.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from symtrs.config import TASKS
from symtrs.registry import (
    REGISTRY,
    ModelSpec,
    RegistryError,
    build_model,
    describe_registry,
    get_spec,
    list_models,
    model_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_files(target: str) -> list:
    """Source files a `module:ClassName` reference could be defined in.

    Usually one file. `symtrs.model.domain` is a package whose `__init__.py` re-exports
    `DomainAdaptation` from `runner.py`, so for a package the search widens to its
    modules rather than assuming the name is defined in the package root.
    """
    module_name, _, class_name = target.partition(":")
    assert class_name, f"{target!r} is not 'module:ClassName'"
    base = REPO_ROOT / Path(*module_name.split("."))
    if base.with_suffix(".py").is_file():
        return [base.with_suffix(".py")]
    assert base.is_dir(), f"{module_name} ({base}) is neither a module nor a package"
    return sorted(base.rglob("*.py"))


def _class_def(target: str) -> ast.ClassDef:
    """The `ClassDef` for a `module:ClassName` reference, parsed from source."""
    _, _, class_name = target.partition(":")
    files = _module_files(target)
    for path in files:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
    raise AssertionError(f"none of {[str(p) for p in files]} defines {class_name}")


def _init_parameters(target: str) -> set:
    """Parameter names accepted by the class's `__init__`, from source.

    `*args`/`**kwargs` are recorded as sentinels so a class that swallows anything is not
    reported as rejecting a name - but see the companion assertion: a registry that
    relies on `**kwargs` to swallow a typo is a bug, not a feature.
    """
    node = _class_def(target)
    init = next(
        (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )
    if init is None:
        raise AssertionError(f"{target} has no __init__; the registry passes kwargs to it")

    names = {a.arg for a in init.args.args} | {a.arg for a in init.args.kwonlyargs}
    names.discard("self")
    if init.args.vararg is not None:
        names.add("*args")
    if init.args.kwarg is not None:
        names.add("**kwargs")
    return names


# --------------------------------------------------------------------- integrity


def test_registry_is_not_empty_and_keys_match_their_entries():
    assert REGISTRY
    for key, spec in REGISTRY.items():
        assert spec.key == key, f"entry {key!r} declares key {spec.key!r}"


def test_every_entry_names_a_task_that_exists():
    for spec in REGISTRY.values():
        assert spec.task in TASKS, f"{spec.key} has task {spec.task!r}"


def test_every_entry_points_at_a_class_that_exists_on_disk():
    for spec in REGISTRY.values():
        _class_def(spec.target)  # asserts the file and the ClassDef


def test_every_entry_names_a_class_that_declares_its_task():
    """`build_model` falls back to `spec.task` only when the class declares none.

    A class that declares a task the registry disagrees with is the dangerous case: the
    registry would win on `.task` while the class's own methods branch on its own value,
    so the two would run different code paths under one name.
    """
    for spec in REGISTRY.values():
        node = _class_def(spec.target)
        declared = None
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(isinstance(t, ast.Name) and t.id == "task" for t in statement.targets):
                declared = ast.literal_eval(statement.value)
        if declared is not None:
            assert declared == spec.task, (
                f"{spec.target} declares task={declared!r} but the registry says "
                f"{spec.task!r}"
            )


def test_default_kwargs_are_parameters_of_the_class_they_are_passed_to():
    """Otherwise the run dies with TypeError after the checkpoint has downloaded.

    This is the check that would have caught `segformer_b1` being absent from the
    registry while the class supported it, and the reverse: a key whose `default_kwargs`
    name a parameter no adapter has.
    """
    for spec in REGISTRY.values():
        accepted = _init_parameters(spec.target)
        for name in spec.default_kwargs:
            assert name in accepted, (
                f"registry entry {spec.key!r} passes {name!r} to {spec.target}, "
                f"whose __init__ takes {sorted(accepted - {'*args', '**kwargs'})}"
            )


@pytest.mark.parametrize(
    "spec", [s for s in REGISTRY.values() if s.task == "depth"], ids=lambda s: s.key
)
def test_depth_entries_declare_a_protocol_and_an_alignment(spec: ModelSpec):
    """A depth model with no protocol cannot be scored: metric and relative models need
    different losses and different post-processing, and guessing is how a table ends up
    mixing two incomparable numbers."""
    assert spec.protocol in {"metric", "relative"}, spec.key
    expected = "none" if spec.protocol == "metric" else {"median", "scale", "scale_shift"}
    if spec.protocol == "metric":
        assert spec.default_align == expected, spec.key
    else:
        assert spec.default_align in expected, spec.key


def test_non_depth_entries_do_not_pretend_to_have_a_protocol():
    for spec in REGISTRY.values():
        if spec.task != "depth":
            assert spec.protocol == "", f"{spec.key} is a {spec.task} model with a depth protocol"
            assert spec.default_align == ""


def test_aliases_resolve_to_their_key_and_never_shadow_another():
    for spec in REGISTRY.values():
        assert get_spec(spec.key).key == spec.key
        for alias in spec.aliases:
            assert get_spec(alias).key == spec.key, alias
            assert alias not in REGISTRY, f"alias {alias!r} collides with a registry key"


def test_a_metric_and_a_relative_model_both_exist_per_family_where_it_matters():
    """The run matrix needs a metric and a relative baseline. ZoeDepth and DA3 each ship
    one of each, and losing one of them silently narrows the comparison."""
    keys = set(REGISTRY)
    assert {"zoedepth_n", "zoedepth_k"} <= keys  # relative and metric, same family
    assert {"da3_large", "da3_metric_large"} <= keys


# ----------------------------------------------------------------------- lookups


def test_unknown_model_suggests_the_closest_key():
    """A typo costs a lookup, not a documentation read."""
    with pytest.raises(RegistryError) as info:
        get_spec("zoedpth_k")
    message = str(info.value)
    assert "zoedepth_k" in message
    assert "symtrs models" in message


def test_a_completely_unknown_model_still_gets_a_usable_message():
    with pytest.raises(RegistryError, match="unknown model"):
        get_spec("definitely_not_a_model")


def test_empty_model_key_is_refused():
    with pytest.raises(RegistryError, match="model key is required"):
        get_spec("")


def test_list_models_filters_by_task():
    for task in ("depth", "segmentation", "domain_adapt"):
        keys = model_keys(task)
        assert keys
        assert all(REGISTRY[k].task == task for k in keys)
    assert len(model_keys()) == len(REGISTRY)
    with pytest.raises(RegistryError, match="unknown task"):
        list_models("depth_estimation")


def test_describe_registry_covers_every_key():
    text = describe_registry()
    for key in REGISTRY:
        assert key in text, key
    assert "protocol" in text  # the column that makes the metric/relative split visible


def test_build_model_rejects_an_unknown_key_before_importing_anything():
    with pytest.raises(RegistryError):
        build_model("nope_not_a_model")


def test_registry_import_does_not_pull_in_a_heavy_optional_dependency():
    """`symtrs models` and `symtrs doctor` are the commands you run when something is
    missing, so importing the registry must not itself require the missing thing.
    """
    import subprocess
    import sys

    probe = (
        "import sys, symtrs.registry as r;"
        "loaded = [m for m in ('transformers', 'depth_pro', 'timm', 'torchvision') "
        "if m in sys.modules];"
        "print(','.join(loaded))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"importing symtrs.registry pulled in {completed.stdout.strip()}, which would "
        "break `symtrs models` on a machine where that extra is not installed"
    )
