"""The legacy super-resolution tree lives inside the package, and reaches outside it never.

The move that put it there is only worth something if it is checkable. The point is that
`models/superresolution/`, `utils/superresolution/` and `dataloader/` became
`symtrs.legacy.*` without a line of implementation changing, and that nothing in the tree
reaches for a top-level name (`models`, `utils`, `dataloader`) a caller might also have --
`utils` is why the trainer used to be launched as a script from the repository root instead
of imported as a module.

Everything here is path and source-text work, so it runs on a machine with no torch: that
is the machine state you are in when you are checking a fresh checkout, which is where a
file lost by a move is found. Two tests at the end are marked rather than torch-free, and
deliberately: they run the shim and import the moved package, and the trainer imports torch
at module level. There is no torch-free way to ask "does this command work", and stubbing
torch out would be testing the stub.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY = REPO_ROOT / "symtrs" / "legacy"

#: Every file that moved, at the path it moved to. A move that silently loses a file is the
#: failure this list exists to catch.
MOVED = (
    "superres/models/__init__.py",
    "superres/models/srcnn.py",
    "superres/models/autoencoder_sr.py",
    "superres/models/srgan.py",
    "superres/models/swinir.py",
    "superres/utils/__init__.py",
    "superres/utils/losses.py",
    "superres/utils/metrics.py",
    "superres/utils/image_utils.py",
    "superres/utils/train_utils.py",
    "superres/train.py",
    "dataloader/__init__.py",
    "dataloader/utils.py",
    "dataloader/split.py",
    "dataloader/superres_dataset.py",
    "dataloader/depth_dataset.py",
    "dataloader/domain_adapt_dataset.py",
    "dataloader/README.md",
)

PYTHON_FILES = tuple(name for name in MOVED if name.endswith(".py"))

#: Top-level names the legacy tree must not import. Checked by the *root* of an absolute
#: import, so `from torch.utils.data import DataLoader` is fine and `from utils import x`
#: is not; a relative import (`level > 0`) is the tree importing itself, which is what
#: makes the move possible at all.
#:
#: Three names, not an allowlist of permitted ones: `models`, `utils` and `dataloader` are
#: the collision that caused the problem, and an allowlist would fail on every new
#: third-party import while catching nothing this does not.
TOP_LEVEL = {"models", "utils", "dataloader"}


def read(path: Path) -> str:
    """Read a source file whole, BOM or not.

    The moved files kept their BOMs, because keeping them is the byte-identity rule, and a
    BOM read as text is a stray character that breaks `ast.parse` while leaving `import`
    working.
    """
    return path.read_text(encoding="utf-8-sig")


def test_every_moved_file_is_where_it_moved_to() -> None:
    missing = [name for name in MOVED if not (LEGACY / name).is_file()]
    assert missing == [], f"the move lost these files: {missing}"


def test_the_moved_files_still_parse_despite_their_boms() -> None:
    for name in PYTHON_FILES:
        text = read(LEGACY / name)
        assert not text.startswith("﻿"), name
        ast.parse(text)  # a BOM left in place by read_text() would raise here


def test_no_module_in_the_legacy_tree_reaches_for_a_top_level_name() -> None:
    offenders = []
    for name in PYTHON_FILES:
        for node in ast.walk(ast.parse(read(LEGACY / name))):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            offenders += [(name, root) for root in roots if root in TOP_LEVEL]
    assert offenders == [], (
        f"the legacy tree still imports top-level names {offenders}. It has to be "
        "importable without the repository root on sys.path -- that is the whole reason "
        "the trainer was launched as a script rather than imported."
    )


def writes_to_sys_path(source: str) -> bool:
    """Whether a module mutates `sys.path`, as opposed to mentioning it in a docstring.

    The distinction matters because this package is now free to *write about* `sys.path`:
    the reason the legacy trainer used to be launched as a script rather than imported was
    a top-level `utils` name, and that story belongs in a docstring or a README. Asserting
    on the string would punish every sentence that explains the move, which is the opposite
    of the point.
    """
    for node in ast.walk(ast.parse(source)):
        # sys.path.append(...) / .insert(...) / .extend(...) - any method at all
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "path"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sys"
        ):
            return True
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            targets = []
        for target in targets:
            # sys.path = [...] / sys.path += [...]
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "path"
                and isinstance(target.value, ast.Name)
                and target.value.id == "sys"
            ):
                return True
    return False


def test_the_sys_path_detector_detects_a_write() -> None:
    """The guard below is only worth having if the detector can fail.

    A guard for an invariant that already holds passes the moment it is written, so the
    watch-it-fail step has to happen here instead - on the inputs that separate the two
    things that look alike: a real mutation, and a sentence describing one.
    """
    assert writes_to_sys_path("import sys\nsys.path.append('/repo')\n") is True
    assert writes_to_sys_path("import sys\nsys.path.insert(0, '/repo')\n") is True
    assert writes_to_sys_path("import sys\nsys.path = ['/repo']\n") is True
    assert writes_to_sys_path('"""It used to put the root on `sys.path`."""\n') is False
    assert writes_to_sys_path("import sys\nprint(sys.path[0])\n") is False


def test_nothing_in_the_package_writes_to_sys_path() -> None:
    """The legacy trainer's `sys.path.append(ROOT)` is the trap the move removes.

    Those two lines are what made the trainer a script rather than a module: it reached
    `models/`, `utils/` and `dataloader/` by putting the repository root on the path, and a
    top-level `utils` inherited by an import would shadow the caller's. The move deletes
    them, and this is the test that notices if it did not.
    """
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "symtrs").rglob("*.py")
        if writes_to_sys_path(read(path))
    ]
    assert offenders == []


def test_the_root_script_is_a_shim() -> None:
    """`train_superresolution.py` may delegate; it may not contain the trainer."""
    text = read(REPO_ROOT / "train_superresolution.py")
    assert "from symtrs.legacy.superres.train import main" in text
    # `sys.path` is checked by what the module *does*, not by the string: the shim's own
    # docstring explains that it does no `sys.path` work, and a substring check would punish
    # the sentence rather than the write. Same distinction as
    # `test_nothing_in_the_package_writes_to_sys_path`, and for the same reason.
    assert not writes_to_sys_path(text), "the shim has grown sys.path logic"
    for forbidden in ("argparse", "torch", "class ", "def main_worker"):
        assert forbidden not in text, f"the shim has grown logic: {forbidden}"
    assert len(text.splitlines()) < 25


def test_the_root_dataloader_package_is_a_shim() -> None:
    """The documented `from dataloader import ...` has to keep resolving, to the moved code."""
    text = read(REPO_ROOT / "dataloader" / "__init__.py")
    assert "symtrs.legacy.dataloader" in text
    assert "def " not in text and "class " not in text


def test_the_root_models_package_is_gone() -> None:
    """`models/` is removed, so `from models.superresolution import SRCNN` stops working.

    Nothing in this repository, in the configs, in the tests or in the README used that
    import, which is why removing the directory is a decision rather than a risk - and the
    one exposure it does have (an out-of-repo script of the author's) is written down in
    `symtrs/legacy/README.md`. This test is here so that decision stays a decision.
    """
    assert not (REPO_ROOT / "models").exists()
    assert (LEGACY / "superres" / "models" / "srcnn.py").is_file()


def test_genai_stayed_where_it_was() -> None:
    """`utils/genai/` is referenced by nothing and is still not this task's to delete.

    The move takes `dataloader/` wholesale and brings two unused datasets with it, on the
    principle that "unused here" is not "unused". `utils/genai/` is the same call one
    directory over: it stays exactly where the author left it until they say otherwise.
    """
    assert (REPO_ROOT / "utils" / "genai").is_dir()
    assert not (LEGACY / "genai").exists()
    assert not (LEGACY / "dataloader" / "genai").exists()


def test_the_root_script_runs_from_any_directory(tmp_path: Path) -> None:
    """`cd /tmp && python3 /repo/train_superresolution.py --help` has to work.

    It does because a script's own directory is what Python puts on `sys.path`, which is
    also why the shim needs no path work of its own -- the check is here so that a future
    edit cannot quietly make the shim depend on the working directory.
    """
    pytest.importorskip("torch", reason="torch is not installed")
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_superresolution.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--lr_root" in completed.stdout


def test_the_shim_imports_the_moved_code_and_not_a_copy(tmp_path: Path) -> None:
    """`from dataloader import SuperResDataset` must land on `symtrs.legacy.dataloader`."""
    pytest.importorskip("torch", reason="torch is not installed")
    completed = subprocess.run(
        [sys.executable, "-c", "import dataloader; print(dataloader.SuperResDataset.__module__)"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "symtrs.legacy.dataloader.superres_dataset"


#: Phrases that name code where it used to be. A stale path in a comment is how the next
#: reader is sent to a file that is not there.
STALE = (
    "models/superresolution",
    "utils/superresolution",
    "train_superresolution.py at the repository root",
)


def test_nothing_but_the_provenance_note_points_at_the_old_locations() -> None:
    """`symtrs/` and `configs/` may not name the old paths; `symtrs/legacy/` must.

    The legacy tree is the one place that has to name them - that is what the provenance
    table is - so it is excluded rather than allowlisted file by file.
    """
    scanned = [
        path
        for path in list((REPO_ROOT / "symtrs").rglob("*.py"))
        + list((REPO_ROOT / "configs").rglob("*"))
        if path.is_file()
        and "legacy" not in path.relative_to(REPO_ROOT).parts
        and path.suffix in {".py", ".yaml", ".md"}
    ]
    offenders = [
        (str(path.relative_to(REPO_ROOT)), phrase)
        for path in scanned
        for phrase in STALE
        if phrase in read(path)
    ]
    assert offenders == [], f"these still point at the old locations: {offenders}"
