"""Locating and driving the vendored CycleGAN/pix2pix code.

The *complete* `junyanz/pytorch-CycleGAN-and-pix2pix` source tree is committed in this
repository, at `pytorch-CycleGAN-and-pix2pix/`, alongside this package. Nothing is
cloned, fetched or added to `sys.path` at run time: a run needs no network access and
`pip install -e .` is enough to reproduce a paper's domain-adaptation numbers on a
machine that has never seen GitHub.

This module is the only place that knows where that tree is and whether it is usable.

Two upstream behaviours have to be worked around rather than fixed, because the whole
point of this task is to use the original training code:

1. **The auto-split trap.** `UnalignedDataset.__init__` takes the plain
   `dataroot/<phase>A` and `dataroot/<phase>B` path *only* when those directories exist.
   When either is missing it falls through to the modified auto-splitting branch, which
   re-splits `--raw_A_dir`/`--raw_B_dir` by `--train_ratio` and `--split_seed` - ignoring
   the SyMTRS train/val/test split entirely. Passing `--raw_A_dir` and `--raw_B_dir`
   therefore does *not* disable the split. `build_staging_dir` avoids the branch
   completely by materialising `trainA`, `trainB`, `valA`, `valB`, `testA`, `testB` as
   symlinks to the real split folders.

2. **Phase naming.** Upstream indexes phases as `dataroot/<phase>A`, so a staging
   directory named for the SyMTRS split (`train`) produces `trainA`/`trainB` for free.
   No flags needed.

Because the staging directory contains only symlinks, the dataset is never copied.

`data/unaligned_dataset.py` and `data/aligned_dataset.py` in the vendored tree are this
repository's contributions - they are what adds the auto-split and tiling options above.
Upstream's originals are kept side by side under `upstream_originals/`, so the delta is
readable without a network fetch. `symtrs setup-adapt --fetch` re-derives that delta
against the pinned revision.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Upstream project.
UPSTREAM_REPO = "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git"

#: The revision the vendored tree was taken from, verified by cloning it and reading
#: `git rev-parse HEAD`. Pinning to a SHA rather than a branch is what makes the
#: vendored copy checkable: `symtrs setup-adapt --fetch` compares the tree here against
#: exactly this commit, so "the code we ran" is a fact rather than a date.
UPSTREAM_REVISION = "2a7afba2895d52556dd5dfe07e8555ef657ced6f"
UPSTREAM_COMMIT_DATE = "2025-08-06"

#: Directories under the upstream root that are *not* vendored. `imgs/` is the README's
#: teaser gallery (7.4MB, almost all of it one animated GIF) and is not code; upstream's
#: `README.md` is kept as `README.upstream.md` so the licence and provenance travel with
#: the tree even though the live README is SyMTRS-specific.
VENDORED_EXCLUDES = ("imgs",)

#: Files this repository deliberately changed, relative to the upstream root, and why.
#: `symtrs setup-adapt --fetch` treats anything *not* in here that differs from the
#: pinned revision as a warning, which is what makes the check worth running: it turns
#: "we think this is upstream" into something that can fail.
DEVIATIONS = {
    "data/unaligned_dataset.py": "CycleGAN loader: auto-split fallback, tiling",
    "data/aligned_dataset.py": "pix2pix loader: auto-split fallback, tiling, basename pairing",
    "README.md": "replaced by the SyMTRS-facing README (upstream's is README.upstream.md)",
}

#: The subset of `DEVIATIONS` that carry code. Their upstream originals are kept in
#: `upstream_originals/` so the delta is readable without fetching anything.
MODIFIED_FILES = tuple(
    name for name in DEVIATIONS if name.endswith(".py")
)

#: Every translation SyMTRS runs, as data: `direction -> (domain A role, domain B role,
#: upstream --direction)`. The roles are `symtrs.data.splits` roles, i.e. the `rgb_<role>`
#: folder staged into `<split>A` / `<split>B`. A pair is always staged the same way whichever
#: way it is translated - A is `rgb_day` - and the direction only says which of the two
#: upstream treats as input, so the day/night pair and the shadow/shadow-free pair share one
#: staging, one launch and one scorer instead of a copy each.
DIRECTION_ROLES: Dict[str, Tuple[str, str, str]] = {
    "day2night": ("day", "night", "AtoB"),
    "night2day": ("day", "night", "BtoA"),
    "shadow2noshadow": ("day", "noshadow", "AtoB"),
    "noshadow2shadow": ("day", "noshadow", "BtoA"),
}


def direction_spec(direction: str) -> Tuple[str, str, str]:
    """`(A role, B role, upstream direction)` of a SyMTRS direction, or a clear error."""
    try:
        return DIRECTION_ROLES[direction]
    except KeyError:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of {tuple(DIRECTION_ROLES)}"
        ) from None


def target_role(direction: str) -> str:
    """The `rgb_<role>` folder holding the real images this direction should generate."""
    a_role, b_role, upstream_side = direction_spec(direction)
    return b_role if upstream_side == "AtoB" else a_role


#: What upstream imports before it parses anything. `train.py` and `test.py` both reach
#: `util.visualizer`, which imports these two at module level, so a domain-adaptation run
#: cannot start without them installed - they are not optional extras of upstream's HTML
#: visualiser in practice. `pip install -e '.[adapt]'` is what installs them.
REQUIRED_IMPORTS = ("dominate", "wandb")

#: Files that must exist for the vendored tree to be usable. This is a sample of the
#: upstream entry points rather than every file: it is here to catch a half-copied or
#: half-deleted tree with a clear message, not to checksum the whole thing.
REQUIRED_FILES = (
    "train.py",
    "test.py",
    "LICENSE",
    "models/__init__.py",
    "models/base_model.py",
    "models/networks.py",
    "models/cycle_gan_model.py",
    "models/pix2pix_model.py",
    "options/__init__.py",
    "options/base_options.py",
    "options/train_options.py",
    "options/test_options.py",
    "data/__init__.py",
    "data/base_dataset.py",
    "data/image_folder.py",
    "data/unaligned_dataset.py",
    "data/aligned_dataset.py",
    "util/__init__.py",
    "util/util.py",
    "util/visualizer.py",
)

#: The SyMTRS splits, and the folder inside each split that each domain lives in.
SPLITS = ("train", "val", "test")


class UpstreamError(RuntimeError):
    """The vendored upstream tree is missing, incomplete, or could not be prepared."""


def default_upstream() -> Path:
    """Where the upstream tree lives, unless overridden.

    `SYMTRS_UPSTREAM` wins so that a shared cluster install can be pointed at a
    separately-managed checkout without editing anything; otherwise the vendored folder
    in this repository is used, which is complete as committed and needs no setup step.
    """
    override = os.environ.get("SYMTRS_UPSTREAM", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "pytorch-CycleGAN-and-pix2pix"


def missing_files(root: Path) -> List[str]:
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def is_complete(root: Optional[Path] = None) -> bool:
    root = Path(root) if root else default_upstream()
    return root.is_dir() and not missing_files(root)


def verify(root: Optional[Path] = None) -> Path:
    """Return the upstream root, or explain precisely what is missing."""
    root = Path(root).expanduser() if root else default_upstream()
    if not root.is_dir():
        raise UpstreamError(
            f"no vendored upstream tree at {root}. The complete CycleGAN/pix2pix source "
            f"is committed to this repository, so this means the checkout is damaged; "
            f"`git checkout -- pytorch-CycleGAN-and-pix2pix` restores it. To use a "
            f"separately-managed copy instead, point SYMTRS_UPSTREAM at it."
        )
    missing = missing_files(root)
    if missing:
        raise UpstreamError(
            f"the vendored tree at {root} is incomplete; missing {missing}. "
            "These files are committed to this repository, so a missing one means the "
            "checkout is damaged rather than that a setup step was skipped."
        )
    return root


def setup(
    target: Optional[str] = None,
    force: bool = False,
    revision: str = UPSTREAM_REVISION,
    fetch: bool = False,
) -> Path:
    """Check the vendored tree, and optionally compare it against upstream.

    There is no clone step any more: the source is committed, so the common case is a
    no-op that verifies the tree is present and complete. `fetch=True` goes to the
    network on purpose, to re-derive which files differ from the pinned revision.

    `force` is accepted for call-site compatibility and no longer changes anything on
    disk - nothing here writes to the tree, because the two files that differ from
    upstream are this repository's own work and overwriting them would be a silent
    regression in the benchmark's data loading.
    """
    root = Path(target).expanduser() if target else default_upstream()
    verify(root)

    if fetch:
        report = compare_with_upstream(root, revision)
        _print_drift(root, report, revision)
    else:
        print(f"[symtrs] vendored upstream ready at {root}")
        print(
            f"[symtrs]   {UPSTREAM_REPO} @ {UPSTREAM_REVISION[:9]} ({UPSTREAM_COMMIT_DATE})"
        )
        print(
            f"[symtrs]   modified here: {', '.join(MODIFIED_FILES)}"
            " (originals in upstream_originals/)"
        )
    return root


def compare_with_upstream(root: Path, revision: str = UPSTREAM_REVISION) -> Dict[str, List[str]]:
    """Clone the pinned revision into a scratch directory and diff the tree against it.

    Returns `{"changed", "only_local", "only_upstream", "excluded"}`, all relative paths.
    Nothing is written to `root`: this is a report, and the point of it is to be able to
    answer "is the code in this repository the code we think it is" before a paper
    submission, not to update anything.
    """
    if shutil.which("git") is None:
        raise UpstreamError("git is not on PATH, so upstream cannot be fetched for comparison")

    scratch = Path(tempfile.mkdtemp(prefix="symtrs-upstream-"))
    try:
        print(f"[symtrs] fetching {UPSTREAM_REPO} @ {revision[:9]} to compare")
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", revision, UPSTREAM_REPO,
             str(scratch / "upstream")],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            # A shallow clone cannot always be asked for a bare SHA, so fall back to a
            # full clone of the default branch and check the SHA out explicitly.
            completed = subprocess.run(
                ["git", "clone", UPSTREAM_REPO, str(scratch / "upstream")],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise UpstreamError(
                    f"git clone failed with exit code {completed.returncode}: "
                    f"{completed.stderr.strip()}\n"
                    "Comparing against upstream needs network access; the vendored tree "
                    "itself does not, so this is only a problem for this check."
                )
            subprocess.run(
                ["git", "-C", str(scratch / "upstream"), "checkout", "--quiet", revision],
                check=True,
            )

        upstream_root = scratch / "upstream"
        return _diff_trees(root, upstream_root)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _diff_trees(root: Path, upstream_root: Path) -> Dict[str, List[str]]:
    changed: List[str] = []
    only_local: List[str] = []
    only_upstream: List[str] = []
    excluded: List[str] = []

    def relevant(relative: Path) -> bool:
        if relative.parts and relative.parts[0] in VENDORED_EXCLUDES:
            return False
        # Local bookkeeping that has no upstream counterpart by design.
        if relative.parts and relative.parts[0] == "upstream_originals":
            return False
        if relative.name in {"README.upstream.md", ".gitignore.upstream"}:
            return False
        return True

    for path in sorted(upstream_root.rglob("*")):
        relative = path.relative_to(upstream_root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if not relevant(relative):
            if path.is_file():
                excluded.append(str(relative))
            continue
        if path.is_dir():
            continue
        local = root / relative
        if not local.is_file():
            only_upstream.append(str(relative))
        elif not filecmp.cmp(path, local, shallow=False):
            changed.append(str(relative))

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if relative.parts and relative.parts[0] in VENDORED_EXCLUDES:
            continue
        if (upstream_root / relative).is_file():
            continue
        only_local.append(str(relative))

    return {
        "changed": changed,
        "only_local": only_local,
        "only_upstream": only_upstream,
        "excluded": excluded,
    }


def _print_drift(root: Path, report: Dict[str, List[str]], revision: str) -> None:
    changed = report["changed"]
    print(f"[symtrs] compared {root} against {UPSTREAM_REPO} @ {revision[:9]}")

    unexpected = [name for name in changed if name not in DEVIATIONS]
    for name in sorted(changed):
        marker = "modified here" if name in DEVIATIONS else "DIFFERS"
        print(f"[symtrs]   {marker}: {name}")

    if report["only_upstream"]:
        print(f"[symtrs]   not vendored ({len(report['only_upstream'])}): "
              + ", ".join(report["only_upstream"][:5])
              + (" ..." if len(report["only_upstream"]) > 5 else ""))
    if report["only_local"]:
        print(f"[symtrs]   added by this repository ({len(report['only_local'])}): "
              + ", ".join(sorted(report["only_local"])[:5])
              + (" ..." if len(report["only_local"]) > 5 else ""))
    if report["excluded"]:
        print(f"[symtrs]   excluded on purpose: {', '.join(sorted(report['excluded']))}")

    if unexpected:
        print(
            "[symtrs] WARNING: files differ from the pinned revision and are not in "
            f"DEVIATIONS: {', '.join(sorted(unexpected))}. Either that change was not "
            "intentional and should be reverted, or it belongs in DEVIATIONS in "
            "symtrs/model/domain/upstream.py."
        )
    else:
        print("[symtrs] only the documented deviations differ from upstream")


# ------------------------------------------------------------------- staging


def link_file(source: Path, destination: Path) -> str:
    """Make `destination` point at `source`; return how ("symlink", "hardlink", "copy").

    A symlink is preferred (nothing is copied), but on Windows `os.symlink` needs Developer
    Mode or an elevated shell and fails with `OSError` otherwise. A hardlink costs no disk
    either but only works within one volume; across volumes the last resort is a real copy.
    The fallbacks are silent to upstream - it just opens files - and reported by the caller,
    so a surprising disk bill has a visible cause.
    """
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    try:
        destination.symlink_to(source)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        pass
    shutil.copy2(source, destination)
    return "copy"


def staged_name(path: Path) -> str:
    """The name a file is staged under: its pairing key plus its extension.

    pix2pix's `AlignedDataset` pairs A and B by *exact basename*, but the real files are
    `rgb_day_2048_<key>.png` and `rgb_night_2048_<key>.png`. Staging both under `<key>.png`
    makes the names agree without touching the vendored tree.
    """
    from symtrs.data.io import normalize_stem

    return normalize_stem(path.stem) + path.suffix.lower()


def build_staging_dir(
    root: str,
    run_dir: str,
    split: str = "train",
    require_all: bool = False,
    direction: str = "day2night",
    aligned: bool = False,
) -> Path:
    """Stage `trainA/trainB/valA/valB/testA/testB` as folders of per-file links.

    Which folders become A and B comes from `DIRECTION_ROLES`: `rgb_day` / `rgb_night` for
    the day-night directions, `rgb_day` / `rgb_noshadow` for the shadow ones.

    `aligned=True` (pix2pix) also writes `<staging>/<split>/<split>/<key>.png`, each A image
    next to its B image: see `write_aligned_pairs` for why the links alone are not enough.

    `dataroot` is then the staging directory and upstream's `--phase` selects the split
    as it always did. Links mean (normally) no image is copied, and because the
    directories exist, upstream never reaches its auto-splitting branch. Each file is linked
    under its normalised key (`staged_name`) so day and night names match for pix2pix.
    """
    from symtrs.data.io import list_images
    from symtrs.data.splits import resolve_split_dirs

    a_role, b_role, _ = direction_spec(direction)
    staging = Path(run_dir) / "staging" / split
    staging.mkdir(parents=True, exist_ok=True)

    for name in SPLITS if require_all else (split,):
        try:
            directories = resolve_split_dirs(root, name, 2048)
        except Exception as exc:
            if name == split:
                raise
            print(f"[symtrs] staging skipped split {name!r}: {exc}")
            continue

        keys = {}
        for domain, folder in (
            ("A", getattr(directories, a_role)), ("B", getattr(directories, b_role))
        ):
            link = staging / f"{name}{domain}"
            if folder is None:
                print(f"[symtrs] {name}: no folder for domain {domain}; skipping")
                continue
            if not Path(folder).is_dir():
                raise UpstreamError(f"the {name} split has no {folder}")
            # An earlier run staged this as a directory symlink (or junction); a real
            # directory of links is rebuilt from scratch.
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                shutil.rmtree(link)
            link.mkdir(parents=True)

            methods = set()
            names = set()
            for image in list_images(folder):
                target = staged_name(image)
                # Absolute source: the run directory is typically moved (or written to
                # shared storage and read from a different mount) after training, and a
                # relative link would break the moment it is.
                methods.add(link_file(image.resolve(), link / target))
                names.add(target)
            keys[domain] = names
            print(
                f"[symtrs] staged {link.name}: {len(names)} files from {folder} "
                f"({'/'.join(sorted(methods)) or 'empty'})"
            )
            if methods - {"symlink"}:
                print(
                    "[symtrs]   symlinks are unavailable here (on Windows: enable Developer "
                    "Mode); fell back to hardlinks/copies, which behave the same for upstream"
                )

        if len(keys) == 2:
            shared = keys["A"] & keys["B"]
            if not shared:
                raise UpstreamError(
                    f"the {name} split has no {a_role}/{b_role} pair with a matching name "
                    f"({len(keys['A'])} {a_role}, {len(keys['B'])} {b_role} files); run "
                    "`symtrs doctor --data-root <dataset root>` for a coverage report"
                )
            if len(shared) < len(keys["A"]) or len(shared) < len(keys["B"]):
                print(
                    f"[symtrs] WARNING: {name}: only {len(shared)} names appear in both "
                    "domains; pix2pix (aligned) will fail on the others"
                )
            if aligned:
                write_aligned_pairs(staging, name, sorted(shared))

    return staging


def write_aligned_pairs(staging: Path, name: str, keys: List[str]) -> int:
    """Write `<staging>/<name>/<key>.png` as `[A | B]` side by side, for pix2pix.

    The vendored `AlignedDataset` never looks at `<phase>A` / `<phase>B`. It reads
    `<dataroot>/<phase>/*.png` as upstream's own pix2pix format - one image holding A on its
    left half and B on its right, what upstream's `combine_A_and_B.py` produces - and, only
    when that folder is empty, falls back to auto-splitting `raw_A_dir` / `raw_B_dir` by
    `--train_ratio`, which cannot take everything (the ratios must sum to less than 1) and so
    silently drops part of the SyMTRS split. The concatenated folder is the supported path
    that keeps the split intact, and unlike the links it costs disk: one lossless PNG per
    scene. A file is reused when it is newer than both of its sources, so a second phase or a
    re-score does not pay for it again.
    """
    from PIL import Image

    folder = staging / name
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {Path(key).stem + ".png" for key in keys}
    for stale in folder.iterdir():  # a pair whose source left the split must not be trained on
        if stale.is_file() and stale.name not in wanted:
            stale.unlink()
    written = 0
    for key in keys:
        source_a = (staging / f"{name}A" / key).resolve()
        source_b = (staging / f"{name}B" / key).resolve()
        target = folder / (Path(key).stem + ".png")
        newest = max(source_a.stat().st_mtime, source_b.stat().st_mtime)
        if target.is_file() and target.stat().st_mtime >= newest:
            continue
        with Image.open(source_a) as left, Image.open(source_b) as right:
            left, right = left.convert("RGB"), right.convert("RGB")
            if left.size != right.size:
                raise UpstreamError(
                    f"{key}: the A image is {left.size[0]}x{left.size[1]} but the B image is "
                    f"{right.size[0]}x{right.size[1]}; pix2pix needs aligned pairs of one size"
                )
            pair = Image.new("RGB", (left.width * 2, left.height))
            pair.paste(left, (0, 0))
            pair.paste(right, (left.width, 0))
        pair.save(target, format="PNG")
        written += 1
    print(f"[symtrs] staged {folder.name}/: {len(keys)} A|B pairs for pix2pix ({written} written)")
    return written


def count_staged(staging: Path, split: str) -> Tuple[int, int]:
    """`(images in splitA, images in splitB)`, or `(-1, -1)` if the links are missing."""
    counts = []
    for domain in ("A", "B"):
        folder = staging / f"{split}{domain}"
        if not folder.is_dir():
            counts.append(-1)
            continue
        counts.append(sum(1 for p in folder.iterdir() if p.is_file()))
    return counts[0], counts[1]


# ------------------------------------------------------------------ commands


def epoch_split(epochs: int, decay_epochs: Optional[int] = None) -> Tuple[int, int]:
    """Split a total epoch budget into `(n_epochs, n_epochs_decay)`.

    Upstream decides its schedule from *two* flags, `--n_epochs` and `--n_epochs_decay`,
    and only the first is ever passed by a SyMTRS run. Upstream's default for the second
    is 100, so `--epochs 40` would silently train for 140 epochs (~3.5x the requested
    work, at 2048px, on a CycleGAN). `cfg.train.epochs` means the total here, as it does
    for every other task in this package, and this is where that is made true.

    The split follows CycleGAN's own shape: a flat learning-rate phase, then a linear
    decay to zero. Two thirds flat is enough for the cycle-consistency objective to
    settle before the decay starts shrinking what it learned.
    """
    total = max(1, int(epochs))
    if decay_epochs is not None:
        decay = max(0, int(decay_epochs))
        return max(1, total - decay), decay
    decay = total // 3
    return total - decay, decay


def missing_imports() -> List[str]:
    """Upstream's start-up imports that are not installed, if any.

    `train.py` and `test.py` both reach `util.visualizer`, which imports `dominate` and
    `wandb` at module level, before a single option is parsed. Neither is a dependency of
    `symtrs` itself - they belong to upstream's HTML visualiser - so they cannot be
    required by the package, and without them the failure is an upstream traceback that
    says nothing about how to fix it. This is the check that turns it into a sentence.
    """
    import importlib.util

    missing: List[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):  # a broken install reports as missing
            found = False
        if not found:
            missing.append(name)
    return missing


def upstream_direction(direction: str) -> str:
    """`day2night` -> `AtoB`, `night2day` -> `BtoA`, and likewise for the shadow pair.

    Stated once (in `DIRECTION_ROLES`), because it is used by both the training command
    and the generation command: a translation trained one way and scored the other way
    produces a run that completes happily and generates the wrong domain. Upstream reads
    this flag to decide which network is the generator (`TestModel` builds `netG` from
    `input_nc`, which the option setter derives from the direction).
    """
    return direction_spec(direction)[2]


def launcher(gpus: int = 1) -> List[str]:
    """The argv prefix that starts one of upstream's scripts.

    Upstream at this revision picks its device *inside* the process:
    `util.util.init_ddp()` reads `WORLD_SIZE`/`LOCAL_RANK` when it is started under DDP
    and otherwise takes `cuda:0` if a GPU is visible and `cpu` if not. There is no
    `--gpu_ids` flag any more, so multi-GPU means DDP, and DDP means starting the script
    the way upstream expects to be started. `torch.distributed.run` sets exactly the
    environment variables `init_ddp` reads, so this is upstream's own launcher rather
    than a reimplementation of one.
    """
    if gpus and gpus > 1:
        return [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(gpus),
        ]
    return [sys.executable]


def train_command(
    upstream: Path,
    staging: Path,
    experiment: str,
    direction: str,
    dataset_mode: str,
    upstream_model: str,
    epochs: int,
    batch_size: int,
    checkpoints_dir: str = "",
    extra_args: Optional[List[str]] = None,
    decay_epochs: Optional[int] = None,
    gpus: int = 1,
) -> List[str]:
    """argv for upstream `train.py`, translating a SyMTRS direction into A/B.

    `day2night` means "translate domain A (day) into domain B (night)", which in
    CycleGAN's terms is `--direction AtoB`; see `upstream_direction`.

    No `--epoch` is passed: upstream's `train.py` defaults to `--epoch latest`, which
    continues `checkpoints/<name>` if it exists and starts fresh if it does not. That is
    the behaviour a SyMTRS run wants, since `--name` carries the full configuration.

    `epochs` is the total, split by `epoch_split` into upstream's two flags.

    Only flags `options/base_options.py` and `options/train_options.py` actually define
    are emitted: upstream's parser *is* the interface, and a flag it has dropped
    (`--gpu_ids`, `--results_dir`) makes the run fail before the first batch. The device
    is steered through the environment instead - see `launcher` and
    `symtrs.model.domain.runner.device_env`.
    """
    n_epochs, n_epochs_decay = epoch_split(epochs, decay_epochs)
    command = launcher(gpus) + [
        str(upstream / "train.py"),
        "--dataroot", str(staging),
        "--name", experiment,
        "--model", upstream_model,
        "--dataset_mode", dataset_mode,
        "--direction", upstream_direction(direction),
        "--n_epochs", str(n_epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--batch_size", str(batch_size),
        # Upstream writes `latest_net_G_*.pth` at the end of an epoch only when
        # `epoch % save_epoch_freq == 0`, and its default is 5. A short run - any smoke
        # test, or `--epochs 3` on a small dataset - then trains happily and leaves no
        # checkpoint for the scoring step to load, which reads as a scoring bug. One
        # epoch between saves is what makes "the weights the run ended with" always
        # exist. Put `--save_epoch_freq 5` in `model_kwargs.extra_args` to thin it out:
        # these flags are appended after these, and upstream's parser takes the last.
        "--save_epoch_freq", "1",
    ]
    # Upstream defaults this to ./checkpoints relative to the working directory. Pinning
    # it to the run directory keeps a SyMTRS run self-contained instead of scattering
    # weights into whichever directory the process happened to start in. Training
    # samples go to <checkpoints_dir>/<name>/web/ - upstream's `train.py` has no
    # `--results_dir`; only `test.py` does, and that is what `test_command` sets.
    if checkpoints_dir:
        command += ["--checkpoints_dir", checkpoints_dir]
    return command + list(extra_args or [])


def test_command(
    upstream: Path,
    staging: Path,
    experiment: str,
    split: str,
    dataset_mode: str,
    upstream_model: str,
    direction: str = "day2night",
    checkpoint: str = "",
    checkpoints_dir: str = "",
    results_dir: str = "",
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """argv for upstream `test.py`, which generates the images that get scored.

    `--model cycle_gan --dataset_mode unaligned` rather than upstream's `--model test`:
    the unaligned dataset loads both domains, so CycleGAN's own `get_current_visuals()`
    writes `fake_A` *and* `fake_B` (plus `real_*`, `rec_*`) into
    `results/<name>/<phase>_<epoch>/images/`. The requested translation is `fake_B` in
    both directions: `--direction BtoA` swaps the *inputs* in `set_input` and `fake_B` is
    still `G(real_A)`, while `fake_A` is the opposite translation; see
    `symtrs.model.domain.runner.GENERATED_LABEL`. `--num_test` is raised out of the way
    because upstream defaults it to 50 and silently truncates a larger test split.
    """
    command = [
        sys.executable, str(upstream / "test.py"),
        "--dataroot", str(staging),
        "--name", experiment,
        "--model", upstream_model,
        "--dataset_mode", dataset_mode,
        "--direction", upstream_direction(direction),
        "--phase", split,
        "--num_test", "100000",
    ]
    if checkpoints_dir:
        command += ["--checkpoints_dir", checkpoints_dir]
    if results_dir:
        command += ["--results_dir", results_dir]
    if checkpoint:
        command += ["--epoch", checkpoint]
    return command + list(extra_args or [])


__all__ = [
    "DEVIATIONS",
    "DIRECTION_ROLES",
    "MODIFIED_FILES",
    "REQUIRED_FILES",
    "REQUIRED_IMPORTS",
    "SPLITS",
    "UPSTREAM_COMMIT_DATE",
    "UPSTREAM_REPO",
    "UPSTREAM_REVISION",
    "VENDORED_EXCLUDES",
    "UpstreamError",
    "build_staging_dir",
    "compare_with_upstream",
    "count_staged",
    "default_upstream",
    "direction_spec",
    "epoch_split",
    "is_complete",
    "launcher",
    "missing_files",
    "missing_imports",
    "setup",
    "target_role",
    "test_command",
    "train_command",
    "upstream_direction",
    "verify",
    "write_aligned_pairs",
]
