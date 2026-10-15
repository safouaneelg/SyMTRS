"""Resolution-to-directory mapping for the revised SyMTRS layout.

    <root>/train/ | val/ | test/
        rgb_day/            rgb_night/
        rgb_day_lr/x2|x4|x8
        depth/              depth_lr/x2|x4|x8
        semantic/           semantic_lr/x2|x4|x8

A resolution names the *pixel* size of the data, which is what the benchmark cares
about, and maps to a directory suffix:

    2048 -> native            (rgb_day, depth, semantic)
    1024 -> _lr/x2
     512 -> _lr/x4
     256 -> _lr/x8

`rgb_night` has no low-resolution variant, which is why domain adaptation is
2048-only.

Every accessor raises `MissingDirectoryError` naming the exact path it looked for.
There is deliberately no fallback to a different resolution: silently training at
1024 when you asked for 2048 is worse than stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

#: Named resolutions, largest first.
RESOLUTIONS: Tuple[int, ...] = (2048, 1024, 512, 256)

#: PyTorch convention is a scale factor; the layout uses directory suffixes.
RESOLUTION_TO_SCALE = {2048: 1, 1024: 2, 512: 4, 256: 8}
RESOLUTION_TO_SUFFIX = {2048: None, 1024: "x2", 512: "x4", 256: "x8"}

SPLITS: Tuple[str, ...] = ("train", "val", "test")


class MissingDirectoryError(FileNotFoundError):
    """A directory the run needs does not exist."""

    def __init__(self, role: str, path: Path, resolution: int, split: str) -> None:
        super().__init__(
            f"no {role} directory for split={split!r} resolution={resolution}: "
            f"expected {path}\n"
            f"  run `symtrs doctor --data-root <dataset root>` to see what is present"
        )
        self.role = role
        self.path = Path(path)
        self.resolution = resolution
        self.split = split


def scale_for_resolution(resolution: int) -> int:
    try:
        return RESOLUTION_TO_SCALE[resolution]
    except KeyError:
        raise ValueError(
            f"unsupported resolution {resolution!r}; expected one of {RESOLUTIONS}"
        ) from None


def subdir_for(role: str, resolution: int) -> str:
    """Relative subdirectory for `role` at `resolution`.

    role is one of "day", "night", "depth", "semantic".
    """
    suffix = RESOLUTION_TO_SUFFIX.get(resolution)
    if suffix is None and resolution not in RESOLUTION_TO_SUFFIX:
        raise ValueError(
            f"unsupported resolution {resolution!r}; expected one of {RESOLUTIONS}"
        )

    if role == "day":
        return "rgb_day" if suffix is None else f"rgb_day_lr/{suffix}"
    if role == "night":
        return "rgb_night"  # native only
    if role == "depth":
        return "depth" if suffix is None else f"depth_lr/{suffix}"
    if role == "semantic":
        return "semantic" if suffix is None else f"semantic_lr/{suffix}"
    raise ValueError(f"unknown role {role!r}; expected day, night, depth or semantic")


@dataclass(frozen=True)
class SplitDirs:
    """Resolved directory paths for one (split, resolution) pair.

    A field is None when that role does not exist at this resolution, which is only
    ever true of `night` below 2048.
    """

    root: Path
    split: str
    resolution: int
    day: Path
    night: Optional[Path]
    depth: Optional[Path]
    semantic: Optional[Path]

    @property
    def scale(self) -> int:
        return scale_for_resolution(self.resolution)

    @property
    def is_native(self) -> bool:
        return self.resolution == max(RESOLUTIONS)

    def path_for(self, role: str) -> Path:
        """The path for `role`, raising if the directory is missing on disk."""
        path = getattr(self, role, None)
        if path is None:
            raise MissingDirectoryError(role, self.root / subdir_for(role, self.resolution), self.resolution, self.split)
        if not path.is_dir():
            raise MissingDirectoryError(role, path, self.resolution, self.split)
        return path

    def as_dict(self) -> dict:
        return {
            "day": self.day,
            "night": self.night,
            "depth": self.depth,
            "semantic": self.semantic,
        }


def resolve_split_dirs(root: str, split: str, resolution: int) -> SplitDirs:
    """Map (root, split, resolution) to concrete directories.

    This does not require the directories to exist; call `SplitDirs.path_for` when
    you need one and want a clear error if it is absent.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root_path}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    split_root = root_path / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"split directory does not exist: {split_root}\n"
            f"  the revised layout expects {root_path}/{{train,val,test}}"
        )

    night = (split_root / subdir_for("night", resolution)) if resolution == max(RESOLUTIONS) else None

    return SplitDirs(
        root=root_path,
        split=split,
        resolution=resolution,
        day=split_root / subdir_for("day", resolution),
        night=night,
        depth=split_root / subdir_for("depth", resolution),
        semantic=split_root / subdir_for("semantic", resolution),
    )


def available_resolutions(root: str, split: str = "train") -> Tuple[int, ...]:
    """Which resolutions are actually present, by looking for the day directory.

    Used by `symtrs inspect` and to let `sweep` skip grid cells that cannot run.
    """
    present = []
    for resolution in RESOLUTIONS:
        try:
            dirs = resolve_split_dirs(root, split, resolution)
        except (FileNotFoundError, ValueError):
            continue
        if dirs.day.is_dir():
            present.append(resolution)
    return tuple(present)


def iter_roles() -> Iterator[str]:
    yield "day"
    yield "night"
    yield "depth"
    yield "semantic"


def describe_layout(root: str) -> str:
    """Human-readable dump of what exists, for `symtrs inspect`."""
    lines = [f"dataset root: {root}"]
    for split in SPLITS:
        split_root = Path(root) / split
        if not split_root.is_dir():
            lines.append(f"  {split}/  MISSING")
            continue
        lines.append(f"  {split}/")
        for resolution in RESOLUTIONS:
            try:
                dirs = resolve_split_dirs(root, split, resolution)
            except (FileNotFoundError, ValueError):
                continue
            marks = []
            for role in iter_roles():
                path = getattr(dirs, role)
                if path is None:
                    marks.append(f"{role}=-")
                elif path.is_dir():
                    marks.append(f"{role}=ok")
                else:
                    marks.append(f"{role}=missing")
            lines.append(f"    {resolution:>5}: " + "  ".join(marks))
    return "\n".join(lines)
