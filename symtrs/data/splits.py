"""Resolution-to-directory mapping for the revised SyMTRS layouts.

Flat layout:
    <root>/train/ | val/ | test/
        rgb_day/            rgb_night/          rgb_noshadow/
        rgb_day_lr/x2|x4|x8
        depth/              depth_lr/x2|x4|x8
        semantic/           semantic_lr/x2|x4|x8

View-organized layout:
    <root>/views/train/loc<NNN>/
        rgb_day_<resolution>_<key>.png
        depth_day_<resolution>_<key>.exr
        seg_day_<resolution>_<key>.png

A resolution names the *pixel* size of the data, which is what the benchmark cares
about, and maps to a directory suffix:

    2048 -> native            (rgb_day, depth, semantic)
    1024 -> _lr/x2
     512 -> _lr/x4
     256 -> _lr/x8

`rgb_night` and `rgb_noshadow` have no low-resolution variant, which is why domain
adaptation (and any shadow-removal task) is 2048-only. `path_for` validates the split
container, and `files_for` selects the requested family in either layout. There is
deliberately no fallback to a different resolution: silently training at 1024 when you
asked for 2048 is worse than stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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


def spread_indices(total: int, count: int) -> List[int]:
    """`count` indices spread evenly over `range(total)` (all of them when count is 0 or >= total).

    The validation subset that picks `best.pt` (`train.val_subset`): evenly spaced over the
    sorted split rather than its first `count` files, because the files are sorted by location
    and the first few hundred would all come from a handful of scenes.
    """
    if count <= 0 or count >= total:
        return list(range(total))
    return [int(i * total / count) for i in range(count)]


def scale_for_resolution(resolution: int) -> int:
    try:
        return RESOLUTION_TO_SCALE[resolution]
    except KeyError:
        raise ValueError(
            f"unsupported resolution {resolution!r}; expected one of {RESOLUTIONS}"
        ) from None


def subdir_for(role: str, resolution: int) -> str:
    """Relative subdirectory for `role` at `resolution`.

    role is one of "day", "night", "noshadow", "depth", "semantic".

    `noshadow` exists at native resolution only and raises below it: there is no
    downsampled shadow-free imagery, and approximating it would change what a
    shadow-removal number means.
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
    if role == "noshadow":
        if suffix is not None:
            raise ValueError(
                f"role 'noshadow' exists at native resolution only (2048), not "
                f"{resolution}: there is no rgb_noshadow_lr folder. Use --resolution 2048."
            )
        return "rgb_noshadow"
    if role == "depth":
        return "depth" if suffix is None else f"depth_lr/{suffix}"
    if role == "semantic":
        return "semantic" if suffix is None else f"semantic_lr/{suffix}"
    raise ValueError(f"unknown role {role!r}; expected day, night, noshadow, depth or semantic")


@dataclass(frozen=True)
class SplitDirs:
    """Resolved directory paths for one (split, resolution) pair.

    A field is None when that role does not exist at this resolution, which is only
    ever true of `night` and `noshadow` below 2048.
    """

    root: Path
    split: str
    resolution: int
    day: Path
    night: Optional[Path]
    depth: Optional[Path]
    semantic: Optional[Path]
    noshadow: Optional[Path] = None
    view_root: Optional[Path] = None

    @property
    def scale(self) -> int:
        return scale_for_resolution(self.resolution)

    @property
    def is_native(self) -> bool:
        return self.resolution == max(RESOLUTIONS)

    def path_for(self, role: str) -> Path:
        """The role directory, or its shared view container, if it exists."""
        path = getattr(self, role, None)
        if path is None:
            # subdir_for raises its own ValueError for a native-only role below 2048.
            raise MissingDirectoryError(role, self.root / self.split / subdir_for(role, self.resolution), self.resolution, self.split)
        if not path.is_dir():
            raise MissingDirectoryError(role, path, self.resolution, self.split)
        return path

    def files_for(self, role: str) -> List[Path]:
        """List files for one role in either supported on-disk layout."""
        folder = self.path_for(role)
        if self.view_root is None:
            extensions = {".exr", ".npy"} if role == "depth" else {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }
            return sorted(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )

        if role not in ("day", "night", "noshadow", "depth", "semantic"):
            raise ValueError(f"unknown role {role!r}")
        if role in ("night", "noshadow") and self.resolution != 2048:
            return []

        resolution = 2048 if role in ("night", "noshadow") else self.resolution
        modality, condition = {
            "day": ("rgb", "day"),
            "night": ("rgb", "night"),
            "noshadow": ("rgb", "noshadow"),
            "depth": ("depth", "day"),
            "semantic": ("seg", "day"),
        }[role]
        extensions = (".exr",) if role == "depth" else (".png",)
        modalities = ("seg", "semantic") if role == "semantic" else (modality,)
        return sorted(
            path
            for name in modalities
            for extension in extensions
            for path in self.view_root.glob(
                f"loc*/{name}_{condition}_{resolution}_*{extension}"
            )
            if path.is_file()
        )

    def filename_prefix(self, role: str) -> str:
        """Filename prefix for a role in the view-organized layout."""
        if self.view_root is None:
            raise ValueError("filename prefixes are only used by the view-organized layout")
        if role not in ("day", "night", "noshadow", "depth", "semantic"):
            raise ValueError(f"unknown role {role!r}")
        resolution = 2048 if role in ("night", "noshadow") else self.resolution
        modality, condition = {
            "day": ("rgb", "day"),
            "night": ("rgb", "night"),
            "noshadow": ("rgb", "noshadow"),
            "depth": ("depth", "day"),
            "semantic": ("seg", "day"),
        }[role]
        return f"{modality}_{condition}_{resolution}_"

    def as_dict(self) -> dict:
        return {
            "day": self.day,
            "night": self.night,
            "noshadow": self.noshadow,
            "depth": self.depth,
            "semantic": self.semantic,
        }


def resolve_split_dirs(root: str, split: str, resolution: int) -> SplitDirs:
    """Map (root, split, resolution) to directories in either supported layout.

    Flat roots retain role-specific directories. View-organized roots use the split's
    `views/<split>` directory and select role files through `SplitDirs.files_for`.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root_path}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    scale_for_resolution(resolution)
    split_root = root_path / split
    view_root = root_path / "views" / split
    if not split_root.is_dir() and not view_root.is_dir():
        raise FileNotFoundError(
            f"split directory does not exist: {split_root}\n"
            f"  expected {root_path}/{{train,val,test}} or {root_path}/views/{{train,val,test}}"
        )

    if not split_root.is_dir():
        return SplitDirs(
            root=root_path,
            split=split,
            resolution=resolution,
            day=view_root,
            night=view_root if resolution == max(RESOLUTIONS) else None,
            noshadow=view_root if resolution == max(RESOLUTIONS) else None,
            depth=view_root,
            semantic=view_root,
            view_root=view_root,
        )

    night = (split_root / subdir_for("night", resolution)) if resolution == max(RESOLUTIONS) else None

    noshadow = (
        (split_root / subdir_for("noshadow", resolution)) if resolution == max(RESOLUTIONS) else None
    )

    return SplitDirs(
        root=root_path,
        split=split,
        resolution=resolution,
        day=split_root / subdir_for("day", resolution),
        night=night,
        noshadow=noshadow,
        depth=split_root / subdir_for("depth", resolution),
        semantic=split_root / subdir_for("semantic", resolution),
    )


def available_resolutions(root: str, split: str = "train") -> Tuple[int, ...]:
    """Which resolutions are actually present, by looking for day files.

    Used by `symtrs inspect` and to let `sweep` skip grid cells that cannot run.
    """
    present = []
    for resolution in RESOLUTIONS:
        try:
            dirs = resolve_split_dirs(root, split, resolution)
        except (FileNotFoundError, ValueError):
            continue
        has_day = bool(dirs.files_for("day")) if dirs.view_root else dirs.day.is_dir()
        if has_day:
            present.append(resolution)
    return tuple(present)


def iter_roles() -> Iterator[str]:
    yield "day"
    yield "night"
    yield "noshadow"
    yield "depth"
    yield "semantic"


def describe_layout(root: str) -> str:
    """Human-readable dump of what exists, for `symtrs inspect`."""
    lines = [f"dataset root: {root}"]
    for split in SPLITS:
        split_root = Path(root) / split
        view_root = Path(root) / "views" / split
        if not split_root.is_dir() and not view_root.is_dir():
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
                else:
                    present = bool(dirs.files_for(role)) if dirs.view_root else path.is_dir()
                    marks.append(f"{role}={'ok' if present else 'missing'}")
            lines.append(f"    {resolution:>5}: " + "  ".join(marks))
    return "\n".join(lines)
