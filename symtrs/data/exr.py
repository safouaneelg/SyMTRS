"""Reading single-channel metric depth.

The revised dataset stores depth as single-channel EXR in metres. `.npy` is also
accepted so the whole pipeline can be exercised before EXR tooling is settled (and
so the test suite needs no EXR writer).

Backends, tried in order by `backend="auto"`:

  * `openexr`  - the `OpenEXR` package. Exact, and handles the float32 channel
    directly. Tries the classic `InputFile` API and then the 3.x `File` API.
  * `cv2`      - `opencv-python-headless`. Reads EXR when OpenCV was built with
    OpenEXR support, which the wheels are.
  * `imageio`  - `imageio.v3`.

Whichever backend is used, the returned array is float32 `[H, W]` in the file's own
row order. EXR row order depends on the writer; `symtrs inspect --save-previews`
writes RGB/depth side-by-side images so a vertically flipped depth map is visible
immediately rather than mistaken for a bad model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEPTH_EXTENSIONS = (".exr", ".npy")
VALID_BACKENDS = ("auto", "openexr", "cv2", "imageio", "numpy")


class DepthReadError(RuntimeError):
    """Raised when no available backend can read a depth file."""


def list_depth_files(root) -> list:
    """Depth files directly inside `root`, sorted. Matches `.exr` and `.npy`.

    Not recursive: every depth file in the layout lives in a flat per-split,
    per-resolution folder, and a recursive walk would quietly fold a `depth_lr/x2` tree
    into the native one.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    files = [p for p in root.iterdir() if p.suffix.lower() in DEPTH_EXTENSIONS]
    return sorted(files)


def read_depth_file(
    path,
    channel: int = 0,
    backend: str = "auto",
) -> np.ndarray:
    """Read depth as a float32 `[H, W]` array. NaN/Inf are preserved as-is."""
    path = Path(path)
    if not path.is_file():
        raise DepthReadError(f"depth file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _read_npy(path)

    if suffix != ".exr":
        raise DepthReadError(
            f"unsupported depth extension {suffix!r} for {path}; "
            f"expected one of {DEPTH_EXTENSIONS}"
        )

    if backend == "numpy":
        raise DepthReadError(
            f"backend 'numpy' only reads .npy files, but {path} is EXR. Use "
            "'openexr', 'cv2' or 'imageio', or install the OpenEXR extra."
        )

    attempts = []
    order = _backend_order(backend)
    for name in order:
        try:
            if name == "openexr":
                return _read_exr_openexr(path, channel)
            if name == "cv2":
                return _read_exr_cv2(path, channel)
            if name == "imageio":
                return _read_exr_imageio(path, channel)
        except Exception as exc:  # noqa: BLE001 - we want the chain to continue
            attempts.append(f"  {name}: {type(exc).__name__}: {exc}")

    raise DepthReadError(
        f"could not read {path} with any available backend.\n"
        + "\n".join(attempts)
        + "\n  install one of:  pip install OpenEXR   |   pip install opencv-python-headless"
    )


def _backend_order(backend: str) -> Tuple[str, ...]:
    if backend == "auto":
        return ("openexr", "cv2", "imageio")
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"unknown depth backend {backend!r}; expected one of {VALID_BACKENDS}"
        )
    return (backend,)


def _read_npy(path: Path) -> np.ndarray:
    array = np.load(path)
    array = np.asarray(array, dtype=np.float32)

    # Writers emit both [1, H, W] (a saved batch) and [H, W, 1] (a channel-last image).
    # Dropping axis 0 unconditionally would turn the second into [W, 1] - a silently
    # transposed and truncated depth map, which reads as a bad model rather than a bug.
    # So only singleton axes go, from either end, and anything still not 2-D is refused.
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim == 1:
        raise DepthReadError(f"depth array in {path} is 1-D with shape {array.shape}")
    if array.ndim != 2:
        raise DepthReadError(
            f"depth array in {path} has shape {array.shape}; expected [H, W] "
            "(or [1, H, W] / [H, W, 1])"
        )
    return np.ascontiguousarray(array)


def _read_exr_openexr(path: Path, channel: int) -> np.ndarray:
    """Classic OpenEXR python API, with a fallback to the 3.x object API."""
    try:
        import OpenEXR  # noqa: F401
        import Imath  # noqa: F401
    except ImportError as exc:
        raise ImportError("OpenEXR and Imath are not installed") from exc

    try:
        return _openexr_classic(path, channel)
    except Exception:
        return _openexr_modern(path, channel)


def _openexr_classic(path: Path, channel: int) -> np.ndarray:
    import Imath
    import OpenEXR

    handle = OpenEXR.InputFile(str(path))
    header = handle.header()

    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1

    names = list(header["channels"].keys())
    if not names:
        raise ValueError("EXR has no channels")
    name = names[channel] if isinstance(channel, int) else str(channel)

    raw = handle.channel(name, Imath.PixelType(Imath.PixelType.FLOAT))
    array = np.frombuffer(raw, dtype=np.float32)
    if array.size != width * height:
        raise ValueError(
            f"channel {name!r} has {array.size} values, expected {width * height}"
        )
    return np.ascontiguousarray(array.reshape(height, width))


def _openexr_modern(path: Path, channel: int) -> np.ndarray:
    import OpenEXR  # OpenEXR >= 3.x

    with OpenEXR.File(str(path)) as handle:
        parts = list(handle.parts)
        if not parts:
            raise ValueError("EXR has no parts")
        header = parts[0].header
        # `header.channels()` returns a dict on the newer API; `channels` a list.
        names = None
        if hasattr(header, "channels"):
            names = header.channels
            if callable(names):
                names = names()
        if isinstance(names, dict):
            names = list(names.keys())
        elif names is None:
            names = [c.name for c in header.get("channels", [])]

        name = names[channel] if isinstance(channel, int) else str(channel)
        pixels = parts[0].channels[name].pixels
        array = np.asarray(pixels, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"channel {name!r} is not 2-D: shape {array.shape}")
        return np.ascontiguousarray(array)


def _read_exr_cv2(path: Path, channel: int) -> np.ndarray:
    import cv2

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if array is None:
        raise ValueError("cv2.imread returned None (OpenCV may lack OpenEXR support)")
    if array.ndim == 3:
        array = array[..., channel]
    return np.ascontiguousarray(array.astype(np.float32))


def _read_exr_imageio(path: Path, channel: int) -> np.ndarray:
    import imageio.v3 as iio

    array = np.asarray(iio.imread(path))
    if array.ndim == 3:
        array = array[..., channel]
    return np.ascontiguousarray(array.astype(np.float32))


# ------------------------------------------------------------------- conversion


def depth_to_tensor(
    array: np.ndarray, scale: float = 1.0, max_depth: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a raw depth array to (metres, valid mask), both float32.

    A pixel is valid when it is finite and strictly positive. Zero is the usual
    "no geometry / sky" marker in rendered depth and must never be treated as a
    distance of zero metres, which would otherwise drag every metric toward the
    origin.

    `scale` converts file units to metres: 1.0 if the EXR already holds metres, 0.01
    if it holds centimetres (the real dataset).

    `max_depth` is in metres, after scaling. The real renders mark sky with huge finite
    values rather than zero, so "finite and positive" alone would count the sky as
    geometry kilometres away; anything beyond `max_depth` is invalid instead. The cut is
    applied to the scaled value so the same number means the same thing whatever the file
    unit is.
    """
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array) & (array > 0.0)
    scaled = array * np.float32(scale)
    if max_depth is not None:
        valid &= scaled <= float(max_depth)
    metres = np.where(valid, scaled, 0.0).astype(np.float32)
    return metres, valid


def depth_summary(array: np.ndarray, scale: float = 1.0) -> dict:
    """Statistics for `symtrs inspect`. Never raises on all-invalid input."""
    metres, valid = depth_to_tensor(array, scale=scale)
    total = int(valid.size)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "shape": tuple(array.shape),
            "min": float("nan"),
            "p1": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "invalid_fraction": 1.0,
            "negative_fraction": float(np.mean(np.asarray(array) < 0)) if total else 1.0,
        }
    values = metres[valid]
    quantiles = np.quantile(values, [0.01, 0.5, 0.99])
    return {
        "shape": tuple(array.shape),
        "min": float(values.min()),
        "p1": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "max": float(values.max()),
        "invalid_fraction": 1.0 - valid_count / max(total, 1),
        "negative_fraction": float(np.mean(np.asarray(array) < 0)),
    }


def guess_scale_from_summary(summary: dict) -> Optional[float]:
    """Suggest a unit conversion from the median depth.

    Purely advisory for `symtrs inspect`: a city-scale aerial median of ~50-500
    reads as metres, ~5000-50000 reads as centimetres (what the real EXRs hold, so
    0.01 is also the `data.depth_scale` default). Sky pixels (huge values) pull the
    median up, which only strengthens the centimetre reading. It never changes a run
    by itself.
    """
    median = summary.get("p50")
    if median is None or not np.isfinite(median) or median <= 0:
        return None
    if median > 3000.0:
        return 0.01
    return 1.0
