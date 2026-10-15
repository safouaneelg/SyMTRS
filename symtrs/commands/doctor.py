"""`symtrs doctor` - check the environment, and check that a model actually works.

    symtrs doctor                          # environment and dependency report
    symtrs doctor --data-root /data/SyMTRS # also checks the dataset layout
    symtrs doctor --model zoedepth_k --build --smoke

The last form is the important one. Several of these models are loaded through
`torch.hub` entrypoints whose exact return type cannot be checked without downloading
them, so the adapter is written defensively and this command is how you find out which
branch was actually taken:

  * `--build` constructs the model, reporting its parameter counts and whether the
    freeze policy did what you meant.
  * `--smoke` also runs one forward pass on a random image at the target resolution and
    prints the output shape and value range, which is the fastest way to discover that a
    network returns a dict, a tuple, or metres where you expected disparity.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from typing import List, Optional

#: module name -> what it enables, and the extra that installs it.
OPTIONAL = {
    "torch": ("the whole package", "core"),
    "torchvision": ("DeepLabV3 segmentation", "core"),
    "PIL": ("reading images and writing previews", "core"),
    "yaml": ("YAML run configs", "core"),
    "numpy": ("everything", "core"),
    "matplotlib": ("depth colour maps and training curves", "core"),
    "tqdm": ("progress bars", "core"),
    "OpenEXR": ("reading 32-bit EXR depth", "exr"),
    "Imath": ("OpenEXR's own dependency", "exr"),
    "cv2": ("EXR fallback reader, and fast image IO", "cv2"),
    "imageio": ("EXR fallback reader", "imageio"),
    # timm is listed once, for every family that needs it. The MiDaS and ZoeDepth
    # checkpoints come from torch.hub, whose `hubconf.py` reaches `import timm` through
    # `midas.blocks` -> `backbones.beit` - so it is a load-time requirement of the published
    # weights, not only of our own SegFormer encoders.
    "timm": ("MiDaS, ZoeDepth and SegFormer", "timm"),
    "transformers": ("Depth Anything 3", "da3"),
    "depth_pro": ("Apple Depth Pro", "depthpro"),
    # The vendored CycleGAN/pix2pix `train.py` imports both at module level (through
    # `util/visualizer.py`), before it parses an option, so a domain-adaptation run
    # cannot start without them even though the SyMTRS wrapper itself never imports them.
    "dominate": ("domain adaptation (upstream HTML visualiser)", "adapt"),
    "wandb": ("domain adaptation (upstream logging import)", "adapt"),
}


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("doctor", help="check the environment and smoke-test a model")
    parser.add_argument("--task", default=None, help="restrict the model report to one task")
    parser.add_argument("--model", default="", help="comma-separated registry keys to check")
    parser.add_argument("--freeze", default="", help="freeze policy to report the effect of")
    parser.add_argument("--build", action="store_true",
                        help="construct the models (downloads weights); off by default")
    parser.add_argument("--smoke", action="store_true",
                        help="run one forward pass per model; implies --build")
    parser.add_argument("--size", type=int, default=0,
                        help="input size for the smoke test (default: the model's native size, else 256)")
    parser.add_argument("--data-root", dest="data_root", default="", help="also check the dataset layout")
    parser.add_argument("--depth-backend", dest="depth_backend", default="auto",
                        choices=["auto", "openexr", "cv2", "imageio"])
    parser.add_argument("--device", default=None)
    parser.set_defaults(handler=_run)


def _report_environment() -> None:
    print("environment")
    print(f"  python           {platform.python_version()} ({sys.executable})")
    print(f"  platform         {platform.platform()}")

    try:
        import torch

        print(f"  torch            {torch.__version__}")
        print(f"  cuda available   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda version     {torch.version.cuda}")
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                print(
                    f"  gpu {index}            {properties.name}, "
                    f"{properties.total_memory / 1024 ** 3:.1f} GiB"
                )
        else:
            print("  note             no CUDA device; training will be extremely slow")
    except ImportError:
        print("  torch            NOT INSTALLED")


def _report_dependencies() -> List[str]:
    print("dependencies")
    missing_extras: List[str] = []
    for module, (purpose, extra) in OPTIONAL.items():
        available = _installed(module)
        mark = "ok  " if available else "MISS"
        print(f"  [{mark}] {module:<14} {purpose}")
        if not available and extra not in ("core",) and extra not in missing_extras:
            missing_extras.append(extra)
    if missing_extras:
        print(f"  install the missing ones with: pip install -e '.[{','.join(missing_extras)}]'")
    return missing_extras


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


#: ZoeDepth's own `environment.yml` pins this, and it is the version its released weights and
#: vendored backbone were built against. Measured: `ZoeD_N` builds, runs and predicts under
#: 0.6.12; under 0.9.16 and 1.0.30 the checkpoint's `relative_position_index` buffers are no
#: longer persisted and the forward then raises `AttributeError: 'Block' object has no
#: attribute 'drop_path'`. SegFormer is the other way round - `mit_b0` does not exist in
#: 0.6.12 - so the two cannot share one environment, which is worth knowing before a
#: resolution x model grid starts downloading weights.
_ZOEDEPTH_TIM_PIN = "0.6.12"


def _timm_notes() -> List[str]:
    """Warnings about the installed timm, for the models whose upstream code pins a version."""
    if not _installed("timm"):
        return []

    notes: List[str] = []
    try:
        import timm

        version = getattr(timm, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - a broken timm is reported by the table above
        return notes

    if version != _ZOEDEPTH_TIM_PIN:
        notes.append(
            f"timm {version} is installed, but the ZoeDepth adapters need "
            f"timm=={_ZOEDEPTH_TIM_PIN} (ZoeDepth's own environment.yml pin). They are "
            "verified working there and verified failing on 0.9.16 and 1.0.30."
        )
    try:
        has_mit = "mit_b0" in timm.list_models()
    except Exception:  # noqa: BLE001
        has_mit = True
    if not has_mit:
        notes.append(
            f"timm {version} has no `mit_b0`, so the SegFormer adapters cannot be built. "
            "SegFormer and ZoeDepth need separate environments."
        )
    return notes


def _report_dataset(root: str, backend: str) -> None:
    print("dataset")
    from symtrs.data.exr import read_depth_file
    from symtrs.data.io import list_images
    from symtrs.data.splits import available_resolutions, describe_layout, resolve_split_dirs

    print(describe_layout(root))

    for split in ("train", "val", "test"):
        resolutions = available_resolutions(root, split)
        if not resolutions:
            print(f"  [{split}] no resolution folders found")
            continue
        print(f"  [{split}] resolutions present: {resolutions}")

    try:
        dirs = resolve_split_dirs(root, "train", 2048)
    except Exception as exc:
        print(f"  could not resolve the 2048 training split: {exc}")
        return

    rgb = list_images(dirs.day)
    depth_files = [p for p in dirs.depth.iterdir()] if dirs.depth.is_dir() else []
    labels = list_images(dirs.semantic)
    print(f"  rgb_day files    {len(rgb)}")
    print(f"  depth files      {len(depth_files)}")
    print(f"  semantic files   {len(labels)}")

    if depth_files:
        try:
            array = read_depth_file(depth_files[0], backend=backend)
            finite = array[array > 0]
            import numpy as np

            print(
                f"  depth sample     {depth_files[0].name}: shape={array.shape} "
                f"dtype={array.dtype} min={float(finite.min()) if finite.size else 'n/a'} "
                f"median={float(np.median(finite)) if finite.size else 'n/a'} "
                f"max={float(finite.max()) if finite.size else 'n/a'}"
            )
            if finite.size and float(np.median(finite)) > 3000:
                print(
                    "  WARNING          median depth is above 3000. If the EXR stores "
                    "millimetres, set --depth-scale 0.01 or the metrics will be off by 100x."
                )
        except Exception as exc:
            print(f"  depth sample     FAILED: {exc}")


def _report_models(args: argparse.Namespace) -> None:
    from symtrs.registry import list_models, model_keys

    if args.model:
        keys = [item.strip() for item in args.model.split(",") if item.strip()]
    else:
        keys = model_keys(args.task)

    print("models")
    for key in keys:
        spec = next((s for s in list_models() if s.key == key or key in s.aliases), None)
        if spec is None:
            print(f"  {key}: not in the registry")
            continue
        problems = [m for m in spec.requires if not _installed(m)]
        state = f"needs {', '.join(problems)}" if problems else "dependencies ok"
        print(f"  {key:<26} {spec.task:<13} {state}")
        if args.build and not problems:
            _build_one(key, args)
        elif args.build and problems:
            print(f"    skipped: install {', '.join(problems)} first")


def _build_one(key: str, args: argparse.Namespace) -> None:
    import torch

    from symtrs.registry import build_model

    kwargs = {"freeze": args.freeze} if args.freeze else {}
    try:
        model = build_model(key, **kwargs)
    except Exception as exc:
        print(f"    BUILD FAILED: {type(exc).__name__}: {exc}")
        return

    print(f"    {model.describe()}")
    if getattr(model, "protocol", ""):
        print(
            f"    protocol={model.protocol} align={getattr(model, 'align', '-')} "
            f"native_input={getattr(model, 'native_input', 0)}"
        )

    if not args.smoke:
        return

    size = args.size or getattr(model, "native_input", 0) or 256
    size = max(64, int(round(size / 32.0)) * 32)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()
    image = torch.rand(1, 3, size, size, device=device)
    try:
        with torch.no_grad():
            prediction = model.forward(model.preprocess_image(image))
        print(f"    smoke: forward({size}x{size}) -> {tuple(prediction.shape)} "
              f"range [{float(prediction.min()):.3g}, {float(prediction.max()):.3g}]")
        with torch.no_grad():
            final = model.predict(image)
        print(f"           predict -> {tuple(final.shape)} "
              f"range [{float(final.min()):.3g}, {float(final.max()):.3g}]")
    except Exception as exc:
        print(f"    SMOKE FAILED: {type(exc).__name__}: {exc}")
        # The adapter file is derived from the spec's own target, not guessed from the key.
        # Guessing went wrong: `'depth' in 'midas_small'` is False, so a MiDaS failure told
        # the reader to go and fix `symtrs/model/semantic.py`.
        module = spec.target.partition(":")[0]
        print(
            f"    the adapter is {module.replace('.', '/')}.py; "
            "re-run with --traceback for the full stack"
        )


def _run(args: argparse.Namespace) -> int:
    if args.smoke:
        args.build = True

    _report_environment()
    print()
    missing = _report_dependencies()
    if args.data_root:
        print()
        _report_dataset(args.data_root, args.depth_backend)
    print()
    _report_models(args)

    print()
    if missing:
        print(f"[symtrs] {len(missing)} optional extra(s) missing; the models needing them are listed above")
    else:
        print("[symtrs] every optional dependency is installed")
    for note in _timm_notes():
        print(f"[symtrs] timm: {note}")
    return 0


__all__ = ["add_parser"]
