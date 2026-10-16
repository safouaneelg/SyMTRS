"""`symtrs inspect` - read a checkpoint's metadata block without loading the weights.

    symtrs inspect runs/depth/midas_dpt_hybrid/1024/weights/best.pt

Prints exactly what `autoloader` would use to rebuild the model: the registry key, the
constructor kwargs, the depth protocol, the alignment convention and the resolved run
config. If a checkpoint turns out not to be reproducible, this is where it shows.
"""

from __future__ import annotations

import argparse
import json

from symtrs.common.checkpoint import read_checkpoint_metadata, resolve_checkpoint_path


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("inspect", help="print a checkpoint's metadata block")
    parser.add_argument("checkpoint", help="a .pt file or a run directory")
    parser.add_argument("--state", action="store_true",
                        help="also summarise the tensors in the state dict")
    parser.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    path = resolve_checkpoint_path(args.checkpoint)
    if path is None:
        raise FileNotFoundError(f"no checkpoint at {args.checkpoint}")
    metadata = read_checkpoint_metadata(path)

    if args.as_json:
        print(json.dumps(metadata, indent=2, default=str))
        return 0

    print(f"[symtrs] {path}")
    headline = [
        ("format_version", "the metadata schema"),
        ("task", "which task this model does"),
        ("model_key", "the registry key, which is enough to rebuild the architecture"),
        ("resolution", "the pixel size it was trained and evaluated at"),
        ("protocol", "depth only: metric (metres) or relative (affine-invariant)"),
        ("default_align", "depth only: the alignment convention this model is compared under"),
        ("align", "depth only: the alignment actually used"),
        ("num_classes", "segmentation only: the size of the prediction head"),
        ("label_step", "segmentation only: grey-level step of the masks (class = round(value / step))"),
        ("ignore_index", "segmentation only: the label excluded from the loss and metrics"),
        ("seed", "the seed the run started from"),
        ("tag", "best or last"),
    ]
    for key, note in headline:
        if key in metadata:
            print(f"  {key:<16} {metadata[key]}   ({note})")

    if metadata.get("metrics"):
        print("  metrics")
        for key, value in metadata["metrics"].items():
            print(f"    {key:<24} {value}")

    kwargs = metadata.get("model_kwargs") or {}
    print(f"  model_kwargs     {json.dumps(kwargs, default=str)}")

    config = metadata.get("config") or {}
    data = config.get("data", {})
    print("  config")
    print(f"    data.root        {data.get('root', '-')}")
    print(f"    train.epochs     {config.get('train', {}).get('epochs', '-')}")
    print(f"    eval.tile_size   {config.get('eval', {}).get('tile_size', '-')}")

    if args.state:
        from symtrs.common.checkpoint import load_checkpoint

        payload = load_checkpoint(path)
        state = payload.get("model") or {}
        total = sum(t.numel() for t in state.values() if hasattr(t, "numel"))
        print(f"  state_dict       {len(state)} tensors, {total / 1e6:.1f}M parameters")
        print(f"  has optimizer    {payload.get('optimizer') is not None}")
        print(f"  epoch            {payload.get('epoch')}")
    return 0
