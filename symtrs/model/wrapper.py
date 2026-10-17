"""The SDK surface: `autoloader`, `SymTRSModel`, and `build_from_config`.

    from symtrs.model import autoloader

    model = autoloader("runs/depth/midas_dpt_hybrid/1024/weights/best.pt")
    depth = model.predict("scene.png")          # metres, HxW float array
    model.train(epochs=5, data={"root": "/data/SyMTRS"})

`SymTRSModel` deliberately does **not** subclass `nn.Module`. PyTorch's `nn.Module.train`
takes a boolean that switches the module into training mode; a `SymTRSModel.train(cfg)`
that instead starts a training run would be a genuine trap for anyone who has used
PyTorch before. The network stays reachable as `model.net`, and mode switching is
`model.net.train()` / `model.net.eval()` exactly as usual.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from symtrs.common.checkpoint import (
    CheckpointError,
    is_symtrs_checkpoint,
    load_checkpoint,
    read_checkpoint_metadata,
    resolve_checkpoint_path,
    save_checkpoint,
)
from symtrs.common.seed import resolve_device
from symtrs.config import RunConfig
from symtrs.registry import build_model

#: Short names accepted by `SymTRSModel.train` / `.evaluate`, mapped to config paths.
SHORTHAND = {
    "epochs": "train.epochs",
    "lr": "train.lr",
    "batch_size": "train.batch_size",
    "freeze": "train.freeze",
    "loss": "train.loss",
    "optim": "train.optim",
    "data_root": "data.root",
    "task": "task",
    "model": "model",
    "resolution": "resolution",
    "seed": "seed",
    "out_dir": "out_dir",
}


class SymTRSModel:
    """A built model plus everything needed to run it.

    Holds:
        net          the `TaskModel` itself
        config       the `RunConfig` this model was built or trained under, if known
        metadata     the checkpoint's `symtrs` block, if it was loaded from one
        source       where it came from, for the repr
    """

    def __init__(
        self,
        net,
        config: Optional[RunConfig] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        source: str = "",
    ) -> None:
        self.net = net
        self.config = config
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.source = source

    # ------------------------------------------------------------------ properties

    @property
    def task(self) -> str:
        return getattr(self.net, "task", "") or self.metadata.get("task", "")

    @property
    def key(self) -> str:
        return getattr(self.net, "key", "") or self.metadata.get("model_key", "")

    @property
    def resolution(self) -> int:
        if self.config is not None:
            return self.config.resolution
        return int(self.metadata.get("resolution", 0))

    @property
    def protocol(self) -> str:
        return getattr(self.net, "protocol", "") or self.metadata.get("protocol", "")

    @property
    def align(self) -> str:
        return (
            getattr(self.net, "align", "")
            or self.metadata.get("align", "")
            or self.metadata.get("default_align", "")
        )

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    @property
    def num_classes(self) -> int:
        if self.config is not None:
            return self.config.data.num_classes
        return int(self.metadata.get("num_classes", 0))

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """The arguments needed to rebuild this architecture, without the weights."""
        kwargs = dict(self.metadata.get("model_kwargs", {}))
        kwargs.update(getattr(self.net, "build_kwargs", {}) or {})
        return kwargs

    def to(self, device: Union[str, torch.device] = "cuda") -> "SymTRSModel":
        self.net.to(torch.device(device) if isinstance(device, str) else device)
        return self

    def describe(self) -> str:
        return self.net.describe()

    # --------------------------------------------------------------------- infer

    def predict(
        self,
        image,
        as_numpy: bool = True,
        tile_size: int = 0,
        tile_overlap: int = 128,
        **kwargs,
    ):
        """Run the model on one image.

        Args:
            image: a path, a `PIL.Image`, a numpy array, or a `[B, 3, H, W]` tensor in
                `[0, 1]`.
            tile_size: sliding-window inference for images too large for one forward
                pass; 0 runs the whole image at once.
            tile_overlap: how much neighbouring windows share when tiling.

        Returns:
            Depth: `HxW` float array of metres.
            Segmentation: `HxW` int64 array of class indices.

        Single images come back without a batch dimension; a batched input keeps it.

        Tiling is handled here rather than inside each adapter so that both tasks get it,
        and so that the stitching happens in the model's own output space - see
        `symtrs.engine.tiling` for why that matters.
        """
        tensor, batched = to_tensor(image)
        tensor = tensor.to(self.device)

        from symtrs.engine.loop import forward_native

        was_training = self.net.training
        self.net.eval()
        try:
            native = forward_native(self.net, tensor, tile_size, tile_overlap)
            if self.task == "depth":
                prediction = self.net.to_metres(native)
            else:
                prediction = native.argmax(dim=1)
        finally:
            if was_training:
                self.net.train()

        if not batched and prediction.shape[0] == 1:
            prediction = prediction[0]

        if not as_numpy:
            return prediction
        prediction = prediction.detach().cpu()
        # Depth is a continuous field, so it comes back as float; class indices must
        # stay integral or a downstream `==` comparison against a label map fails.
        return prediction.float().numpy() if self.task == "depth" else prediction.numpy()

    def predict_batch(self, images, **kwargs) -> np.ndarray:
        """Run on a list of images and stack the results."""
        arrays = [self.predict(image, **kwargs) for image in images]
        return np.stack(arrays, axis=0)

    # -------------------------------------------------------------------- running

    def _resolve_config(self, cfg: Optional[Any] = None, overrides: Optional[Mapping[str, Any]] = None) -> RunConfig:
        if cfg is None:
            if "config" in self.metadata:
                base = RunConfig.from_dict(self.metadata["config"])
            else:
                base = RunConfig(task=self.task, model=self.key)
        elif isinstance(cfg, RunConfig):
            base = cfg
        elif isinstance(cfg, argparse.Namespace):
            base = RunConfig(**vars(cfg))
        elif isinstance(cfg, Mapping):
            base = RunConfig.from_dict(cfg)
        elif isinstance(cfg, str):
            base = RunConfig.from_yaml(cfg)
        else:
            raise TypeError(
                f"cfg must be a RunConfig, a dict, an argparse.Namespace or a YAML path; "
                f"got {type(cfg).__name__}"
            )

        merged: Dict[str, Any] = {}
        for dotted, value in dict(overrides or {}).items():
            merged[SHORTHAND.get(dotted, dotted)] = value
        return base.with_overrides(merged).validate() if merged else base

    def train(self, cfg: Optional[Any] = None, **overrides: Any) -> Dict[str, Any]:
        """Finetune on the training split, then evaluate on the test split.

        `cfg` may be a `RunConfig`, a YAML path, a dict, an `argparse.Namespace`, or
        omitted entirely, in which case the config recorded in this checkpoint is used.
        Bare keywords such as `epochs=5` and dotted paths such as `data.root="..."` are
        both accepted.
        """
        from symtrs.engine.trainer import train as run_training

        config = self._resolve_config(cfg, overrides)
        return run_training(config, model=self.net, mode="finetune")

    def evaluate(self, cfg: Optional[Any] = None, split: str = "", **overrides: Any) -> Dict[str, Any]:
        """Evaluate the current weights. `split` defaults to the config's test split."""
        from symtrs.engine.trainer import evaluate_run

        config = self._resolve_config(cfg, overrides)
        return evaluate_run(config, model=self.net, split=split or None)

    # ------------------------------------------------------------------ checkpoint

    def save(self, path: str, metrics: Optional[Mapping[str, float]] = None) -> str:
        """Write a self-describing checkpoint that `autoloader` can read back."""
        from symtrs.engine.trainer import checkpoint_metadata

        if self.config is not None:
            metadata = checkpoint_metadata(self.net, self.config, {"metrics": dict(metrics or {})})
        else:
            metadata = dict(self.metadata)
            metadata.setdefault("task", self.task)
            metadata.setdefault("model_key", self.key)
            metadata.setdefault("model_kwargs", self.model_kwargs)
        save_checkpoint(str(path), {"model": self.net.state_dict(), "metrics": dict(metrics or {})}, metadata)
        return str(path)

    def __repr__(self) -> str:
        origin = f" from {self.source}" if self.source else ""
        return f"SymTRSModel({self.net.describe()}{origin})"


def _fill_camera_kwargs(cfg: RunConfig) -> None:
    """Give a metric model that needs the focal length the one the data implies.

    Depth Anything 3 metric returns *canonical* depth (the depth a camera of focal length 300
    px would see) and `metres = canonical * focal_px / 300`. Left unset it was scoring that
    canonical depth as metres, ~25x too small here (abs_rel 0.91-0.96, d1 = 0). The focal
    length is a fact about the data level, `(resolution / 2) / tan(hfov / 2)`, so it is derived
    from `cfg.resolution` and `data.camera_hfov_deg` and follows the resolution axis without a
    per-file number that could be left behind (a 1024 run with the 2048 value would be 2x off).
    Cropping and tiling do not change it: a crop keeps the pixel size.

    The value is written back into `cfg.model_kwargs` so that it lands in the checkpoint
    metadata and in `config.yaml`, which is what `autoloader` rebuilds the model from; an
    explicit number in `model_kwargs` always wins.
    """
    from symtrs.config import focal_px_for_resolution
    from symtrs.registry import get_spec

    focal_kwarg = get_spec(cfg.model).focal_kwarg
    if not focal_kwarg or cfg.model_kwargs.get(focal_kwarg) is not None:
        return
    cfg.model_kwargs[focal_kwarg] = focal_px_for_resolution(
        cfg.resolution, cfg.data.camera_hfov_deg
    )


def build_from_config(cfg: RunConfig):
    """Build the model a run config asks for, with the config's loss, align and freeze.

    The registry supplies the architecture; the config supplies the run-time policy.
    Keeping that split in one function is what stops the CLI, the YAML loader and the
    SDK from each assembling a model slightly differently.
    """
    overrides: Dict[str, Any] = dict(cfg.model_kwargs)

    if cfg.task == "depth":
        _fill_camera_kwargs(cfg)
        overrides = dict(cfg.model_kwargs)
        overrides.setdefault("loss", cfg.train.loss)
        overrides.setdefault("align", cfg.eval.align)
        overrides.setdefault("freeze", cfg.train.freeze)
        if cfg.train.loss_kwargs:
            overrides.setdefault("loss_kwargs", dict(cfg.train.loss_kwargs))
    elif cfg.task == "segmentation":
        overrides.setdefault("num_classes", cfg.data.num_classes)
        overrides.setdefault("ignore_index", cfg.data.ignore_index)
        overrides.setdefault("loss", cfg.train.loss)
        if cfg.train.loss_kwargs:
            overrides.setdefault("loss_kwargs", dict(cfg.train.loss_kwargs))
        overrides.setdefault("freeze", cfg.train.freeze)

    return build_model(cfg.model, **overrides)


def autoloader(
    source: str,
    device: Optional[str] = None,
    **overrides: Any,
) -> SymTRSModel:
    """Load a model from a checkpoint path, a run directory, or a registry key.

    * A `.pt` written by symtrs, or a run directory containing one -> the architecture
      is rebuilt from the checkpoint's metadata and the weights are loaded.
    * A registry key such as `"midas_dpt_hybrid"` -> the pretrained model, untrained on
      SyMTRS. That is the zero-shot half of the run matrix, and it is what
      `symtrs eval --model <key>` uses.

    `**overrides` are forwarded to the model constructor, which is how a checkpoint can
    be rebuilt with a different `freeze` policy or a different number of classes.
    """
    if not source:
        raise ValueError("autoloader needs a checkpoint path, a run directory, or a model key")

    path = resolve_checkpoint_path(source)

    if path is not None and is_symtrs_checkpoint(path):
        metadata = read_checkpoint_metadata(path, map_location="cpu")
        config = None
        if isinstance(metadata.get("config"), Mapping):
            try:
                config = RunConfig.from_dict(metadata["config"])
            except Exception as exc:
                print(f"[symtrs] checkpoint config could not be parsed ({exc}); using metadata only")

        kwargs: Dict[str, Any] = dict(metadata.get("model_kwargs", {}))
        # Segmentation needs its class count to rebuild the head; a checkpoint that
        # forgot it would otherwise silently rebuild a 21-class VOC head.
        for key in ("num_classes", "ignore_index"):
            if key in metadata:
                kwargs.setdefault(key, metadata[key])
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

        net = build_model(metadata["model_key"], **kwargs)
        if device:
            net.to(resolve_device(device))
        payload = load_checkpoint(path, map_location="cpu")
        state = payload.get("model")
        if state is None:
            raise CheckpointError(f"checkpoint {path} has no 'model' state dict")
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[symtrs] warning: {len(missing)} missing and {len(unexpected)} unexpected "
                f"parameters loading {path}"
            )
            if missing:
                print(f"[symtrs]   missing: {sorted(missing)[:8]}")
            if unexpected:
                print(f"[symtrs]   unexpected: {sorted(unexpected)[:8]}")
        net.eval()
        return SymTRSModel(net, config=config, metadata=metadata, source=path)

    looks_like_a_path = str(source).endswith((".pt", ".pth", ".ckpt")) or "/" in str(source) or "\\" in str(source)
    if looks_like_a_path:
        raise CheckpointError(
            f"{source!r} is not a readable symtrs checkpoint"
            + (f" (found {path}, but it has no 'symtrs' metadata block)" if path else " (no such file)")
        )

    net = build_model(source, **{k: v for k, v in overrides.items() if v is not None})
    if device:
        net.to(resolve_device(device))
    print(f"[symtrs] loaded {source!r} from the registry; these are the released weights, "
          "not a SyMTRS-finetuned checkpoint")
    return SymTRSModel(net, metadata={"model_key": net.key, "task": net.task}, source=source)


# --------------------------------------------------------------------- input glue


def to_tensor(image, mode: str = "rgb") -> tuple:
    """Anything image-shaped to `([1, 3, H, W]` in `[0, 1]`, was-it-batched)`.

    Accepts a path, a `PIL.Image`, a numpy array in either layout, or a tensor. The
    point is that `model.predict(...)` should not care which of these the caller has,
    because in practice a dataset sample, a `cv2.imread` result and a file path all
    show up in the same script.
    """
    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            # Ambiguous between [C,H,W] and [H,W,C]; channels-first wins because that is
            # what every other function in this package produces.
            if tensor.shape[0] in (1, 3):
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0), False

    if isinstance(image, (str, Path)):
        from symtrs.data.io import read_image

        return read_image(image, mode=mode).unsqueeze(0), False

    if isinstance(image, np.ndarray):
        array = image
    else:
        # Duck-type PIL rather than importing it: this is the one place PIL might be
        # absent, and a clear TypeError beats an ImportError from three frames down.
        if hasattr(image, "convert") and hasattr(image, "size"):
            array = np.array(image.convert("RGB" if mode == "rgb" else "L"))
        else:
            raise TypeError(
                f"cannot read an image from {type(image).__name__}; pass a path, a PIL "
                "image, a numpy array or a torch tensor"
            )

    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] not in (1, 3):
        raise ValueError(
            f"cannot tell which axis is the channel axis in shape {tuple(tensor.shape)}"
        )
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).unsqueeze(0), False


__all__ = ["SHORTHAND", "SymTRSModel", "autoloader", "build_from_config", "to_tensor"]
