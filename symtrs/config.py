"""Run configuration: the single source of truth for every SyMTRS run.

Three front-ends build the same `RunConfig`:

  * a YAML file      -> ``RunConfig.from_yaml(path)``
  * command line     -> ``RunConfig.from_overrides(cfg, {...})`` on top of a YAML base
  * python kwargs    -> ``RunConfig(task=..., model=..., ...)`` then ``with_overrides``

Nothing else in the package defines configuration defaults. Keeping one schema is
what stops the YAML files and the CLI flags from drifting apart.

Override keys may be dotted paths into the nested dataclasses, e.g.
``{"train.epochs": 30, "data.root": "/data/SyMTRS"}``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from symtrs import CONFIG_FORMAT_VERSION
# The reader owns the list of backend names; validating against a second copy here would
# let the two drift, and the symptom would be a backend that works but the config refuses.
from symtrs.data.exr import VALID_BACKENDS

TASKS: tuple = ("superresolution", "depth", "segmentation", "domain_adapt", "shadow_removal")

#: Resolutions are named by their pixel dimension, not by a scale factor.
RESOLUTIONS: tuple = (2048, 1024, 512, 256)

#: Evaluation modes. "zero_shot" evaluates released pretrained weights with no
#: training on SyMTRS; "finetune" trains on train/ first. Both share one eval path.
MODES: tuple = ("zero_shot", "finetune")


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent or unknown."""


def focal_px_for_resolution(resolution: int, hfov_deg: float) -> float:
    """Focal length in pixels of a square image `resolution` pixels wide.

    `f = (W / 2) / tan(hfov / 2)`. `resolution` names the pixel size of the data (ground rule
    1 in AGENTS.md), so the focal length follows it by proportion: 1773.6 px at 2048 and 60
    degrees, half of that at 1024, and so on. It is a property of the data level, not of the
    network input, which is why a 512px training crop or a tile still uses the same value.
    """
    import math

    return (resolution / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


@dataclass
class DataConfig:
    """Where a run reads its data from."""

    root: str = ""
    split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # Depth reading.
    # File units -> metres. The real EXRs store CENTIMETRES, so 0.01 is the default; a
    # metric model scored against unconverted centimetres is wrong by a constant 100x.
    depth_scale: float = 0.01
    # Metres, after depth_scale. Pixels farther than this are invalid. The real depth marks
    # sky with huge values rather than zero, which would dominate every loss and metric if
    # it counted as geometry. None keeps every positive finite pixel.
    max_depth: Optional[float] = None
    depth_channel: int = 0
    # Horizontal field of view of the capture camera, in degrees, for the metric models whose
    # output needs the focal length in pixels (Depth Anything 3 metric: metres = canonical
    # depth * focal_px / 300). ASSUMPTION, not a measurement: 60 is the default of the capture
    # actor (`FieldOfView = 60.f`, SyMTRSDatasetCaptureActor.h) and it is editable per level, so
    # confirm it against the level that rendered the dataset. The images are square, so the
    # vertical FOV is the same. `focal_px_for_resolution` turns it into pixels at the data's
    # resolution; cropping and tiling do not change it, because a crop keeps the pixel size.
    camera_hfov_deg: float = 60.0
    # Which library reads the EXR files. "auto" tries them in order; "numpy" reads .npy
    # only and is rejected on an EXR rather than silently returning something else. The
    # list of names lives in the reader, not here, so the two cannot disagree about what
    # is supported.
    depth_backend: str = "auto"

    # Segmentation.
    # The real masks hold class_id * label_step as 8-bit gray (current export: 63, so
    # 0/63/126/189/252 -> background, building, green, road, car; older ones used 51 with 6
    # classes). Pixels not within a couple of grey levels of a multiple become ignore_index.
    # make-dummy writes the same encoding, so a config that runs on it runs on the real data.
    num_classes: int = 5
    label_step: int = 63
    ignore_index: int = 255

    # Loading.
    num_workers: int = 4
    pin_memory: bool = True
    max_samples: int = 0  # 0 = all
    limit_train_batches: int = 0  # 0 = all; useful for smoke runs
    limit_val_batches: int = 0

    def validate(self) -> None:
        if not self.root:
            raise ConfigError("data.root is required (path to the SyMTRS dataset root)")
        if self.split == self.val_split or self.split == self.test_split:
            raise ConfigError(
                f"data.split ({self.split!r}) must differ from the evaluation splits "
                f"({self.val_split!r}, {self.test_split!r})"
            )
        if self.depth_backend not in VALID_BACKENDS:
            raise ConfigError(
                f"unknown data.depth_backend: {self.depth_backend!r}; expected one of "
                f"{VALID_BACKENDS}"
            )
        if self.depth_scale <= 0:
            raise ConfigError("data.depth_scale must be positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ConfigError("data.max_depth must be positive (metres) or unset")
        if not 0.0 < self.camera_hfov_deg < 180.0:
            raise ConfigError(
                f"data.camera_hfov_deg must be in (0, 180) degrees; got {self.camera_hfov_deg}"
            )
        if self.num_classes < 2:
            raise ConfigError("data.num_classes must be at least 2")
        if self.label_step < 1:
            raise ConfigError(
                "data.label_step must be at least 1 (1 = masks hold raw class ids)"
            )


@dataclass
class TrainConfig:
    """How a run trains. Ignored entirely for zero-shot evaluation."""

    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    optim: str = "adamw"  # adamw | adam | sgd
    sched: str = "cosine"  # cosine | poly | none
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.01

    # Random-crop patch training. 0 trains on the full image, which only fits at
    # the lower resolutions.
    patch_size: int = 0
    # Evaluation geometry, val AND test alike. 0 = the whole image (never `patch_size`: the
    # training crop is a memory device, not part of the protocol). A positive value is a
    # deterministic CENTRE crop, only for speed; images that do not fit in one pass are
    # scored with `eval.tile_size` instead, which still covers every pixel.
    val_patch_size: int = 0

    amp: bool = True
    gpus: int = 1
    grad_clip: float = 0.0
    grad_accum: int = 1

    # "" -> the model's registry default for its protocol.
    loss: str = ""
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    # "" | "encoder" | "head" | "last_n:<k>"
    freeze: str = ""

    log_every: int = 50
    save_samples_every: int = 1
    resume: str = ""
    device: str = "cuda"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigError("train.epochs must be at least 1")
        if self.batch_size < 1:
            raise ConfigError("train.batch_size must be at least 1")
        if self.lr <= 0:
            raise ConfigError("train.lr must be positive")
        if self.optim not in {"adamw", "adam", "sgd"}:
            raise ConfigError(f"unknown train.optim: {self.optim!r}")
        if self.sched not in {"cosine", "poly", "none"}:
            raise ConfigError(f"unknown train.sched: {self.sched!r}")
        if self.gpus < 1:
            raise ConfigError("train.gpus must be at least 1")
        if self.grad_accum < 1:
            raise ConfigError("train.grad_accum must be at least 1")
        if self.freeze and not (
            self.freeze in {"encoder", "head"} or self.freeze.startswith("last_n:")
        ):
            raise ConfigError(
                "train.freeze must be '', 'encoder', 'head' or 'last_n:<k>'; "
                f"got {self.freeze!r}"
            )


@dataclass
class EvalConfig:
    """How a run is evaluated. Used by zero-shot eval and at the end of training."""

    # Does not filter what is computed - the task's scorer always reports everything it
    # can. It selects which metric decides "best": the first name here that the task
    # knows how to minimise or maximise wins, and if none match, the task default in
    # `symtrs.engine.trainer.MONITORS` applies. This matters more than it sounds, since
    # the checkpoint kept for the paper is whichever one won here.
    metrics: List[str] = field(default_factory=list)  # [] -> task defaults
    batch_size: int = 1
    amp: bool = True
    # Sliding-window inference for images that do not fit at full resolution.
    tile_size: int = 0  # 0 = whole image in one forward pass
    tile_overlap: int = 128
    # "" -> the model's own published alignment convention.
    align: str = ""
    num_visualizations: int = 4

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ConfigError("eval.batch_size must be at least 1")
        if self.tile_size < 0:
            raise ConfigError("eval.tile_size must be >= 0")
        if self.tile_size and self.tile_overlap >= self.tile_size:
            raise ConfigError("eval.tile_overlap must be smaller than eval.tile_size")


@dataclass
class ShadowConfig:
    """Settings that exist only for `shadow_removal`; ignored by every other task."""

    # The dataset has no shadow masks, and ShadowFormer / ShadowMaskFormer need one as an input while the
    # shadow / non-shadow metric split needs one for every model. The mask is derived once,
    # at staging, from the pair itself: a pixel is shadow when the shadow-free image is
    # brighter than the shadowed one by more than this fraction of its own luminance,
    # (Y_noshadow - Y_day) / max(Y_noshadow, eps) > mask_threshold. 0.1 is above 8-bit
    # rounding and render noise (the two renders differ only by the shadows) while keeping
    # penumbra; changing it changes the regional metrics, so it is recorded in the summary.
    mask_threshold: float = 0.1
    # Side of the square structuring element of the opening that removes isolated pixels
    # from the mask. 0 disables it.
    mask_open: int = 3
    # Training length. 0 -> `train.epochs`, where one epoch is one random crop from every
    # training image (ceil(n_train / batch_size) iterations). A positive value wins, which
    # is how a ~500 iteration validation run is asked for: `--set shadow.iterations=500`.
    iterations: int = 0
    # Select "best" on a fixed set of random validation crops every `val_every` iterations
    # (full 2048 images are only run once, at the end). 0 validates only at the end.
    val_every: int = 250
    val_crops: int = 32
    # Optional upstream checkpoint to initialise from. "" trains from scratch. Only tensors
    # whose name and shape match are loaded, and the count is printed.
    init_weights: str = ""
    # Skip the final full-image prediction / scoring (smoke tests of the training loop).
    skip_final_eval: bool = False

    def validate(self) -> None:
        if not 0.0 < self.mask_threshold < 1.0:
            raise ConfigError("shadow.mask_threshold must be in (0, 1)")
        if self.mask_open < 0 or self.iterations < 0 or self.val_every < 0 or self.val_crops < 0:
            raise ConfigError(
                "shadow.mask_open, iterations, val_every and val_crops must not be negative"
            )



@dataclass
class RunConfig:
    """A complete description of one run."""

    task: str = "depth"
    model: str = ""
    resolution: int = 2048
    seed: int = 42
    out_dir: str = ""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    # Forwarded verbatim to the model builder; recorded in the checkpoint so the
    # architecture can be rebuilt without the YAML file.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Extra arguments forwarded to the vendored upstream script (domain_adapt only).
    upstream_args: List[str] = field(default_factory=list)

    format_version: int = CONFIG_FORMAT_VERSION

    # ---------------------------------------------------------------- validation

    def validate(self) -> "RunConfig":
        if self.task not in TASKS:
            raise ConfigError(f"unknown task {self.task!r}; expected one of {TASKS}")
        if not self.model:
            raise ConfigError("model is required")
        if self.resolution not in RESOLUTIONS:
            raise ConfigError(
                f"resolution must be one of {RESOLUTIONS}; got {self.resolution}"
            )
        if self.format_version != CONFIG_FORMAT_VERSION:
            raise ConfigError(
                f"config format_version {self.format_version} does not match this "
                f"build ({CONFIG_FORMAT_VERSION})"
            )
        self.data.validate()
        self.train.validate()
        self.eval.validate()
        self.shadow.validate()
        return self

    # ------------------------------------------------------------------ defaults

    @property
    def resolved_out_dir(self) -> Path:
        """Where this run writes. Explicit `out_dir` wins, otherwise a canonical path."""
        if self.out_dir:
            return Path(self.out_dir)
        return Path("runs") / self.task / self.model / str(self.resolution)

    def anchored(self) -> "RunConfig":
        """A copy whose caller-relative paths are absolute, resolved against *this* process's cwd.

        Why: shadow removal and domain adaptation launch their subprocess with the upstream
        checkout as cwd (upstream's imports and relative `./checkpoints` need it), and the
        shadow worker re-reads this config from `config.yaml` there. A relative
        `--out-dir runs/validation/x` therefore pointed into the upstream tree (or nowhere)
        and both tasks died at the first path they opened (2026-10-17 validation queue,
        which passed relative out roots). The paths a user types are relative to *their*
        shell, so they are fixed once, here, before any subprocess exists - never inside a
        launcher, where the cwd is already the wrong one. An empty `out_dir` is anchored
        too: its canonical default (`runs/<task>/<model>/<res>`) is relative as well.
        """
        cfg = copy.deepcopy(self)
        cfg.out_dir = str(self.resolved_out_dir.resolve())
        for owner, name in (
            (cfg.data, "root"),
            (cfg.shadow, "init_weights"),
            (cfg.train, "resume"),
        ):
            value = getattr(owner, name)
            if value:
                setattr(owner, name, str(Path(value).resolve()))
        return cfg

    # --------------------------------------------------------------- conversion

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError(f"config must be a mapping, got {type(raw).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(
                f"unknown top-level config keys: {sorted(unknown)}; known keys are "
                f"{sorted(known)}"
            )
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            if key == "data":
                kwargs["data"] = _build_section(DataConfig, value, "data")
            elif key == "train":
                kwargs["train"] = _build_section(TrainConfig, value, "train")
            elif key == "eval":
                kwargs["eval"] = _build_section(EvalConfig, value, "eval")
            elif key == "shadow":
                kwargs["shadow"] = _build_section(ShadowConfig, value, "shadow")
            else:
                kwargs[key] = value
        return cls(**kwargs)

    # ---------------------------------------------------------------------- yaml

    def to_yaml(self, path: Optional[str] = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is None:
            raise ConfigError(f"config file is empty: {path}")
        return cls.from_dict(raw).validate()

    # ----------------------------------------------------------------- overrides

    def with_overrides(self, overrides: Mapping[str, Any]) -> "RunConfig":
        """Return a copy with dotted-path overrides applied.

        Values are coerced to the type of the field they replace, so a CLI string
        ``"30"`` lands in an ``int`` field. ``None`` values are ignored, which lets
        callers pass a whole parsed namespace without filtering it first.
        """
        cfg = copy.deepcopy(self)
        for dotted, value in overrides.items():
            if value is None:
                continue
            _set_dotted(cfg, dotted, value)
        return cfg

    def set_many(self, **overrides: Any) -> "RunConfig":
        return self.with_overrides(overrides)


# --------------------------------------------------------------------- helpers


def _build_section(cls, raw: Any, name: str):
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config section {name!r} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"unknown keys in {name!r}: {sorted(unknown)}; known keys are {sorted(known)}"
        )
    return cls(**dict(raw))


def _set_dotted(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target = obj
    for part in parts[:-1]:
        if isinstance(target, dict):
            # Free-form sections such as model_kwargs and loss_kwargs accept new keys,
            # so a nested path into one is created rather than rejected.
            nested = target.get(part)
            if not isinstance(nested, (dict,)):
                nested = {}
                target[part] = nested
            target = nested
            continue
        if not hasattr(target, part):
            raise ConfigError(f"unknown config key {dotted!r} (no attribute {part!r})")
        target = getattr(target, part)
        if not is_dataclass(target) and not isinstance(target, dict):
            raise ConfigError(f"config key {dotted!r} does not point into a section")

    leaf = parts[-1]
    if isinstance(target, dict):
        target[leaf] = _coerce_unknown(value)
        return
    if not hasattr(target, leaf):
        raise ConfigError(f"unknown config key {dotted!r}")
    current = getattr(target, leaf)
    optional = _optional_hint(target, leaf)
    if optional is not _NOT_OPTIONAL and isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        # `None`/`null` unset an Optional field again, e.g. `--set data.max_depth=none`.
        setattr(target, leaf, None)
        return
    if current is None:
        # Nothing to read a type from: `data.max_depth` is `Optional[float] = None`, so
        # `--set data.max_depth=1000` used to be stored as the string "1000" and only failed
        # later, in `validate()`, as a `TypeError` comparing a str with 0. The dataclass
        # annotation is the type.
        if optional in (bool, int, float, str):
            # Reuse the typed branches of `_coerce` by handing it a prototype of that type.
            setattr(target, leaf, _coerce(optional(), value, dotted))
        else:
            setattr(target, leaf, _coerce_unknown(value))
        return
    setattr(target, leaf, _coerce(current, value, dotted))


_NOT_OPTIONAL = object()


def _optional_hint(target: Any, leaf: str) -> Any:
    """The declared type inside `Optional[X]` for a dataclass field, or `_NOT_OPTIONAL`.

    A field annotated as a plain type returns `_NOT_OPTIONAL`; `Optional[X]` returns `X`
    (or `None` when it is not a single plain member). Annotations are strings here
    (`from __future__ import annotations`), hence `get_type_hints`.
    """
    import typing

    try:
        hint = typing.get_type_hints(type(target)).get(leaf)
    except Exception:  # noqa: BLE001 - an unresolvable annotation just means "not Optional"
        return _NOT_OPTIONAL
    if typing.get_origin(hint) is not typing.Union or type(None) not in typing.get_args(hint):
        return _NOT_OPTIONAL
    members = [arg for arg in typing.get_args(hint) if arg is not type(None)]
    return members[0] if len(members) == 1 else None


def _coerce_unknown(value: Any) -> Any:
    """Guess a type for a value going into a free-form section.

    There is no schema to read a type from inside `model_kwargs`, so `--set
    model_kwargs.metric=true` has to become a bool and `--set loss_kwargs.threshold=0.2`
    a float. The guess is conservative: anything that is not a boolean or a number
    stays a string, so a model id or a path is never mangled.
    """
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _coerce(current: Any, value: Any, dotted: str) -> Any:
    """Coerce an override value to the type of the field being replaced."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"1", "true", "yes", "on"}:
                return True
            if low in {"0", "false", "no", "off"}:
                return False
            raise ConfigError(f"cannot read {value!r} as a boolean for {dotted!r}")
        return bool(value)
    if isinstance(current, list):
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ConfigError(f"cannot read {value!r} as a list for {dotted!r}")
    if isinstance(current, dict):
        if isinstance(value, Mapping):
            return dict(value)
        raise ConfigError(f"cannot read {value!r} as a mapping for {dotted!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"cannot read {value!r} as an integer for {dotted!r}") from None
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"cannot read {value!r} as a number for {dotted!r}") from None
    if isinstance(current, str):
        return str(value)
    if is_dataclass(current) and isinstance(value, Mapping):
        # `train={...}` - the shape the YAML file itself has, and the first thing an SDK user
        # reaches for, since the config they just read is nested. Assigning it would put a
        # plain dict where a section object belongs; nothing complains until `validate()` is
        # called on it several frames later, and the error there is
        # `AttributeError: 'dict' object has no attribute 'validate'`, which names neither the
        # key nor the fix. Naming the dotted form is cheaper than making the reader work it
        # out.
        flattened = ", ".join(f"{dotted}.{key}" for key in value)
        raise ConfigError(
            f"{dotted!r} is a config section, so a nested mapping cannot replace it. "
            f"Set the fields by dotted path instead: {flattened}"
        )
    return value


def load_config(
    yaml_path: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> RunConfig:
    """Build a validated config from a YAML base, then overrides, then kwargs."""
    cfg = RunConfig.from_yaml(yaml_path) if yaml_path else RunConfig()
    if overrides:
        cfg = cfg.with_overrides(overrides)
    if kwargs:
        cfg = cfg.with_overrides(kwargs)
    return cfg.validate()
