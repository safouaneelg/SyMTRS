"""The engine's one job that is easy to get wrong invisibly: output shape.

Every supervised task here compares a prediction against a ground truth that came from the
dataset, and the two only line up if the prediction is brought back to the size of the *raw*
image. That is not the size the network was handed: `prepare_input` may resize (MiDaS's hub
transform resizes everything it is given to 256 or 384) and may pad (the sides are rounded up
to a multiple of 32). Both paths existed and both were wrong, and neither was caught by a test
- they were caught by running `symtrs eval` against a small dummy dataset, where the sizes
happen not to coincide.

`torch` is needed, so this module skips on a machine without it. Nothing here downloads
weights or reads a dataset.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

import torch.nn.functional as F  # noqa: E402

from symtrs.engine.loop import forward_native  # noqa: E402


class _ResizingAdapter(torch.nn.Module):
    """Stands in for MiDaS: preprocessing changes the spatial size.

    Its hub transform resizes whatever it is handed to the checkpoint's input size, so the
    tensor the network sees - and therefore the size it naturally returns - has nothing to do
    with the ground truth.

    `forward_at_size` is not decoration. Every real depth adapter inherits it from
    `DepthAdapter`, and the engine takes a different branch when it is present - the branch
    that was broken. A stand-in without it exercises the *working* branch and would pass with
    or without the fix, which is a test that guards nothing.
    """

    size_multiple = 1
    protocol = "relative"
    default_align = "scale"

    def __init__(self, target: int = 64) -> None:
        super().__init__()
        self.target = target
        # A parameter, so `nn.Module` is genuinely a module and `.eval()`/`.to()` behave.
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image, size=(self.target, self.target), mode="bilinear", align_corners=False
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1]

    def forward_at_size(self, image: torch.Tensor) -> torch.Tensor:
        """The real contract, copied from `DepthAdapter`: resize to the tensor handed in.

        That tensor is the preprocessed one, which is the whole trap - the adapter is being
        faithful to its documented contract and still returns something the ground truth
        cannot be compared against.
        """
        return F.interpolate(
            self.forward(image).float(),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class _PaddingAdapter(torch.nn.Module):
    """Stands in for the base adapter: preprocessing pads the sides up to a multiple.

    `DepthAdapter._pad_to_multiple` is what the engine looks for, so it is implemented here
    with the same name and signature rather than inherited - importing a real adapter would
    pull in its weights.
    """

    size_multiple = 32
    protocol = "metric"
    default_align = "none"

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def _pad_to_multiple(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        if pad_h == 0 and pad_w == 0:
            return image
        return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1]

    def forward_at_size(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            self.forward(image).float(),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


# ------------------------------------------- the prediction matches the ground truth


def test_a_resizing_adapter_still_returns_the_raw_image_size():
    """The size the ground truth is at, not the size the network saw.

    This returned `(64, 64)` - the transform's size - and every MiDaS training and evaluation
    run then died in `compute_depth_metrics` with "prediction shape (1, 1, 384, 384) does not
    match target (1, 1, 2048, 2048)". It was invisible at 2048 in principle and invisible in
    every unit test in practice, because nothing else in the suite calls `forward_native`.
    """
    model = _ResizingAdapter(target=64).eval()
    image = torch.rand(2, 3, 40, 56)

    with torch.no_grad():
        prediction = forward_native(model, image)

    assert prediction.shape[-2:] == (40, 56)


def test_a_padding_adapter_still_returns_the_unpadded_size():
    """The same failure by the other route: 40x56 pads to 64x64, and the prediction came back
    at 64x64 against a 40x56 ground truth."""
    model = _PaddingAdapter().eval()
    image = torch.rand(1, 3, 40, 56)

    with torch.no_grad():
        prediction = forward_native(model, image)

    assert prediction.shape[-2:] == (40, 56)


def test_an_image_already_the_right_size_is_left_alone():
    """The resize is a no-op when it should be - no interpolation softens a correct output."""
    model = _PaddingAdapter().eval()
    image = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        prediction = forward_native(model, image)

    assert prediction.shape[-2:] == (64, 64)


def test_a_resizing_adapter_can_actually_be_scored():
    """The composition the two live paths use, end to end: the shape `forward_native` returns
    is the shape `compute_depth_metrics` insists on. This is the assertion that would have
    failed loudly on the real dataset instead of several frames deep in the metrics."""
    from symtrs.metrics.depth import DEPTH_METRIC_NAMES, compute_depth_metrics

    model = _ResizingAdapter(target=64).eval()
    image = torch.rand(1, 3, 40, 56)
    ground_truth = torch.rand(1, 1, 40, 56) + 0.1
    valid = torch.ones_like(ground_truth, dtype=torch.bool)

    with torch.no_grad():
        prediction = forward_native(model, image)

    metrics = compute_depth_metrics(prediction, ground_truth, valid, align="none")
    assert set(DEPTH_METRIC_NAMES) <= set(metrics)


@pytest.mark.parametrize("tile_size", [0, 32])
def test_the_tiled_and_untiled_paths_agree_on_the_output_size(tile_size):
    """Tiling already resized to the raw size, which is why the bug showed up only on the
    untiled path - and why the two must be compared, not tested in isolation."""
    model = _ResizingAdapter(target=64).eval()
    image = torch.rand(1, 3, 40, 56)

    with torch.no_grad():
        prediction = forward_native(model, image, tile_size=tile_size, tile_overlap=8)

    assert prediction.shape[-2:] == (40, 56)


# ------------------------------------------------------- previews accept what they get


def test_the_depth_preview_accepts_the_ranks_the_eval_loop_holds():
    """The evaluation loop hands over `image[index]` as `[3, H, W]` and the rest as
    `[1, H, W]`. Collapsing only some of those left `tgt` two-dimensional and `mask`
    three-dimensional, so `tgt[mask]` raised
    `IndexError: too many indices for tensor of dimension 2` and every preview panel was lost
    - while the failure was reported as a missing matplotlib, which was installed.
    """
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")
    import tempfile
    from pathlib import Path

    from symtrs.engine.loop import _save_depth_visual

    image = torch.rand(1, 3, 32, 32)
    metres = torch.rand(1, 1, 32, 32)
    ground_truth = torch.rand(1, 1, 32, 32) + 0.1
    valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "panel.png"
        _save_depth_visual(image[0], metres[0], ground_truth[0], valid[0], str(path))

        assert path.is_file()
        assert path.stat().st_size > 0


# ----------------------------------------- a batch of one is explained, not leaked


#: Verbatim from torch, on torchvision's DeepLabV3: its ASPP pools globally and then runs a
#: BatchNorm, so that BatchNorm is always handed a 1x1 map and has only the batch axis left.
_TORCH_BATCHNORM_MESSAGE = (
    "Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])"
)


class _AnyModel(torch.nn.Module):
    pass


def test_a_single_image_batch_is_explained_in_terms_of_the_batch_size():
    """torch's own message names neither the batch size the reader set nor the model, and the
    only fix is a flag they control. Measured: DeepLabV3 builds and runs at batch 2 and 4, and
    fails at 1."""
    from symtrs.config import ConfigError
    from symtrs.engine.loop import _as_batch_size_error

    explained = _as_batch_size_error(ValueError(_TORCH_BATCHNORM_MESSAGE), _AnyModel(), 1)

    assert isinstance(explained, ConfigError), (
        "the CLI reports ConfigError as a one-line message; anything else is announced as an "
        "unexpected internal bug and sent off to a bug report"
    )
    assert "--batch-size" in str(explained)


def test_the_batch_size_advice_is_not_given_for_a_batch_that_is_already_fine():
    """The guard is on the batch size as well as the message, so the same torch error with a
    larger batch - which would mean something else entirely - is left alone."""
    from symtrs.engine.loop import _as_batch_size_error

    original = ValueError(_TORCH_BATCHNORM_MESSAGE)
    assert _as_batch_size_error(original, _AnyModel(), 2) is original


def test_an_unrelated_value_error_is_not_reinterpreted():
    """Only BatchNorm's own check is translated; a genuine shape bug must stay a shape bug."""
    from symtrs.engine.loop import _as_batch_size_error

    original = ValueError("prediction shape (1, 1, 384, 384) does not match target (1, 1, 2048, 2048)")
    assert _as_batch_size_error(original, _AnyModel(), 1) is original

