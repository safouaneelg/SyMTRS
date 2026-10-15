"""Contracts the depth adapters owe the rest of the pipeline.

These are narrow on purpose. Every one of them pins something that was wrong in a way no
test noticed: a declared dependency that `doctor` then reported as satisfied, and a
preprocessing step that produced a five-dimensional tensor which only failed several frames
later, inside `conv2d`.

The dependencies here are the ones that come from *upstream* code, so they cannot be found by
reading our own adapters - nothing in `symtrs/` imports timm or cv2 for a MiDaS forward pass.
They are load-time requirements of the published checkpoints, and they are recorded next to
the reason so that the next person does not remove them as unused.

Torch is needed, so this module skips on a machine without it. Nothing here downloads
weights; the forward pass is exercised by `symtrs doctor --model <key> --smoke`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")


# --------------------------------------------------- declared dependencies are complete


#: adapter target -> modules that must be declared. Each is required by code we do not own:
#:   timm     - the MiDaS hub repo's `hubconf.py` imports `midas.blocks`, which imports
#:              `backbones.beit`, which does `import timm` at module level. So it is needed
#:              to load the weights at all, not just to use our own encoders. The upstream
#:              `hubconf.py` declares `dependencies = ["torch"]`, so nothing warns you.
#:              ZoeDepth needs it for the same reason: its entrypoint builds that same MiDaS
#:              core, so a ZoeDepth build fails without timm even though `zoedepth/` contains
#:              no `import timm` of its own. Verified by blocking timm in `sys.meta_path` and
#:              watching `ZoeD_N` fail at the `backbones/beit.py` import.
#:   cv2      - the hub's `transforms()` entry point imports cv2, and the MiDaS adapter calls
#:              it, deliberately, so that the normalisation matches the checkpoint. ZoeDepth
#:              does *not* list it: it never calls `transforms()`, and a pretrained ZoeD_N
#:              forward pass was measured to run with cv2 blocked from `sys.meta_path`.
#: The timm gap was invisible: `symtrs doctor` reported every MiDaS and ZoeDepth entry as
#: "dependencies ok" on a machine where none of them could be built.
_REQUIRED_BY_UPSTREAM = {
    "symtrs.model.depth.midas:MiDaSAdapter": ("torch", "timm", "cv2"),
    "symtrs.model.depth.zoedepth:ZoeDepthAdapter": ("torch", "timm"),
}


@pytest.mark.parametrize("target,expected", sorted(_REQUIRED_BY_UPSTREAM.items()))
def test_the_adapter_declares_what_the_upstream_code_needs(target, expected):
    import importlib

    module_name, _, class_name = target.partition(":")
    adapter = getattr(importlib.import_module(module_name), class_name)

    declared = set(adapter.requires)
    assert set(expected) <= declared, (
        f"{class_name}.requires is {sorted(declared)}, missing "
        f"{sorted(set(expected) - declared)}; `doctor` reads this attribute, so a gap here "
        "is reported as a satisfied dependency"
    )


@pytest.mark.parametrize("target,expected", sorted(_REQUIRED_BY_UPSTREAM.items()))
def test_the_registry_entry_agrees_with_the_adapter(target, expected):
    """The two copies drift silently: `doctor`'s per-model report reads the registry, while
    the runtime ImportError reads the class. A gap in either one is a wrong answer."""
    from symtrs.registry import list_models

    spec = next(s for s in list_models() if s.target == target)
    assert set(expected) <= set(spec.requires), (
        f"the registry entry for {spec.key!r} declares {sorted(spec.requires)}"
    )


# ------------------------------------------------------------- preprocessing shape


def test_a_hub_transform_output_is_not_batched_a_second_time():
    """The MiDaS transforms end in `PrepareForNet`, which adds the batch axis itself, so one
    call on one image returns `[1, 3, H, W]`. Reading that as unbatched and `stack`ing a list
    of them produced `[B, 1, 3, H, W]`; `conv2d` then rejected it with "Expected 3D
    (unbatched) or 4D (batched) input ... got [1, 1, 3, 257, 257]", which points at neither
    the transform nor this line. Every MiDaS forward pass failed this way.
    """
    from symtrs.model.depth.midas import MiDaSAdapter

    already_batched = torch.zeros(1, 3, 256, 256)
    assert MiDaSAdapter._as_batched(already_batched).shape == (1, 3, 256, 256)

    unbatched = torch.zeros(3, 256, 256)
    assert MiDaSAdapter._as_batched(unbatched).shape == (1, 3, 256, 256)


def test_a_batch_of_transform_outputs_concatenates_to_four_dimensions():
    """The composition, not just the helper: three images in, `[3, 3, H, W]` out. `cat` and
    not `stack`, which is the whole of the fix."""
    from symtrs.model.depth.midas import MiDaSAdapter

    per_image = [MiDaSAdapter._as_batched(torch.zeros(1, 3, 256, 256)) for _ in range(3)]
    assert torch.cat(per_image, dim=0).shape == (3, 3, 256, 256)


def test_a_non_tensor_transform_output_is_accepted():
    """A transform that returns a numpy array or a list still has to work; `torch.cat` on
    those would raise a confusing TypeError inside the forward pass.

    The shapes are worth spelling out: `_as_batched` adds an axis only for a single image,
    `[C, H, W]` -> `[1, C, H, W]`. A two-dimensional `[H, W]` is passed through unchanged, so
    asserting `(1, 4, 4)` on one would be asserting the wrong contract.
    """
    from symtrs.model.depth.midas import MiDaSAdapter

    one_image = [[[0.0] * 4] * 4] * 2  # [C, H, W], as a list this time rather than a tensor
    assert MiDaSAdapter._as_batched(one_image).shape == (1, 2, 4, 4)

    # An unbatched `[H, W]` stays as it is: adding a leading axis here would turn a 2-D
    # array into a 3-D one and quietly change what the caller sees.
    assert MiDaSAdapter._as_batched([[0.0] * 4] * 4).shape == (4, 4)
