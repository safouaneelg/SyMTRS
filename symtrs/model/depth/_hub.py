"""One compatibility shim for checkpoints loaded out of `torch.hub`.

Both MiDaS and ZoeDepth are loaded by calling an entry point in someone else's repository,
and that entry point does the `load_state_dict` itself. So when a released checkpoint no
longer loads, there is no call site of ours to fix - the failure surfaces from inside
`torch.hub`, naming neither the cause nor a way out.

The case handled here is specific and provable. MiDaS's DPT-BEiT backbones and every ZoeDepth
variant store `relative_position_index` in their released weights. It is not a learned
parameter: it is an index table derived purely from the window size, and recent timm registers
it with `persistent=False`, which keeps it out of `state_dict()` - so a strict load rejects the
checkpoint's copy as an unexpected key. Recomputing it, which is what the freshly built model
already did, gives the same table, so the key is safe to drop.

That reasoning is only sound for these keys, so the shim checks the result of the relaxed load
and re-raises the original error if anything else is missing or unexpected. A genuine
architecture mismatch therefore still fails, loudly, as it must.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch

#: The only buffer this shim will drop. Anything else is a real mismatch.
_BENIGN_KEY = "relative_position_index"


def _is_dropped_position_index(exc: BaseException) -> bool:
    """Whether a failure is the known benign `relative_position_index` mismatch.

    Deliberately reads PyTorch's message rather than catching anything narrower, because
    `load_state_dict` raises a plain `RuntimeError` for both this and a real mismatch. The
    two are told apart by the key names, which is stable across versions; the surrounding
    prose is not.
    """
    text = str(exc)
    if _BENIGN_KEY not in text:
        return False
    if "Missing key" in text:
        # Something the model needs is absent from the checkpoint. That is a real mismatch
        # and must not be relaxed - whatever else the message happens to mention.
        return False
    return "Unexpected key" in text


@contextlib.contextmanager
def load_dropping_position_index() -> Iterator[None]:
    """Let a strict `load_state_dict` inside a hub entry point drop `relative_position_index`.

    Scoped to the single hub call it wraps, and restored in `finally`, so nothing outside it
    is affected. `torch.hub` construction is not concurrent with training, so the global
    patch is safe here; it is not something to reach for in general.
    """
    original = torch.nn.Module.load_state_dict

    def tolerant(self, state_dict, *args, **kwargs):
        try:
            return original(self, state_dict, *args, **kwargs)
        except RuntimeError as exc:
            if not _is_dropped_position_index(exc):
                raise

            kwargs.pop("strict", None)
            report = original(self, state_dict, *args, strict=False, **kwargs)
            offending = [k for k in report.unexpected_keys if _BENIGN_KEY not in k]
            if report.missing_keys or offending:
                # The relaxed load "succeeded" only because strict was off. Put the original
                # error back: it is the honest description of what is wrong.
                raise exc
            print(
                f"[symtrs] {type(self).__name__}: dropped {len(report.unexpected_keys)} "
                f"`{_BENIGN_KEY}` buffer(s) from the checkpoint. Recent timm stops persisting "
                "them; they are recomputed from the window size at construction, so the values "
                "match. Note this makes the *load* succeed only - see the timm warning in "
                "`symtrs doctor`, because some of these models also need the older timm to run."
            )
            return report

    torch.nn.Module.load_state_dict = tolerant
    try:
        yield
    finally:
        torch.nn.Module.load_state_dict = original


__all__ = ["load_dropping_position_index"]
