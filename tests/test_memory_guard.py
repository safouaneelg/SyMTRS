"""The CUDA memory guard exists because WDDM spills to system RAM instead of raising OOM."""

import pytest

from tests.conftest import requires_torch


@requires_torch
def test_guard_is_off_without_the_variable(monkeypatch):
    import torch
    from symtrs.common import seed

    monkeypatch.delenv(seed.MEMORY_FRACTION_ENV, raising=False)
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a))
    seed.apply_memory_guard(torch.device("cuda", 0))
    assert calls == []  # no variable, no cap: other machines behave exactly as before


@requires_torch
@pytest.mark.parametrize("raw", ["0", "1.5", "-0.2"])
def test_guard_rejects_a_fraction_outside_0_1(monkeypatch, raw):
    import torch
    from symtrs.common import seed

    monkeypatch.setenv(seed.MEMORY_FRACTION_ENV, raw)
    with pytest.raises(ValueError, match="must be in \(0, 1\]"):
        seed.apply_memory_guard(torch.device("cuda", 0))
