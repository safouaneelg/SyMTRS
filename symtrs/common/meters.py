"""Running averages and per-metric accumulation."""

from __future__ import annotations

from typing import Dict, Mapping


class AverageMeter:
    """Weighted running mean. Weight is normally the batch size."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AverageMeter(avg={self.avg:.6f}, n={self.count})"


class MetricTracker:
    """A named bag of AverageMeters.

    Metrics are accumulated over an epoch and then formatted for the CSV row. The
    order is insertion order, which keeps the CSV columns stable across epochs and
    is what `symtrs tables` relies on.
    """

    def __init__(self) -> None:
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value, n: int = 1) -> None:
        meter = self._meters.get(name)
        if meter is None:
            meter = self._meters[name] = AverageMeter()
        meter.update(value, n=n)

    def update_many(self, values: Mapping[str, float], n: int = 1) -> None:
        for name, value in values.items():
            self.update(name, value, n=n)

    def averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self._meters.items()}

    def get(self, name: str) -> float:
        meter = self._meters.get(name)
        return meter.avg if meter else 0.0

    def reset(self) -> None:
        for meter in self._meters.values():
            meter.reset()

    def __contains__(self, name: str) -> bool:
        return name in self._meters

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        inner = ", ".join(f"{k}={v:.4f}" for k, v in self.averages().items())
        return f"MetricTracker({inner})"
