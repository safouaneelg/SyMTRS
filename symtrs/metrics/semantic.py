"""Segmentation metrics built on a confusion matrix.

Accumulating a confusion matrix over the epoch and dividing at the end is the only
correct way to compute mIoU over a dataset: averaging per-batch IoU weights a batch
containing a rare class exactly as heavily as one that does not, which is how mIoU
numbers end up unreproducible. `ConfusionMatrix` exists to make the correct thing
the easy thing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

#: Canonical metric order for anything that logs or tabulates segmentation results.
SEGMENTATION_METRIC_NAMES = ("miou", "pixel_acc", "mean_acc", "mean_dice", "fwiou")

EPS = 1e-8


class ConfusionMatrix:
    """A `num_classes x num_classes` confusion matrix accumulated on CPU.

    Rows are ground truth, columns are predictions. Pixels equal to `ignore_index`
    in the target are skipped, as are targets outside `[0, num_classes)`.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch. `prediction` is class indices, `target` is labels."""
        prediction = prediction.detach().reshape(-1).cpu().long()
        target = target.detach().reshape(-1).cpu().long()
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction has {prediction.numel()} pixels, target has {target.numel()}"
            )

        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        if not valid.any():
            return
        target = target[valid]
        # A prediction outside the label range cannot be right; count it as an error
        # against the true class rather than silently dropping it.
        prediction = prediction[valid].clamp(0, self.num_classes - 1)

        flat = target * self.num_classes + prediction
        counts = torch.bincount(flat, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes).double()

    def reset(self) -> None:
        self.matrix.zero_()

    # ------------------------------------------------------------------ derived

    def intersection(self) -> torch.Tensor:
        return torch.diag(self.matrix)

    def union(self) -> torch.Tensor:
        return self.matrix.sum(dim=0) + self.matrix.sum(dim=1) - self.intersection()

    def per_class_iou(self) -> torch.Tensor:
        """IoU per class, NaN for classes with no ground truth and no prediction."""
        union = self.union()
        iou = self.intersection() / union.clamp_min(EPS)
        iou[union == 0] = float("nan")
        return iou

    def per_class_dice(self) -> torch.Tensor:
        intersection = self.intersection()
        denominator = self.matrix.sum(dim=0) + self.matrix.sum(dim=1)
        dice = 2.0 * intersection / denominator.clamp_min(EPS)
        dice[denominator == 0] = float("nan")
        return dice

    def miou(self) -> float:
        """Mean IoU over classes present in the data.

        Classes absent from both prediction and ground truth (union == 0) are excluded
        rather than counted as zero, which would make the score depend on how many unused
        classes the label space happens to declare. A class absent from the ground truth
        that the model nevertheless *predicts* has union > 0, so its IoU is a real 0 and
        does count: predicting something that is not there is an error.

        On the SyMTRS capture class 0 (background) has no pixel in any split - the masks
        only hold the grey levels of classes 1-4 - so in practice this is the mean over the
        classes present, and `miou_classes` in `summary()` says how many that was. A table
        of mIoUs from runs with different `miou_classes` is not a like-for-like table.
        """
        iou = self.per_class_iou()
        valid = ~torch.isnan(iou)
        if not valid.any():
            return 0.0
        return float(iou[valid].mean())

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        if total <= 0:
            return 0.0
        return float(self.intersection().sum() / total)

    def mean_accuracy(self) -> float:
        """Mean per-class recall: each class weighted equally regardless of size."""
        per_class_total = self.matrix.sum(dim=1)
        recall = self.intersection() / per_class_total.clamp_min(EPS)
        valid = per_class_total > 0
        if not valid.any():
            return 0.0
        return float(recall[valid].mean())

    def mean_dice(self) -> float:
        dice = self.per_class_dice()
        valid = ~torch.isnan(dice)
        if not valid.any():
            return 0.0
        return float(dice[valid].mean())

    def frequency_weighted_iou(self) -> float:
        """IoU weighted by class frequency. Penalises errors on common classes more."""
        per_class_total = self.matrix.sum(dim=1)
        total = per_class_total.sum()
        if total <= 0:
            return 0.0
        weights = per_class_total / total
        iou = self.per_class_iou()
        iou = torch.nan_to_num(iou, nan=0.0)
        return float((weights * iou).sum())

    def classes_averaged(self) -> int:
        """How many classes `miou()` averaged over (those with a non-empty union)."""
        return int((~torch.isnan(self.per_class_iou())).sum())

    def summary(self) -> Dict[str, float]:
        """The canonical metric dict, keys in `SEGMENTATION_METRIC_NAMES` order.

        A class with an empty union has `iou_<k>` / `dice_<k>` of NaN, written as `null`
        in `summary.json`: "not measurable" is not the same statement as "IoU 0", and a
        0.0 here made a class that does not exist in the data look like a model failure.
        `miou_classes` is the number of classes `miou` was averaged over.
        """
        per_class_iou = self.per_class_iou()
        per_class_dice = self.per_class_dice()
        summary = {
            "miou": self.miou(),
            "pixel_acc": self.pixel_accuracy(),
            "mean_acc": self.mean_accuracy(),
            "mean_dice": self.mean_dice(),
            "fwiou": self.frequency_weighted_iou(),
            "miou_classes": self.classes_averaged(),
        }
        for index in range(self.num_classes):
            summary[f"iou_{index}"] = float(per_class_iou[index])
            summary[f"dice_{index}"] = float(per_class_dice[index])
        return summary

    def class_names(self) -> List[str]:
        return [f"class_{index}" for index in range(self.num_classes)]

    def format_report(self, names: Optional[Sequence[str]] = None) -> str:
        """A per-class table, for the end of an evaluation run."""
        names = list(names) if names else self.class_names()
        per_class_iou = self.per_class_iou()
        per_class_dice = self.per_class_dice()
        support = self.matrix.sum(dim=1)

        lines = [
            f"{'class':<14}{'IoU':>8}{'Dice':>8}{'pixels':>14}",
            "-" * 44,
        ]
        for index, name in enumerate(names):
            iou = per_class_iou[index].item()
            dice = per_class_dice[index].item()
            iou_text = "n/a" if iou != iou else f"{iou:.4f}"
            dice_text = "n/a" if dice != dice else f"{dice:.4f}"
            lines.append(f"{name:<14}{iou_text:>8}{dice_text:>8}{int(support[index].item()):>14}")
        lines.append("-" * 44)
        summary = self.summary()
        lines.append(
            f"{'mIoU':<14}{summary['miou']:>8.4f}"
            f"{'pixel_acc':>12}{summary['pixel_acc']:>8.4f}"
            f"{'mean_acc':>12}{summary['mean_acc']:>8.4f}"
        )
        return "\n".join(lines)
