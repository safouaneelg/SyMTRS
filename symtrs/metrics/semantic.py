"""Segmentation metrics built on a confusion matrix.

Accumulating a confusion matrix over the epoch and dividing at the end is the only
correct way to compute mIoU over a dataset: averaging per-batch IoU weights a batch
containing a rare class exactly as heavily as one that does not, which is how mIoU
numbers end up unreproducible. `ConfusionMatrix` exists to make the correct thing
the easy thing.

Which classes the class-averaged scores (`miou`, `mean_dice`, `mean_acc`) average over is a
property of the DATA, not of the model: the classes with ground-truth pixels in the evaluated
split (`supported_classes`). The head may declare more outputs than the data uses - the SyMTRS
masks never contain class 0 - and a class-averaged score whose denominator moved with whatever
the model happened to predict would put different models of one table on different scales.
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

    def supported_classes(self) -> torch.Tensor:
        """Boolean mask of the classes with at least one ground-truth pixel.

        Taken from the whole confusion matrix, i.e. the union over the entire evaluated split
        and never per image, so it is identical for every model scored on that split.
        """
        return self.matrix.sum(dim=1) > 0

    def per_class_iou(self) -> torch.Tensor:
        """IoU per class, NaN for classes with no ground truth and no prediction.

        A class with no ground truth that the model nevertheless predicts has union > 0 and a
        real IoU of 0. It is reported (so the false positives are visible, see `fp_pixels_k` in
        `summary()`) but it is NOT part of any class average: see `supported_classes`.
        """
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
        """Mean IoU over the classes that have ground-truth pixels in the evaluated split.

        The denominator is a property of the data (`supported_classes`), identical for every
        model scored on the same split. The previous rule - average every class with a non-empty
        union - made it depend on the model: class 0 (background) has no ground-truth pixel in
        any SyMTRS split while the head has an output for it, so a model that predicted a single
        class-0 pixel got IoU_0 = 0 and its mIoU was divided by 5 instead of 4, a ~20% relative
        drop for one stray pixel. Predicting a class that is not in the data is still punished -
        the pixels it steals from the real classes are false negatives there, and `fp_pixels_k`
        counts them - but it no longer rescales the average.

        `miou_classes` in `summary()` is how many classes this averaged over. A table of mIoUs
        from different splits (or a different label space) is not a like-for-like table.
        """
        return self._mean_over_supported(self.per_class_iou())

    def _mean_over_supported(self, values: torch.Tensor) -> float:
        supported = self.supported_classes() & ~torch.isnan(values)
        if not supported.any():
            return 0.0
        return float(values[supported].mean())

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        if total <= 0:
            return 0.0
        return float(self.intersection().sum() / total)

    def mean_accuracy(self) -> float:
        """Mean per-class recall over the supported classes (see `supported_classes`)."""
        per_class_total = self.matrix.sum(dim=1)
        recall = self.intersection() / per_class_total.clamp_min(EPS)
        valid = per_class_total > 0
        if not valid.any():
            return 0.0
        return float(recall[valid].mean())

    def mean_dice(self) -> float:
        """Mean Dice over the supported classes, for the same reason as `miou`."""
        return self._mean_over_supported(self.per_class_dice())

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
        """How many classes `miou()` averaged over (those with ground-truth pixels)."""
        return int(self.supported_classes().sum())

    def summary(self) -> Dict[str, float]:
        """The canonical metric dict, keys in `SEGMENTATION_METRIC_NAMES` order.

        A class with an empty union has `iou_<k>` / `dice_<k>` of NaN, written as `null`
        in `summary.json`: "not measurable" is not the same statement as "IoU 0", and a
        0.0 here made a class that does not exist in the data look like a model failure.
        A class with no ground truth that is predicted anyway keeps `iou_<k>` = 0.0 but is not
        averaged; `fp_pixels_k` (pixels predicted k whose ground truth is another class) shows
        how much of it there was.

        `miou_classes` is the number of classes the class averages cover, and `support_k` the
        ground-truth pixel count of each class, so the averaged class ids are `k` with
        `support_k > 0`. (Kept numeric: this dict goes straight into the metrics CSV.)
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
        row_totals = self.matrix.sum(dim=1)
        false_positives = self.matrix.sum(dim=0) - self.intersection()
        for index in range(self.num_classes):
            summary[f"support_{index}"] = float(row_totals[index])
            summary[f"fp_pixels_{index}"] = float(false_positives[index])
        return summary

    def class_names(self) -> List[str]:
        return [f"class_{index}" for index in range(self.num_classes)]

    def format_report(self, names: Optional[Sequence[str]] = None) -> str:
        """A per-class table, for the end of an evaluation run."""
        names = list(names) if names else self.class_names()
        per_class_iou = self.per_class_iou()
        per_class_dice = self.per_class_dice()
        support = self.matrix.sum(dim=1)
        false_positives = self.matrix.sum(dim=0) - self.intersection()

        lines = [
            f"{'class':<14}{'IoU':>8}{'Dice':>8}{'pixels':>14}{'fp_pixels':>14}",
            "-" * 58,
        ]
        for index, name in enumerate(names):
            iou = per_class_iou[index].item()
            dice = per_class_dice[index].item()
            iou_text = "n/a" if iou != iou else f"{iou:.4f}"
            dice_text = "n/a" if dice != dice else f"{dice:.4f}"
            lines.append(
                f"{name:<14}{iou_text:>8}{dice_text:>8}{int(support[index].item()):>14}"
                f"{int(false_positives[index].item()):>14}"
            )
        lines.append("-" * 58)
        summary = self.summary()
        lines.append(
            f"{'mIoU':<14}{summary['miou']:>8.4f}"
            f"{'pixel_acc':>12}{summary['pixel_acc']:>8.4f}"
            f"{'mean_acc':>12}{summary['mean_acc']:>8.4f}"
        )
        return "\n".join(lines)
