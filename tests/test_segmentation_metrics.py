"""The segmentation confusion matrix and the metrics derived from it.

Two things are being protected here. The first is that mIoU is computed over an
accumulated confusion matrix rather than averaged per batch - getting that wrong is how
a mIoU number becomes unreproducible while still looking plausible. The second is that a
class with no ground truth is *excluded* from mIoU rather than counted as zero, so the
score does not depend on how many unused classes a label space declares.

The expectations are hand-computed from the confusion matrix rather than captured from a
run. Torch is needed, so this module skips on a machine without it.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs.metrics.semantic import (  # noqa: E402
    SEGMENTATION_METRIC_NAMES,
    ConfusionMatrix,
)


#: 8 pixels over 3 classes, one of them ignored. Rows are ground truth, columns
#: predictions, so the matrix is
#:
#:     [[1, 1, 0],
#:      [1, 1, 0],
#:      [1, 0, 2]]
#:
#: The four errors are the interesting part: two class-1/class-0 confusions and one
#: class-2/class-0 confusion, so no class is perfect and no class is hopeless.
PREDICTION = [0, 1, 2, 0, 1, 255, 2, 0]
TARGET = [0, 1, 2, 1, 0, 255, 2, 2]


def _matrix(num_classes: int = 3) -> ConfusionMatrix:
    matrix = ConfusionMatrix(num_classes=num_classes, ignore_index=255)
    matrix.update(torch.tensor(PREDICTION), torch.tensor(TARGET))
    return matrix


def _built(prediction, target, num_classes: int = 3) -> ConfusionMatrix:
    matrix = ConfusionMatrix(num_classes=num_classes, ignore_index=255)
    matrix.update(torch.tensor(prediction), torch.tensor(target))
    return matrix


def test_the_matrix_is_rows_ground_truth_columns_prediction():
    """Swapping the two axes is the classic slip, and it leaves mIoU unchanged while
    quietly turning precision into recall in `mean_acc`."""
    matrix = _matrix().matrix
    assert matrix.tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 2.0]]


def test_the_ignored_pixel_is_dropped_from_the_total():
    """7 of the 8 pixels count. Including the ignored one would depress every fraction
    by a factor that depends on how much stuff the labellers deliberately skipped."""
    assert float(_matrix().matrix.sum()) == 7.0


def test_every_metric_matches_hand_computation():
    summary = _matrix().summary()

    # IoU = diag / (col + row - diag) = [1/4, 1/3, 2/3]
    assert summary["miou"] == pytest.approx((0.25 + 1 / 3 + 2 / 3) / 3, abs=1e-9)
    # 4 of 7 pixels land on the diagonal.
    assert summary["pixel_acc"] == pytest.approx(4 / 7, abs=1e-9)
    # per-class recall = diag / row sum = [1/2, 1/2, 2/3], each class weighted equally
    assert summary["mean_acc"] == pytest.approx((0.5 + 0.5 + 2 / 3) / 3, abs=1e-9)
    # Dice = 2*diag / (col + row) = [0.4, 0.5, 0.8]
    assert summary["mean_dice"] == pytest.approx((0.4 + 0.5 + 0.8) / 3, abs=1e-9)
    # frequency-weighted: weights are the ground-truth shares [2/7, 2/7, 3/7]
    assert summary["fwiou"] == pytest.approx(
        (2 / 7) * 0.25 + (2 / 7) * (1 / 3) + (3 / 7) * (2 / 3), abs=1e-9
    )


def test_the_named_metrics_are_exactly_the_ones_reported():
    summary = _matrix().summary()
    for name in SEGMENTATION_METRIC_NAMES:
        assert name in summary, name


def test_per_class_metrics_are_reported_alongside_the_means():
    """For a six-class benchmark the per-class breakdown is the thing you actually look
    at when the mean moves, so it has to survive into the summary dict."""
    summary = _matrix().summary()
    assert summary["iou_0"] == pytest.approx(0.25, abs=1e-9)
    assert summary["iou_1"] == pytest.approx(1 / 3, abs=1e-9)
    assert summary["iou_2"] == pytest.approx(2 / 3, abs=1e-9)
    assert summary["dice_2"] == pytest.approx(0.8, abs=1e-9)


# ------------------------------------------------------- accumulation, not averaging


def test_accumulating_in_batches_equals_a_single_update():
    """The whole point of the class. Averaging per-batch IoU gives a different - and
    batch-size-dependent - answer, which is why an evaluation has to accumulate."""
    all_at_once = _built(PREDICTION, TARGET)

    matrix = ConfusionMatrix(num_classes=3, ignore_index=255)
    for index in range(0, len(PREDICTION), 3):
        matrix.update(
            torch.tensor(PREDICTION[index : index + 3]),
            torch.tensor(TARGET[index : index + 3]),
        )

    assert torch.equal(matrix.matrix, all_at_once.matrix)
    assert matrix.miou() == pytest.approx(all_at_once.miou(), abs=1e-12)


def test_batch_slices_accumulate_into_the_same_matrix_as_one_update():
    matrix = ConfusionMatrix(num_classes=3, ignore_index=255)
    for index in range(0, len(PREDICTION), 3):
        matrix.update(
            torch.tensor(PREDICTION[index : index + 3]),
            torch.tensor(TARGET[index : index + 3]),
        )
    assert torch.equal(matrix.matrix, _matrix().matrix)


# ------------------------------------------------------------------- edge behaviour


def test_a_class_absent_from_both_prediction_and_target_is_excluded_not_zeroed():
    """Otherwise the score would depend on how many unused classes the label space
    declares - and a 6-class benchmark would score lower than the same predictions under
    a 3-class one."""
    matrix = _built([0, 0, 1, 1], [0, 1, 1, 1], num_classes=4)

    assert matrix.per_class_iou()[2].isnan() and matrix.per_class_iou()[3].isnan()
    # (0.5 + 2/3) / 2, not divided by 4.
    assert matrix.miou() == pytest.approx((0.5 + 2 / 3) / 2, abs=1e-9)
    # The per-class entries are NaN (null in summary.json), not 0.0: "not measurable" read
    # as "IoU 0" made a class that is nowhere in the data look like a model failure. The
    # number of classes the mean covers is reported next to it.
    assert math.isnan(matrix.summary()["iou_2"])
    assert math.isnan(matrix.summary()["dice_3"])
    assert matrix.summary()["miou_classes"] == 2


def test_a_prediction_outside_the_label_range_counts_as_an_error():
    """A model that emits class 7 under a 2-class label space is wrong, not invisible.
    Dropping the pixel would raise the score for being wrong in a new way."""
    matrix = _built([7], [0], num_classes=2)
    assert float(matrix.matrix.sum()) == 1.0
    assert matrix.pixel_accuracy() == 0.0
    assert float(matrix.matrix[0, 1]) == 1.0


def test_a_batch_that_is_entirely_ignored_is_a_no_op():
    """The last batch of an epoch is often short and can be all ignore_index in a tile
    with no labels. It must not raise, and it must not change the matrix."""
    matrix = _matrix()
    before = matrix.matrix.clone()
    matrix.update(torch.tensor([255, 255]), torch.tensor([255, 255]))
    assert torch.equal(matrix.matrix, before)


def test_a_mismatched_pixel_count_is_refused():
    matrix = ConfusionMatrix(num_classes=3)
    with pytest.raises(ValueError, match="pixels"):
        matrix.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1]))


def test_an_empty_matrix_reports_zero_rather_than_nan():
    """A run that aborts after evaluating nothing still writes a CSV row."""
    matrix = ConfusionMatrix(num_classes=3)
    assert matrix.miou() == 0.0
    assert matrix.pixel_accuracy() == 0.0
    assert matrix.mean_accuracy() == 0.0
    assert matrix.mean_dice() == 0.0
    assert matrix.frequency_weighted_iou() == 0.0


def test_reset_clears_the_accumulation():
    matrix = _matrix()
    matrix.reset()
    assert float(matrix.matrix.sum()) == 0.0


def test_five_class_report_names_every_class():
    """The capture has five classes; the report is what a reader checks a run against."""
    names = ["background", "building", "green", "road", "car"]
    matrix = ConfusionMatrix(num_classes=5, ignore_index=255)
    matrix.update(torch.arange(5), torch.arange(5))
    report = matrix.format_report(names)

    assert "miou" not in report  # the table spells it "mIoU"
    assert "mIoU" in report and "pixel_acc" in report
    for name in names:
        assert name in report, name
    # Every class was predicted perfectly, so mIoU is 1 and the per-class rows say so.
    assert matrix.miou() == pytest.approx(1.0, abs=1e-9)
    assert "1.0000" in report


# ------------------------------------- the class average is a property of the data, not the model


def test_class_averages_cover_the_classes_with_ground_truth_whatever_the_model_predicts():
    """mIoU, mean_dice and mean_acc average over the classes that have ground-truth pixels in the
    split. Class 0 has none (SyMTRS masks hold classes 1-4) but the head has an output for it; a
    model that emits one class-0 pixel must not change the denominator, or two models in one
    table are averaged over different numbers of classes."""
    target = [1, 1, 2, 2, 3, 3, 4, 4]
    clean = _built(target, target, num_classes=5)
    stray = _built([0, 1, 2, 2, 3, 3, 4, 4], target, num_classes=5)

    for matrix in (clean, stray):
        assert matrix.classes_averaged() == 4
        assert matrix.supported_classes().tolist() == [False, True, True, True, True]
    assert clean.miou() == pytest.approx(1.0)
    assert stray.miou() == pytest.approx((0.5 + 3.0) / 4)  # not divided by 5
    assert stray.mean_dice() == pytest.approx((2 / 3 + 3.0) / 4)
    assert stray.mean_accuracy() == pytest.approx((0.5 + 3.0) / 4)


def test_the_unsupported_but_predicted_class_stays_visible_per_class():
    stray = _built([0, 0, 1], [1, 1, 1], num_classes=3)
    summary = stray.summary()
    assert summary["iou_0"] == 0.0, "reported, not hidden"
    assert summary["fp_pixels_0"] == 2.0 and summary["support_0"] == 0.0
    assert summary["support_1"] == 3.0 and summary["fp_pixels_1"] == 0.0
    assert summary["miou_classes"] == 1
    assert summary["miou"] == pytest.approx(1 / 3), "class 1: 1 of 3 pixels found"


def test_the_supported_set_is_the_union_over_the_whole_split_not_per_image():
    """An image that happens to lack class 2 must not remove it from the average."""
    matrix = ConfusionMatrix(num_classes=3, ignore_index=255)
    matrix.update(torch.tensor([0, 0]), torch.tensor([0, 0]))  # image 1: only class 0
    matrix.update(torch.tensor([2, 2]), torch.tensor([2, 2]))  # image 2: only class 2
    assert matrix.classes_averaged() == 2
    assert matrix.supported_classes().tolist() == [True, False, True]
