import pytest

from tinytune.metrics import accuracy, compute_metrics, f1_binary, f1_macro


def test_accuracy_perfect_and_half():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0
    assert accuracy([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5


def test_accuracy_rejects_mismatched_or_empty():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1])
    with pytest.raises(ValueError):
        accuracy([], [])


def test_f1_binary_known_value():
    # tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3, f1=2/3
    assert f1_binary([1, 1, 1, 0], [1, 1, 0, 1]) == pytest.approx(2 / 3)


def test_f1_binary_no_true_positives_is_zero():
    assert f1_binary([0, 0, 0], [1, 1, 1]) == 0.0


def test_f1_macro_averages_classes():
    preds = [1, 1, 1, 0]
    labels = [1, 1, 0, 1]
    expected = (f1_binary(preds, labels, 1) + f1_binary(preds, labels, 0)) / 2
    assert f1_macro(preds, labels) == pytest.approx(expected)


def test_compute_metrics_keys_and_rounding():
    m = compute_metrics([1, 0, 1, 0], [1, 0, 0, 0])
    assert set(m) == {"accuracy", "f1"}
    assert m["accuracy"] == 0.75
