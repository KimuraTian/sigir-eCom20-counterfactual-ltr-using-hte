from pytest import approx
import numpy as np

from counterfactual_evaluation.datasets import make_tfr_input


def test_make_tfr_input():
    X = np.array([[2, 3, 5],
                  [3, 3, 6],
                  [1, 2, 3],
                  [2, 2, 5],
                  [6, 3, 1]])
    y = np.array([1, 1, 1, 0, 1])
    qid = np.array([1, 1, 2, 2, 3])
    feature_columns = ['1', '2', '3']
    expected_feature_dict = {
        '1': np.array([[[2], [3]], [[1], [2]], [[6], [0]]]),
        '2': np.array([[[3], [3]], [[2], [2]], [[3], [0]]]),
        '3': np.array([[[5], [6]], [[3], [5]], [[1], [0]]]),
    }
    expected_labels = np.array([[1, 1], [1, 0], [1, -1]])
    actual_features, actual_labels = make_tfr_input(
        X, y, qid, feature_columns, list_size=2)
    for i in feature_columns:
        assert actual_features[i] == approx(expected_feature_dict[i])
    assert actual_labels == approx(expected_labels)
