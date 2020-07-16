import numpy as np


def _dcg(labels, discount=np.log):
    labels = np.nan_to_num(labels)
    ranks = np.arange(1, labels.shape[-1] + 1)
    disc = 1 / discount(ranks + 1)
    numerator = np.power(2, labels) - 1
    return np.dot(numerator, disc)


def nDCG(labels, predictions, topn=3, discount=np.log):
    """
    Computes the nDCG metric for each example.

    Args:
        labels (array-like): The labels with a shape of [batch_size, list_size]
        predictions (array-like): The predictions with the same shape as the
            labels
        topn (int): Top-N metric. Default: 3
        discount (numpy.ufunc): The discount function. Default: np.log

    Returns:
        numpy.array: example-wise nDCG scores.
    """
    labels = np.asarray(labels, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    labels, predictions = _validate_label_and_prediction(labels, predictions)
    topn = np.minimum(labels.shape[-1], topn)
    reverse_sorted = np.flip(np.argsort(predictions), -1)[..., :topn]
    sorted_labels_by_pred = np.take_along_axis(labels, reverse_sorted, -1)
    dcg_score = _dcg(sorted_labels_by_pred, discount)
    sorted_labels = np.flip(np.sort(labels), -1)[..., :topn]
    ideal_dcg_score = _dcg(sorted_labels, discount)
    # 0 if divided by 0
    out = np.zeros_like(ideal_dcg_score)
    return np.divide(dcg_score, ideal_dcg_score,
                     out=out, where=ideal_dcg_score != 0)


def compute_reciprocal_rank(labels, predictions):
    labels, predictions = _validate_label_and_prediction(labels, predictions)
    list_size = labels.shape[-1]
    reverse_sorted = np.flip(np.argsort(predictions), -1)
    sorted_labels_by_pred = np.take_along_axis(labels, reverse_sorted, -1)
    rel = np.greater_equal(sorted_labels_by_pred, 1.0, dtype=np.float32)
    reciprocal_rank = 1.0 / np.arange(1, list_size + 1, dtype=np.float32)
    return np.amax(rel*reciprocal_rank, axis=1)


def _validate_label_and_prediction(labels, predictions):
    """
    Validates labels and predictions by setting any negative labels to 0 and
    setting corresponding predictions to the minimum predictions

    Args:
        labels (array-like): The labels with a shape of [batch_size, list_size]
        predictions (array-like): The predictions with the same shape as labels

    Returns:
        array-like: The labels after validation
        array-like: The predictions
    """
    labels = np.asarray(labels, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    assert np.array_equal(labels.shape, predictions.shape)
    valid_labels = np.greater_equal(labels, 0.0)
    labels = np.where(valid_labels, labels, np.zeros_like(labels))
    predictions = np.where(
        valid_labels, predictions, -1e-6 * np.ones_like(predictions) +
        np.amin(predictions, axis=1, keepdims=True))
    return labels, predictions
