import pytest
from pytest import approx
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

from counterfactual_evaluation.metrics import nDCG, compute_reciprocal_rank
from counterfactual_evaluation.datasets import rank_list_by_pred_np


@pytest.mark.repeat(10)
def test_ndcg():
    """tests ndcg scores close to tfr implementation"""
    with tf.Session() as sess:
        labels = np.random.randint(-1, 3, size=(10, 5)).astype(np.float32)
        preds = np.random.randn(10, 5).astype(np.float32)
        actual_ndcg = []
        metric_ops = []
        update_ops = []
        for topn in range(1, 6):
            for i in range(10):
                label = labels[[i]]
                pred = preds[[i]]
                metric_op, update_op = (
                    tfr.metrics.normalized_discounted_cumulative_gain(
                        label, pred, topn=topn))
                metric_ops.append(metric_op)
                update_ops.append(update_op)
            actual_ndcg.append(nDCG(labels, preds, topn))

        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(update_ops)
        expected_ndcg = sess.run(metric_ops)
        expected_ndcg = np.array(expected_ndcg).reshape((5, 10))
        actual_ndcg = np.array(actual_ndcg)
        assert actual_ndcg == approx(expected_ndcg)


@pytest.mark.repeat(10)
def test_reciprocal_rank():
    """tests reciprocal rank scores close to tfr implementation"""
    labels = np.random.randint(-1, 3, size=(10, 5)).astype(np.float32)
    preds = np.random.randn(10, 5).astype(np.float32)
    metric_ops = []
    update_ops = []
    for i in range(10):
        label = labels[[i]]
        pred = preds[[i]]
        metric_op, update_op = (
            tfr.metrics.mean_reciprocal_rank(
                label, pred))
        metric_ops.append(metric_op)
        update_ops.append(update_op)
    with tf.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(update_ops)
        expected_reciprocal_rank = sess.run(metric_ops)
    expected_reciprocal_rank = np.array(expected_reciprocal_rank)
    actual_reciprocal_rank = compute_reciprocal_rank(labels, preds)
    assert actual_reciprocal_rank == approx(expected_reciprocal_rank)


def test_rank_list_by_pred_np():
    # negative labels are invalid labels. Any documents with invalid labels will
    # be appended at the bottom of the return list
    labels = [[3, 2, 1, 0], [3, 0, -1, -1]]
    prediction_list = [[[3, 2, 1, 5], [2, 1, 3, 3]],
                       [[1, 3, 2, 0], [1, 2, 1, 1]]]
    documents = [['in my feelings', 'drake', 'drake & future', 'nonstop'],
                 ['rock hits', 'classic rock', 'unknown', 'unknown']]
    ranked_documents = rank_list_by_pred_np(labels, prediction_list, documents)
    expected = [
        np.array([['nonstop', 'in my feelings', 'drake', 'drake & future'],
                  ['rock hits', 'classic rock', 'unknown', 'unknown']]),
        np.array([['drake', 'drake & future', 'in my feelings', 'nonstop'],
                  ['classic rock', 'rock hits', 'unknown', 'unknown']])]
    assert np.array_equal(ranked_documents, expected)
