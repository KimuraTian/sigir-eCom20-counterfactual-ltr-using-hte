import pytest
from pytest import approx
import numpy as np

import counterfactual_evaluation as cfeval


@pytest.mark.repeat(10)
def test_position_examination_probabilities():
    eta = np.random.choice([0, 1, 2])
    pos_exam = cfeval.simulator.PositionExamination(eta)
    # test a single probability
    rank = np.random.randint(1, 10)
    expected_prob = (1 / rank) ** eta
    actual_prob = pos_exam.probabilities(rank)
    assert expected_prob == actual_prob
    # test probabilities
    ranks = np.random.randint(1, 11, 100)
    expected_probs = np.power(1 / ranks, eta)
    actual_probs = pos_exam.probabilities(ranks)
    assert actual_probs == approx(expected_probs)
    # test truncation at rank `trunc_k`
    trunc_k = np.random.randint(1, 11)
    pos_exam.set_params(trunc_at_rank=trunc_k)
    rank = np.random.randint(1, 100)
    expected_prob = (1 / rank) ** eta if rank <= trunc_k else 0
    actual_prob = pos_exam.probabilities(rank)
    assert expected_prob == actual_prob
    # test probabilities with truncation
    ranks = np.random.randint(1, 20, 100)
    expected_probs = np.power(1 / ranks, eta)
    expected_probs[ranks > trunc_k] = 0
    actual_probs = pos_exam.probabilities(ranks)
    assert actual_probs == approx(expected_probs)


@pytest.mark.repeat(10)
def test_position_examination_values():
    seed = np.random.randint(1, 20)
    eta = np.random.choice([0, 1, 2])
    pos_exam = cfeval.simulator.PositionExamination(eta)
    # test a single draw
    rank = np.random.randint(1, 10)
    prob = (1 / rank) ** eta
    np.random.seed(seed)
    expected_value = np.random.binomial(1, prob)
    np.random.seed(seed)
    actual_value = pos_exam.values(rank)
    assert actual_value == expected_value
    # test multiple draws
    ranks = np.random.randint(1, 11, 100)
    expected_probs = np.power(1 / ranks, eta)
    np.random.seed(seed)
    expected_values = np.random.binomial(1, expected_probs)
    np.random.seed(seed)
    actual_values = pos_exam.values(ranks)
    assert actual_values == approx(expected_values)
    # test truncation at rank `trunc_k`
    trunc_k = np.random.randint(1, 11)
    pos_exam.set_params(trunc_at_rank=trunc_k)
    rank = np.random.randint(1, 100)
    expected_prob = (1 / rank) ** eta if rank <= trunc_k else 0
    np.random.seed(seed)
    expected_value = np.random.binomial(1, expected_prob)
    np.random.seed(seed)
    actual_value = pos_exam.values(rank)
    assert actual_value == expected_value
    # test multiple draws with truncation
    ranks = np.random.randint(1, 20, 100)
    expected_probs = np.power(1 / ranks, eta)
    expected_probs[ranks > trunc_k] = 0
    np.random.seed(seed)
    expected_values = np.random.binomial(1, expected_probs)
    np.random.seed(seed)
    actual_values = pos_exam.values(ranks)
    assert actual_values == approx(expected_values)


@pytest.mark.repeat(10)
def test_click_noise_values():
    seed = np.random.randint(1, 20)
    perfect_eps = dict([(0, 0.0), (1, 0.2), (2, 0.4), (3, 0.8), (4, 1.0)])
    noise_model = cfeval.simulator.ClickNoise(perfect_eps)
    # test a single draw
    rel = np.random.randint(5)
    prob = perfect_eps.get(rel)
    np.random.seed(seed)
    expected_value = np.random.binomial(1, prob)
    np.random.seed(seed)
    actual_value = noise_model.values(rel)
    assert actual_value == expected_value
    # test multiple draws
    rels = np.random.randint(5, size=100)
    expected_probs = np.array([perfect_eps.get(r) for r in rels])
    np.random.seed(seed)
    expected_values = np.random.binomial(1, expected_probs)
    np.random.seed(seed)
    actual_values = noise_model.values(rels)
    assert actual_values == approx(expected_values)


def test_position_based_click_model_values():
    k = range(5)
    v = [0.0, 0.2, 0.4, 0.8, 1.0]
    perfect_eps = dict(zip(k, v))
    for eta in range(3):
        pos_exam = cfeval.simulator.PositionExamination(eta)
        noise_model = cfeval.simulator.ClickNoise(perfect_eps)
        pbm = cfeval.simulator.PositionBasedClickModel(pos_exam, noise_model)
        # test a single draw
        for rank in range(1, 6):
            expected_clicks = []
            actual_clicks = []
            for i in range(10000):
                prob = (1 / rank) ** eta
                expected_examination = np.random.binomial(1, prob)
                exp_click = [0, 0, 0, 0, 0]
                if expected_examination:
                    exp_click = np.random.binomial(1, v)
                expected_clicks.append(exp_click)
                act_click = pbm.values(rank, k, context=None)
                actual_clicks.append(act_click)
            # test click rate
            assert np.mean(actual_clicks, axis=0) == approx(
                np.mean(expected_clicks, axis=0), abs=0.05)

    # test multiple draws
    ndraws = 10000
    ranks = np.random.randint(1, 11, 100)
    for eta in range(3):
        expected_probs = np.power(1 / ranks, eta)
        expected_examination = np.random.binomial(
            1, expected_probs, size=(ndraws, expected_probs.shape[0]))
        expected_perceived_rel = np.random.binomial(1, v, size=(ndraws, len(v)))
        expected_clicks = (expected_examination[:, :, None] *
                           expected_perceived_rel[:, None, :])
        pos_exam = cfeval.simulator.PositionExamination(eta)
        noise_model = cfeval.simulator.ClickNoise(perfect_eps)
        pbm = cfeval.simulator.PositionBasedClickModel(pos_exam, noise_model)
        actual_clicks = []
        for rel in k:
            act_clicks = []
            for d in range(ndraws):
                act_click = pbm.values(ranks, rel, context=None)
                act_clicks.append(act_click)
            actual_clicks.append(np.array(act_clicks))
        actual_clicks = np.stack(actual_clicks, axis=2)
        assert np.mean(actual_clicks, axis=0) == approx(
            np.mean(expected_clicks, axis=0), abs=0.05)
