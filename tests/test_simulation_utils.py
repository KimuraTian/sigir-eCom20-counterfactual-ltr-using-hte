import numpy as np
import pandas as pd

from counterfactual_evaluation.simulation_utils import split_data_by_groups


def test_split_data_by_groups():
    data = pd.DataFrame({'partition': ['train'] * 6 + ['test'] * 4,
                         'qid': list(range(6)) + list(range(4))})
    groups = ['partition', 'qid']
    train, test = split_data_by_groups(data, groups, 0.2)
    assert train.merge(test).empty
    combined = pd.concat([train, test])
    columns = data.columns.values.tolist()
    assert combined.shape == data.shape
    combined_values = combined.sort_values(columns).values
    data_values = data.sort_values(columns).values
    assert np.array_equal(combined_values, data_values)
    # duplicate qids in subgroup
    data = pd.DataFrame({'partition': ['train'] * 6 + ['test'] * 4,
                         'qid': [1, 1, 1, 2, 3, 3] + [1, 2, 2, 2],
                         'values': np.random.random_sample(10)})
    train, test = split_data_by_groups(data, groups, 0.2)
    assert train.merge(test).empty
    combined = pd.concat([train, test])
    columns = data.columns.values.tolist()
    assert combined.shape == data.shape
    combined_values = combined.sort_values(columns).values
    data_values = data.sort_values(columns).values
    assert np.array_equal(combined_values, data_values)
