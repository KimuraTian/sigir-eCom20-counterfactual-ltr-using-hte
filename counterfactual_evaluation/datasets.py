"""
Data preparation and pre-processing.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

from counterfactual_evaluation.metrics import _validate_label_and_prediction

project_path = Path(__file__).parents[1]
data_path = project_path / 'data'
build_path = project_path / 'build'


def load_libsvm_data(path, feature_columns,
                     nfeatures=137, zeros_based=False,
                     list_size=10, padding_label=-1):
    """
    Loads LibSVM data set. It assumes that query and feature ids are in
    ascending order.

    Args:
        path (path-like): The data path.
        feature_columns (list of str): A list of feature names that used to map
            str feature names to numerical feature names in LibSVM data.
            This is useful when ranking models read in features that
            have different column names than numerical names.
        nfeatures (int): The number of features
        zeros_based (bool): Whether or not the feature names start with 0
        list_size (int): The list size. a result list that is not equal to
            `list_size` is truncated or padded with zeros for features and with
            `padding_label` for labels.
        padding_label (int, float): The padding label. Default -1.

    Returns:
        dict of numpy.array: The feature dictionary used by model input
            function. Keys are the feature names and each value is a
            numpy.array with a shape of [num_result_lists, list_size, 1].
            The last dimension 1 is used for compatible with
            tensorflow_ranking groupwise ranking functions.
        numpy.array: The labels with a shape of [num_result_lists, list_size].
    """
    from sklearn.datasets import load_svmlight_file
    from collections import OrderedDict
    X, y, qid = load_svmlight_file(path, n_features=nfeatures,
                                   query_id=True, zero_based=zeros_based)
    unique_qid, qid_count = np.unique(qid, return_counts=True)
    # initialize features and labels
    features = OrderedDict((k, []) for k in feature_columns)
    labels = []
    # ending index for each query
    split_indices = np.delete(np.cumsum(qid_count), -1)
    feature_arr = np.array_split(X.toarray(), split_indices)
    label_arr = np.array_split(y, split_indices)

    for qid, f_arr in enumerate(feature_arr):
        truncated_list_size = min(len(f_arr), list_size)
        # fill label values
        l_arr = label_arr[qid]
        label = np.ones([list_size], dtype=np.float32) * padding_label
        label[:truncated_list_size] = l_arr[:truncated_list_size]
        labels.append(label)

        # fill feature values
        for i, k in enumerate(feature_columns):
            # feature
            feature = np.zeros((list_size, 1), dtype=np.float32)
            feature[:truncated_list_size, 0] = f_arr[:truncated_list_size, i]
            features[k].append(feature)
    # convert to numpy.array
    for k, v in features.items():
        features[k] = np.array(v)
    return features, np.array(labels)


def load_datasets(data_paths, nfeatures, list_sizes, feature_columns=None,
                  name=None):
    """
    Loads several LibSVM data sets sequentially.

    Args:
        data_paths (:obj:`list` of :obj:`path-like`): The list of paths of
            data sets.
        nfeatures (:obj:`list` of :obj:`int`): The list of number of features
        list_sizes (:obj:`list` of :obj:`int`): The list of list sizes of data
            sets.
        feature_columns (None or :obj:`list` of :obj:`list` of :obj:`str`):
            feature names for each data set. Defaults to None. If None,
            feature names will be set to numerical names starts from 1 to
            `nfeatures`.
        name (None or :obj:`list` of :obj:`str`): The names of data sets.

    Returns:
        :obj:`list` of :obj:`tuple`: A list of features and labels tuples

    """
    if feature_columns is None:
        feature_columns = []
        for nf in nfeatures:
            fc = [str(x+1) for x in range(nf)]
            feature_columns.append(fc)
    data_list = []
    for dp, fc, nf, lsz in zip(data_paths, feature_columns, nfeatures,
                               list_sizes):
        tf.compat.v1.logging.info(
            f'Loading {name or "data"} from {dp} '
            f'Specs: nfeatures={nf}; '
            f'feature_columns=[{fc[0]}...{fc[-1]}]; '
            f'list_size={lsz}')
        features_and_labels = load_libsvm_data(
            path=dp, feature_columns=fc,
            nfeatures=nf, list_size=lsz)
        data_list.append(features_and_labels)
    return data_list


def rank_list_by_pred_np(labels, prediction_list, documents, topn=None):
    """
    Ranks documents by prediction scores

    Args:
        labels (numpy.array): The labels
        prediction_list (list of numpy.array): The predictions by multiple
            algorithms. Predictions from each algorithms have the same shape
            of [batch_size, list_size]
        documents (numpy.array): The documents with the same shape as
            predictions in prediction_list
        topn (int): The truncation number

    Returns:
        list of numpy.array: The list of documents ranked by multiple
            algorithms.
    """
    documents = np.asarray(documents)
    labels = np.asarray(labels)
    prediction_list = [np.asarray(pred) for pred in prediction_list]
    sorted_documents = []
    if topn is None:
        topn = labels.shape[-1]
    topn = np.minimum(labels.shape[-1], topn)
    for pred in prediction_list:
        label, prediction = _validate_label_and_prediction(
            labels, pred)
        reverse_sorted = np.flip(np.argsort(prediction), -1)[..., :topn]
        sorted_document = np.take_along_axis(
            documents, reverse_sorted, -1)
        sorted_documents.append(sorted_document)
    return sorted_documents


def make_tfr_input(features, y, qid, feature_columns, list_size=5,
                   padding_label=-1):
    """
    Converts feature array's shape from [num_examples, num_features] to
    [num_result_lists, list_size, 1], and label array's shape from
    [num_examples] to [num_result_lists, list_size]. It assumes that query and
    feature ids are in ascending order.

    Args:
        features (numpy.array): The feature array with a shape of
        [num_examples, num_features].
        y (numpy.array): The label array with a size of num_examples.
        qid (numpy.array): The array of query ids with a size of num_examples.
        feature_columns (list of str): A list of feature names that used to map
            str feature names to numerical feature names in LibSVM data.
            This is useful when ranking models read in features that
            have different column names than numerical names.
        list_size (int): The list size. a result list that is not equal to
            `list_size` is truncated or padded with zeros for features and with
            `padding_label` for labels.
        padding_label (int, float): The padding label. Default -1.

    Returns:
        dict of numpy.array: The feature dictionary used by model input
            function. Keys are the feature names and each value is a
            numpy.array with a shape of [num_result_lists, list_size, 1].
            The last dimension 1 is used for compatible with
            tensorflow_ranking groupwise ranking functions.
        numpy.array: The labels with a shape of [num_result_lists, list_size].
    """
    from collections import OrderedDict
    unique_qid, qid_count = np.unique(qid, return_counts=True)
    # initialize features and labels
    feature_dict = OrderedDict((k, []) for k in feature_columns)
    labels = []
    # ending index for each query
    split_indices = np.delete(np.cumsum(qid_count), -1)
    feature_arr = np.array_split(features, split_indices)
    label_arr = np.array_split(y, split_indices)

    for qid, f_arr in enumerate(feature_arr):
        truncated_list_size = min(len(f_arr), list_size)
        # fill label values
        l_arr = label_arr[qid]
        label = np.ones([list_size], dtype=np.float32) * padding_label
        label[:truncated_list_size] = l_arr[:truncated_list_size]
        labels.append(label)

        # fill feature values
        for i, k in enumerate(feature_columns):
            # feature
            feature = np.zeros((list_size, 1), dtype=np.float32)
            feature[:truncated_list_size, 0] = f_arr[:truncated_list_size, i]
            feature_dict[k].append(feature)
    # convert to numpy.array
    for k, v in feature_dict.items():
        feature_dict[k] = np.array(v)
    return feature_dict, np.array(labels)
