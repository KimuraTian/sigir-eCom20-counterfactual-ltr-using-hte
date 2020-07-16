import pandas as pd


def load_libsvm_to_dataframe(data_paths, feature_column_names=None,
                             keys=('train', 'vali',),
                             nfeatures=136, reindex_qid=True):
    from sklearn.datasets import load_svmlight_file
    ds_list = []
    for p in data_paths:
        X, y, qid = load_svmlight_file(p, n_features=nfeatures,
                                       query_id=True, zero_based=False)
        df = pd.DataFrame(X.toarray(), columns=feature_column_names)
        df['y'] = y
        df['qid'] = qid
        ds_list.append(df)
    ds = (pd.concat(ds_list, keys=keys, names=['partition', 'qdid'])
          .reset_index(level='partition'))
    if reindex_qid:
        ds['qid'] = ds.groupby(['partition', 'qid']).ngroup() + 1
    return ds


def normalize_datasets(data, order_by, binarize_label=True,
                       truncate_rank=10):
    if binarize_label:
        data['y'] = (data['y'] > 2).astype(int)
    data = data.sort_values(['qid', 'y'], ascending=[True, False])
    # remove lists that don't have relevant documents
    qids = data[data['y'] > 0]['qid'].unique()
    data = data[data['qid'].isin(qids)].copy()
    data['rank'] = data.groupby(order_by).cumcount()
    # truncate ranks
    data = data[data['rank'] < truncate_rank].copy()
    return data


def split_data_by_groups(data, groups, test_size):
    unique_queries = data[groups].drop_duplicates().reset_index(drop=True)
    test = unique_queries.sample(frac=test_size)
    train = unique_queries[~unique_queries.index.isin(test.index)]
    test = test.merge(data, how='left')
    train = train.merge(data, how='left')
    return train, test


def dump_dataframe_to_libsvm(data, feature_columns, label_column,
                             partition_column, query_id,
                             save_to_dir, prefix):
    from itertools import combinations
    from pathlib import Path
    from sklearn.datasets import dump_svmlight_file
    partition_names = data[partition_column].sort_values().unique()
    ncomb = len(partition_names)
    combs = list()
    for i in range(1, ncomb + 1):
        combs.extend(list(combinations(partition_names, i)))
    saveto = Path(save_to_dir)
    for comb in combs:
        partition_name = '_'.join(comb)
        partition = data[data[partition_column].isin(comb)].copy()
        partition[query_id] = partition.groupby(
            [partition_column, query_id]).ngroup() + 1
        partition = partition.sort_values(query_id)
        partition_saveto = saveto / f'{prefix}_{partition_name}.txt'
        print(f'Dumping {prefix} partition {partition_name} to'
              f' {partition_saveto}')
        dump_svmlight_file(X=partition[feature_columns].values,
                           y=partition[label_column].values,
                           query_id=partition[query_id].values,
                           f=partition_saveto,
                           zero_based=False)


def generate_ranks(features, labels, ranker):
    from counterfactual_evaluation.train_eval_opts import model_input_fn
    input_fn, iter_hook = model_input_fn(
        features, labels, batch_size=None, is_train=False)
    # predict
    pred_iter = ranker.predict(input_fn=input_fn, hooks=[iter_hook])
    pred_df = pd.DataFrame.from_records(pred_iter)
    pred_df = pred_df.stack().rename('prediction').rename_axis(
        index=['qid', 'item_seq'])
    label_df = pd.DataFrame(labels)
    label_df = label_df.stack().rename('label').rename_axis(
        index=['qid', 'item_seq'])
    ranking_df = pd.concat([pred_df, label_df], axis=1)
    # remove dummpy documents
    ranking_df = ranking_df[ranking_df['label'] != -1].reset_index()
    # rank by predictions
    ranking_df['pred_rank'] = ranking_df.groupby('qid')['prediction'].rank(
        method='first', ascending=False)
    return ranking_df
