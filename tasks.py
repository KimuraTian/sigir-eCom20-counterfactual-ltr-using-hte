import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from invoke import task
import tensorflow as tf

import counterfactual_evaluation as cfeval

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


@task(iterable=['data_paths'])
def normalize_and_split_raw_data(ctx, data_paths, save_to_dir):
    if not Path(save_to_dir).exists():
        Path(save_to_dir).mkdir(parents=True)
    data = cfeval.simulation_utils.load_libsvm_to_dataframe(
        data_paths, keys=['train', 'vali', 'test'])
    data = cfeval.simulation_utils.normalize_datasets(
        data, order_by=['partition', 'qid'])
    train_vali = data[data['partition'].isin(['train', 'vali'])]
    test = data[data['partition'] == 'test']
    # sample 1% data for simulating production ranker
    exp_train_vali, prod_train_vali = (
        cfeval.simulation_utils.split_data_by_groups(
            train_vali, ['partition', 'qid'], 0.01))
    # save splits
    feature_columns = list(range(136))
    cfeval.simulation_utils.dump_dataframe_to_libsvm(
        test, feature_columns, 'y', 'partition', 'qid', save_to_dir, 'sim')
    # production ranker
    cfeval.simulation_utils.dump_dataframe_to_libsvm(
        prod_train_vali, feature_columns, 'y', 'partition', 'qid',
        save_to_dir, 'sim_prod_ranker'
    )
    # experiment data
    exp_train_vali['qid'] = (exp_train_vali.groupby(['partition', 'qid'])
                             .ngroup() + 1)
    exp_train_vali = exp_train_vali.sort_values('qid')
    exp_train_vali_path = Path(save_to_dir) / 'sim_exp_train_vali.csv'
    exp_train_vali.to_csv(exp_train_vali_path, index=False)


@task(iterable=['data_paths'])
def train_and_serve_production_ranker(ctx, data_paths, tmp_sweep_model_dir,
                                      prod_ranker_dir, exp_train_vali_path,
                                      pred_ranking_dir,
                                      nfeatures, list_size):
    nfeatures = int(nfeatures)
    list_size = int(list_size)
    # sweep parameters on train and vali sets
    train_path, vali_path, train_vali_path = data_paths
    ((train_features, train_labels),
     (vali_features, vali_labels)) = cfeval.datasets.load_datasets(
        [train_path, vali_path],
        [nfeatures, nfeatures],
        [list_size, list_size]
    )
    learning_rate = 0.1
    batch_sizes = [32]
    regularizer = 'l2'
    regularizer_scales = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0,
                          5.0, 10.0, 20.0, 30.0]
    best_metric = 0
    best_epoch = 0
    best_batch_size = 0
    best_reg_scale = 0
    for bs in batch_sizes:
        for reg_scale in regularizer_scales:
            tf.compat.v1.logging.info(f'Tuning batch size: {bs}; reg_scale: '
                                      f'{reg_scale}')
            curr_model_dir = (Path(tmp_sweep_model_dir) /
                              'ProdRanker' /
                              f'params_{learning_rate}_{bs}_'
                              f'{regularizer}_{reg_scale}')
            metric, epoch = (
                cfeval.train_eval_opts.train_and_eval_with_early_stopping(
                    train_features, train_labels, vali_features, vali_labels,
                    curr_model_dir, epochs=1000,
                    train_validate_nfeatures=nfeatures,
                    learning_rate=learning_rate, batch_size=bs, nruns=1,
                    model_name='linear', regularizer=regularizer,
                    regularizer_scale=reg_scale, metric_name='metric/mrr'))
            tf.compat.v1.logging.info(f'Done tuning batch size: {bs}; '
                                      f'reg_scale: {reg_scale}; '
                                      f'metric values: {metric}; '
                                      f'epoch: {epoch}')
            if metric > best_metric:
                tf.compat.v1.logging.info(
                    f'Current best batch size: {best_batch_size}; '
                    f'best reg_scale: {best_reg_scale}; '
                    f'best metric values: {best_metric}; '
                    f'best epoch: {best_epoch}')
                best_metric = metric
                best_epoch = epoch
                best_batch_size = bs
                best_reg_scale = reg_scale
                tf.compat.v1.logging.info(
                    f'Updating to best batch size: {best_batch_size}; '
                    f'best reg_scale: {best_reg_scale}; '
                    f'best metric values: {best_metric}; '
                    f'best epoch: {best_epoch}')

    # re-train models with the best parameters on train and vali sets
    del train_features, train_labels, vali_features, vali_labels

    feature_columns = [str(x+1) for x in range(nfeatures)]
    train_vali_features, train_vali_labels = cfeval.datasets.load_libsvm_data(
        train_vali_path, feature_columns, nfeatures, list_size=list_size
    )
    train_vali_input_fn, train_vali_iter_hook = (
        cfeval.train_eval_opts.model_input_fn(
            train_vali_features, train_vali_labels,
            batch_size=best_batch_size, is_train=True)
    )
    hparams = cfeval.train_eval_opts.make_hparams(
        learning_rate=learning_rate,
        nfeatures=nfeatures, loss='pairwise_hinge_loss',
        train_weights_feature_name=None,
        eval_weights_feature_name=None)
    train_steps = int(train_vali_labels.shape[0] // best_batch_size + 1)
    prod_ranker = cfeval.train_eval_opts.get_estimator(
        hparams, model_dir=prod_ranker_dir,
        save_checkpoints_steps=train_steps,
        model_name='linear',
        regularizer='l2',
        regularizer_scale=best_reg_scale)
    tf.compat.v1.logging.info(
        f'Re-training production ranker with '
        f'{train_steps}*{best_epoch}={train_steps * best_epoch} steps')
    prod_ranker.train(input_fn=train_vali_input_fn,
                      hooks=[train_vali_iter_hook],
                      steps=train_steps * best_epoch)
    exp_train_vali = pd.read_csv(exp_train_vali_path)
    feature_columns_zero_based = [str(x) for x in range(nfeatures)]
    features, labels = cfeval.datasets.make_tfr_input(
        features=exp_train_vali[feature_columns_zero_based].values,
        y=exp_train_vali['y'].values, qid=exp_train_vali['qid'].values,
        feature_columns=feature_columns, list_size=list_size)
    pred_rankings = cfeval.simulation_utils.generate_ranks(
        features, labels, prod_ranker)
    exp_train_vali['pred_rank'] = pred_rankings['pred_rank'].values
    exp_train_vali['pred_score'] = pred_rankings['prediction'].values
    exp_train_vali.to_csv(pred_ranking_dir, index=False)


@task
def sweep_causal_forests(ctx, avg_clicks, nq, total_nqueries, fold,
                         model_dir, algorithm_name,
                         train_weights_feature_name=None,
                         eval_weights_feature_name=None,
                         max_train_steps=None,
                         train_list_size=10,
                         validate_list_size=10,
                         epochs=100,
                         train_validate_nfeatures=137,
                         loss='pairwise_hinge_loss',
                         learning_rate=0.1,
                         model_name='linear',
                         regularizer='l2',
                         batch_size=32,
                         reg_scale=0.1,
                         metric_name='metric/mrr'):
    print('Reading random swapping and LTR clicks')
    avg_clicks = int(avg_clicks)
    nq = int(nq)
    total_nqueries = int(total_nqueries)
    ltr_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    swap_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    print('Loading swapping query ids')
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{nq}.pkl',
              'rb') as f:
        swap_qids = pickle.load(f)
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{total_nqueries}.pkl',
              'rb') as f:
        total_qids = pickle.load(f)
    ltr_qids = np.setdiff1d(total_qids, swap_qids)
    print('Merging swapping and LTR clicks')
    exp_clicks = pd.concat([swap_clicks[swap_clicks['qid'].isin(swap_qids)],
                            ltr_clicks[ltr_clicks['qid'].isin(ltr_qids)]])
    exp_clicks = exp_clicks.sort_values(['list_id', 'swapped_rank'])
    exp_clicks = exp_clicks.rename(columns={'swapped_rank': 'exam_position'})
    exp_clicks['exam_position'] = exp_clicks['exam_position'].astype(int)
    print('Reading propensity estimations')
    causal_forests_estimations = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_causal_forests_ltr_test_results_{avg_clicks}_{nq}.csv')
    curr_causal_forests_estimations = causal_forests_estimations[
        ['partition', 'qd_id', 'treatment_rank', 'tau_pred']].rename(
        columns={'treatment_rank': 'exam_position'})
    curr_causal_forests_estimations['exam_position'] = (
            curr_causal_forests_estimations['exam_position'] + 1)
    print('Merging estimations with clicks')
    exp_clicks = exp_clicks.merge(curr_causal_forests_estimations, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])
    exp_clicks = exp_clicks.fillna(0.0)
    observed_impression = (exp_clicks.groupby(
        ['partition', 'qd_id', 'exam_position'])['click'].mean()
                           .rename('ctr@k').reset_index())
    exp_clicks = exp_clicks.merge(observed_impression, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])

    exp_clicks['corrected_ctr'] = exp_clicks['ctr@k'] + exp_clicks['tau_pred']

    exp_clicks.loc[exp_clicks['corrected_ctr'] > 1, 'corrected_ctr'] = 1
    exp_clicks.loc[exp_clicks['corrected_ctr'] < 0, 'corrected_ctr'] = 0

    exp_clicks['binomial_click'] = np.random.binomial(1, exp_clicks[
        'corrected_ctr'].values)
    exp_ltr_features = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_features.csv')
    exp_clicks_features = exp_clicks.merge(exp_ltr_features,
                                           on=['partition', 'qd_id'],
                                           how='left')
    exp_clicks_features = exp_clicks_features.sort_values(
        ['list_id', 'exam_position'])
    # train data
    sample_train_rankings = exp_clicks_features[
        exp_clicks_features['partition'] == 'train'].copy()
    sample_train_rankings['list_id'] = sample_train_rankings.groupby(
        'list_id').ngroup() + 1
    sample_train_rankings = sample_train_rankings.sort_values(
        ['list_id', 'exam_position'])
    # vali data
    sample_vali_rankings = exp_clicks_features[
        exp_clicks_features['partition'] == 'vali'].copy()
    sample_vali_rankings['list_id'] = sample_vali_rankings.groupby(
        'list_id').ngroup() + 1
    sample_vali_rankings = sample_vali_rankings.sort_values(
        ['list_id', 'exam_position'])
    del exp_clicks, exp_ltr_features, exp_clicks_features
    feature_columns = [str(x) for x in range(train_validate_nfeatures - 1)]
    feature_weights_columns = [
        str(x + 1) for x in range(train_validate_nfeatures)]
    train_features, train_labels = cfeval.datasets.make_tfr_input(
        sample_train_rankings[[*feature_columns, 'binomial_click']].values,
        sample_train_rankings['binomial_click'].values,
        sample_train_rankings['list_id'].values,
        feature_weights_columns,
        list_size=train_list_size)

    vali_features, vali_labels = cfeval.datasets.make_tfr_input(
        sample_vali_rankings[[*feature_columns, 'binomial_click']].values,
        sample_vali_rankings['binomial_click'].values,
        sample_vali_rankings['list_id'].values,
        feature_weights_columns,
        list_size=validate_list_size)

    curr_model_dir = (Path(model_dir) /
                      algorithm_name /
                      f'avg_clicks_{avg_clicks}_nq_{nq}' /
                      f'params_{learning_rate}_{batch_size}_'
                      f'{regularizer}_{reg_scale}')
    metric, epoch = cfeval.train_eval_opts.train_and_eval_with_early_stopping(
        train_features, train_labels, vali_features, vali_labels,
        curr_model_dir,
        train_weights_feature_name, eval_weights_feature_name,
        max_train_steps, epochs,
        train_validate_nfeatures, loss, learning_rate, batch_size, nruns=1,
        model_name=model_name, regularizer=regularizer,
        regularizer_scale=reg_scale, metric_name=metric_name)
    eval_results = pd.DataFrame({'algorithm': [algorithm_name],
                                 metric_name: [metric],
                                 'epoch': [epoch],
                                 'batch_size': [batch_size],
                                 'reg_scale': [reg_scale]})
    print(f'Done sweeping algorithm {algorithm_name} with {metric_name}: '
          f'{metric}. epoch: {epoch}, batch_size: {batch_size}, reg_scale: '
          f'{reg_scale}')
    eval_results.to_csv(curr_model_dir / 'sweep_results.csv', index=False)


@task
def sweep_x_learner(ctx, avg_clicks, nq, total_nqueries, fold,
                    model_dir, algorithm_name,
                    train_weights_feature_name=None,
                    eval_weights_feature_name=None,
                    max_train_steps=None,
                    train_list_size=10,
                    validate_list_size=10,
                    epochs=100,
                    train_validate_nfeatures=137,
                    loss='pairwise_hinge_loss',
                    learning_rate=0.1,
                    model_name='linear',
                    regularizer='l2',
                    batch_size=32,
                    reg_scale=0.1,
                    metric_name='metric/mrr'):
    print('Reading random swapping and LTR clicks')
    avg_clicks = int(avg_clicks)
    nq = int(nq)
    total_nqueries = int(total_nqueries)
    ltr_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    swap_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    print('Loading swapping query ids')
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{nq}.pkl',
              'rb') as f:
        swap_qids = pickle.load(f)
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{total_nqueries}.pkl',
              'rb') as f:
        total_qids = pickle.load(f)
    ltr_qids = np.setdiff1d(total_qids, swap_qids)
    print('Merging swapping and LTR clicks')
    exp_clicks = pd.concat([swap_clicks[swap_clicks['qid'].isin(swap_qids)],
                            ltr_clicks[ltr_clicks['qid'].isin(ltr_qids)]])
    exp_clicks = exp_clicks.sort_values(['list_id', 'swapped_rank'])
    exp_clicks = exp_clicks.rename(columns={'swapped_rank': 'exam_position'})
    exp_clicks['exam_position'] = exp_clicks['exam_position'].astype(int)
    print('Reading propensity estimations')
    x_learner_estimations = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_xlearner_ltr_test_results_{avg_clicks}_{nq}.csv')
    curr_x_learner_estimations = x_learner_estimations[
        ['partition', 'qd_id', 'treatment_rank', 'tau_pred']].rename(
        columns={'treatment_rank': 'exam_position'})
    curr_x_learner_estimations['exam_position'] = (
            curr_x_learner_estimations['exam_position'] + 1)
    print('Merging estimations with clicks')
    exp_clicks = exp_clicks.merge(curr_x_learner_estimations, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])
    exp_clicks = exp_clicks.fillna(0.0)
    observed_impression = (exp_clicks.groupby(
        ['partition', 'qd_id', 'exam_position'])['click'].mean()
                           .rename('ctr@k').reset_index())
    exp_clicks = exp_clicks.merge(observed_impression, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])

    exp_clicks['corrected_ctr'] = exp_clicks['ctr@k'] + exp_clicks['tau_pred']

    exp_clicks.loc[exp_clicks['corrected_ctr'] > 1, 'corrected_ctr'] = 1
    exp_clicks.loc[exp_clicks['corrected_ctr'] < 0, 'corrected_ctr'] = 0

    exp_clicks['binomial_click'] = np.random.binomial(1, exp_clicks[
        'corrected_ctr'].values)
    exp_ltr_features = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_features.csv')
    exp_clicks_features = exp_clicks.merge(exp_ltr_features,
                                           on=['partition', 'qd_id'],
                                           how='left')
    exp_clicks_features = exp_clicks_features.sort_values(
        ['list_id', 'exam_position'])
    # train data
    sample_train_rankings = exp_clicks_features[
        exp_clicks_features['partition'] == 'train'].copy()
    sample_train_rankings['list_id'] = sample_train_rankings.groupby(
        'list_id').ngroup() + 1
    sample_train_rankings = sample_train_rankings.sort_values(
        ['list_id', 'exam_position'])
    # vali data
    sample_vali_rankings = exp_clicks_features[
        exp_clicks_features['partition'] == 'vali'].copy()
    sample_vali_rankings['list_id'] = sample_vali_rankings.groupby(
        'list_id').ngroup() + 1
    sample_vali_rankings = sample_vali_rankings.sort_values(
        ['list_id', 'exam_position'])
    del exp_clicks, exp_ltr_features, exp_clicks_features
    feature_columns = [str(x) for x in range(train_validate_nfeatures - 1)]
    feature_weights_columns = [
        str(x + 1) for x in range(train_validate_nfeatures)]
    train_features, train_labels = cfeval.datasets.make_tfr_input(
        sample_train_rankings[[*feature_columns, 'binomial_click']].values,
        sample_train_rankings['binomial_click'].values,
        sample_train_rankings['list_id'].values,
        feature_weights_columns,
        list_size=train_list_size)

    vali_features, vali_labels = cfeval.datasets.make_tfr_input(
        sample_vali_rankings[[*feature_columns, 'binomial_click']].values,
        sample_vali_rankings['binomial_click'].values,
        sample_vali_rankings['list_id'].values,
        feature_weights_columns,
        list_size=validate_list_size)

    curr_model_dir = (Path(model_dir) /
                      algorithm_name /
                      f'avg_clicks_{avg_clicks}_nq_{nq}' /
                      f'params_{learning_rate}_{batch_size}_'
                      f'{regularizer}_{reg_scale}')
    metric, epoch = cfeval.train_eval_opts.train_and_eval_with_early_stopping(
        train_features, train_labels, vali_features, vali_labels,
        curr_model_dir,
        train_weights_feature_name, eval_weights_feature_name,
        max_train_steps, epochs,
        train_validate_nfeatures, loss, learning_rate, batch_size, nruns=1,
        model_name=model_name, regularizer=regularizer,
        regularizer_scale=reg_scale, metric_name=metric_name)
    eval_results = pd.DataFrame({'algorithm': [algorithm_name],
                                 metric_name: [metric],
                                 'epoch': [epoch],
                                 'batch_size': [batch_size],
                                 'reg_scale': [reg_scale]})
    print(f'Done sweeping algorithm {algorithm_name} with {metric_name}: '
          f'{metric}. epoch: {epoch}, batch_size: {batch_size}, reg_scale: '
          f'{reg_scale}')
    eval_results.to_csv(curr_model_dir / 'sweep_results.csv', index=False)


@task
def sweep_cpbm_ltr(ctx, avg_clicks, nq, total_nqueries, fold,
                   model_dir, algorithm_name,
                   train_weights_feature_name=None,
                   eval_weights_feature_name=None,
                   max_train_steps=None,
                   train_list_size=10,
                   validate_list_size=10,
                   epochs=100,
                   train_validate_nfeatures=137,
                   loss='pairwise_hinge_loss',
                   learning_rate=0.1,
                   model_name='linear',
                   regularizer='l2',
                   batch_size=32,
                   reg_scale=0.1,
                   metric_name='metric/mrr'):
    print('Reading random swapping and LTR clicks')
    avg_clicks = int(avg_clicks)
    nq = int(nq)
    total_nqueries = int(total_nqueries)
    ltr_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    swap_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    print('Loading swapping query ids')
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{nq}.pkl',
              'rb') as f:
        swap_qids = pickle.load(f)
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{total_nqueries}.pkl',
              'rb') as f:
        total_qids = pickle.load(f)
    ltr_qids = np.setdiff1d(total_qids, swap_qids)
    print('Merging swapping and LTR clicks')
    exp_clicks = pd.concat([swap_clicks[swap_clicks['qid'].isin(swap_qids)],
                            ltr_clicks[ltr_clicks['qid'].isin(ltr_qids)]])
    exp_clicks = exp_clicks.sort_values(['list_id', 'swapped_rank'])
    exp_clicks = exp_clicks.rename(columns={'swapped_rank': 'exam_position'})
    exp_clicks['exam_position'] = exp_clicks['exam_position'].astype(int)
    print('Reading propensity estimations')
    cpbm_estimations = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_{algorithm_name}_test_results.csv')
    curr_cpbm_estimations = cpbm_estimations[
        (cpbm_estimations['nqueries'] == nq) &
        (cpbm_estimations['avg_clicks'] == avg_clicks)]
    curr_cpbm_estimations = curr_cpbm_estimations.drop(
        ['avg_clicks', 'nqueries'], axis=1)
    print('Merging estimations with clicks')
    exp_clicks = exp_clicks.merge(curr_cpbm_estimations, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])
    exp_clicks = exp_clicks.fillna(1.0)
    exp_clicks.loc[
        exp_clicks['click'] == 0, 'inverse_cpbm_propensity_ratio_hat'] = 0
    print('Merging with features')
    exp_ltr_features = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_features.csv')
    exp_clicks_features = exp_clicks.merge(exp_ltr_features,
                                           on=['partition', 'qd_id'],
                                           how='left')
    exp_clicks_features = exp_clicks_features.sort_values(
        ['list_id', 'exam_position'])
    # train data
    sample_train_rankings = exp_clicks_features[
        exp_clicks_features['partition'] == 'train'].copy()
    sample_train_rankings['list_id'] = sample_train_rankings.groupby(
        'list_id').ngroup() + 1
    sample_train_rankings = sample_train_rankings.sort_values(
        ['list_id', 'exam_position'])
    # vali data
    sample_vali_rankings = exp_clicks_features[
        exp_clicks_features['partition'] == 'vali'].copy()
    sample_vali_rankings['list_id'] = sample_vali_rankings.groupby(
        'list_id').ngroup() + 1
    sample_vali_rankings = sample_vali_rankings.sort_values(
        ['list_id', 'exam_position'])
    del exp_clicks, exp_ltr_features, exp_clicks_features, cpbm_estimations
    feature_columns = [str(x) for x in range(train_validate_nfeatures - 1)]
    feature_weights_columns = [
        str(x + 1) for x in range(train_validate_nfeatures)]
    train_features, train_labels = cfeval.datasets.make_tfr_input(
        sample_train_rankings[[*feature_columns, 'inverse_cpbm_propensity_ratio_hat']].values,
        sample_train_rankings['click'].values,
        sample_train_rankings['list_id'].values,
        feature_weights_columns,
        list_size=train_list_size)

    vali_features, vali_labels = cfeval.datasets.make_tfr_input(
        sample_vali_rankings[[*feature_columns, 'inverse_cpbm_propensity_ratio_hat']].values,
        sample_vali_rankings['click'].values,
        sample_vali_rankings['list_id'].values,
        feature_weights_columns,
        list_size=validate_list_size)

    curr_model_dir = (Path(model_dir) /
                      algorithm_name /
                      f'avg_clicks_{avg_clicks}_nq_{nq}' /
                      f'params_{learning_rate}_{batch_size}_'
                      f'{regularizer}_{reg_scale}')
    del sample_train_rankings, sample_vali_rankings
    metric, epoch = cfeval.train_eval_opts.train_and_eval_with_early_stopping(
        train_features, train_labels, vali_features, vali_labels,
        curr_model_dir,
        train_weights_feature_name, eval_weights_feature_name,
        max_train_steps, epochs,
        train_validate_nfeatures, loss, learning_rate, batch_size, nruns=1,
        model_name=model_name, regularizer=regularizer,
        regularizer_scale=reg_scale, metric_name=metric_name)
    eval_results = pd.DataFrame({'algorithm': [algorithm_name],
                                 metric_name: [metric],
                                 'epoch': [epoch],
                                 'batch_size': [batch_size],
                                 'reg_scale': [reg_scale]})
    print(f'Done sweeping algorithm {algorithm_name} with {metric_name}: '
          f'{metric}. epoch: {epoch}, batch_size: {batch_size}, reg_scale: '
          f'{reg_scale}')
    eval_results.to_csv(curr_model_dir / 'sweep_results.csv', index=False)


@task
def train_and_test_causal_forests_ltr(ctx, avg_clicks, nq, total_nqueries, fold,
                                      model_dir, eval_result_dir,
                                      algorithm_name, test_data_path,
                                      train_weights_feature_name=None,
                                      eval_weights_feature_name=None,
                                      train_list_size=10,
                                      test_list_size=10,
                                      train_steps=None,
                                      epochs=100,
                                      train_nfeatures=137,
                                      test_nfeatures=136,
                                      loss='pairwise_hinge_loss',
                                      learning_rate=0.1,
                                      model_name='linear',
                                      regularizer='l2',
                                      batch_size=32,
                                      reg_scale=0.1,
                                      nruns=1):
    print('Reading random swapping and LTR clicks')
    avg_clicks = int(avg_clicks)
    nq = int(nq)
    total_nqueries = int(total_nqueries)
    ltr_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    swap_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    print('Loading swapping query ids')
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{nq}.pkl',
              'rb') as f:
        swap_qids = pickle.load(f)
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{total_nqueries}.pkl',
              'rb') as f:
        total_qids = pickle.load(f)
    ltr_qids = np.setdiff1d(total_qids, swap_qids)
    print('Merging swapping and LTR clicks')
    exp_clicks = pd.concat([swap_clicks[swap_clicks['qid'].isin(swap_qids)],
                            ltr_clicks[ltr_clicks['qid'].isin(ltr_qids)]])
    exp_clicks = exp_clicks.sort_values(['list_id', 'swapped_rank'])
    exp_clicks = exp_clicks.rename(columns={'swapped_rank': 'exam_position'})
    exp_clicks['exam_position'] = exp_clicks['exam_position'].astype(int)
    print('Reading propensity estimations')
    causal_forests_estimations = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_causal_forests_ltr_test_results_{avg_clicks}_{nq}.csv')
    curr_causal_forests_estimations = causal_forests_estimations[
        ['partition', 'qd_id', 'treatment_rank', 'tau_pred']].rename(
        columns={'treatment_rank': 'exam_position'})
    curr_causal_forests_estimations['exam_position'] = (
            curr_causal_forests_estimations['exam_position'] + 1)
    print('Merging estimations with clicks')
    exp_clicks = exp_clicks.merge(curr_causal_forests_estimations, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])
    del causal_forests_estimations
    exp_clicks = exp_clicks.fillna(0.0)
    observed_impression = (exp_clicks.groupby(
        ['partition', 'qd_id', 'exam_position'])['click'].mean()
                           .rename('ctr@k').reset_index())
    exp_clicks = exp_clicks.merge(observed_impression, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])

    exp_clicks['corrected_ctr'] = exp_clicks['ctr@k'] + exp_clicks['tau_pred']

    exp_clicks.loc[exp_clicks['corrected_ctr'] > 1, 'corrected_ctr'] = 1
    exp_clicks.loc[exp_clicks['corrected_ctr'] < 0, 'corrected_ctr'] = 0

    exp_clicks['binomial_click'] = np.random.binomial(1, exp_clicks[
        'corrected_ctr'].values)
    exp_ltr_features = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_features.csv')
    exp_clicks_features = exp_clicks.merge(exp_ltr_features,
                                           on=['partition', 'qd_id'],
                                           how='left')
    exp_clicks_features = exp_clicks_features.sort_values(
        ['list_id', 'exam_position'])
    feature_columns = [str(x) for x in range(train_nfeatures - 1)]
    feature_weights_columns = [
        str(x + 1) for x in range(train_nfeatures)]
    train_features, train_labels = cfeval.datasets.make_tfr_input(
        exp_clicks_features[[*feature_columns, 'binomial_click']].values,
        exp_clicks_features['binomial_click'].values,
        exp_clicks_features['list_id'].values,
        feature_weights_columns,
        list_size=train_list_size)
    del exp_clicks, exp_ltr_features, exp_clicks_features
    test_feature_columns = [str(x + 1) for x in range(test_nfeatures)]
    test_features, test_labels = cfeval.datasets.load_libsvm_data(
        test_data_path, test_feature_columns, test_nfeatures,
        list_size=test_list_size)
    model_dir = (Path(model_dir) / algorithm_name /
                 f'avg_clicks_{avg_clicks}_nq_{nq}')
    eval_result_dir = (Path(eval_result_dir) / algorithm_name /
                       f'avg_clicks_{avg_clicks}_nq_{nq}')
    cfeval.train_eval_opts.train_and_test_multiple_runs(
        train_features, train_labels,
        test_features, test_labels,
        model_dir, eval_result_dir, algorithm_name,
        train_weights_feature_name, eval_weights_feature_name,
        train_steps, epochs, train_nfeatures, loss, learning_rate, batch_size,
        nruns, model_name, regularizer, reg_scale)


@task
def train_and_test_cpbm_ltr(ctx, avg_clicks, nq, total_nqueries, fold,
                            model_dir, eval_result_dir,
                            algorithm_name, test_data_path,
                            train_weights_feature_name=None,
                            eval_weights_feature_name=None,
                            train_list_size=10,
                            test_list_size=10,
                            train_steps=None,
                            epochs=100,
                            train_nfeatures=137,
                            test_nfeatures=136,
                            loss='pairwise_hinge_loss',
                            learning_rate=0.1,
                            model_name='linear',
                            regularizer='l2',
                            batch_size=32,
                            reg_scale=0.1,
                            nruns=1):
    print('Reading random swapping and LTR clicks')
    avg_clicks = int(avg_clicks)
    nq = int(nq)
    total_nqueries = int(total_nqueries)
    ltr_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    swap_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    print('Loading swapping query ids')
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{nq}.pkl',
              'rb') as f:
        swap_qids = pickle.load(f)
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{total_nqueries}.pkl',
              'rb') as f:
        total_qids = pickle.load(f)
    ltr_qids = np.setdiff1d(total_qids, swap_qids)
    print('Merging swapping and LTR clicks')
    exp_clicks = pd.concat([swap_clicks[swap_clicks['qid'].isin(swap_qids)],
                            ltr_clicks[ltr_clicks['qid'].isin(ltr_qids)]])
    exp_clicks = exp_clicks.sort_values(['list_id', 'swapped_rank'])
    exp_clicks = exp_clicks.rename(columns={'swapped_rank': 'exam_position'})
    exp_clicks['exam_position'] = exp_clicks['exam_position'].astype(int)
    print('Reading propensity estimations')
    cpbm_estimations = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_cpbm_ltr_test_results.csv')
    curr_cpbm_estimations = cpbm_estimations[
        (cpbm_estimations['nqueries'] == nq) &
        (cpbm_estimations['avg_clicks'] == avg_clicks)]
    curr_cpbm_estimations = curr_cpbm_estimations.drop(
        ['avg_clicks', 'nqueries'], axis=1)
    print('Merging estimations with clicks')
    exp_clicks = exp_clicks.merge(curr_cpbm_estimations, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])
    exp_clicks = exp_clicks.fillna(1.0)
    exp_clicks.loc[
        exp_clicks['click'] == 0, 'inverse_cpbm_propensity_ratio_hat'] = 0
    print('Merging with features')
    exp_ltr_features = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_features.csv')
    exp_clicks_features = exp_clicks.merge(exp_ltr_features,
                                           on=['partition', 'qd_id'],
                                           how='left')
    exp_clicks_features = exp_clicks_features.sort_values(
        ['list_id', 'exam_position'])
    del exp_clicks, exp_ltr_features, cpbm_estimations
    feature_columns = [str(x) for x in range(train_nfeatures - 1)]
    feature_weights_columns = [
        str(x + 1) for x in range(train_nfeatures)]
    train_features, train_labels = cfeval.datasets.make_tfr_input(
        exp_clicks_features[[*feature_columns, 'inverse_cpbm_propensity_ratio_hat']].values,
        exp_clicks_features['click'].values,
        exp_clicks_features['list_id'].values,
        feature_weights_columns,
        list_size=train_list_size)
    del exp_clicks_features
    test_feature_columns = [str(x + 1) for x in range(test_nfeatures)]
    test_features, test_labels = cfeval.datasets.load_libsvm_data(
        test_data_path, test_feature_columns, test_nfeatures,
        list_size=test_list_size)
    model_dir = (Path(model_dir) / algorithm_name /
                 f'avg_clicks_{avg_clicks}_nq_{nq}')
    eval_result_dir = (Path(eval_result_dir) / algorithm_name /
                       f'avg_clicks_{avg_clicks}_nq_{nq}')
    cfeval.train_eval_opts.train_and_test_multiple_runs(
        train_features, train_labels,
        test_features, test_labels,
        model_dir, eval_result_dir, algorithm_name,
        train_weights_feature_name, eval_weights_feature_name,
        train_steps, epochs, train_nfeatures, loss, learning_rate, batch_size,
        nruns, model_name, regularizer, reg_scale)


@task
def train_and_test_x_learner_ltr(ctx, avg_clicks, nq, total_nqueries, fold,
                                 model_dir, eval_result_dir,
                                 algorithm_name, test_data_path,
                                 train_weights_feature_name=None,
                                 eval_weights_feature_name=None,
                                 train_list_size=10,
                                 test_list_size=10,
                                 train_steps=None,
                                 epochs=100,
                                 train_nfeatures=137,
                                 test_nfeatures=136,
                                 loss='pairwise_hinge_loss',
                                 learning_rate=0.1,
                                 model_name='linear',
                                 regularizer='l2',
                                 batch_size=32,
                                 reg_scale=0.1,
                                 nruns=1):
    print('Reading random swapping and LTR clicks')
    avg_clicks = int(avg_clicks)
    nq = int(nq)
    total_nqueries = int(total_nqueries)
    ltr_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    swap_clicks = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_train_vali_clicks_avg_clicks_{avg_clicks}.csv')
    print('Loading swapping query ids')
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{nq}.pkl',
              'rb') as f:
        swap_qids = pickle.load(f)
    with open(f'build/simulation/{fold}/sim_exp_swap_query_ids_{total_nqueries}.pkl',
              'rb') as f:
        total_qids = pickle.load(f)
    ltr_qids = np.setdiff1d(total_qids, swap_qids)
    print('Merging swapping and LTR clicks')
    exp_clicks = pd.concat([swap_clicks[swap_clicks['qid'].isin(swap_qids)],
                            ltr_clicks[ltr_clicks['qid'].isin(ltr_qids)]])
    exp_clicks = exp_clicks.sort_values(['list_id', 'swapped_rank'])
    exp_clicks = exp_clicks.rename(columns={'swapped_rank': 'exam_position'})
    exp_clicks['exam_position'] = exp_clicks['exam_position'].astype(int)
    print('Reading propensity estimations')
    xlearner_estimations = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_swap_xlearner_ltr_test_results_{avg_clicks}_{nq}.csv')
    curr_xlearner_estimations = xlearner_estimations[
        ['partition', 'qd_id', 'treatment_rank', 'tau_pred']].rename(
        columns={'treatment_rank': 'exam_position'})
    curr_xlearner_estimations['exam_position'] = (
            curr_xlearner_estimations['exam_position'] + 1)
    print('Merging estimations with clicks')
    exp_clicks = exp_clicks.merge(curr_xlearner_estimations, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])
    del xlearner_estimations
    exp_clicks = exp_clicks.fillna(0.0)
    observed_impression = (exp_clicks.groupby(
        ['partition', 'qd_id', 'exam_position'])['click'].mean()
                           .rename('ctr@k').reset_index())
    exp_clicks = exp_clicks.merge(observed_impression, how='left',
                                  on=['partition', 'qd_id', 'exam_position'])

    exp_clicks['corrected_ctr'] = exp_clicks['ctr@k'] + exp_clicks['tau_pred']

    exp_clicks.loc[exp_clicks['corrected_ctr'] > 1, 'corrected_ctr'] = 1
    exp_clicks.loc[exp_clicks['corrected_ctr'] < 0, 'corrected_ctr'] = 0

    exp_clicks['binomial_click'] = np.random.binomial(1, exp_clicks[
        'corrected_ctr'].values)
    exp_ltr_features = pd.read_csv(
        f'build/simulation/{fold}/sim_exp_ltr_features.csv')
    exp_clicks_features = exp_clicks.merge(exp_ltr_features,
                                           on=['partition', 'qd_id'],
                                           how='left')
    exp_clicks_features = exp_clicks_features.sort_values(
        ['list_id', 'exam_position'])
    feature_columns = [str(x) for x in range(train_nfeatures - 1)]
    feature_weights_columns = [
        str(x + 1) for x in range(train_nfeatures)]
    train_features, train_labels = cfeval.datasets.make_tfr_input(
        exp_clicks_features[[*feature_columns, 'binomial_click']].values,
        exp_clicks_features['binomial_click'].values,
        exp_clicks_features['list_id'].values,
        feature_weights_columns,
        list_size=train_list_size)
    del exp_clicks, exp_ltr_features, exp_clicks_features
    test_feature_columns = [str(x + 1) for x in range(test_nfeatures)]
    test_features, test_labels = cfeval.datasets.load_libsvm_data(
        test_data_path, test_feature_columns, test_nfeatures,
        list_size=test_list_size)
    model_dir = (Path(model_dir) / algorithm_name /
                 f'avg_clicks_{avg_clicks}_nq_{nq}')
    eval_result_dir = (Path(eval_result_dir) / algorithm_name /
                       f'avg_clicks_{avg_clicks}_nq_{nq}')
    cfeval.train_eval_opts.train_and_test_multiple_runs(
        train_features, train_labels,
        test_features, test_labels,
        model_dir, eval_result_dir, algorithm_name,
        train_weights_feature_name, eval_weights_feature_name,
        train_steps, epochs, train_nfeatures, loss, learning_rate, batch_size,
        nruns, model_name, regularizer, reg_scale)


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()
