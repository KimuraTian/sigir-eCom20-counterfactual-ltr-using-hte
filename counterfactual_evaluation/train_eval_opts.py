"""
Train and Evaluation options
"""
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import tensorflow as tf
import tensorflow_ranking as tfr

from .datasets import load_libsvm_data
from .metrics import nDCG
from .metrics import compute_reciprocal_rank


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    def __init__(self):
        super().__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        del coord
        self.iterator_initializer_fn(session)


def model_input_fn(features, labels, batch_size, is_train=True):
    """
    Training input function for tf.Estimator

    Args:
        features (dict): The feature dictionary that maps feature name to
            training data with a shape of [num_examples, list_size, 1]
        labels (numpy.array): The labels with a shape of
            [num_examples, list_size]
        batch_size (int, or None): The batch size
        is_train (bool): The mode key for the input data. If True,
            the dataset will be shuffled, repeated and set with a batch_size.
            If False, `model_input_fn` will return an iterator that iterates
            through once each example in the data.

    Returns:
        function: The training input function for tf.Estimator
        tf.estimator.SessionRunHook: The iterator initializer hook used for
            initializing input data iterator
    """

    iter_init_hook = IteratorInitializerHook()

    def _model_input_fn():
        # feature placeholder
        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in features.items()
        }
        # label placeholder
        labels_placeholder = tf.compat.v1.placeholder(labels.dtype,
                                                      labels.shape)
        if is_train:
            dataset = tf.data.Dataset.from_tensor_slices(
                (features_placeholder, labels_placeholder))
            dataset = dataset.shuffle(1024).repeat().batch(batch_size)
        else:
            dataset = tf.data.Dataset.from_tensors(
                (features_placeholder, labels_placeholder))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        # feed_dict
        feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k]
             for k in features_placeholder})
        iter_init_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _model_input_fn, iter_init_hook


def example_feature_columns(nfeatures=55):
    """
    Returns an OrderedDict that maps example feature name to feature columns
    It assumes that the last column is the weights column.
    """
    feature_names = [f'{i + 1}' for i in range(nfeatures - 1)]
    return OrderedDict((name,
                        tf.compat.v1.feature_column.numeric_column(
                            name, shape=(1,), default_value=0.0))
                       for name in feature_names)


def make_transform_fn(train_weights_feature_name, nfeatures, model_name='nn'):
    example_feature_cols = example_feature_columns(nfeatures)
    if train_weights_feature_name is None:
        example_feature_cols = example_feature_columns(nfeatures + 1)

    def _transform_fn(features, mode):
        """Defines transform_fn."""
        context_features, example_features = tfr.feature.encode_listwise_features(
            features=features,
            context_feature_columns=None,
            example_feature_columns=example_feature_cols,
            mode=mode,
            scope="transform_layer")
        return context_features, example_features

    if model_name == 'nn':
        return _transform_fn
    else:
        return None


def make_score_fn(model_name='nn', regularizer='l2', regularizer_scale=0.01):
    """Returns a scoring function to build `EstimatorSpec`."""
    def _unpack_group_features(group_features,
                               train_weights_feature_name,
                               nfeatures):
        example_feature_cols = example_feature_columns(nfeatures)
        if train_weights_feature_name is None:
            example_feature_cols = example_feature_columns(nfeatures + 1)
        example_input = [
            tf.compat.v1.layers.flatten(group_features[name])
            for name in example_feature_cols]
        input_layer = tf.compat.v1.concat(example_input, 1)
        return input_layer

    def _create_regularizer_fn(reg_name, reg_scale):
        if reg_name not in ('l1', 'l2'):
            raise ValueError(f'{regularizer} regularizer not supported')
        if reg_name == 'l2':
            return tf.contrib.layers.l2_regularizer(reg_scale)
        if reg_name == 'l1':
            return tf.contrib.layers.l1_regularizer(reg_scale)

    def _linear_score_fn(context_features, group_features, mode, params,
                         config):
        """Defines linear function to score a document."""
        del config
        input_layer = _unpack_group_features(group_features,
                                             params.train_weights_feature_name,
                                             params.nfeatures)
        reg_fn = _create_regularizer_fn(regularizer, regularizer_scale)
        logits = tf.compat.v1.layers.dense(input_layer, units=1,
                                           kernel_regularizer=reg_fn)
        return logits

    def _nn_score_fn(context_features, group_features, mode, params, config):
        """Defines neural network to score a document."""
        del config
        input_layer = _unpack_group_features(group_features,
                                             params.train_weights_feature_name,
                                             params.nfeatures)
        reg_fn = _create_regularizer_fn(regularizer, regularizer_scale)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.compat.v1.layers.batch_normalization(
            input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in ['256', '128', '64']):
            cur_layer = tf.compat.v1.layers.dense(
                cur_layer, units=layer_width, kernel_regularizer=reg_fn)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training)
            cur_layer = tf.compat.v1.nn.relu(cur_layer)
            cur_layer = tf.compat.v1.layers.dropout(
                cur_layer, rate=0.2, training=is_training)
        logits = tf.compat.v1.layers.dense(
            cur_layer, units=1, kernel_regularizer=reg_fn)
        return logits

    if model_name == 'linear':
        return _linear_score_fn

    return _nn_score_fn


def make_regularized_loss_fn(
        loss_keys, loss_weights=None,
        weights_feature_name=None,
        lambda_weight=None,
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None,
        seed=None,
        extra_args=None):
    """A wrapper function of tensorflow_ranking.losses.make_loss_fn"""

    def _loss_fn(labels, logits, features):
        ranking_loss_fn = tfr.losses.make_loss_fn(
            loss_keys=loss_keys,
            loss_weights=loss_weights,
            weights_feature_name=weights_feature_name,
            lambda_weight=lambda_weight,
            reduction=reduction,
            name=name,
            seed=seed,
            extra_args=extra_args
        )
        regularization_losses = tf.compat.v1.losses.get_regularization_losses()
        regularization_loss = tf.compat.v1.reduce_sum(regularization_losses)
        loss = ranking_loss_fn(labels, logits, features)
        return loss + regularization_loss
    return _loss_fn


def make_hparams(learning_rate=0.1, nfeatures=136, loss='pairwise_hinge_loss',
                 train_weights_feature_name=None,
                 eval_weights_feature_name=None):
    """
    Sets up a hype-parameter object for models

    Args:
        learning_rate (float): The learning rate.
        nfeatures (int): The number of features.
        loss (str): The name of pairwise loss.
        train_weights_feature_name (str or None): The name of weights feature
            for training.
        eval_weights_feature_name (str or None): The name of weights feature
            for evaluation.

    Returns:
        tf.contrib.training.HParams: The hyper-parameter object.
    """
    return tf.contrib.training.HParams(
        learning_rate=learning_rate, nfeatures=nfeatures, loss=loss,
        train_weights_feature_name=train_weights_feature_name,
        eval_weights_feature_name=eval_weights_feature_name,
    )


def eval_metric_fns(weights_feature_name='55'):
    """
    Returns a function for weighted metrics
    Args:
        weights_feature_name (str or None): The name of weight feature. If
            None, compute normal metrics without weights.

    Returns:
        dict: A dict that maps metric name to tfr.metrics
    """
    metric_fns = {}
    metric_fns.update({
        f'metric/ndcg@{topn}':
        tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn,
            weights_feature_name=weights_feature_name)
        for topn in [1, 2, 3, 4, 5, 10, 15, 20, 25]})
    metric_fns.update({
        f'metric/mrr':
        tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.MRR,
            weights_feature_name=weights_feature_name)})
    metric_fns.update({
        f'metric/arp':
        tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.ARP,
            weights_feature_name=weights_feature_name)})
    return metric_fns


def get_estimator(hparams, model_dir, save_checkpoints_steps=None,
                  model_name='nn', regularizer='l2', regularizer_scale=0.01):

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.compat.v1.train.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer=tf.compat.v1.train.AdadeltaOptimizer)

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=make_regularized_loss_fn(
            loss_keys=hparams.loss,
            weights_feature_name=hparams.train_weights_feature_name,
            reduction='weighted_sum'),
        eval_metric_fns=eval_metric_fns(hparams.eval_weights_feature_name),
        train_op_fn=_train_op_fn)

    return tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(model_name,
                                         regularizer=regularizer,
                                         regularizer_scale=regularizer_scale),
            group_size=1,
            transform_fn=make_transform_fn(
                hparams.train_weights_feature_name,
                hparams.nfeatures,
                model_name),
            ranking_head=ranking_head),
        params=hparams,
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=6))


def train_and_eval_with_early_stopping(
        train_features, train_labels,
        vali_features, vali_labels,
        model_dir,
        train_weights_feature_name=None,
        eval_weights_feature_name=None,
        max_train_steps=None, epochs=100,
        train_validate_nfeatures=55,
        loss='pairwise_hinge_loss',
        learning_rate=0.1, batch_size=32, nruns=3, model_name='linear',
        regularizer='l2', regularizer_scale=0.01, metric_name='metric/mrr'):
    tf.compat.v1.logging.info('Experiment starts')

    one_epoch_steps = int(train_labels.shape[0] // batch_size + 1)
    if max_train_steps is None:
        max_train_steps = int(one_epoch_steps * epochs)
        tf.compat.v1.logging.info(f'Argument `max_train_step` is None. '
                                  f'Setting to {max_train_steps}')
    else:
        max_train_steps = int(max_train_steps)

    # train input function
    train_input_fn, train_iter_hook = model_input_fn(
        train_features, train_labels, batch_size=batch_size, is_train=True)
    # validation input function
    validate_input_fn, validate_iter_hook = (
        model_input_fn(
            vali_features, vali_labels, batch_size=None, is_train=False)
    )

    hparams = make_hparams(
        learning_rate=learning_rate,
        nfeatures=train_validate_nfeatures, loss=loss,
        train_weights_feature_name=train_weights_feature_name,
        eval_weights_feature_name=eval_weights_feature_name)
    metric_values = []
    best_train_steps = []
    for i in range(nruns):
        tf.compat.v1.logging.info(f'Starts {i+1}/{nruns} run')
        curr_model_dir = Path(model_dir) / f'run_{i}'
        tf.compat.v1.logging.info(f'Creates model at {curr_model_dir}')
        ranker = get_estimator(hparams, model_dir=curr_model_dir,
                               save_checkpoints_steps=one_epoch_steps,
                               model_name=model_name,
                               regularizer=regularizer,
                               regularizer_scale=regularizer_scale)
        early_stopping_hook = (
            tf.compat.v1.estimator.experimental.stop_if_no_increase_hook(
                ranker, metric_name=metric_name,
                max_steps_without_increase=5*one_epoch_steps,
                run_every_secs=None, run_every_steps=one_epoch_steps))
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=max_train_steps,
            hooks=[train_iter_hook, early_stopping_hook])
        validate_spec = tf.estimator.EvalSpec(
            input_fn=validate_input_fn,
            hooks=[validate_iter_hook],
            start_delay_secs=0,
            throttle_secs=30)
        tf.compat.v1.logging.info(f'Training for max {max_train_steps} steps '
                                  f'and evaluating for every '
                                  f'{one_epoch_steps} steps. Early stopping '
                                  f'if {metric_name} does not increase on '
                                  f'the validation data set for '
                                  f'{5*one_epoch_steps} steps')
        tf.estimator.train_and_evaluate(ranker, train_spec, validate_spec)

        tf.compat.v1.logging.info(f'Train and validation finished')
        tf.compat.v1.logging.info('Test on ground truth data')

        tf.compat.v1.logging.info('Generating predictions')
        # the earliest saved model is the best model
        best_model_ckpt = sorted(
            list(curr_model_dir.glob('model.ckpt*')), reverse=True).pop()
        # remove suffix
        best_model_ckpt = '.'.join(best_model_ckpt.name.split('.')[:-1])
        best_train_step = int(best_model_ckpt.split('-')[1])
        best_model_ckpt = (curr_model_dir / best_model_ckpt).as_posix()
        metric_value = ranker.evaluate(
            input_fn=validate_input_fn,
            hooks=[validate_iter_hook],
            checkpoint_path=best_model_ckpt)[metric_name]
        metric_values.append(metric_value)
        best_train_steps.append(best_train_step)
        tf.compat.v1.logging.info(f'Done {i+1}/{nruns} run with '
                                  f'{best_train_step} steps and '
                                  f'{metric_name}: {metric_value}')
    mean_metric = sum(metric_values) / nruns
    mean_epochs = (sum(best_train_steps) // nruns) // one_epoch_steps
    return mean_metric, mean_epochs


def train_and_test_multiple_runs(
        train_features, train_labels,
        test_features, test_labels,
        model_dir, eval_result_dir, algorithm_name,
        train_weights_feature_name=None,
        eval_weights_feature_name=None,
        train_steps=None, epochs=100,
        train_nfeatures=55,
        loss='pairwise_hinge_loss',
        learning_rate=0.1, batch_size=32, nruns=3, model_name='linear',
        regularizer='l2', regularizer_scale=0.01):
    tf.compat.v1.logging.info('Experiment starts')

    if train_steps is None:
        train_steps = int(train_labels.shape[0] // batch_size + 1)
        tf.compat.v1.logging.info(f'Argument `train_step` is None. '
                                  f'Setting to {train_steps}')
    else:
        train_steps = int(train_steps)

    # train input function
    train_input_fn, train_iter_hook = model_input_fn(
        train_features, train_labels, batch_size=batch_size, is_train=True)

    # test input function
    test_input_fn, test_iter_hook = (
        model_input_fn(
            test_features, test_labels, batch_size=None, is_train=False)
    )

    hparams = make_hparams(
        learning_rate=learning_rate,
        nfeatures=train_nfeatures, loss=loss,
        train_weights_feature_name=train_weights_feature_name,
        eval_weights_feature_name=eval_weights_feature_name)

    for i in range(nruns):
        tf.compat.v1.logging.info(f'Starts {i+1}/{nruns} run')
        curr_model_dir = Path(model_dir) / f'run_{i}'
        curr_eval_result_dir = Path(eval_result_dir) / f'run_{i}'
        if not curr_eval_result_dir.is_dir():
            curr_eval_result_dir.mkdir(parents=True)
        tf.compat.v1.logging.info(f'Creates model at {curr_model_dir}')
        ranker = get_estimator(hparams, model_dir=curr_model_dir,
                               save_checkpoints_steps=train_steps,
                               model_name=model_name,
                               regularizer=regularizer,
                               regularizer_scale=regularizer_scale)

        for epoch in range(epochs):
            tf.compat.v1.logging.info(f'Train for {epoch+1}/{epochs} epochs '
                                      f'with steps {train_steps} per epoch')
            ranker.train(input_fn=train_input_fn, hooks=[train_iter_hook],
                         steps=train_steps)
            # ranker.evaluate(input_fn=test_input_fn, hooks=[test_iter_hook])
        tf.compat.v1.logging.info(f'Train finished')
        tf.compat.v1.logging.info('Test on ground truth data')
        tf.compat.v1.logging.info('Generating predictions')
        predictions = [
            pred for pred in ranker.predict(input_fn=test_input_fn,
                                            hooks=[test_iter_hook])]

        eval_result = pd.DataFrame()
        tf.compat.v1.logging.info('Compute query-level metrics')
        eval_result['reciprocal_rank'] = compute_reciprocal_rank(
            test_labels, predictions)
        for topn in [1, 2, 3, 4, 5, 10, 15, 20, 25]:
            ndcg_score = nDCG(test_labels, predictions, topn)
            eval_result[f'ndcg@{topn}'] = ndcg_score
        tf.compat.v1.logging.info('Metric computation finished')
        mean_metrics = eval_result.mean()
        mean_metrics = ', '.join(
            [f'{k} = {mean_metrics[k]:.8f}' for k in mean_metrics.index])
        tf.compat.v1.logging.info(f'Test results: {mean_metrics}')
        tf.compat.v1.logging.info('Saving evaluation results')
        eval_result['algo'] = algorithm_name
        save_to = Path(curr_eval_result_dir) / f'{algorithm_name}.csv'
        tf.compat.v1.logging.info(f'Saving evaluation results to {save_to}')
        eval_result.to_csv(save_to, index=False)
        tf.compat.v1.logging.info(f'Done {i+1}/{nruns} run')
