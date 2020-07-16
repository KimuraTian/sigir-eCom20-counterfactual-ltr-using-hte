"""
Estimators of position bias.
"""
import tensorflow as tf


class ContextualPositionBasedModel(object):
    # TODO: come up with a better implementation using tf.Estimator

    def __init__(self, nfeatures, np_units, nr_units, nranks, optimizer,
                 eps=1e-6, seed=None):
        self.features = tf.compat.v1.placeholder(
            tf.float32, shape=(tf.newaxis, nfeatures))
        self.y_pos = tf.compat.v1.placeholder(
            tf.float32, shape=(tf.newaxis, nranks, nranks))
        self.y_neg = tf.compat.v1.placeholder(
            tf.float32, shape=(tf.newaxis, nranks, nranks))
        self._p_logits, self._r_logits, self._logits = self._create_model(
            np_units, nr_units, nranks, eps, seed)
        # propensity ratio p_k / p_1
        self._norm_p_logits = tf.math.divide(
            self._p_logits, tf.compat.v1.expand_dims(self._p_logits[:, 0], -1))
        self._loss = self._create_loss()
        self._train_op = optimizer.minimize(
            self._loss,
            global_step=tf.compat.v1.train.get_or_create_global_step())

    def _create_model(self, np_units, nr_units, nranks, eps, seed=None):
        k_init = tf.compat.v1.constant_initializer(0.1)
        b_init = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                           seed=seed)
        # examination model
        p_hidden_layer = tf.compat.v1.layers.dense(
            self.features, units=np_units, activation=tf.compat.v1.nn.sigmoid,
            kernel_initializer=k_init, bias_initializer=b_init)
        p_logits = tf.compat.v1.layers.dense(
            p_hidden_layer, units=nranks,
            activation=tf.compat.v1.nn.sigmoid,
            kernel_initializer=k_init, bias_initializer=b_init)
        # average relevance model
        r_hidden_layer = tf.compat.v1.layers.dense(
            self.features, units=nr_units, activation=tf.compat.v1.nn.sigmoid,
            kernel_initializer=k_init, bias_initializer=b_init)
        r_logits = tf.compat.v1.layers.dense(
            r_hidden_layer, units=nranks * nranks,
            activation=tf.compat.v1.nn.sigmoid,
            kernel_initializer=k_init, bias_initializer=b_init)
        r_logits_matrix = tf.compat.v1.reshape(r_logits, (-1, nranks, nranks))
        r_logits_matrix_t = tf.linalg.matrix_transpose(r_logits_matrix)
        r_symmetric_logits = tf.math.divide(
            tf.compat.v1.add(r_logits_matrix, r_logits_matrix_t), 2.0)
        logits = tf.compat.v1.expand_dims(p_logits, -1) * r_symmetric_logits
        logits = tf.compat.v1.clip_by_value(logits, eps, 1 - eps)
        return p_logits, r_symmetric_logits, logits

    def _create_loss(self):
        loss = -tf.compat.v1.reduce_sum(
            tf.compat.v1.add(self.y_pos * tf.compat.v1.log(self.logits),
                             self.y_neg * tf.compat.v1.log(1 - self.logits)))
        return loss

    @property
    def logits(self):
        return self._logits

    @property
    def p_logits(self):
        return self._p_logits

    @property
    def norm_p_logits(self):
        return self._norm_p_logits

    @property
    def r_logits(self):
        return self._r_logits

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op
