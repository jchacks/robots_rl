"""
Copied from stable baselines
https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops


class Pd(object):
    """
    Base class for describing a probability distribution.
    """

    def __init__(self):
        super(Pd, self).__init__()

    def flatparam(self):
        """
        Return the direct probabilities
        :return: ([float]) the probabilities
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability
        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood
        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leibler divergence from the given probability distribution
        :param other: ([float]) the distribution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns Shannon's entropy of the probability
        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probability distribution
        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood
        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return -self.neglogp(x)


class CategoricalPd(Pd):
    def __init__(self, logits):
        """
        Probability distributions from categorical input
        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits
        super(CategoricalPd, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.stop_gradient(one_hot_actions)
        )

    def kl(self, other):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(
            p_0 * (a_0 - tf.math.log(z_0) - a_1 + tf.math.log(z_1)), axis=-1
        )

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_mean(
            tf.reduce_sum(p_0 * (tf.math.log(z_0) - a_0), axis=-1), axis=-1
        )

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random.uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.math.log(-tf.math.log(uniform)), axis=-1)

    def prob(self):
        return tf.nn.softmax(self.logits)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values
        :param flat: ([float]) the categorical logits input
        :return: (Pd) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input
        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalPd, tf.split(flat, nvec, axis=-1)))
        super(MultiCategoricalPd, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.stack([p.mode() for p in self.categoricals], axis=-1)

    def neglogp(self, x):
        return tf.add_n(
            [p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))]
        )

    def kl(self, other):
        return tf.add_n(
            [p.kl(q) for p, q in zip(self.categoricals, other.categoricals)]
        )

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.stack([p.sample() for p in self.categoricals], axis=-1)

    def prob(self):
        return [p.prob() for p in self.categoricals]

    def logits(self):
        return [p.logits for p in self.categoricals]


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)

    @property
    def mean(self):
        return self.ps

    def mode(self):
        return tf.cast(tf.round(self.ps), tf.int64)

    def prob(self):
        return self.ps

    def neglogp(self, x):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(x, tf.float32)
        )

    def kl(self, other):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=other.logits, labels=self.ps
            ),
            axis=-1,
        ) - tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps),
            axis=-1,
        )

    def entropy(self):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps),
            axis=-1,
        )

    def sample(self):
        u = tf.random.uniform(tf.shape(self.ps))
        return tf.cast(u < self.ps, tf.int64)


class MaskedBernoulliPd(Pd):
    def __init__(self, logits, mask):
        self.mask = mask
        self.logits = tf.where(mask, tf.float32.min, logits)
        self.ps = tf.sigmoid(logits)

    @property
    def mean(self):
        return self.ps

    def mode(self):
        return tf.cast(tf.round(self.ps), tf.int64)

    def prob(self):
        return self.ps

    def neglogp(self, x):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(x, tf.float32)
        )

    def kl(self, other):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=other.logits, labels=self.ps
            ),
            axis=-1,
        ) - tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps),
            axis=-1,
        )

    def masked_kl(self):
        return

    def nonmasked_entropy(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=self.ps
        )

    def entropy(self):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits[~self.mask], labels=self.ps[~self.mask]
            ),
            axis=-1,
        )

    def sample(self):
        u = tf.random.uniform(tf.shape(self.ps))
        return tf.cast(u < self.ps, tf.int64)


class GroupedPd(Pd):
    def __init__(self, dists):
        self.distributions = dists
        super(GroupedPd, self).__init__()

    def mode(self):
        return tf.stack([p.mode() for p in self.distributions], axis=-1)

    def neglogp(self, x):
        return tf.add_n(
            [p.neglogp(px) for p, px in zip(self.distributions, tf.unstack(x, axis=-1))]
        )

    def kl(self, other):
        return tf.add_n(
            [p.kl(q) for p, q in zip(self.distributions, other.distributions)]
        )

    def entropy(self):
        return tf.reduce_sum([p.entropy() for p in self.distributions])

    def sample(self):
        return tf.stack([p.sample() for p in self.distributions], axis=-1)

    def prob(self):
        return [p.prob() for p in self.distributions]

    def logits(self):
        return [p.logits for p in self.distributions]
