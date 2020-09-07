import logging
from collections import namedtuple

import numpy as np
import numba as nb
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

print(tf.__version__, tfp.__version__)
tf.disable_v2_behavior()

__all__ = [
    "tf",
    "tfp",
    "add_to_collection",
    "lstm",
    "fully",
    "conv1d",
    "dropout",
    "FeedFetch",
    "mish",
    "discount_with_dones",
]

logger = logging.getLogger(__name__)

FeedFetch = namedtuple("FeedFetch", ["feed", "fetch"])


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(self.mu, self.sigma)


def add_to_collection(name, tensors):
    for t in tensors:
        tf.add_to_collection(name, t)


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[: shape[0], : shape[1]]).astype(np.float32)

    return _ortho_init


def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


def seq_to_batch(h, flat=False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def lstm(xs, ms, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh * 4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh * 4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))

    # Split the state and output
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        # Mask the state and output
        c = c * (1 - m)
        h = h * (1 - m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c + i * u
        h = o * tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s


def fully(
    inp,
    out_size,
    summary_w=False,
    summary_b=False,
    reg=False,
    infer_shapes=False,
    scope=None,
    activation=None,
    w_init=(0, 0.01),
    **other_kwargs
):
    assert scope is not None, "'scope' must be given"
    for k, v in other_kwargs.items():
        logger.warning("Kwarg %s: %s, is not valid and will be ignored." % (k, v))

    activation_name = activation.__name__ if activation else None
    logger.info(
        "Constructing full layer '%s'; inp: %s, out_size: %s, activation: %s"
        % (scope, inp.name, out_size, activation_name)
    )

    with tf.variable_scope("full_layer_%s" % scope):
        inp_shape = inp.get_shape()
        inp_shape_t = tf.shape(inp)

        w = tf.get_variable("w", shape=(inp_shape[-1], out_size), initializer=tf.initializers.random_normal(*w_init))
        b = tf.get_variable("b", shape=(out_size,), initializer=tf.zeros_initializer())
        if reg:
            tf.add_to_collection("full_weights", w)

        if summary_w:
            tf.summary.histogram("w", w)
        if summary_b:
            tf.summary.histogram("b", b)

        if infer_shapes:
            logger.debug("Infer Shapes: inp shape %s" % (tuple(inp.get_shape()),))
            inp = tf.reshape(inp, shape=tf.stack([tf.reduce_prod(inp_shape_t[:-1]), inp_shape[-1]], 0))
            out = tf.matmul(inp, w) + b
            out = tf.reshape(out, shape=tf.stack([-1, *inp_shape[1:-1], out_size], 0))
            logger.debug("Infer Shapes: out shape %s" % (tuple(out.get_shape()),))
        else:
            out = tf.matmul(inp, w) + b

        if activation is None:
            logger.warning("No activation function applied to full layer '%s'" % (scope,))
        else:
            out = activation(out)

    return out


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def mish(x):
    return tf.nn.tanh(tf.nn.softplus(x)) * x


def conv1d(inp, out_size, width=4, stride=2, scope=None, activation=None):
    assert scope is not None, "'scope' must be given"
    logger.info(
        "Constructing conv1d layer; inp: %s, out_size: %s, width: %s, stride: %s" % (inp.name, out_size, width, stride)
    )
    with tf.variable_scope("conv1d_%s" % scope):
        w = tf.get_variable(
            "w", (width, inp.get_shape()[-1], out_size), initializer=tf.initializers.random_uniform(-0.1, 0.1)
        )
        out = tf.nn.conv1d(inp, w, stride=stride, padding="SAME")

        if activation is None:
            logger.warning("No activation function applied to conv1d layer '%s'" % (scope,))
        else:
            out = activation(out)
        return out


def dropout(inp, pctg, training=True):
    if training:
        return tf.nn.dropout(inp, keep_prob=pctg)
    else:
        return inp


@nb.njit()
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.0 - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]
