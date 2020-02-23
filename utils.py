import logging
from collections import namedtuple

import tensorflow as tf
import numpy as np
__all__ = ['add_to_collection', 'lstm', 'fully', 'conv1d', 'dropout', 'FeedFetch']

logger = logging.getLogger(__name__)

FeedFetch = namedtuple('FeedFetch', ['feed', 'fetch'])


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def add_to_collection(name, tensors):
    for t in tensors:
        tf.add_to_collection(name, t)


def lstm(inp, sequence, layers, name=None):
    assert name is not None, "'name' cannot be None"
    with tf.variable_scope(name + '_lstm'):
        inp_shape = inp.get_shape()
        inp_shape_t = tf.shape(inp)
        batch_size = inp_shape_t[0]
        steps = inp_shape[1]
        logger.info("Constructing lstm %s; layers: %s, steps: %s" % (name, layers, steps))

        if len(layers) > 1:
            cells = [tf.contrib.rnn.BasicLSTMCell(shape) for shape in layers]
            lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(layers[0])

        state = z_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        outputs = [tf.zeros(shape=(batch_size, layers[-1]), dtype=tf.float32)]
        states = [z_state]

        for time_step in range(steps):
            cell_output, state = lstm_cell(inp[:, time_step, :], state)
            outputs.append(cell_output)
            states.append(state)

        with tf.variable_scope('output'):
            outputs = tf.stack(outputs)
            lstm_output = tf.gather_nd(tf.transpose(outputs, [1, 0, 2]),
                                       tf.stack((tf.range(batch_size), sequence), axis=1))
            state_output = tf.gather_nd(tf.transpose(states, [1, 0, 2]),
                                        tf.stack((tf.range(batch_size), sequence), axis=1))
            tf.summary.histogram('lstm_out', lstm_output)
    return lstm_output, (state_output, z_state)


def fully(inp, out_size, summary_w=False, summary_b=False, reg=False, infer_shapes=False, scope=None, activation=None,
          w_init=(0, .01), **other_kwargs):
    assert scope is not None, "'scope' must be given"
    for k, v in other_kwargs.items():
        logger.warning('Kwarg %s: %s, is not valid and will be ignored.' % (k, v))

    activation_name = activation.__name__ if activation else None
    logger.info("Constructing full layer \'%s\'; inp: %s, out_size: %s, activation: %s" %
                (scope, inp.name, out_size, activation_name))

    with tf.variable_scope("full_layer_%s" % scope):
        inp_shape = inp.get_shape()
        inp_shape_t = tf.shape(inp)

        w = tf.get_variable('w', shape=(inp_shape[-1], out_size), initializer=tf.initializers.random_normal(*w_init))
        b = tf.get_variable('b', shape=(out_size,), initializer=tf.zeros_initializer())
        tf.add_to_collection('full_weights', w)

        if summary_w:
            tf.summary.histogram('w', w)
        if summary_b:
            tf.summary.histogram('b', b)

        if infer_shapes:
            logger.debug("Infer Shapes: inp shape %s" % (tuple(inp.get_shape()),))
            inp = tf.reshape(inp, shape=tf.stack([tf.reduce_prod(inp_shape_t[:-1]), inp_shape[-1]], 0))
            out = tf.matmul(inp, w) + b
            out = tf.reshape(out, shape=tf.stack([-1, *inp_shape[1:-1], out_size], 0))
            logger.debug("Infer Shapes: out shape %s" % (tuple(out.get_shape()),))
        else:
            out = tf.matmul(inp, w) + b

        if activation is None:
            logger.warning("No activation function applied to full layer \'%s\'" % (scope,))
        else:
            out = activation(out)

    return out


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def conv1d(inp, out_size, width=4, stride=2, scope=None, activation=None):
    assert scope is not None, "'scope' must be given"
    logger.info(
        "Constructing conv1d layer; inp: %s, out_size: %s, width: %s, stride: %s" % (inp.name, out_size, width, stride))
    with tf.variable_scope('conv1d_%s' % scope):
        w = tf.get_variable("w", (width, inp.get_shape()[-1], out_size),
                            initializer=tf.initializers.random_uniform(-.1, .1))
        out = tf.nn.conv1d(inp, w, stride=stride, padding="SAME")

        if activation is None:
            logger.warning("No activation function applied to conv1d layer \'%s\'" % (scope,))
        else:
            out = activation(out)
        return out


def dropout(inp, pctg, training=True):
    if training:
        return tf.nn.dropout(inp, keep_prob=pctg)
    else:
        return inp
