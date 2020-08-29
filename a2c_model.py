import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfp

from utils import *


class Model(object):
    def __init__(self, state_features, num_actions=2, restore=True):
        self.is_training = 1.0
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.num_actions = num_actions
        self.model(state_features=state_features)
        self.loss()
        self.init()
        if restore:
            self.restore()
        self.summary()
        self.save_path = 'checkpoint/model'

    def restore(self):
        chkp = tf.train.latest_checkpoint('./checkpoint')
        if chkp is not None:
            print("Restoring chkp: %s " % (chkp,))
            self._saver.restore(self._sess, chkp)

    def init(self):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self._sess = tf.Session(config=config)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(tf.trainable_variables(), save_relative_paths=True)
        self._summ_writer = tf.summary.FileWriter('./train/{0}'.format(int(time.time())), self._sess.graph)

    def model(self, state_features):
        with tf.variable_scope('model'):
            self.X = state = tf.placeholder('float32', (None, state_features), 'state')

            with tf.variable_scope('shared'):
                shared = fully(state, 64, scope='s1', summary_w=False, reg=True, activation=mish)
                shared = fully(shared, 128, scope='s2', summary_w=False, reg=True, activation=mish)

            with tf.variable_scope('actor'):
                out = fully(shared, 64, scope='a1', summary_w=False, reg=True, activation=mish)

                mu = fully(shared, 32, scope='a_mu', summary_w=True, reg=True, activation=mish)
                self.mu = mu = fully(mu, self.num_actions, scope='mu', summary_w=True, activation=tf.nn.tanh)
                tf.summary.histogram('mu', mu)

                chol = fully(out, 32, scope='a_chol', summary_w=False, reg=True, activation=mish)
                tf.summary.histogram('chol', chol)

                # chol_matrix = fully(chol, self.num_actions ** 2, scope='chol', summary_w=False, activation=None)
                # chol_matrix = tf.reshape(chol_matrix, (-1, self.num_actions, self.num_actions))
                # self.chol = chol = tfp.matrix_diag_transform(chol_matrix, transform=tf.nn.softplus)

                # self.normal_dist = tfp.MultivariateNormalTriL(mu, chol, allow_nan_stats=False)
                chol = fully(out, self.num_actions, scope='var', summary_w=False, reg=True, activation=None)
                self.normal_dist = tfp.Normal(mu, tf.clip_by_value(tf.exp(chol), 1e-3, 50), allow_nan_stats=False)
                self.sampled_action = tf.clip_by_value(self.normal_dist.sample(), -1.0, 1.0)

            with tf.variable_scope('critic'):
                out = fully(shared, 64, scope='v1', summary_w=True, activation=mish)
                out = fully(out, 32, scope='v2', summary_w=True, activation=mish)
                self.value = fully(out, 1, summary_w=True, activation=None, scope='value')
                tf.summary.histogram('value', self.value)

    def loss(self):
        self.actor_lr = tf.placeholder_with_default(0.0002, (), 'actor_lr')
        self.critic_lr = tf.placeholder_with_default(0.0003, (), 'critic_lr')

        self.advantage = advantage = tf.placeholder('float32', (None, 1), 'advantage')
        self.action = action = tf.placeholder('float32', (None, self.num_actions), 'action')
        self.td_target = tf.placeholder('float32', (None, 1), 'td_target')
        tf.summary.histogram('advantage', self.advantage)
        tf.summary.histogram('action', self.action)
        tf.summary.histogram('td_target', self.td_target)

        # Actor loss
        self.log_prob = self.normal_dist.log_prob(action)
        actor_loss = tf.reduce_mean(-self.log_prob * advantage)
        tf.summary.scalar('actor_loss', actor_loss)
        self.entropy = tf.reduce_mean(self.normal_dist.entropy())
        tf.summary.scalar('entropy', self.entropy)

        reg_loss = 0  # -1e-2 * self.entropy
        self.actor_loss = actor_loss + reg_loss
        tf.summary.scalar('reg_loss', reg_loss)

        # Actor Optimiser
        tvars = tf.trainable_variables("model/actor")
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.actor_loss, tvars), 0.5)
        optimiser = tf.train.RMSPropOptimizer(self.actor_lr)
        self.minimise_actor = optimiser.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # self.sig_loss = tf.reduce_mean(tf.nn.relu(self.sig - 0.5) ** 2)
        # self.mu_loss = tf.reduce_mean(self.mu ** 2) * 1e2

        # Critic loss
        self.critic_loss = tf.reduce_mean((self.value - self.td_target) ** 2)
        tf.summary.scalar('critic_loss', self.critic_loss)

        # Critic Optimiser
        tvars = tf.trainable_variables("model/critic")
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.critic_loss, tvars), 0.5)
        optimiser = tf.train.RMSPropOptimizer(self.critic_lr)
        with tf.control_dependencies([self.minimise_actor]):
            self.minimise_critic = optimiser.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            # self.minimiser = optimiser.minimize(self.loss, global_step=self.global_step)

    def run(self, **kwargs):
        action, value = self._sess.run(
            [self.sampled_action, self.value], {
                self.X: kwargs['obs'],
            })
        return action, value

    def test(self, obs):
        """
        Takes the mean prediction for each value
        :param kwargs:
        :return:
        """
        return self._sess.run([self.mu, self.value], {self.X: obs, })

    def train(self, X, **kwargs):
        train_fetches = {
            'critic_loss': self.critic_loss,
            'actor_loss': self.actor_loss,
            'summary': self.summ,
            'minimiser': self.minimise_critic,
            'step': self.global_step
        }

        res = self._sess.run(train_fetches, {
            self.X: X,
            self.advantage: np.expand_dims(kwargs['advantage'], -1),
            self.td_target: np.expand_dims(kwargs['td_target'], -1),
            self.action: kwargs['action']
        })
        self._summ_writer.add_summary(res['summary'], global_step=res['step'])
        if res['step'] % 100 == 0:
            self._saver.save(self._sess, self.save_path, global_step=res['step'])

        return res

    def get_value(self, obs):
        """
        Returns the value of state at obs
        :param obs:
        :return:
        """
        return self._sess.run(self.value, {self.X: obs})

    def summary(self):
        self.summ = tf.summary.merge_all()
        return self.summ
