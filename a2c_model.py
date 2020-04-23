import time
from collections import deque
import tensorflow as tf
import tensorflow.contrib.distributions as tfp

from utils import *

learning_rate = 0.002


class Model(object):
    def __init__(self):
        self.is_training = 1.0
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.model(13, 5)
        self.loss()
        self.init()
        self.summary()
        self.to_console = deque(maxlen=20)

    def restore(self):
        chkp = tf.train.latest_checkpoint('./checkpoint/')
        print("Restoring chkp: %s " % (chkp,))
        self._saver.restore(self._sess, chkp)

    def init(self):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self._sess = tf.Session(config=config)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(tf.trainable_variables(), save_relative_paths=True)
        self._summ_writer = tf.summary.FileWriter('./train/{0}'.format(int(time.time())), self._sess.graph)

    def run(self, data):
        return self._sess.run([self.action, self.value], {(self.state, self.seq): data})

    def train(self, state, advantage, td_target, action):
        train_fetches = {
            'summary': self.summ,
            'minimiser': self.minimiser,
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
            'mu_loss': self.mu_loss,
            'sig_loss': self.sig_loss,
            'entropy_loss': self.entropy_loss,
            'step': self.global_step
        }
        res = self._sess.run(train_fetches, {
            (self.state, self.seq): state,
            self.advantage: advantage,
            self.td_target: td_target,
            self.action: action
        })
        self._summ_writer.add_summary(res['summary'], global_step=res['step'])
        return res['actor_loss'], res['critic_loss'], res['entropy_loss'], res['mu_loss'], res['sig_loss']

    def model(self, state_features, num_actions=5):
        self.state = state = tf.placeholder('float32', (None, 25, state_features), 'state')
        self.seq = seq = tf.placeholder('int32', (None,), 'state_seq')

        with tf.variable_scope('model'):
            lstm_out = lstm(state, seq, [256, 256], name='state')

            out = fully(lstm_out, 512, scope='a1', reg=True, activation=tf.nn.relu)
            out = fully(out, 256, scope='a2', reg=True, activation=tf.nn.relu)
            out = fully(out, 128, scope='a3', reg=True, activation=tf.nn.relu)

            self.mu = mu = fully(out, num_actions, scope='mu', activation=tf.nn.tanh)
            self.sig = sig = fully(out, num_actions, scope='sig', activation=tf.nn.softplus) + 1e-5
            tf.summary.histogram('mu', mu)
            tf.summary.histogram('sig', sig)

            self.normal_dist = tfp.MultivariateNormalDiag(mu, sig)
            action = self.normal_dist.sample()
            self.action = tf.clip_by_value(action, -1.0, 1.0)
            self.log_prob = self.normal_dist.log_prob(self.action)
            out = fully(lstm_out, 128, scope='v1', activation=tf.nn.relu)
            value_out = fully(out, 64, scope='v2', activation=tf.nn.relu)
            self.value = fully(value_out, 1, activation=None, scope='out')

            tf.summary.histogram('value', self.value)

    def loss(self):
        self.advantage = advantage = tf.placeholder('float32', (None,), 'advantage')
        self.td_target = tf.placeholder('float32', (None,), 'td_target')
        tf.summary.histogram('advantage', self.advantage)
        tf.summary.histogram('td_target', self.td_target)

        self.critic_loss = tf.reduce_mean((self.value - self.td_target) ** 2)
        tf.summary.scalar('critic_loss', self.critic_loss)

        self.actor_loss = tf.reduce_mean(-self.log_prob * advantage)
        tf.summary.scalar('actor_loss', self.actor_loss)
        entropy = self.normal_dist.entropy()
        self.entropy_loss = tf.reduce_mean(- 1e-1 * entropy)
        tf.summary.scalar('entropy', tf.reduce_mean(entropy))
        tf.summary.scalar('entropy_loss', self.entropy_loss)

        self.sig_loss = tf.reduce_mean(self.sig)
        self.mu_loss = tf.reduce_mean(self.mu ** 2) * 1e2

        reg_loss = self.entropy_loss + self.sig_loss + self.mu_loss
        self.loss = self.critic_loss + self.actor_loss + reg_loss

        self.lr = lr = tf.placeholder_with_default(learning_rate, (), 'actor_lr')
        optimiser = tf.train.AdamOptimizer(lr)
        self.minimiser = optimiser.minimize(self.loss, global_step=self.global_step)

    def summary(self):
        self.summ = tf.summary.merge_all()
        return self.summ
