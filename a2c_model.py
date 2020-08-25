import time
from collections import deque, defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfp

from utils import *

learning_rate = 2e-4


class Memory(object):
    pass


class Runner(object):
    def __init__(self):
        self.model = Model(8, 2)
        self.memory = defaultdict(list)
        self.iteration = 0

    def prep_data(self):
        state_, value_tar_, action_, advantage_ = [], [], [], []
        skipped = 0
        steps = 10

        for robot in self.memory.keys():
            # Tick, state, value, action, energy, done
            tick, state, value, action, energy, done = zip(*self.memory[robot])
            if len(state) == 0:
                continue

            value = np.concatenate(value)
            last_value = value[-1]

            value = value[:-1]
            done = np.array(done)[:-1]
            state = np.stack(state)[:-1]
            action = np.stack(action)[:-1]
            energy = np.array(energy)
            # Reward per state is difference in energy
            # Last state has 0 reward for winning.
            rewards = np.concatenate([energy[1:] - energy[:-1]])

            # Truncated reward value over next `steps` and add the offset predicted value
            truncated_sum = np.cumsum(rewards[::-1])
            # # Moving window of 10 cumsum
            # truncated_sum[steps:] = truncated_sum[steps:] - truncated_sum[:-steps]

            offset_value = value[steps:]
            offset_value = np.concatenate([offset_value, [last_value] * (len(state) - len(offset_value))])
            true_value = truncated_sum[::-1] + offset_value
            advantage = rewards + true_value - value

            state_.append(state)
            value_tar_.append(true_value)
            action_.append(action)
            advantage_.append(advantage)

        state_ = np.concatenate(state_)
        value_tar_ = np.concatenate(value_tar_)
        action_ = np.concatenate(action_)
        advantage_ = np.concatenate(advantage_)

        # Shuffle with permutation
        perm = np.random.permutation(len(state_))
        state_ = state_[perm]
        value_tar_ = value_tar_[perm]
        action_ = action_[perm]
        advantage_ = advantage_[perm]

        self.memory = defaultdict(list)
        return state_, value_tar_, action_, advantage_

    def run(self, tick, data):
        robots = list(data.keys())
        states, energy, done = [], [], []
        for robot in robots:
            st, e, d = data[robot]
            states.append(st)
            energy.append(e)
            done.append(d)

        action, value = self.model.run(state=np.stack(states))

        out = {}
        for i, robot in enumerate(robots):
            self.memory[robot].append((
                int(tick),
                states[i],
                value[i],
                action[i],
                energy[i],
                done[i]
            ))
            if not done[i]:
                out[robot] = action[i]

        return out

    def test(self, tick, data):
        robots = list(data.keys())
        states, energy = [], []
        for robot in robots:
            st, e = data[robot]
            states.append(st)
            energy.append(e)

        action, value = self.model.test(state=np.stack(states))
        out = {}
        for i, robot in enumerate(robots):
            self.memory[robot].append((
                int(tick),
                states[i],
                value[i],
                action[i],
                energy[i]))
            out[robot] = action[i]
        return out

    def train(self):
        state, value_tar, action, advantage = self.prep_data()
        step = 512
        for i in range(0, state.shape[0], step):
            res = self.model.train(state=state[i:i + step],
                                   advantage=advantage[i:i + step],
                                   td_target=value_tar[i:i + step],
                                   action=action[i:i + step])
        print(self.iteration,
              'advantage:', advantage.mean(),
              'value_tar:', round(value_tar.mean(), 3), round(res['critic_loss'].mean(), 3),
              'samples:', len(state))
        self.iteration += 1


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
        self.to_console = deque(maxlen=20)
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
            self.state = state = tf.placeholder('float32', (None, state_features), 'state')

            with tf.variable_scope('actor'):
                out = fully(state, 64, scope='a1', summary_w=False, reg=True, activation=mish)
                out = fully(out, 128, scope='a2', summary_w=False, reg=True, activation=mish)

                mu = fully(out, 32, scope='a_mu', summary_w=True, reg=True, activation=mish)
                self.mu = mu = fully(mu, self.num_actions, scope='mu', summary_w=True, activation=tf.nn.tanh)

                chol = fully(out, 32, scope='a_chol', summary_w=False, reg=True, activation=mish)
                chol_matrix = fully(chol, self.num_actions ** 2, scope='chol', summary_w=False, activation=None)
                chol_matrix = tf.reshape(chol_matrix, (-1, self.num_actions, self.num_actions))

                self.chol = chol = tfp.matrix_diag_transform(chol_matrix, transform=tf.nn.softplus)
                tf.summary.histogram('mu', mu)
                tf.summary.histogram('chol', chol)

                self.normal_dist = tfp.MultivariateNormalTriL(mu, chol, allow_nan_stats=False)
                self.sampled_action = tf.clip_by_value(self.normal_dist.sample(), -1.0, 1.0)

            with tf.variable_scope('actor'):
                out = fully(state, 64, scope='v1', summary_w=True, activation=mish)
                out = fully(out, 32, scope='v2', summary_w=True, activation=mish)
                self.value = fully(out, 1, summary_w=True, activation=None, scope='value')
                tf.summary.histogram('value', self.value)

    def loss(self):
        self.actor_lr = tf.placeholder_with_default(learning_rate, (), 'actor_lr')
        self.critic_lr = tf.placeholder_with_default(learning_rate, (), 'critic_lr')

        self.advantage = advantage = tf.placeholder('float32', (None,), 'advantage')
        self.action = action = tf.placeholder('float32', (None, self.num_actions), 'action')
        self.td_target = tf.placeholder('float32', (None,), 'td_target')
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
        tvars = tf.trainable_variables("model/actor")
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.actor_loss, tvars), 0.5)
        optimiser = tf.train.RMSPropOptimizer(self.actor_lr)
        self.minimiser_actor = optimiser.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # self.sig_loss = tf.reduce_mean(tf.nn.relu(self.sig - 0.5) ** 2)
        # self.mu_loss = tf.reduce_mean(self.mu ** 2) * 1e2

        # Citic loss
        self.critic_loss = tf.reduce_mean((self.value - self.td_target) ** 2)
        tf.summary.scalar('critic_loss', self.critic_loss)

        tvars = tf.trainable_variables("model/critic")
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.critic_loss, tvars), 0.5)
        optimiser = tf.train.RMSPropOptimizer(self.actor_lr)
        self.minimiser_critic = optimiser.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        # self.minimiser = optimiser.minimize(self.loss, global_step=self.global_step)

    def run(self, **kwargs):
        action, value = self._sess.run(
            [self.sampled_action, self.value], {
                self.state: kwargs['state'],
            })
        return action, value

    def test(self, **kwargs):
        """
        Takes the mean prediction for each value
        :param kwargs:
        :return:
        """
        action, value = self._sess.run(
            [self.mu, self.value], {
                self.state: kwargs['state'],
            })
        return action, value

    def train(self, **kwargs):
        train_fetches = {
            'critic_loss': self.critic_loss,
            'actor_loss': self.actor_loss,
            'summary': self.summ,
            'minimiser': self.minimiser,
            'step': self.global_step
        }

        res = self._sess.run(train_fetches, {
            self.state: kwargs['state'],
            self.advantage: kwargs['advantage'],
            self.td_target: kwargs['td_target'],
            self.action: kwargs['action']
        })
        self._summ_writer.add_summary(res['summary'], global_step=res['step'])
        if res['step'] % 100 == 0:
            self._saver.save(self._sess, self.save_path, global_step=res['step'])

        return res

    def summary(self):
        self.summ = tf.summary.merge_all()
        return self.summ
