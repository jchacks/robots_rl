import tensorflow as tf

from utils import *

actor_learning_rate = 0.00025
critic_learning_rate = 0.001


def summary():
    return tf.summary.merge_all()


class Model(object):
    def __init__(self):
        self.model(11, 5)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self._sess = tf.Session(config=config)
        self._saver = tf.train.Saver(tf.trainable_variables('actor'), save_relative_paths=True)

    def restore(self):
        chkp = tf.train.latest_checkpoint('./checkpoint/')
        print("Restoring chkp: %s " % (chkp,))
        self._saver.restore(self._sess, chkp)

    def run(self, data):
        return self._sess.run(self.policy, {(self.state, self.seq): data})

    def model(self, state_features, num_actions=5):
        self.state = state = tf.placeholder('float32', (None, 10, state_features), 'state')
        self.seq = seq = tf.placeholder('int32', (None,), 'state_seq')

        with tf.variable_scope('model'):
            lstm_out, (out_state, z_state) = lstm(state, seq, [256], name='state')
            tf.summary.histogram('lstm_out', lstm_out)

            out = fully(lstm_out, 256, scope='1', activation=tf.nn.relu)
            out = fully(out, 128, scope='2', activation=tf.nn.relu)

            policy_out = fully(out, num_actions, scope='out', activation=tf.nn.relu)
            self.policy = tf.tanh(policy_out )
            value_out = fully(out, 64, scope='o2', activation=tf.nn.relu)
            self.value = fully(value_out, 1, activation=None, scope='out')

        self.lstm_out = lstm_out
        self.lstm_state = (out_state, z_state)

    def train(self, score_prediction):
        self.lr = lr = tf.placeholder_with_default(actor_learning_rate, (), 'actor_lr')
        optimiser = tf.train.AdamOptimizer(lr)

        with tf.variable_scope('critic_loss'):
            score_target = tf.placeholder(tf.float32, [None, 1], 'score')
            critic_loss = tf.reduce_mean((score_target - score_prediction) ** 2)  # + critic_reg
            tf.summary.scalar('mse', critic_loss)

        with tf.variable_scope('actor_loss'):
            actor_loss = tf.reduce_mean((100 - score_prediction) ** 2)  # + self.actor_reg / 100000
            tf.summary.scalar('mse', actor_loss)

        self.train = optimiser.minimize(actor_loss)
