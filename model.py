import tensorflow as tf

from utils import *

actor_learning_rate = 0.00025
critic_learning_rate = 0.001


class Model():
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

        with tf.variable_scope('actor'):
            lstm_out, (out_state, z_state) = lstm(state, seq, [256], name='state')
            tf.summary.histogram('lstm_out', lstm_out)

            act_out = fully(lstm_out, 256, scope='1', activation=tf.nn.relu)
            act_out = fully(act_out, 128, scope='2', activation=tf.nn.relu)
            out = fully(act_out, num_actions, scope='out', activation=tf.nn.relu)
            self.policy = tf.tanh(out)

        with tf.variable_scope('critic'):
            action = tf.placeholder('float32', (None, num_actions), 'action')
            a1 = fully(action, 128, scope='a1', activation=tf.nn.relu)
            conc = tf.concat([act_out, a1], axis=1)
            out = fully(conc, 128, scope='o1', activation=tf.nn.relu)
            out = fully(out, 64, scope='o2', activation=tf.nn.relu)
            self.value = fully(out, 1, activation=None, scope='out')

    def train(self):
        self.critic_lr = critic_lr = tf.placeholder_with_default(critic_learning_rate, (), 'actor_lr')
        self.actor_lr = actor_lr = tf.placeholder_with_default(actor_learning_rate, (), 'actor_lr')

        with tf.variable_scope('critic_loss'):
            score_target = tf.placeholder(tf.float32, [None, 1], 'score')
            value_loss = tf.reduce_mean((score_target - self.value) ** 2)  # + critic_reg
            tf.summary.scalar('mse', value_loss)
            critic_opt = tf.train.AdamOptimizer(critic_lr)
            critic_vars = tf.trainable_variables('critic')
            self.train_critic = train_critic = critic_opt.minimize(value_loss, var_list=critic_vars)

        with tf.control_dependencies([train_critic]):
            with tf.variable_scope('actor_loss'):
                policy_loss = tf.reduce_mean((100 - self.value) ** 2)  # + self.actor_reg / 100000
                tf.summary.scalar('mse', policy_loss)
                actor_opt = tf.train.AdamOptimizer(actor_lr)
                self.train_actor = actor_opt.minimize(policy_loss, var_list=tf.trainable_variables('actor'))

    def summary(self):
        return tf.summary.merge_all()
