import tensorflow as tf

from utils import *

actor_learning_rate = 0.00025
critic_learning_rate = 0.001


def model(state_features, num_actions=5):
    state = tf.placeholder('float32', (None, 10, state_features), 'state')
    seq = tf.placeholder('int32', (None,), 'state_seq')

    with tf.variable_scope('actor'):
        lstm_out, (out_state, z_state) = lstm(state, seq, [256], name='state')
        tf.summary.histogram('lstm_out', lstm_out)

        act_out = fully(lstm_out, 256, scope='1', activation=tf.nn.relu)
        act_out = fully(act_out, 128, scope='2', activation=tf.nn.relu)
        out = fully(act_out, num_actions, scope='out', activation=tf.nn.relu)
        policy = tf.tanh(out)

    with tf.variable_scope('critic'):
        action = tf.placeholder('float32', (None, num_actions), 'action')
        a1 = fully(action, 128, scope='a1', activation=tf.nn.relu)
        conc = tf.concat([act_out, a1], axis=1)
        out = fully(conc, 128, scope='o1', activation=tf.nn.relu)
        out = fully(out, 64, scope='o2', activation=tf.nn.relu)
        value = fully(out, 1, activation=None, scope='out')

    return FeedFetch({
        'state': (state, seq),
        'action': action
    }, {
        'state_lstm': (lstm_out, out_state, z_state),
        'policy': policy,
        'value': value,
    })


def train(value):
    critic_lr = tf.placeholder_with_default(critic_learning_rate, (), 'actor_lr')
    actor_lr = tf.placeholder_with_default(actor_learning_rate, (), 'actor_lr')

    with tf.variable_scope('critic_loss'):
        score_target = tf.placeholder(tf.float32, [None, 1], 'score')
        value_loss = tf.reduce_mean((score_target - value) ** 2)  # + critic_reg
        tf.summary.scalar('mse', value_loss)
        train_critic = tf.train.AdamOptimizer(critic_lr).minimize(value_loss, var_list=tf.trainable_variables('critic'))

    with tf.control_dependencies([train_critic]):
        with tf.variable_scope('actor_loss'):
            policy_loss = tf.reduce_mean((100 - value) ** 2)  # + self.actor_reg / 100000
            tf.summary.scalar('mse', policy_loss)
            train_actor = tf.train.AdamOptimizer(actor_lr).minimize(policy_loss,
                                                                    var_list=tf.trainable_variables('actor'))

    return FeedFetch({
        'score_target': score_target,
        'critic_lr': critic_lr,
        'actor_lr': critic_lr
    }, {
        'critic_minimizer': train_critic,
        'actor_minimizer': train_actor,
        'value_loss': value_loss,
        'policy_loss': policy_loss,
    })


def summary():
    return tf.summary.merge_all()


def call():
    m = model(11, 5)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)

    saver = tf.train.Saver(tf.trainable_variables('actor'), save_relative_paths=True)
    chkp = tf.train.latest_checkpoint('./checkpoint/')
    import os
    print(os.getcwd(), chkp)
    saver.restore(sess, chkp)

    while True:
        data = yield
        yield sess.run(m.fetch['policy'], {m.feed['state']: data})
