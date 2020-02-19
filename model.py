import tensorflow as tf
from utils import *

actor_learning_rate = 0.00025
critic_learning_rate = 0.001


def model(state_features, num_actions=5):
    state = tf.placeholder('float32', (None, 10, state_features), 'state')
    seq = tf.placeholder('float32', (None,), 'state_seq')

    with tf.variable_scope('state'):
        state = fully(state, 256, scope='1')

    lstm_out, (out_state, z_state) = lstm(state, seq, [32], name="LSTM")

    with tf.variable_scope("Actor"):
        with tf.variable_scope('f1'):
            act_out = fully(lstm_out, 128, dropout=0.9)

        with tf.variable_scope('fe'):
            raw_action = fully(act_out, num_actions)

            # Define the activations for, do I remove these full layers
            with tf.variable_scope('fa1'):
                a1 = fully(raw_action, 4, activation=tf.nn.tanh)
            with tf.variable_scope('fa2'):
                a2 = fully(raw_action, 2, activation=tf.nn.sigmoid)
            with tf.variable_scope('fa3'):
                a3 = fully(raw_action, 4, activation=tf.nn.softmax)
            action_prediction = tf.concat([a1, a2, a3], 1)

    with tf.variable_scope("critic"):
        action = tf.placeholder('float32', (None, num_actions), 'action')

        with tf.variable_scope('f1a'):
            out = dropout(fully(tf.concat([lstm_out, action], axis=1), 128), 0.75)
        with tf.variable_scope('f3'):
            out = fully(out, 64)
        with tf.variable_scope('fe'):
            score_prediction = fully(out, 1, activation=None, bn=False)

    return FeedFetch({
        'state': (state,seq)
    }, {
        'state_lstm': (lstm_out, out_state, z_state),
        'action_prediction': action_prediction,
        'score_prediction': score_prediction,
    })


def train(score_prediction):
    critic_lr = tf.placeholder_with_default(critic_learning_rate,  (), 'actor_lr')
    actor_lr = tf.placeholder_with_default(actor_learning_rate, (), 'actor_lr')

    score_target = tf.placeholder(tf.float32, [None, 1], 'score')
    # critic_reg = tf.reduce_mean(tf.abs(tf.get_collection('reg_loss', 'critic')))
    critic_loss = tf.reduce_mean((score_target - score_prediction) ** 2) #+ critic_reg

    train_critic = tf.train.AdamOptimizer(critic_lr) \
        .minimize(critic_loss, var_list=tf.trainable_variables('critic'))

    with tf.control_dependencies([train_critic]):
        # actor_reg = tf.reduce_mean(tf.abs(tf.get_collection('reg_loss', 'actor')))
        actor_loss = -score_prediction #+ self.actor_reg / 100000
        train_actor = tf.train.AdamOptimizer(actor_lr) \
            .minimize(actor_loss, var_list=tf.trainable_variables('actor'))

    return FeedFetch({
        'score_target': score_target,
        'critic_lr': critic_lr,
        'actor_lr': critic_lr
    }, {
        'critic_minimizer': train_critic,
        'actor_minimizer': train_actor,
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
    })

