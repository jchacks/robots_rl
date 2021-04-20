import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions

optimiser = tf.keras.optimizers.Adam(learning_rate=9e-4, beta_1=0.5, epsilon=1e-5)

NUM_ACTIONS = 2*3*3*3


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.d3 = layers.Dense(256, activation='relu')
        self.d4 = layers.Dense(64, activation='relu')
        self.o = layers.Dense(1)

    @tf.Module.with_name_scope
    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)
        latent = self.d3(latent)
        latent = self.d4(latent)

        value = self.o(latent)
        return value


class Attention(tf.Module):
    def __init__(self, units, signal, name='attention'):
        super().__init__(name=name)
        self.d1 = layers.Dense(units, activation='sigmoid')
        self.signal = signal

    def __call__(self, query, inp):
        query = self.d1(query)

        query = query * self.signal/tf.reduce_sum(query)
        out = query * inp

class Actor(tf.Module):
    def __init__(self, num_actions, name='actor'):
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.d2 = layers.Dense(1024, activation='relu')
        self.attention = layers.Dense(1024, activation='sigmoid')
        self.d3 = layers.Dense(512, activation='relu')
        self.d4 = layers.Dense(256, activation='relu')
        self.o = layers.Dense(num_actions, activation='softmax')

    @tf.Module.with_name_scope
    def __call__(self, obs):
        latent = self.d1(obs)  # 100
        latent = self.d2(latent)  # 1024
        attention = self.attention(latent)  # 1024
        attention = attention/tf.reduce_sum(attention)  # ! Fix this so its constant signal
        latent = latent * attention
        latent = self.d3(latent)
        latent = self.d4(latent)
        return self.o(latent), attention


class Model(tf.Module):
    def __init__(self, name='model'):
        super().__init__(name=name)
        self.actor = Actor()
        self.critic = Critic()


actor = Actor(NUM_ACTIONS)
critic = Critic()


def sample(observations):
    probs, _ = actor(observations)
    pd_action = distributions.Categorical(probs=probs)
    return pd_action.sample()


def run(observations):
    probs, _ = actor(observations)
    return tf.argmax(probs, axis=1)


def train(observations, rewards, actions, values, norm_advs=False, print_grads=False):
    """[summary]

    Args:
        observations ([type]): [description]
        rewards ([type]): [description]
        actions ([type]): [description]
        values ([type]): From previous values
    """
    observations = tf.cast(observations, tf.float32)
    rewards = tf.cast(rewards, tf.float32)

    advantage = rewards - values
    if norm_advs:
        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

    with tf.GradientTape() as tape:
        probs, attention = actor(observations)
        pd = distributions.Categorical(probs=probs)
        vpred = critic(observations)

        a_losses = -advantage * pd.log_prob(actions)[:, tf.newaxis]
        a_loss = tf.reduce_mean(a_losses)

        # Value function loss
        c_losses = (vpred - rewards) ** 2
        c_loss = tf.reduce_mean(c_losses)

        attention_size = tf.reduce_sum(attention, 1)
        attention_size_reg = tf.reduce_mean(tf.nn.relu(attention_size - 300))

        entropy_reg = tf.reduce_mean(pd.entropy())
        loss = a_loss + c_loss - entropy_reg * 0.5 + attention_size_reg

    training_variables = tape.watched_variables()
    grads = tape.gradient(loss, training_variables)
    if print_grads:
        for g, v in zip(grads, training_variables):
            max_g = tf.reduce_max(g)
            tf.print(v.name, max_g)
    grads_and_vars = zip(grads, training_variables)
    optimiser.apply_gradients(grads_and_vars)
    return (
        loss,
        a_loss,
        c_loss,
        entropy_reg,
        attention_size_reg,
        tf.reduce_mean([tf.reduce_mean(g) for g in grads]))
