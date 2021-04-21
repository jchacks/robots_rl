import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions

optimiser = tf.keras.optimizers.Adam(learning_rate=9e-4, beta_1=0.5, epsilon=1e-5)


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(1024, activation='relu')
        self.d2 = layers.Dense(512, activation='relu')
        self.d3 = layers.Dense(512, activation='relu')
        self.o = layers.Dense(1)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.o(x)


class Actor(tf.Module):
    def __init__(self, num_actions, name='actor'):
        super().__init__(name=name)
        self.d1 = layers.Dense(1024, activation='relu')
        self.d2 = layers.Dense(512, activation='relu')
        self.d3 = layers.Dense(512, activation='relu')
        self.o = layers.Dense(num_actions, activation='softmax')

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.o(x)


class Model(tf.Module):
    def __init__(self, action_space, name='model'):
        super().__init__(name=name)
        self.d1 = layers.Dense(1024, activation='relu')
        self.d2 = layers.Dense(1024, activation='relu')
        self.d3 = layers.Dense(512, activation='relu')
        self.actor = Actor(action_space)
        self.critic = Critic()

    @tf.Module.with_name_scope
    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)
        latent = self.d3(latent)
        return self.actor(latent), self.critic(latent)


model = None


def sample(observations):
    probs, value = model(observations)
    pd_action = distributions.Categorical(probs=probs)
    return pd_action.sample().numpy(), value.numpy()


def run(observations):
    probs, value = model(observations)
    return tf.argmax(probs, axis=1).numpy(), value.numpy()


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
        probs, vpred = model(observations)
        pd = distributions.Categorical(probs=probs)
        a_losses = -advantage * pd.log_prob(actions)[:, tf.newaxis]
        a_loss = tf.reduce_mean(a_losses)

        # Value function loss
        c_losses = (vpred - rewards) ** 2
        c_loss = tf.reduce_mean(c_losses)

        entropy_reg = tf.reduce_mean(pd.entropy())
        loss = a_loss + c_loss - entropy_reg * 0.05

    training_variables = tape.watched_variables()
    grads = tape.gradient(loss, training_variables)
    if print_grads:
        for g, v in zip(grads, training_variables):
            max_g = tf.reduce_max(g)
            tf.print(v.name, max_g)
    grads_and_vars = zip(grads, training_variables)
    optimiser.apply_gradients(grads_and_vars)

    d_grads = tf.reduce_mean([tf.reduce_mean(g) for g in grads])
    d_val = tf.reduce_mean(vpred)
    d_adv = tf.reduce_mean(advantage)
    return (
        loss,
        a_loss,
        c_loss,
        entropy_reg,
        d_adv,
        d_val,
        d_grads)
