from numpy.lib.utils import deprecate
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from distributions import MultiCategoricalProbabilityDistribution


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d2 = layers.Dense(512, activation='relu')
        self.d3 = layers.Dense(512, activation='relu')
        self.o = layers.Dense(1)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d2(x)
        x = self.d3(x)
        return self.o(x)


class Actor(tf.Module):
    def __init__(self, num_actions, name='actor'):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.d2 = layers.Dense(512, activation='relu')
        self.d3 = layers.Dense(512, activation='relu')
        self.o = layers.Dense(num_actions)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d2(x)
        x = self.d3(x)
        return self.o(x)


class Model(tf.Module):
    def __init__(self, action_space, name='model'):
        super().__init__(name=name)
        self.action_space = action_space
        self.num_actions = np.sum(action_space)

        self.d1 = layers.Dense(1024, activation='relu')
        self.d2 = layers.Dense(1024, activation='relu')
        self.d3 = layers.Dense(512, activation='relu')
        self.actor = Actor(self.num_actions)
        self.critic = Critic()

    @tf.Module.with_name_scope
    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)
        latent = self.d3(latent)
        return self.actor(latent), self.critic(latent)

    def distribution(self, logits):
        return MultiCategoricalProbabilityDistribution(self.action_space, logits)

    def sample(self, obs):
        logits, value = self(obs)
        dist = self.distribution(logits)
        return dist.sample().numpy(), value.numpy()

    def prob(self, obs):
        logits, value = self(obs)
        dist = self.distribution(logits)
        return [d.numpy() for d in dist.prob()]

    def sample(self, obs):
        logits, value = self(obs)
        dist = self.distribution(logits)
        return dist.sample().numpy(), value.numpy()

    def run(self, obs):
        logits, value = self(obs)
        dist = self.distribution(logits)
        return dist.mode().numpy(), value.numpy()


model = None
optimiser = tf.keras.optimizers.Adam(learning_rate=7e-4, epsilon=1e-5)


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
        logits, vpred = model(observations)
        pd = model.distribution(logits)
        a_losses = advantage * pd.neglogp(actions)[:, tf.newaxis]
        a_loss = tf.reduce_mean(a_losses)

        # Value function loss
        c_losses = (vpred - rewards) ** 2
        c_loss = tf.reduce_mean(c_losses)

        entropy_reg = tf.reduce_mean(pd.entropy())
        loss = a_loss + (c_loss * 0.5) - (entropy_reg * 0.05)

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
        d_grads
    )
