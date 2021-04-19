import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions

optimiser = tf.keras.optimizers.Adam(learning_rate=7e-4, beta_1=0.5, epsilon=1e-5)


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


class Actor(tf.Module):
    def __init__(self, name='actor'):
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.d3 = layers.Dense(256, activation='relu')
        self.d4 = layers.Dense(256, activation='relu')
        # self.moving = layers.Dense(3, activation='softmax')
        self.turning = layers.Dense(3, activation='softmax')
        self.shoot = layers.Dense(2, activation='softmax')

    @tf.Module.with_name_scope
    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)
        latent = self.d3(latent)
        latent = self.d4(latent)

        moving = None  # self.moving(latent)
        turning = self.turning(latent)
        shoot = self.shoot(latent)
        return moving, turning, shoot

    def sample(self, obs):
        moving, turning, shoot = self(obs)
        # pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)
        pd_shoot = distributions.Categorical(probs=shoot)
        return None, pd_turning.sample(), pd_shoot.sample()

    def log_prob(self, obs, actions):
        amoving, aturning, ashoot = actions
        moving, turning, shoot = self(obs)
        # pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)
        pd_shoot = distributions.Categorical(probs=shoot)
        return (  # pd_moving.log_prob(amoving),
            pd_turning.log_prob(aturning),
            pd_shoot.log_prob(ashoot))

    def entropy(self, obs):
        moving, turning, shoot = self(obs)
        # pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)
        pd_shoot = distributions.Categorical(probs=shoot)
        return tf.reduce_mean(pd_turning.entropy()) + tf.reduce_mean(pd_shoot.entropy())


actor = Actor()
critic = Critic()


def get_distributions(observations):
    moving, turning, shoot = actor(observations)
    pd_turning = distributions.Categorical(probs=turning)
    pd_shoot = distributions.Categorical(probs=shoot)
    return pd_turning, pd_shoot


def actor_loss(observations, actions, advantage):
    a_move, a_turn, a_shoot = actions
    move, turn, shoot = actor(observations)
    pd_turn = distributions.Categorical(probs=turn)
    pd_shoot = distributions.Categorical(probs=shoot)
    
    log_prob_turn = pd_turn.log_prob(a_turn)
    log_prob_shoot = pd_shoot.log_prob(a_shoot)
    shoot_loss = (-advantage * log_prob_shoot)[a_shoot == -1]

    # I am not sure what the correct combination is mb adding them together?
    return -advantage * tf.reduce_sum(log_probs, axis=0)[:, tf.newaxis]


def train(observations, rewards, actions, values, norm_advs=False):
    """[summary]

    Args:
        observations ([type]): [description]
        rewards ([type]): [description]
        actions ([type]): [description]
        values ([type]): From previous values
    """
    observations = tf.cast(observations, tf.float32)
    rewards = tf.cast(rewards, tf.float32)

    advs = rewards - values
    if norm_advs:
        advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)

    with tf.GradientTape() as tape:
        vpred = critic(observations)
        
        a_losses = actor_loss(observations, actions, advs)
        a_loss = tf.reduce_mean(a_losses)

        # Value function loss
        c_losses = (vpred - rewards) ** 2
        c_loss = tf.reduce_mean(c_losses)

        entropy_reg = actor.entropy(observations)
        loss = a_loss + c_loss - entropy_reg * 0.5

    training_variables = tape.watched_variables()
    grads = tape.gradient(loss, training_variables)
    for g, v in zip(grads, training_variables):
        max_g = tf.reduce_max(g)
        tf.print(v.name, max_g)
        if tf.math.is_nan(max_g):
            raise Exception
    grads_and_vars = zip(grads, training_variables)
    optimiser.apply_gradients(grads_and_vars)
    return loss, a_loss, c_loss, entropy_reg, tf.reduce_mean([tf.reduce_mean(g) for g in grads])
