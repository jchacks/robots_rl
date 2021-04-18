import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions

optimiser = optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-5)


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.d2 = layers.Dense(100, activation='relu')
        self.o = layers.Dense(1)

    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)
        value = self.o(latent)
        return value


class Actor(tf.Module):
    def __init__(self, name='actor'):
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.d2 = layers.Dense(100, activation='relu')
        self.moving = layers.Dense(3, activation='softmax')
        self.turning = layers.Dense(3, activation='softmax')
        self.shoot = layers.Dense(2, activation='softmax')

    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)

        moving = self.moving(latent)
        turning = self.turning(latent)
        shoot = self.shoot(latent)
        return moving, turning, shoot

    def sample(self, obs):
        moving, turning, shoot = self(obs)
        pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)
        pd_shoot = distributions.Categorical(probs=shoot)
        return pd_moving.sample(), pd_turning.sample(), pd_shoot.sample()

    def log_prob(self, obs, actions):
        amoving, aturning, ashoot = actions
        moving, turning, shoot = self(obs)
        pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)
        pd_shoot = distributions.Categorical(probs=shoot)
        return (pd_moving.log_prob(amoving),
                pd_turning.log_prob(aturning),
                pd_shoot.log_prob(ashoot))

actor = Actor()
critic = Critic()


def actor_loss(advantage, actions, observations):
    log_probs = actor.log_prob(observations, actions)
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
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    with tf.GradientTape() as tape:
        # Probabilities have to be done here as they need variable batch size
        vpred = critic(observations)

        alosses = actor_loss(advs, actions, observations)
        aloss = tf.reduce_mean(alosses)

        # Value function loss
        closses = (vpred - rewards) ** 2
        closs = tf.reduce_mean(closses)
        loss = aloss + closs

    grads = tape.gradient(loss, tape.watched_variables())
    optimiser.apply_gradients(zip(grads, tape.watched_variables()))

    return loss, aloss, closs
