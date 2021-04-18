import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions

optimiser = optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-5)


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.o = layers.Dense(1, activation='relu')

    def __call__(self, obs):
        latent = self.d1(obs)
        value = self.o(latent)
        return value


class Actor(tf.Module):
    def __init__(self, name='actor'):
        super().__init__(name=name)
        self.d1 = layers.Dense(100, activation='relu')
        self.moving = layers.Dense(3, activation='softmax')
        self.turning = layers.Dense(3, activation='softmax')

    def __call__(self, obs):
        latent = self.d1(obs)

        moving = self.moving(latent)
        turning = self.turning(latent)
        return moving, turning

    def sample(self, obs):
        moving, turning = self(obs)
        pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)
        return pd_moving.sample(), pd_turning.sample()


def actor_loss(advantage, actions, pds):
    a_moving, a_turning = actions
    pd_moving, pd_turning = pds
    log_prob = pd_moving.log_prob(a_moving) + pd_turning.log_prob(a_turning)
    # I am not sure what the correct combination is mb adding them together?
    return -advantage * log_prob[:, tf.newaxis]


actor = Actor()
critic = Critic()


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
        moving, turning = actor(observations)

        # Probabilities have to be done here as they need variable batch size
        pd_moving = distributions.Categorical(probs=moving)
        pd_turning = distributions.Categorical(probs=turning)

        vpred = critic(observations)

        alosses = actor_loss(advs, actions, (pd_moving, pd_turning))
        aloss = tf.reduce_mean(alosses)

        # Value function loss
        closses = (vpred - rewards) ** 2
        closs = tf.reduce_mean(closses)
        loss = aloss + closs

    grads = tape.gradient(loss, tape.watched_variables())
    optimiser.apply_gradients(zip(grads, tape.watched_variables()))

    return loss, aloss, closs
