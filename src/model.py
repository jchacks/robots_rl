from numpy.lib.utils import deprecate
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from distributions import MultiCategoricalProbabilityDistribution


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d2 = layers.Dense(64, activation='relu')
        self.d3 = layers.Dense(64, activation='relu')
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
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(64, activation='relu')
        self.o = layers.Dense(num_actions)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.o(x)


class Model(tf.Module):
    def __init__(self, action_space, name='model'):
        super().__init__(name=name)
        self.action_space = action_space
        self.num_actions = np.sum(action_space)

        self.d1 = layers.Dense(512, activation='relu')
        self.d2 = layers.Dense(512, activation='relu')
        self.actor = Actor(self.num_actions)
        self.critic = Critic()

    @tf.Module.with_name_scope
    def __call__(self, obs):
        latent = self.d1(obs)
        latent = self.d2(latent)
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
        actions = dist.sample().numpy()
        return actions, value.numpy(), dist.neglogp(actions).numpy()

    def run(self, obs):
        logits, value = self(obs)
        dist = self.distribution(logits)
        return dist.mode().numpy(), value.numpy()


class Trainer(object):
    def __init__(self, model, save_path='../ckpts') -> None:
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=7e-4, epsilon=1e-5)
        self.model = model
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimiser,
            model=model
        )
        self.manager = tf.train.CheckpointManager(self.ckpt, save_path, max_to_keep=3)

    def checkpoint(self):
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def train(self,
              observations,
              rewards,
              actions,
              neglogp,
              values,
              norm_advs=False,
              print_grads=False):
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
            logits, vpred = self.model(observations)
            pd = self.model.distribution(logits)

            ratio = tf.exp(neglogp - pd.neglogp(actions)[:, tf.newaxis])
            a_losses = advantage * ratio
            a_loss = tf.reduce_mean(a_losses)

            # Value function loss
            c_losses = (vpred - rewards) ** 2
            c_loss = tf.reduce_mean(c_losses)

            entropy_reg = tf.reduce_mean(pd.entropy())
            loss = a_loss + (c_loss * 0.1) - (entropy_reg * 0.005)

        training_variables = tape.watched_variables()
        grads = tape.gradient(loss, training_variables)
        if print_grads:
            for g, v in zip(grads, training_variables):
                max_g = tf.reduce_max(g)
                tf.print(v.name, max_g)
        grads_and_vars = zip(grads, training_variables)
        self.optimiser.apply_gradients(grads_and_vars)
        if int(self.ckpt.step) % 1000 == 0:
            self.checkpoint()

        self.ckpt.step.assign_add(1)
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

    def restore(self, partial=None):
        status = self.ckpt.restore(self.manager.latest_checkpoint)
        if partial:
            status.expect_partial()
        elif partial is False:
            status.assert_consumed()
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
