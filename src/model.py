from importlib_metadata import distribution
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from distributions import GroupedPd, CategoricalPd, BernoulliPd
from utils import PROJECT_ROOT


class Critic(tf.Module):
    def __init__(self, name='critic') -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(512, activation='relu')
        self.d2 = layers.Dense(512, activation='relu')
        self.d3 = layers.Dense(128, activation='relu')
        self.o = layers.Dense(1)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.o(x)


class Actor(tf.Module):
    def __init__(self, action_space, name='actor'):
        super().__init__(name=name)
        self.num_actions = np.sum(action_space)
        self.d1 = layers.Dense(512, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.d3 = layers.Dense(256, activation='relu')
        self.o = layers.Dense(self.num_actions)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.o(x)


class Model(tf.Module):
    def __init__(self, action_space, name='model'):
        super().__init__(name=name)
        self.action_space = action_space
        self.p1 = layers.Dense(512, activation='relu')
        self.lstm = layers.LSTMCell(units=512,)
        self.s1 = layers.Dense(512, activation='relu')
        self.d1 = layers.Dense(1024, activation='relu')
        self.d2 = layers.Dense(1024, activation='relu')
        self.actor = Actor(self.action_space)
        self.critic = Critic()

    @tf.Module.with_name_scope
    def __call__(self, obs, states):
        pre = self.p1(obs)
        latent, states = self.lstm(pre, states=states)
        obs = self.s1(obs)  # Scaling layer
        latent = tf.concat([latent, obs], axis=-1)
        latent = self.d1(latent)
        latent = self.d2(latent)
        return self.actor(latent), self.critic(latent), tf.stack(states)

    def initial_state(self, batch_size):
        return tf.stack(self.lstm.get_initial_state(batch_size=batch_size, dtype=tf.float32))

    def distribution(self, logits):
        logits = tf.split(logits, self.action_space, axis=-1) 
        return GroupedPd([
            BernoulliPd(logits[0][:, 0]),
            CategoricalPd(logits[1]),
            CategoricalPd(logits[2]),
            CategoricalPd(logits[3]),
        ])

    def prob(self, obs, states):
        logits, value, states = self(obs, states)
        dist = self.distribution(logits)
        return [prob.numpy() for prob in dist.prob()], [logit.numpy() for logit in dist.logits()], value.numpy()

    def sample(self, obs, states):
        logits, value, states = self(obs, states)
        dist = self.distribution(logits)
        actions = dist.sample()
        return actions.numpy(), value.numpy(), dist.neglogp(actions).numpy(), states.numpy()

    def run(self, obs, states):
        logits, value, states = self(obs, states)
        dist = self.distribution(logits)
        return dist.mode().numpy(), value.numpy(), states.numpy()


class Trainer(object):
    def __init__(self,
                 model,
                 old_model,
                 save_path=f"{PROJECT_ROOT}/ckpts",
                 interval=50,
                 critic_scale=0.3,
                 entropy_scale=0.01,
                 learning_rate=7e-4,
                 epsilon=0.2) -> None:
        """Class to manage training a model.
        Contains Optimiser and CheckpointManager.

        Args:
            model ([type]): Model to train.
            save_path (str, optional): Checkpoint path. Defaults to '../ckpts'.
            interval (int, optional): Interval on which to save checkpoints. Defaults to 500.
            critic_scale (float, optional): Scale of critic in the loss function. Defaults to 0.2.
            entropy_scale (float, optional): Scale of entropy in the loss function. Defaults to 0.05.
        """
        self.learning_rate = learning_rate
        self.optimiser = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        # self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-5)
        self.model = model
        self.old_model = old_model

        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimiser,
            model=model
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            save_path,
            max_to_keep=3,
            keep_checkpoint_every_n_hours=1,
            step_counter=self.ckpt.step,
            checkpoint_interval=interval)
        self.critic_scale = critic_scale
        self.entropy_scale = entropy_scale
        self.epsilon = epsilon

    def checkpoint(self):
        self.ckpt.step.assign_add(1)
        save_path = self.manager.save()
        if save_path:
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    # @tf.function
    def train(self,
              observations,
              states,
              rewards,
              actions,
              neglogp,
              values,
              norm_advs=True,
              print_grads=False):
        """[summary]

        Args:
            observations ([type]): Observations of the environment
            states ([type]): LSTM states
            rewards ([type]): TD Rewards
            actions ([type]): Actions taken
            neglogp ([type]): Negative log prob of actions taken.
            values ([type]): [description]
            norm_advs (bool, optional): [description]. Defaults to False.
            print_grads (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        print("Tracing train function")
        observations = tf.cast(observations, tf.float32)
        rewards = tf.cast(rewards, tf.float32)

        advantage = rewards - values
        # Record the mean advs before norm for debugging
        d_adv = tf.reduce_mean(advantage)
        if norm_advs:
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

        oldlogits, _, _ = self.old_model(observations, states=states)
        old_neglogp = self.old_model.distribution(oldlogits).neglogp(actions)

        with tf.GradientTape() as tape:
            logits, vpred, _ = self.model(observations, states=states)

            # Mask out cases when we cant choose to shoot
            mask = actions[:, 0] == -1
            logits[:,0] = tf.where(mask, tf.float32.min, logits[:,0])
            actions[:, 0][mask] = 0

            pd = self.model.distribution(logits)
            # Actor loss
            neglogp = pd.neglogp(actions)
            # PPO
            ratio = tf.exp(old_neglogp - neglogp)[:, tf.newaxis]
            ratio_clipped = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)

            # a_losses = advantage * ratio
            # a_losses = advantage * pd.neglogp(actions)[:, tf.newaxis]  # old loss
            a_losses = tf.minimum(advantage * ratio, advantage * ratio_clipped)
            a_loss = - tf.reduce_mean(a_losses)

            # Value function loss
            c_losses = (vpred - rewards) ** 2
            c_loss = tf.reduce_mean(c_losses)

            entropy_reg = tf.reduce_mean(pd.entropy())
            loss = a_loss + (c_loss * self.critic_scale) - (entropy_reg * self.entropy_scale)

        # Assign current policy to old policy before update
        for v1, v2 in zip(self.model.variables, self.old_model.variables):
            v2.assign(v1)

        training_variables = tape.watched_variables()
        grads = tape.gradient(loss, training_variables)
        if print_grads:
            # Print grads
            for g, v in zip(grads, training_variables):
                max_g = tf.reduce_max(g)
                tf.print(v.name, max_g)

        # Apply grads
        grads_and_vars = zip(grads, training_variables)
        self.optimiser.apply_gradients(grads_and_vars)

        d_grads = tf.reduce_mean([tf.reduce_mean(g) for g in grads])
        d_grads_actor = tf.reduce_mean([tf.reduce_mean(g) for g, v in zip(
            grads, training_variables) if v.name.startswith('actor/')])

        d_val = tf.reduce_mean(vpred)
        return (
            loss,
            a_loss,
            c_loss,
            entropy_reg,
            d_adv,
            d_val,
            d_grads,
            d_grads_actor
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
