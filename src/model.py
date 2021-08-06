import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers
from distributions import GroupedPd, CategoricalPd, MaskedBernoulliPd
from utils import PROJECT_ROOT
from collections import namedtuple
import operator

layers.LSTMCell

Losses = namedtuple(
    "Losses",
    "loss,actor,critic,ratio,ratio_clipped,entropy,entropies,advantage,value,d_grads,d_grads_actor",
)


def map_numpy(func):
    def inner(*args, **kwargs):
        return map(operator.methodcaller("numpy"), func(*args, **kwargs))

    return inner


class LSTM(tf.keras.layers.Layer):
    def __init__(self, units, init_scale=1.0, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.units = int(units)
        self.kernelx_initializer = initializers.Orthogonal(gain=init_scale)
        self.kernelh_initializer = initializers.Orthogonal(gain=init_scale)
        self.bias_initializer = initializers.Zeros()

    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype)
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `LSTM` layer with non-floating point "
                "dtype %s" % (dtype,)
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = int(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `LSTM` "
                "should be defined. Found `None`."
            )

        self.kernelx = self.add_weight(
            "kernelx",
            shape=[last_dim, self.units * 4],
            initializer=self.kernelx_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernelh = self.add_weight(
            "kernelh",
            shape=[self.units, self.units * 4],
            initializer=self.kernelh_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias",
            shape=[
                self.units * 4,
            ],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, states, masks=None):
        c, h = tf.unstack(states)
        masks = 1 - masks if masks is not None else tf.ones(len(inputs))
        masks = tf.expand_dims(masks, -1)
        output = []

        for idx in range(inputs.get_shape().as_list()[0]):
            x, m = inputs[idx], masks[idx]
            c = c * m
            h = h * m
            z = tf.matmul(x, self.kernelx) + tf.matmul(h, self.kernelh)
            i, f, o, u = tf.split(z, axis=1, num_or_size_splits=4)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            u = tf.tanh(u)

            c = f * c + i * u
            h = o * tf.tanh(c)

            output.append(h)
        return tf.stack(output), tf.stack([c, h])

    def get_initial_state(self, batch_size, dtype):
        return tf.zeros(shape=[2, batch_size, self.units], dtype=dtype)


class Critic(tf.Module):
    def __init__(self, name="critic") -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(32, activation="elu")
        # self.d2 = layers.Dense(32, activation="elu")
        # self.d3 = layers.Dense(128, activation="elu")
        self.o = layers.Dense(1)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        # x = self.d2(x)
        # x = self.d3(x)
        return self.o(x)


class Actor(tf.Module):
    def __init__(self, action_space, name="actor"):
        super().__init__(name=name)
        self.num_actions = np.sum(action_space)
        self.d1 = layers.Dense(128, activation="elu")
        # self.d2 = layers.Dense(256, activation="elu")
        # self.d3 = layers.Dense(256, activation="elu")
        self.o = layers.Dense(self.num_actions)

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        # x = self.d2(x)
        # x = self.d3(x)
        return self.o(x)


class Model(tf.Module):
    def __init__(self, action_space, name="model"):
        super().__init__(name=name)
        self.action_space = action_space
        self.p1 = layers.Dense(64, activation="elu")
        self.lstm = LSTM(units=128)
        self.s1 = layers.Dense(128, activation="elu")
        self.d1 = layers.Dense(256, activation="elu")
        # self.d2 = layers.Dense(128, activation="elu")
        self.actor = Actor(self.action_space)
        self.critic = Critic()

    @tf.Module.with_name_scope
    def __call__(self, obs, states, masks=None):
        pre = self.p1(obs)
        latent, states = self.lstm(pre, states, masks)
        obs = self.s1(obs)  # Scaling layer
        latent = tf.concat([latent, obs], axis=-1)
        latent = self.d1(latent)
        # latent = self.d2(latent)
        return self.actor(latent), self.critic(latent), tf.stack(states)

    def initial_state(self, batch_size):
        return tf.stack(
            self.lstm.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        )

    def distribution(self, logits, shoot_mask):
        logits = tf.split(logits, self.action_space, axis=-1)
        return GroupedPd(
            [
                MaskedBernoulliPd(logits[0], shoot_mask),
                CategoricalPd(logits[1]),
                CategoricalPd(logits[2]),
                CategoricalPd(logits[3]),
            ]
        )

    def prob(self, obs, states, shoot_mask):
        logits, value, states = self(obs, states)
        dist = self.distribution(logits, shoot_mask)
        return (
            [prob.numpy() for prob in dist.prob()],
            [logit.numpy() for logit in dist.logits()],
            value.numpy(),
        )

    @map_numpy
    @tf.function
    def sample(self, obs, states, shoot_mask):
        print("Tracing sample")
        logits, value, states = self(obs, states)
        dist = self.distribution(logits, shoot_mask)
        actions = dist.sample()
        return actions, value, dist.neglogp(actions), states

    @map_numpy
    def run(self, obs, states, shoot_mask):
        logits, value, states = self(obs, states)
        dist = self.distribution(logits, shoot_mask)
        return dist.mode(), value, states


class Trainer(object):
    def __init__(
        self,
        model,
        old_model,
        save_path=f"{PROJECT_ROOT}/ckpts",
        interval=10,
        critic_scale=0.6,
        entropy_scale=0.03,
        learning_rate=3e-3,
        epsilon=0.13,
    ) -> None:
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
        self.optimiser = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, momentum=0.5, nesterov=True
        )
        # self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = model
        self.old_model = old_model

        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimiser, model=model
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            save_path,
            max_to_keep=3,
            keep_checkpoint_every_n_hours=1,
            step_counter=self.ckpt.step,
            checkpoint_interval=interval,
        )
        self.critic_scale = critic_scale
        self.entropy_scale = entropy_scale
        self.epsilon = epsilon

    def checkpoint(self):
        self.ckpt.step.assign_add(1)
        save_path = self.manager.save()
        if save_path:
            print(
                "Saved checkpoint for step {}: {}".format(
                    int(self.ckpt.step), save_path
                )
            )

    @tf.function
    def train(
        self,
        observations,
        states,
        advantage,
        returns,
        actions,
        neglogp,
        shoot_masks,
        dones,
        norm_advs=True,
        print_grads=False,
    ):
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
        advantage = tf.cast(advantage, tf.float32)
        returns = tf.cast(returns, tf.float32)
        dones = tf.cast(dones, tf.float32)
        # rewards = tf.cast(rewards, tf.float32)
        # advantage = rewards - values

        # Record the mean advs before norm for debugging
        d_adv = tf.reduce_mean(advantage)
        if norm_advs:
            advantage = (advantage - tf.reduce_mean(advantage)) / (
                tf.math.reduce_std(advantage) + 1e-8
            )

        oldlogits, _, _ = self.old_model(observations, states=states)
        old_neglogp = self.old_model.distribution(oldlogits, shoot_masks).neglogp(
            actions
        )

        with tf.GradientTape() as tape:
            logits, vpred, _ = self.model(observations, states=states, masks=dones)
            pd = self.model.distribution(logits, shoot_masks)
            neglogp = pd.neglogp(actions)
            # Actor loss
            # PPO
            #! e^(-log(pi_old) -- log(pi)) = pi/pi_old
            ratio = tf.exp(old_neglogp - neglogp)
            ratio_clipped = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)

            # a_losses = advantage * ratio
            # a_losses = advantage * pd.neglogp(actions)[:, tf.newaxis]  # old loss
            a_losses = tf.minimum(advantage * ratio, advantage * ratio_clipped)
            a_loss = -tf.reduce_mean(a_losses)

            # Value function loss
            # c_losses = (vpred[:, 0] - rewards) ** 2
            c_losses = (vpred[:, 0] - returns) ** 2
            c_loss = tf.reduce_mean(c_losses)

            entropies = [e * s for e, s in zip(pd.entropy(), [1.2, 1.0, 1.0, 1.0])]
            entropy_reg = tf.reduce_mean(entropies)
            loss = (
                a_loss
                + (c_loss * self.critic_scale)
                - (entropy_reg * self.entropy_scale)
            )

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
        d_grads_actor = tf.reduce_mean(
            [
                tf.reduce_mean(g)
                for g, v in zip(grads, training_variables)
                if v.name.startswith("actor/")
            ]
        )

        d_val = tf.reduce_mean(vpred)
        return Losses(
            loss,
            a_loss,
            c_loss,
            ratio,
            ratio_clipped,
            entropy_reg,
            entropies,
            d_adv,
            d_val,
            d_grads,
            d_grads_actor,
        )

    def copy_to_oldmodel(self):
        # Assign current policy to old policy before update
        for v1, v2 in zip(self.model.variables, self.old_model.variables):
            v2.assign(v1)

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
