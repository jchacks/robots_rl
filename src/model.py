import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers, regularizers
from tensorflow.python.ops.variables import Variable
from distributions import GroupedPd, CategoricalPd, MaskedBernoulliPd, MaskedCategoricalPd
from utils import PROJECT_ROOT
from collections import namedtuple
import operator

import os

if os.environ.get("DEBUG"):
    tf.config.run_functions_eagerly(True)

Losses = namedtuple(
    "Losses",
    "loss,actor,critic,ratio,ratio_clipped,entropy,entropies,"
    "advantage,value,d_grads,d_grads_actor,d_grads_critic,reg_loss",
)

ACTION_DIMS = (1, 3 * 3 * 3)
REG_RATE = 1e-3


def map_numpy(func):
    def inner(*args, **kwargs):
        return map(operator.methodcaller("numpy"), func(*args, **kwargs))

    return inner


class LSTM(tf.keras.layers.Layer):
    def __init__(self, units, init_scale=1.0, **kwargs):
        self.kernelx_regularizer = kwargs.pop("kernelx_regularizer", None)
        self.kernelh_regularizer = kwargs.pop("kernelh_regularizer", None)

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
            regularizer=self.kernelx_regularizer,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernelh = self.add_weight(
            "kernelh",
            shape=[self.units, self.units * 4],
            initializer=self.kernelh_initializer,
            regularizer=self.kernelh_regularizer,
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

    def call(self, inputs, states, dones=None):
        """Process input to LSTM returning outputs/states

        Args:
            inputs ([type]): Sequence of inputs (timesteps, batchsize, features)
            states ([type]): The first state provided for this sequence
            masks ([type], optional): Masks for resetting state. Defaults to None.

        Returns:
            [type]: [description]
        """
        c, h = tf.unstack(states)
        masks = 1 - dones if dones is not None else tf.ones(len(inputs))
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


class LearningNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-2):
        super(LearningNormalization, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = int(input_shape[-1])
        self.means = Variable(tf.zeros(last_dim, self.dtype), trainable=False)
        self.stds = Variable(tf.ones(last_dim, self.dtype), trainable=False)

    def call(self, x: tf.Tensor, training=False):
        if training:
            _x = tf.reshape(x, (-1, x.shape[-1]))
            print("Norm Tracing, training")
            mean = tf.math.reduce_mean(_x, self.axis)
            stddev = tf.math.reduce_std(_x, self.axis)
            self.means.assign_sub((1 - 0.999) * (self.means - mean))
            self.stds.assign_sub((1 - 0.999) * (self.stds - stddev))
        return tf.clip_by_value((x - self.means) / (self.stds + 1e-8),-5.0,5.0)


class Critic(tf.Module):
    def __init__(self, name="critic") -> None:
        super().__init__(name=name)
        self.d1 = layers.Dense(
            16,
            activation="elu",
            kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )
        self.bn1 = layers.BatchNormalization()
        self.d2 = layers.Dense(
            16,
            activation="elu",
            kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )
        self.o = layers.Dense(
            1, kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE)
        )

    @tf.Module.with_name_scope
    def __call__(self, x):
        # x = self.d1(x)
        # x = self.d2(x)
        return self.o(x)

    @property
    def losses(self):
        return self.d1.losses


class Actor(tf.Module):
    def __init__(self, action_space, name="actor"):
        super().__init__(name=name)
        self.num_actions = np.sum(action_space)
        self.d1 = layers.Dense(
            64,
            activation="elu",
            kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )
        self.bn1 = layers.BatchNormalization()
        self.d2 = layers.Dense(
            64,
            activation="elu",
            kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )
        self.o = layers.Dense(
            self.num_actions,
            kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.o(x)

    @property
    def losses(self):
        return self.d1.losses


class Model(tf.Module):
    def __init__(self, action_space, name="model", training=False):
        super().__init__(name=name)
        self.training = training
        self.action_space = action_space
        # self.bn0 = layers.BatchNormalization()
        self.norm = LearningNormalization()
        self.lstm = LSTM(
            units=128,
            kernelx_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
            kernelh_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )
        self.d1 = layers.Dense(
            128,
            activation="elu",
            kernel_regularizer=regularizers.l1_l2(l1=REG_RATE, l2=REG_RATE),
        )

        self.actor = Actor(self.action_space)
        self.critic = Critic()

    @tf.Module.with_name_scope
    def __call__(self, obs, states, dones=None):
        obs = self.norm(obs, training=self.training)
        latent, states = self.lstm(obs, states, dones)
        # latent = self.d1(latent)
        return self.actor(latent), self.critic(latent), tf.stack(states)

    @property
    def losses(self):
        return tf.concat([self.lstm.losses, self.actor.losses, self.critic.losses], 0)

    def initial_state(self, batch_size):
        return tf.stack(
            self.lstm.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        )

    def distribution(self, logits, shoot_mask):
        logits = tf.split(logits, self.action_space, axis=-1)
        return GroupedPd(
            [
                MaskedBernoulliPd(logits[0], shoot_mask),
                # MaskedCategoricalPd(logits[1], shoot_mask[:,tf.newaxis]),
                CategoricalPd(logits[1]),
                # CategoricalPd(logits[2]),
                # CategoricalPd(logits[3]),
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

    @tf.function
    def sample(self, obs, states, shoot_mask):
        print(
            f"Tracing sample, obs {obs.shape} {obs.dtype}, "
            f"states {states.shape} {states.dtype}, "
            f"mask {shoot_mask.shape} {shoot_mask.dtype}"
        )
        logits, value, states = self(obs, states)
        dist = self.distribution(logits, shoot_mask)
        actions = dist.sample()
        return actions, value, states

    def run(self, obs, states, shoot_mask):
        logits, value, states = self(obs, states)
        dist = self.distribution(logits, shoot_mask)
        return dist.mode(), value, states


class ModelManager:
    def __init__(self, model=None) -> None:
        self.model = Model(ACTION_DIMS) if model is None else model
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        self.save_path = f"{PROJECT_ROOT}/ckpts"
        interval = 10
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            self.save_path,
            max_to_keep=20,
            keep_checkpoint_every_n_hours=1,
            step_counter=self.ckpt.step,
            checkpoint_interval=interval,
        )

    def checkpoint(self):
        self.ckpt.step.assign_add(1)
        save_path = self.manager.save()
        if save_path:
            print(
                "Saved checkpoint for step {}: {}".format(
                    int(self.ckpt.step), save_path
                )
            )
            return True
        return False

    def restore(self, partial=None, offset=None):
        if offset is None:
            checkpoint = tf.train.latest_checkpoint(self.save_path)
        else:
            # Check checkpoints otherwise load None
            try:
                checkpoint = self.manager.checkpoints[offset]
            except IndexError:
                checkpoint = None

        status = self.ckpt.restore(checkpoint)

        if partial:
            status.expect_partial()
        elif partial is False:
            status.assert_consumed()
        if checkpoint is not None:
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")


class Trainer(object):
    def __init__(
        self,
        model,
        critic_scale=1.0,
        entropy_scale=3e-2, 
        learning_rate=5e-4,
        epsilon=0.2,
    ) -> None:
        """Class to manage training a model.
        Contains Optimiser and training function.

        Args:
            model ([type]): Model to train.
            save_path (str, optional): Checkpoint path. Defaults to '../ckpts'.
            interval (int, optional): Interval on which to save checkpoints. Defaults to 500.
            critic_scale (float, optional): Scale of critic in the loss function. Defaults to 0.2.
            entropy_scale (float, optional): Scale of entropy in the loss function. Defaults to 0.05.
        """
        self.learning_rate = learning_rate
        # self.optimiser = tf.keras.optimizers.SGD(
        #     learning_rate=self.learning_rate, momentum=0.5, nesterov=True
        # )
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.ret_stddev_mean = None

        self.sample_model = model
        self.train_model = Model(ACTION_DIMS, training=True)

        self.model_manager = ModelManager(self.train_model)
        self.model_manager.restore()
        self.copy_to_current_model()

        self.critic_scale = critic_scale
        self.entropy_scale = entropy_scale
        self.epsilon = epsilon

    def copy_to_current_model(self):
        print("Copying weights to sample_model")
        # Assign current policy to old policy before update
        for v1, v2 in zip(self.train_model.variables, self.sample_model.variables):
            v2.assign(v1)

    @tf.function
    def train(
        self,
        observations,
        states,
        advantage,
        returns,
        actions,
        shoot_masks,
        dones,
        norm_advs=True,
        print_grads=False,
        max_grad_norm=0.5,
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

        ret_stddev = tf.math.reduce_std(returns)
        # Create ret stddev mean and initialise
        if self.ret_stddev_mean is None:
            self.ret_stddev_mean = tf.Variable(
                ret_stddev, trainable=False, dtype=tf.float32
            )
        self.ret_stddev_mean.assign_sub((1 - 0.9) * (self.ret_stddev_mean - ret_stddev))
        returns = returns / (self.ret_stddev_mean + 1e-8)

        # Record the mean advs before norm for debugging
        d_adv = tf.reduce_mean(advantage)
        if norm_advs:
            advantage = (advantage - tf.reduce_mean(advantage)) / (
                tf.math.reduce_std(advantage) + 1e-8
            )

        oldlogits, _, _ = self.sample_model(observations, states=states)
        old_neglogp = self.sample_model.distribution(oldlogits, shoot_masks).neglogp(
            actions
        )

        with tf.GradientTape() as tape:
            logits, vpred, _ = self.train_model(
                observations, states=states, dones=dones
            )
            pd = self.train_model.distribution(logits, shoot_masks)
            neglogp = pd.neglogp(actions)
            # Actor loss
            # PPO
            #! e^(-log(pi_old) -- log(pi)) =e^log(pi/pi_old) = pi/pi_old
            ratio = tf.exp(old_neglogp - neglogp)
            ratio_clipped = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)

            pg_loss1 = -advantage * ratio
            pg_loss2 = -advantage * ratio_clipped
            # a_losses = advantage * pd.neglogp(actions)[:, tf.newaxis]  # old loss
            a_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

            # Value function loss
            c_loss = tf.reduce_mean(tf.square(vpred[:, 0] - returns))

            entropies = pd.entropy()
            entropy_reg = (-self.entropy_scale) * tf.reduce_mean(entropies)
            reg_loss = tf.reduce_mean(self.train_model.losses)
            loss = a_loss + (c_loss * self.critic_scale) + entropy_reg + reg_loss

        training_variables = tape.watched_variables()
        grads = tape.gradient(loss, training_variables)
        if print_grads:
            # Print grads
            for g, v in zip(grads, training_variables):
                max_g = tf.reduce_max(g)
                tf.print(v.name, max_g)

        if max_grad_norm:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        # Apply grads
        grads_and_vars = zip(grads, training_variables)
        self.optimiser.apply_gradients(grads_and_vars)

        d_grads = tf.reduce_mean([tf.reduce_mean(g) for g in grads])
        d_grads_actor = [
            tf.reduce_mean(g)
            for g, v in zip(grads, training_variables)
            if v.name.startswith("actor/")
        ]

        d_grads_critic = [
            tf.reduce_mean(g)
            for g, v in zip(grads, training_variables)
            if v.name.startswith("critic/")
        ]

        d_val = tf.reduce_mean(vpred)
        return Losses(
            loss=loss,
            actor=a_loss,
            critic=c_loss,
            ratio=ratio,
            ratio_clipped=ratio_clipped,
            entropy=entropy_reg,
            entropies=entropies,
            advantage=d_adv,
            value=d_val,
            d_grads=d_grads,
            d_grads_actor=d_grads_actor,
            d_grads_critic=d_grads_critic,
            reg_loss=reg_loss,
        )
