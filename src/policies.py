from utils import batch_to_seq, seq_to_batch, tf, tfp, fully, mish
from baselines.common.distributions import DiagGaussianPd
import utils
import numpy as np


def mlp(width=128):
    def network_fn(
        X,
    ):
        h = tf.layers.flatten(X)
        with tf.variable_scope("shared"):
            shared = fully(h, width, scope="s1", summary_w=False, reg=True, activation=mish)
            shared = fully(shared, width, scope="s2", summary_w=False, reg=True, activation=mish)

        with tf.variable_scope("actor"):
            out = fully(shared, width, scope="a1", summary_w=False, reg=True, activation=mish)
            latent = fully(out, width, scope="action_latent", summary_w=False, reg=True, activation=mish)
            tf.summary.histogram("latent", latent)

        with tf.variable_scope("critic"):
            vf_latent = fully(shared, width, scope="vf_latent", summary_w=True, activation=mish)
            tf.summary.histogram("vf_latent", vf_latent)

        return latent, vf_latent

    return network_fn


def lstm(nlstm=128, layer_norm=False):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        S = tf.placeholder(tf.float32, [nenv, 2 * nlstm], "states")  # states
        M = tf.placeholder(tf.float32, [nbatch], "mask")  # mask (done t-1)

        xs = batch_to_seq(tf.layers.flatten(X), nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope="lnlstm", nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope="lstm", nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {"S": S, "M": M, "state": snew, "initial_state": initial_state}

    return network_fn


class Policy(object):
    def __init__(self, nenvs, n_steps, num_actions, state_features, sess, normal=False):
        with tf.variable_scope("policy"):
            self._sess = sess
            self.X = tf.placeholder("float32", (nenvs * n_steps, state_features), "obs")
            X, model_args = lstm(128)(self.X, nenvs)
            self.__dict__.update(model_args)
            latent, vf_latent = mlp(128)(X)

            self.action_space = action_space = fully(latent, num_actions * 2, scope="mu")
            tf.summary.histogram("ac", action_space)
            self.pd = DiagGaussianPd(action_space)

            # if normal:
            #     chol = fully(
            #         latent,
            #         num_actions,
            #         scope="var",
            #         reg=True,
            #         activation=None,
            #     )
            #     tf.summary.histogram("chol", chol)
            #     self.pd = tfp.distributions.Normal(
            #         mu,
            #         tf.clip_by_value(tf.exp(chol), 1e-3, 50),
            #         allow_nan_stats=False,
            #     )
            # else:
            #     chol_matrix = fully(
            #         latent,
            #         num_actions * (num_actions + 1) // 2,
            #         scope="chol",
            #         summary_w=False,
            #         activation=tf.nn.softplus,
            #     )
            #     tf.summary.histogram("chol", chol_matrix)
            #     chol_matrix = tfp.math.fill_triangular(chol_matrix, upper=False, name=None)
            #     self.pd = tfp.distributions.MultivariateNormalTriL(mu, chol_matrix, allow_nan_stats=False)

            self.action = tf.clip_by_value(self.pd.sample(), -1.0, 1.0)
            tf.summary.histogram("action", self.action)
            self.neglogp = self.pd.neglogp(self.action)
            self.vf = fully(vf_latent, 1, summary_w=True, activation=None, scope="value")
            tf.summary.histogram("value", self.vf)

    def step(self, obs, S, M):
        action, value, state, neglogp = self._sess.run(
            [self.action, self.vf, self.state, self.neglogp],
            {self.X: obs, self.S: S, self.M: M},
        )
        return action, value, state, neglogp

    def test(self, obs, S, M):
        action, value, state, neglogp = self._sess.run(
            [self.pd.mode(), self.vf, self.state, self.neglogp],
            {self.X: obs, self.S: S, self.M: M},
        )
        return action, value, state, neglogp

    def value(self, obs, S, M):
        """
        Returns the value of state at obs
        :param obs:
        :return:
        """
        return self._sess.run(self.vf, {self.X: obs, self.S: S, self.M: M})
