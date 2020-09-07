import time

from baselines.a2c.utils import Scheduler
from utils import fully, tf, tfp
from policies import Policy


class Model(object):
    def __init__(self, nsteps, nenvs, state_features, num_actions=2, restore=True, max_grad_norm=0.4):
        self.is_training = 1.0
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.nsteps = nsteps
        self.nenvs = nenvs
        nbatch = nsteps * nenvs

        self.num_actions = num_actions
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self._sess = tf.Session(config=config)

        self.ent_coef = 0.005
        self.vf_coef = 0.5  # 0.5 default
        self.lr = 7e-4
        self.alpha = 0.99
        self.epsilon = 1e-5
        lrschedule = "constant"
        self.lr = Scheduler(v=self.lr, nvalues=80e6, schedule=lrschedule)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            self.step_model = step_model = Policy(nenvs, 1, num_actions, state_features, self._sess)
            # train_model is used to train our network
            self.train_model = train_model = Policy(nenvs, nsteps, num_actions, state_features, self._sess)

        with tf.variable_scope("optimiser", reuse=tf.AUTO_REUSE):
            self.LR = LR = tf.placeholder(tf.float32, [], "learning_rate")
            self.ADV = ADV = tf.placeholder("float32", [nbatch], "advantage")
            self.A = A = tf.placeholder("float32", [nbatch, self.num_actions], "action")
            self.R = R = tf.placeholder("float32", [nbatch], "reward")

            # Used to scale losses
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder("float32", [nbatch], "old_neglogpac")
            self.OLDVALUE = OLDVALUE = tf.placeholder("float32", [nbatch], "old_value")

            # Cliprange
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
            tf.summary.scalar("lr", LR)

            ### Actor loss
            neglogpac = train_model.pd.neglogp(A)
            # Calculate ratio (pi current policy / pi old policy)
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            # Record some stats
            approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            tf.summary.scalar("approxkl", approxkl)
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            tf.summary.scalar("clipfrac", clipfrac)
            entropy = tf.reduce_mean(train_model.pd.entropy())
            tf.summary.scalar("entropy", entropy)

            # Calculate pg_loss
            pg_losses1 = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            tf.summary.scalar("pg_loss", pg_loss)

            ### Critic loss
            vpred = train_model.vf
            # Clip the value to reduce variability during Critic training
            vpredclipped = OLDVALUE + tf.clip_by_value(train_model.vf - OLDVALUE, -CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)

            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            tf.summary.scalar("vf_loss", vf_loss)

            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            # Optimiser
            params = tf.trainable_variables("model")
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

            self.grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), max_grad_norm)
            grads, var = zip(*self.trainer.compute_gradients(loss, params))
            if max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_var = list(zip(grads, var))
            self.grads = grads
            self.var = var
            self._train_op = self.trainer.apply_gradients(grads_and_var, global_step=self.global_step)

        self.loss_names = ["policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac"]
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        # Map functions from sub models
        self.step = step_model.step
        self.test = step_model.test
        self.value = step_model.value
        self.initial_state = step_model.initial_state

        self.init()
        self.save_path = "../checkpoint/model"
        if restore:
            self.restore()
        self.summary()

    def train(self, obs, states, rewards, masks, actions, values, neglogpacs, cliprange, grads=False, norm_advs=True):
        if not self._summ_writer:
            self._summ_writer = tf.summary.FileWriter("../train/{0}".format(int(time.time())), self._sess.graph)

        train_fetches = {
            "minimiser": self._train_op,
            "step": self.global_step,
            "summary": self.summ,
        }
        train_fetches.update(dict(zip(self.loss_names, self.stats_list)))
        if grads:
            train_fetches["grads"] = self.grads

        advs = rewards - values
        if norm_advs:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for step in range(len(obs)):
            cur_lr = self.lr.value()

        res = self._sess.run(
            train_fetches,
            {
                self.train_model.X: obs,
                self.train_model.S: states,
                self.train_model.M: masks,
                self.LR: cur_lr,
                self.ADV: advs,
                self.R: rewards,
                self.A: actions,
                self.CLIPRANGE: cliprange,
                self.OLDNEGLOGPAC: neglogpacs,
                self.OLDVALUE: values,
            },
        )

        self._summ_writer.add_summary(res["summary"], global_step=res["step"])
        if res["step"] % 100 == 0:
            self._saver.save(self._sess, self.save_path, global_step=res["step"])
        return res

    def restore(self):
        chkp = tf.train.latest_checkpoint("../checkpoint")
        if chkp is not None:
            print("Restoring chkp: %s " % (chkp,))
            self._saver.restore(self._sess, chkp)

    def init(self):
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(tf.trainable_variables(), save_relative_paths=True)
        self._summ_writer = None

        # self.sig_loss = tf.reduce_mean(tf.nn.relu(self.sig - 0.5) ** 2)
        # self.mu_loss = tf.reduce_mean(self.mu ** 2) * 1e2

    def summary(self):
        self.summ = tf.summary.merge_all(scope="(optimiser/)|(model/policy_1/)")
        return self.summ
