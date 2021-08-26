import numpy as np




OBS_SPACE = 23
ACTION_DIMS = (1, 3, 3, 3)


class Env:
    def __init__(self) -> None:
        pass

    def step(self):
        return


class MultiEnv:
    def __init__(self) -> None:
        pass

class Runner:
    def __init__(self, *, env, model, nsteps, gamma, lam) -> None:
        self.model = model
        self.nenv = nenv = 1
        self.batch_ob_shape = (nenv * nsteps,) + OBS_SPACE
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        m_actions = np.zeros((self.steps, self.n_robots, len(ACTION_DIMS)), dtype=np.uint8)
        m_rewards = np.zeros((self.steps + 1, self.n_robots), dtype=np.float32)
        m_values = np.zeros((self.steps + 1, self.n_robots), dtype=np.float32)
        m_dones = np.zeros((self.steps, self.n_robots), dtype=np.bool)
        m_observations = np.zeros((self.steps, self.n_robots, 23), dtype=np.float32)
        m_states = np.zeros((self.steps, 2, self.n_robots, 32), dtype=np.float32)
        m_shoot_masks = np.zeros((self.steps, self.n_robots), dtype=np.bool)

        for i in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)