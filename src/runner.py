import time
from threading import Event

import numpy as np
from robots.battle import MultiBattle
from utils import discount_with_dones


class Memory(object):
    def __init__(self):
        self.mb_obs = []
        self.mb_rewards = []
        self.mb_actions = []
        self.mb_values = []
        self.mb_dones = []
        self.mb_states = []
        self.mb_neglogpac = []

    def append(
        self,
        obs=None,
        rewards=None,
        actions=None,
        values=None,
        dones=None,
        neglogpac=None,
    ):
        if obs is not None:
            self.mb_obs.append(obs)
        if rewards is not None:
            self.mb_rewards.append(rewards)
        if actions is not None:
            self.mb_actions.append(actions)
        if values is not None:
            self.mb_values.append(values)
        if dones is not None:
            self.mb_dones.append(dones)
        if neglogpac is not None:
            self.mb_neglogpac.append(neglogpac)

    def clear(self):
        self.mb_obs = []
        self.mb_rewards = []
        self.mb_actions = []
        self.mb_values = []
        self.mb_dones = []
        self.mb_states = []
        self.mb_neglogpac = []


class Runner(object):
    def __init__(self, env, model, train_steps, render=False):
        self.model = model
        self.memory = None
        self.train_steps = train_steps
        self.iteration = 0
        self.batch_action_shape = 2
        self.batch_ob_shape = 9
        self.gamma = 0.99
        self.env = env
        self.robots = env.robots
        self.render = render
        self.states = self.model.initial_state

        self.should_stop = Event()

    def prepare(self):
        mb_obs = np.asarray(self.memory.mb_obs, dtype="float32").swapaxes(1, 0).reshape(-1, self.batch_ob_shape)
        mb_rewards = np.asarray(self.memory.mb_rewards, dtype="float32").swapaxes(1, 0)
        mb_actions = np.asarray(self.memory.mb_actions, dtype="float32").swapaxes(1, 0)
        mb_values = np.asarray(self.memory.mb_values, dtype="float32").swapaxes(1, 0)
        mb_dones = np.asarray(self.memory.mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_neglogpacs = np.asarray(self.memory.mb_neglogpac, dtype=np.float32).swapaxes(1, 0).reshape(-1)
        mb_states = self.memory.mb_states
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).flatten().tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                # rewards = rewards.tolist()
                # dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(
                        np.array(rewards.tolist() + [value]), np.array(dones.tolist() + [0]), self.gamma
                    )[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards
        mb_actions = mb_actions.reshape(-1, self.batch_action_shape)
        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_neglogpacs

    def run(self):
        self.memory = Memory()
        self.obs = self.env.get_obs()
        self.dones = self.env.get_dones()
        # Oddly the state is reset inside of the lstm that is used on Done.
        self.memory.mb_states = self.states

        for i in range(self.train_steps):
            actions, values, states, neglogpac = self.model.step(self.obs, S=self.states, M=self.dones)
            obs, rewards, dones = self.env.step(actions)
            # Why is this copy(obs)
            self.memory.append(np.copy(self.obs), rewards, actions, values, self.dones, neglogpac)
            self.obs = obs
            self.dones = dones
            self.states = states

        self.memory.mb_dones.append(self.dones)
        return self.prepare()

    def test(self, simrate=30):
        self.obs = self.env.get_obs()
        self.dones = self.env.get_dones()
        interval = 1 / simrate
        last_sim = 0
        while not self.should_stop.is_set():
            if (time.time() - last_sim) >= interval:
                actions, values, states, _ = self.model.test(self.obs, S=self.states, M=self.dones)
                obs, rewards, dones = self.env.step(actions=actions)
                print(actions.flatten(), self.obs.flatten())
                self.obs = obs
                self.dones = dones
                self.states = states
                last_sim = time.time()

    def train(self):
        for update in range(1, 100000):
            t = time.time()
            obs, states, rewards, masks, actions, values, neglogpac = self.run()
            t = time.time() - t
            res = self.model.train(obs, states, rewards, masks, actions, values, neglogpac, 0.2)
            print(
                self.iteration,
                "reward:",
                round(rewards.mean(), 5),
                "critic:",
                round(res["value_loss"].mean(), 5),
                "fps:",
                self.train_steps / t,
                "samples:",
                len(obs),
            )
            self.iteration += 1


class Env(MultiBattle):
    def __init__(self, *args, **kwargs):
        super(Env, self).__init__(*args, **kwargs)
        # Get ticks from battle use to check rewards
        self.tick = 0
        self.previous_energies = None

    @property
    def all_robots(self):
        return [robot for battle in self.battles for robot in battle.robots]

    def get_obs(self):
        return np.stack([robot.get_obs() for robot in self.all_robots])

    def get_rewards(self):
        return np.stack([robot.energy - robot.previous_energy for robot in self.all_robots])

    def get_dones(self):
        return np.stack([robot.get_done() for robot in self.all_robots])

    def delta(self, actions=None):
        """
        Returns observations, rewards, dones after running the previous actions
        """
        self.previous_energies = np.stack([robot.energy for robot in self.all_robots])
        # Tick all robots
        for i, robot in enumerate(self.all_robots):
            robot.delta(self.tick, actions[i])

        # TODO figure out how to do this before reset
        # # Give a reward for winning
        # for battle in self.battles:
        #     if battle.check_round_over():
        #         for robot in battle.robots:
        #             if not robot.dead:
        #                 robot.energy += 100

        rewards = self.get_rewards()
        for robot in self.all_robots:
            robot.previous_energy = robot.energy
            robot.energy -= 0.01

        return self.get_obs(), rewards, self.get_dones()
