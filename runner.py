import time

import numpy as np
from robots.battle import MultiBattle
from threading import Event
from utils import discount_with_dones

class Memory(object):
    def __init__(self):
        self.mb_obs = []
        self.mb_rewards = []
        self.mb_actions = []
        self.mb_values = []
        self.mb_dones = []

    def append(self, obs, rewards, actions, values, dones):
        self.mb_obs.append(obs)
        self.mb_rewards.append(rewards)
        self.mb_actions.append(actions)
        self.mb_values.append(values)
        self.mb_dones.append(dones)

    def clear(self):
        self.mb_obs = []
        self.mb_rewards = []
        self.mb_actions = []
        self.mb_values = []
        self.mb_dones = []


class Runner(object):
    def __init__(self, env, model, train_steps=50, render=False):
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
        self.should_stop = Event()

    def prepare(self):
        mb_obs = np.asarray(self.memory.mb_obs, dtype='float32').swapaxes(1, 0).reshape(-1, self.batch_ob_shape)
        mb_rewards = np.asarray(self.memory.mb_rewards, dtype='float32').swapaxes(1, 0)
        mb_actions = np.asarray(self.memory.mb_actions, dtype='float32').swapaxes(1, 0)
        mb_values = np.asarray(self.memory.mb_values, dtype='float32').swapaxes(1, 0)
        mb_dones = np.asarray(self.memory.mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.get_value(self.obs).flatten().tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards
        mb_actions = mb_actions.reshape(-1, self.batch_action_shape)
        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, None, mb_rewards, mb_masks, mb_actions, mb_values

    def run(self):
        self.memory = Memory()
        self.obs = self.env.get_obs()
        self.dones = self.env.get_dones()

        for i in range(self.train_steps):
            actions, values = self.model.run(obs=self.obs)
            obs, rewards, dones = self.env.step(actions)
            # Why is this copy(obs)

            self.memory.append(np.copy(self.obs), rewards, actions, values, self.dones)
            self.obs = obs
            self.dones = dones
        self.memory.mb_dones.append(self.dones)
        return self.prepare()

    def test(self, simrate=30):
        self.obs = self.env.get_obs()
        interval = 1/simrate
        last_sim = 0
        while not self.should_stop.is_set():
            if (time.time() - last_sim) >= interval:
                actions, values = self.model.test(obs=self.obs)
                obs, rewards, dones = self.env.step(actions=actions)
                self.obs = obs

    def train(self):
        for update in range(1, 100000):
            t = time.time()
            obs, states, rewards, masks, actions, values = self.run()
            t = time.time() - t
            advs = rewards - values
            res = self.model.train(X=obs,
                                   advantage=advs,
                                   td_target=rewards,
                                   action=actions)
            print(self.iteration,
                  'advantage:', advs.mean(),
                  'reward:', round(rewards.mean(), 3),
                  'critic:', round(res['critic_loss'].mean(), 3),
                  'fps:', self.train_steps / t,
                  'samples:', len(obs))
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
        return np.stack([robot.energy for robot in self.all_robots]) - self.previous_energies

    def get_dones(self):
        return np.stack([robot.get_done() for robot in self.all_robots])

    def delta(self, actions=None):
        self.previous_energies = np.stack([robot.energy for robot in self.all_robots])
        for i, robot in enumerate(self.all_robots):
            robot.delta(self.tick, actions[i])
        return self.get_obs(), self.get_rewards(), self.get_dones()
