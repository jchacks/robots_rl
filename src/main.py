from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *
from model import Model, Trainer
import numpy as np
import tensorflow as tf
from wrapper import Dummy, AITrainingBattle
from utils import Memory, discounted, TURNING, MOVING
import time
import wandb

ACTION_DIMS = (2, 3, 3, 3)
model = Model(ACTION_DIMS)
trainer = Trainer(model)
trainer.restore()


def make_eng():
    robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
    size = (600, 600)
    eng = Engine(robots, size)

    # Simplify battles
    eng.ENERGY_DECAY_ENABLED = True
    eng.GUN_HEAT_ENABLED = True
    eng.BULLET_COLLISIONS_ENABLED = False
    return eng


class Runner(object):
    def __init__(self, num) -> None:
        self.num = num
        self.engines = [make_eng() for _ in range(num)]
        for eng in self.engines:
            eng.init()
            eng.memories = {r: Memory('rewards,action,neglogp,values,obs,state,dones') for r in eng.robots}
            eng.states = tf.stack(model.lstm.get_initial_state(batch_size=2, dtype=tf.float32))
            eng.total_reward = {r: 0 for r in eng.robots}

    def get_obs(self):
        return tf.reshape(
            tf.stack([[r.get_obs() for r in eng.robots] for eng in self.engines]),
            (self.num*2, -1)
        )

    def get_states(self):
        return tf.reshape(tf.stack([eng.states for eng in self.engines], axis=1), (2, self.num*2, -1))

    def run(self):
        observations = self.get_obs()
        states = self.get_states()
        actions, values, neglogps, new_states = model.sample(observations, tf.unstack(states))

        new_states = tf.reshape(tf.stack(new_states), (2, self.num, 2, -1))
        actions = tf.reshape(actions, (self.num, 2, -1))

        # Assign actions and records the next states
        for i, eng in enumerate(self.engines):
            for j, robot in enumerate(eng.robots):
                robot.assign_actions(actions[i][j])
            eng.states = new_states[:, i]
            eng.step()

        # Reshape back to num_envs, 2 robots, dims
        observations = tf.reshape(observations, (self.num, 2, -1))
        values = tf.reshape(values, (self.num, 2, -1))
        neglogps = tf.reshape(neglogps, (self.num, 2, -1))
        states = tf.reshape(states, (2, self.num, 2, -1))

        for i, eng in enumerate(self.engines):
            for j, robot in enumerate(eng.robots):
                reward = (robot.energy-robot.previous_energy)/100
                if eng.is_finished():
                    if robot.energy > 0:
                        reward += 1
                    else:
                        reward -= 1

                eng.total_reward[robot] += reward
                eng.memories[robot].append(
                    rewards=reward,
                    action=actions[i, j],
                    values=values[i, j],
                    neglogp=neglogps[i, j],
                    obs=observations[i, j],
                    state=states[:, i, j],
                    dones=eng.is_finished()
                )

            if eng.is_finished():
                eng.total_reward = {r: 0 for r in eng.robots}
                eng.init()
                eng.states = model.lstm.get_initial_state(batch_size=2, dtype=tf.float32)

    def train(self,):
        observations = self.get_obs()
        states = self.get_states()
        _, last_values, _ = model.run(observations, states)
        last_values = tf.reshape(last_values, (self.num, 2, -1))

        b_rewards = []
        b_action = []
        b_neglogp = []
        b_values = []
        b_obs = []
        b_states = []

        for i, eng in enumerate(self.engines):
            for j, robot in enumerate(eng.robots):
                mem = eng.memories[robot]
                disc_reward = discounted(np.array(mem['rewards']), np.array(mem['dones']), last_values[i][j], 0.9)
                b_rewards.append(disc_reward)
                b_action.append(mem['action'])
                b_neglogp.append(mem['neglogp'])
                b_values.append(mem['values'])
                b_obs.append(mem['obs'])
                b_states.append(mem['state'])

            # Clear memories of old data
            eng.memories = {r: Memory('rewards,action,neglogp,values,obs,state,dones') for r in eng.robots}

        b_rewards = tf.concat(b_rewards, axis=0)[:, tf.newaxis]
        b_action = tf.concat(b_action, axis=0)
        b_neglogp = tf.concat(b_neglogp, axis=0)
        b_values = tf.concat(b_values, axis=0)
        b_obs = tf.concat(b_obs, axis=0)
        b_states = tf.concat(b_states, axis=0)
        losses = trainer.train(b_obs, tf.unstack(b_states, axis=1), b_rewards, b_action, b_neglogp, b_values)
        wandb.log({
            "loss": losses[0],
            "actor": losses[1],
            "critic": losses[2],
            "entropy": losses[3],
            "advantage": losses[4],
            "values": losses[5]
        })
        if np.isnan(losses[0].numpy()):
            raise RuntimeError
        

max_steps = 30

# Initate WandB before running
wandb.init(project='robots_rl', entity='jchacks')
config = wandb.config
config.rlenv = "all_actions"
config.action_space = "2,3,3,3"
config.max_steps = max_steps
runner = Runner(100)

for iteration in range(1000000):
    # Create a memory per player
    steps = 0
    for i in range(max_steps):
        runner.run()
    runner.train()
    print(iteration)
