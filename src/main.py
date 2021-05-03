import argparse
import time

import numpy as np
import tensorflow as tf
from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *

import wandb
from model import Model, Trainer
from utils import discounted, Timer, cast
from wrapper import Dummy

WANDBOFF = True
timer = Timer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--verbose", action='store_true', help="Print debugging information.")
    parser.add_argument("--wandboff", action='store_true', help="Turn off W&B logging.")
    parser.add_argument('-r', "--render", action='store_true', help="Render battles during training.")
    parser.add_argument('-n', "--envs", type=int, default=25, help="Number of envs to use for training.")
    parser.add_argument('-s', "--steps", type=int, default=100, help="Number of steps to use for training.")
    return parser.parse_args()


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
        self.robot_map = {}
        self.inv_robot_map = {}

        for eng in self.engines:
            eng.init(robot_kwargs={"all_robots": eng.robots})
            for robot in eng.robots:
                idx = len(self.robot_map)
                self.robot_map[idx] = robot
                self.inv_robot_map[robot] = idx
                robot.memory = []

    @cast(tf.float32)
    def get_obs(self):
        return tf.stack([self.robot_map[i].get_obs() for i in range(len(self.robot_map))])

    def init_states(self,):
        self.states = model.initial_state(len(self.robot_map))

    def run(self):
        timer.split("run")
        observations = self.get_obs()
        states = tf.unstack(tf.cast(self.states, tf.float32))
        actions, values, neglogps, new_states = model.sample(observations, states)

        # Assign actions and records the next states
        for i, robot in self.robot_map.items():
            robot.assign_actions(actions[i])

        # Use this to only update non_done robots
        pre_alive = [(i, robot) for i, robot in self.robot_map.items() if robot.alive]

        timer.split("step")
        for i, eng in enumerate(self.engines):
            eng.step()
        timer.add_diff("step")
        timer.split("memetc")

        for idx, robot in pre_alive:
            done = not robot.alive
            reward = (robot.energy-robot.previous_energy)/100
            if done:
                # Zero out the index for the done robot
                new_states[:, idx] = 0
                if robot.energy > 0:
                    reward += 1 + robot.energy  # How about this?
                else:
                    reward -= 1

            timer.split("mem")
            robot.memory.append((
                reward,
                actions[idx],
                values[idx],
                neglogps[idx],
                observations[idx],
                self.states[:, idx],
                done
            ))
            timer.add_diff('mem')

        # Overwrite state tracking with new states
        self.states = new_states

        for eng in self.engines:
            if eng.is_finished():
                eng.init(robot_kwargs={"all_robots": eng.robots})

        timer.add_diff("memetc")
        timer.add_diff("run")

    def train(self):
        observations = self.get_obs()
        states = tf.unstack(tf.cast(self.states, tf.float32))
        _, last_values, _ = model.run(observations, states)

        timer.split("prep")

        b_rewards = []
        b_action = []
        b_neglogp = []
        b_values = []
        b_obs = []
        b_states = []

        for i, robot in self.robot_map.items():
            # Take apart memories.
            (rewards, actions, values, neglogps,  observations, states, dones) = zip(*robot.memory)
            # Clear memories of old data
            robot.memory = []

            disc_reward = discounted(np.array(rewards), np.array(dones), last_values[i], 0.9)
            b_rewards.append(disc_reward)
            b_action.append(actions)
            b_neglogp.append(neglogps)
            b_values.append(values)
            b_obs.append(observations)
            b_states.append(states)

        timer.add_diff("prep")

        b_rewards = tf.concat(b_rewards, axis=0)[:, tf.newaxis]
        b_action = tf.concat(b_action, axis=0)
        b_neglogp = tf.concat(b_neglogp, axis=0)
        b_values = tf.concat(b_values, axis=0)
        b_obs = tf.concat(b_obs, axis=0)
        b_states = tf.concat(b_states, axis=0)
        timer.split("train")
        # Pass data to trainer, managing the model.
        losses = trainer.train(b_obs, tf.unstack(b_states, axis=1), b_rewards, b_action, b_neglogp, b_values)
        # Checkpoint manager will save every x steps
        trainer.checkpoint()
        timer.add_diff("train")

        print("Prepping times", timer.mean_diffs("prep"))
        print("Training times", timer.mean_diffs("train"))
        print("\tRunning times", timer.mean_diffs("run"))
        print("\tStep", timer.mean_diffs("step"))
        print("\tMem Etc.", timer.mean_diffs("memetc"))
        print("\tMem", timer.mean_diffs("mem"))
        if not WANDBOFF:
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


def main(steps, envs, render=False, wandboff=False):
    global WANDBOFF
    WANDBOFF = wandboff
    # Initate WandB before running
    if not WANDBOFF:
        wandb.init(project='robots_rl', entity='jchacks')
        config = wandb.config
        config.rlenv = "all_actions"
        config.action_space = "2,3,3,3"
        config.critic_scale = trainer.critic_scale
        config.entropy_scale = trainer.entropy_scale
        config.max_steps = steps
        config.envs = envs

    runner = Runner(envs)
    if render:
        # Todo clean up this interaction with Engine and Battle
        from robots.app import App
        from wrapper import AITrainingBattle

        app = App(size=(600, 600))
        eng = runner.engines[0]
        battle = AITrainingBattle(eng.robots, (600, 600), eng=eng)
        app.child = battle
        runner.app = app

    for iteration in range(1000000):
        timer.split()
        # reset the lstm states
        runner.init_states()
        for _ in range(steps):
            runner.run()
            if render:
                app.step()
        runner.train()
        print(iteration, timer.add_diff(), timer.mean_diffs())


if __name__ == "__main__":
    args = parse_args()
    main(steps=args.steps,
         envs=args.envs,
         wandboff=args.wandboff,
         render=args.render)
