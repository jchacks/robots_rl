import argparse
import random
import time

import numpy as np
import tensorflow as tf
from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *

import wandb
from model import Model, Trainer
from utils import Timer, cast, discounted
from wrapper import Dummy

WANDBOFF = True
timer = Timer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--verbose", action='store_true', help="Print debugging information.")
    parser.add_argument("--wandboff", action='store_true', help="Turn off W&B logging.")
    parser.add_argument('-r', "--render", action='store_true', help="Render battles during training.")
    parser.add_argument('-n', "--envs", type=int, default=100, help="Number of envs to use for training.")
    parser.add_argument('-s', "--steps", type=int, default=40, help="Number of steps to use for training.")
    return parser.parse_args()


ACTION_DIMS = (1, 3, 3, 3)
model = Model(ACTION_DIMS)
model = Model(ACTION_DIMS)
old_model = Model(ACTION_DIMS, 'old_model')
trainer = Trainer(model, old_model)
trainer.restore()
trainer.copy_to_oldmodel()

class TrainingEngine(Engine):
    def init_robotdata(self, robot):
        robot.position = np.random.uniform(np.array(self.size))
        robot.base_rotation = random.random() * 360
        robot.turret_rotation = random.random() * 360
        robot.radar_rotation = robot.turret_rotation
        robot.energy = random.randint(10, 100)  # Randomize starting hp


def make_eng():
    robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
    size = (600, 600)
    eng = TrainingEngine(robots, size)

    # Simplify battles
    eng.ENERGY_DECAY_ENABLED = True
    eng.GUN_HEAT_ENABLED = True
    eng.BULLET_COLLISIONS_ENABLED = False
    return eng

class Runner(object):
    def __init__(self, nenvs, steps) -> None:
        self.train_iteration = 0
        self.nenvs = nenvs
        self.steps = steps
        self.gamma = 0.97  # discounted factor

        self.m_actions = np.zeros((steps, nenvs, len(ACTION_DIMS)))
        self.m_values = np.zeros((steps, nenvs))
        self.m_rewards = np.zeros((steps, nenvs))
        self.m_neglogps = np.zeros((steps, nenvs))
        self.m_observations = np.zeros((steps, nenvs, 16))
        self.m_shoot_masks = np.zeros((steps, nenvs), np.bool)
        self.m_dones = np.zeros((steps, nenvs), np.bool)

        self.engines = [make_eng() for _ in range(nenvs)]
        self.robots = []
        self.robot_map = {}
        self.inv_robot_map = {}

        for eng in self.engines:
            eng.init(robot_kwargs={"all_robots": eng.robots})
            for robot in eng.robots:
                idx = len(self.robots)
                self.robots.append(robot)
                self.robot_map[idx] = robot
                self.inv_robot_map[robot] = idx
                robot.memory = []

    @cast(tf.float32)
    def get_obs(self):
        return tf.stack([r.get_obs() for r in self.robots])
    
    @cast(tf.bool)
    def get_shoot_mask(self):
        return tf.stack([r.turret_heat > 0 for r in self.robots])

    def init_states(self,):
        self.states = model.initial_state(len(self.robots))

    def run(self):
        timer.start("run")

        # Use this to only update non_done robots
        pre_alive = [(i, robot) for i, robot in self.robot_map.items() if robot.alive]

        observations = self.get_obs()
        states = tf.unstack(tf.cast(self.states, tf.float32))
        shoot_mask = self.get_shoot_mask()
        actions, values, neglogps, new_states = model.sample(observations, states, shoot_mask)

        # Assign actions and records the next states
        for i, robot in enumerate(self.robots):
            robot.assign_actions(actions[i])

        timer.start("step")
        for i, eng in enumerate(self.engines):
            eng.step()
            # # If going too long then kill both
            # if eng.steps >= 500:
            #     for robot in eng.data:
            #         robot.energy = 0

        timer.stop("step")

        timer.start("post")
        for idx, robot in pre_alive:
            done = not robot.alive
            reward = (robot.energy-robot.previous_energy-1)/100
            if done:
                # Zero out the index for the done robot
                new_states[:, idx] = 0
                if robot.energy > 0:
                    reward += 1 + robot.energy/100
                else:
                    reward -= 1
            
            timer.start("mem")
            robot.memory.append((
                reward,
                actions[idx],
                values[idx],
                neglogps[idx],
                observations[idx],
                self.states[:, idx],
                shoot_mask[idx],
                done
            ))
            timer.stop("mem")

        timer.stop("post")

        # Overwrite state tracking with new states
        self.states = new_states

        # Move this out to train
        for eng in self.engines:
            # If finished clean up
            if eng.is_finished():
                # Reset LSTMS states
                for robot in eng.robots:
                    self.states[:, self.inv_robot_map[robot]] = 0
                # Reset the engine
                eng.init(robot_kwargs={"all_robots": eng.robots})

        timer.stop("run")

    def train(self):
        observations = self.get_obs()
        states = tf.unstack(tf.cast(self.states, tf.float32))
        shoot_mask = self.get_shoot_mask()
        _, last_values, _ = model.run(observations, states, shoot_mask)
        if self.train_iteration == 0:
            _ = old_model.run(observations, states, shoot_mask)
        
        timer.start("prep")

        b_rewards = []
        b_action = []
        b_neglogp = []
        b_values = []
        b_obs = []
        b_states = []
        b_shoot_masks = []

        for i, robot in self.robot_map.items():
            # Take apart memories.
            (rewards, actions, values, neglogps,  observations, states, shoot_mask, dones) = zip(*robot.memory)
            # Clear memories of old data
            robot.memory = []

            disc_reward = discounted(np.array(rewards), np.array(dones), last_values[i], self.gamma)
            b_rewards.append(disc_reward)
            b_action.append(actions)
            b_neglogp.append(neglogps)
            b_values.append(values)
            b_obs.append(observations)
            b_states.append(states)
            b_shoot_masks.append(shoot_mask)

        b_rewards = tf.concat(b_rewards, axis=0)[:, tf.newaxis]
        b_action = tf.concat(b_action, axis=0)

        b_neglogp = tf.concat(b_neglogp, axis=0)
        b_values = tf.concat(b_values, axis=0)
        b_obs = tf.concat(b_obs, axis=0)
        b_states = tf.concat(b_states, axis=0)
        b_shoot_masks = tf.concat(b_shoot_masks, axis=0)
        timer.stop("prep")

        timer.start("train")
        # Pass data to trainer, managing the model.
        losses = trainer.train(b_obs, tf.unstack(b_states, axis=1), b_rewards, b_action, b_neglogp, b_values, b_shoot_masks)
        # Checkpoint manager will save every x steps
        trainer.checkpoint()
        timer.stop("train")
        if not WANDBOFF:
            wandb.log({
                "rewards": wandb.Histogram(b_rewards.numpy(), num_bins=256),
                "loss": losses.loss,
                "actor": losses.actor,
                "critic": losses.critic,
                "ratio": wandb.Histogram(losses.ratio.numpy(),num_bins=256),
                "ratio_clipped": wandb.Histogram(losses.ratio_clipped.numpy(), num_bins=256),
                "entropy": losses.entropy,
                "advantage": losses.advantage,
                "values": losses.value
            })
        if np.isnan(losses[0].numpy()):
            raise RuntimeError
        self.train_iteration += 1


def main(steps, envs, render=False, wandboff=False):
    global WANDBOFF
    WANDBOFF = wandboff
    # Initate WandB before running
    if not WANDBOFF:
        wandb.init(project='robots_rl', entity='jchacks')
        config = wandb.config
        config.critic_scale = trainer.critic_scale
        config.entropy_scale = trainer.entropy_scale
        config.learning_rate = trainer.learning_rate
        config.epsilon = trainer.epsilon
        config.max_steps = steps
        config.envs = envs

    runner = Runner(envs, steps)
    if render:
        # Todo clean up this interaction with Engine and Battle
        from robots.app import App

        from wrapper import AITrainingBattle

        app = App(size=(600, 600), fps_target=60)
        eng = runner.engines[0]
        battle = AITrainingBattle(eng.robots, (600, 600), eng=eng)
        app.child = battle
        runner.app = app

    for iteration in range(1000000):
        timer.start()
        # reset the lstm states
        runner.init_states()
        for _ in range(steps):
            runner.run()
            if render:
                app.step()
        runner.train()
        timer.stop()
        print(iteration, timer.log_str())


if __name__ == "__main__":
    args = parse_args()
    main(steps=args.steps,
         envs=args.envs,
         wandboff=args.wandboff,
         render=args.render)
