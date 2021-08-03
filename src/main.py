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
from utils import Timer, cast, discounted, get_advantage
from wrapper import Dummy

WANDBOFF = True
timer = Timer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debugging information."
    )
    parser.add_argument("--wandboff", action="store_true", help="Turn off W&B logging.")
    parser.add_argument(
        "-r", "--render", action="store_true", help="Render battles during training."
    )
    parser.add_argument(
        "-n",
        "--envs",
        type=int,
        default=500,
        help="Number of envs to use for training.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=30,
        help="Number of steps to use for training.",
    )
    return parser.parse_args()


ACTION_DIMS = (1, 3 * 3 * 3)
model = Model(ACTION_DIMS)
model = Model(ACTION_DIMS)
old_model = Model(ACTION_DIMS, "old_model")
trainer = Trainer(model, old_model)
trainer.restore()
trainer.copy_to_oldmodel()


class TrainingEngine(Engine):
    def init_robotdata(self, robot):
        robot.position = np.random.uniform(np.array(self.size))
        robot.base_rotation = random.random() * 360
        robot.turret_rotation = random.random() * 360
        robot.radar_rotation = robot.turret_rotation
        robot.energy = random.randint(30, 100)  # Randomize starting hp


def make_eng():
    robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
    size = (600, 600)
    eng = TrainingEngine(robots, size)

    # Simplify battles
    eng.ENERGY_DECAY_ENABLED = False
    eng.GUN_HEAT_ENABLED = True
    eng.BULLET_COLLISIONS_ENABLED = False
    return eng


class Runner(object):
    def __init__(self, nenvs, steps) -> None:
        # Callback to render during training
        self.on_step = None

        self.train_iteration = 0
        self.nenvs = nenvs
        self.steps = steps
        self.gamma = 0.99
        self.lmbda = 0.95  # discounted factor

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

    def init_states(self):
        self.states = model.initial_state(len(self.robots))

    def run(self):
        timer.start("run")

        timer.start("tfstep")
        observations = self.get_obs()
        states = tf.unstack(tf.cast(self.states, tf.float32))
        shoot_mask = self.get_shoot_mask()
        actions, values, neglogps, new_states = model.sample(
            observations[tf.newaxis], states, shoot_mask
        )
        timer.stop("tfstep")

        # Assign actions and records the next states
        for i, robot in enumerate(self.robots):
            robot.assign_actions(actions[0][i])

        timer.start("step")
        for i, eng in enumerate(self.engines):
            eng.step()
            # # If going too long then kill both
            # if eng.steps >= 500:
            #     for robot in eng.data:
            #         robot.energy = 0

        timer.stop("step")

        dones = []
        rewards = []

        for idx, robot in enumerate(self.robots):
            done = not robot.alive
            reward = (robot.energy - robot.previous_energy - 1) / 100
            if done:
                # Zero out the index for the done robot
                new_states[:, idx] = 0
                if robot.energy > 0:
                    reward += 5 + robot.energy / 100
                else:
                    reward -= 5

            dones.append(done)
            rewards.append(reward)

        dones = np.array(dones)
        rewards = np.array(rewards)

        memory = (
            rewards,
            actions,
            values,
            neglogps,
            observations,
            self.states,
            shoot_mask,
            dones,
        )
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
        return memory

    def train(self):
        n_robots = self.nenvs * 2
        m_actions = np.zeros((self.steps, n_robots, len(ACTION_DIMS)), dtype=np.uint8)
        m_rewards = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_values = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_dones = np.zeros((self.steps, n_robots), dtype=np.bool)
        m_neglogps = np.zeros((self.steps, n_robots), dtype=np.float32)
        m_observations = np.zeros((self.steps, n_robots, 12), dtype=np.float32)
        m_states = np.zeros((self.steps, 2, n_robots, 512), dtype=np.float32)
        m_shoot_masks = np.zeros((self.steps, n_robots), dtype=np.bool)

        for i in range(self.steps):
            (
                rewards,
                actions,
                values,
                neglogps,
                observations,
                states,
                shoot_masks,
                dones,
            ) = self.run()
            m_actions[i] = actions
            m_values[i] = values[:, :, 0]
            m_rewards[i] = rewards
            m_neglogps[i] = neglogps
            m_observations[i] = observations
            m_states[i] = states
            m_shoot_masks[i] = shoot_masks
            m_dones[i] = dones
            if self.on_step is not None:
                self.on_step()

        observations = self.get_obs()
        states = tf.unstack(tf.cast(self.states, tf.float32))
        shoot_mask = self.get_shoot_mask()
        _, last_values, _ = model.run(observations[tf.newaxis], states, shoot_mask)

        # Insert into last slot the last values
        m_values[-1] = last_values[:, :, 0]
        m_rewards[-1] = last_values[:, :, 0]

        if self.train_iteration == 0:
            _ = old_model.run(observations[tf.newaxis], states, shoot_mask)

        # p, l, v = model.prob(observations, states, shoot_mask)
        # print([p[0:4] for p in p], [l[0:4] for l in l], v[0:4])

        # disc_reward = discounted(m_rewards, m_dones, self.gamma)
        advs, rets = get_advantage(
            m_rewards, m_values, ~m_dones, self.gamma, self.lmbda
        )

        # Assign current policy to old policy before update
        trainer.copy_to_oldmodel()

        epochs = False
        if epochs:
            num_batches = 8
            batch_size = (n_robots // num_batches) + 1
            for _ in range(4):
                order = np.arange(n_robots)
                np.random.shuffle(order)
                for i in range(num_batches):
                    slc = slice(i * batch_size, (i + 1) * batch_size)
                    # Pass data to trainer, managing the model.
                    losses = trainer.train(
                        m_observations[order][slc],
                        tf.unstack(m_states[0][order][slc], axis=1),
                        advs,
                        rets,
                        m_actions[order][slc],
                        m_neglogps[order][slc],
                        m_shoot_masks[order][slc],
                        m_dones,
                    )
        else:
            # Pass data to trainer, managing the model.
            losses = trainer.train(
                m_observations,
                tf.unstack(m_states[0]),
                advs,
                rets,
                m_actions,
                m_neglogps,
                m_shoot_masks,
                m_dones,
            )

        # Checkpoint manager will save every x steps
        trainer.checkpoint()
        if not WANDBOFF:
            wandb.log(
                {
                    "rewards": wandb.Histogram(m_rewards, num_bins=256),
                    "mean_reward": m_rewards.mean(),
                    "loss": losses.loss,
                    "actor": losses.actor,
                    "critic": losses.critic,
                    "ratio": wandb.Histogram(losses.ratio.numpy(), num_bins=256),
                    "ratio_clipped": wandb.Histogram(
                        losses.ratio_clipped.numpy(), num_bins=256
                    ),
                    "entropy": losses.entropy,
                    "advantage": losses.advantage,
                    "values": losses.value,
                }
            )
        if np.isnan(losses[0].numpy()):
            raise RuntimeError
        self.train_iteration += 1


def main(steps, envs, render=False, wandboff=False):
    global WANDBOFF
    WANDBOFF = wandboff
    # Initate WandB before running
    if not WANDBOFF:
        wandb.init(project="robots_rl", entity="jchacks")
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
        runner.on_step = app.step

    for iteration in range(1000000):
        timer.start()
        # reset the lstm states
        runner.init_states()
        runner.train()
        timer.stop()
        print(iteration, timer.log_str())


if __name__ == "__main__":
    args = parse_args()
    main(steps=args.steps, envs=args.envs, wandboff=args.wandboff, render=args.render)
