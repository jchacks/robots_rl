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
        default=1000,
        help="Number of envs to use for training.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=10,
        help="Number of steps to use for training.",
    )
    return parser.parse_args()


ACTION_DIMS = (1, 3, 3, 3)
model = Model(ACTION_DIMS)
model = Model(ACTION_DIMS)
old_model = Model(ACTION_DIMS, "old_model")
trainer = Trainer(model, old_model)
trainer.restore()
trainer.copy_to_oldmodel()


class TrainingEngine(Engine):
    def __init__(self):
        robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]

        super().__init__(
            robots,
            (600, 600),
            bullet_collisions_enabled=False,
            gun_heat_enabled=True,
            energy_decay_enabled=False,
            rate=-1,
        )

    def init(self):
        super().init(robot_kwargs={"all_robots": self.robots})
        self.lstm_states = tf.zeros(
            (2, 2, 128), dtype=tf.float32
        )  # 2[c,h], 2 robots, 128 hidden

    def init_robotdata(self, robot):
        robot.position = np.random.uniform(np.array(self.size))
        robot.base_rotation = random.random() * 360
        robot.turret_rotation = random.random() * 360
        robot.radar_rotation = robot.turret_rotation
        robot.energy = random.randint(30, 100)  # Randomize starting hp

    def get_obs(self):
        return np.stack([robot.get_obs() for robot in self.robots])

    def get_action_mask(self):
        return np.stack([r.turret_heat > 0 for r in self.robots])

    def step(self, actions):
        for robot, action in zip(self.robots, actions):
            robot.assign_actions(action)
            robot.prev_action = action
        
        timer.start("super_step")
        super().step()
        timer.stop("super_step")

        rewards = []
        for robot in self.robots:
            reward = (robot.energy - robot.previous_energy) / 100 - 0.1
            if self.is_finished():
                if robot.energy > 0:
                    reward += 5 + robot.energy / 100
                else:
                    reward -= 5
            rewards.append(reward)

        return rewards, self.get_obs(), self.is_finished()


class Runner(object):
    def __init__(self, nenvs, steps) -> None:
        self.on_step = None

        self.train_iteration = 0
        self.nenvs = nenvs
        self.steps = steps
        self.gamma = 0.97
        self.lmbda = 0.95  # discounted factor

        self.engines = [TrainingEngine() for i in range(nenvs)]
        for eng in self.engines:
            eng.init()
        self.observations = tf.concat([eng.get_obs() for eng in self.engines], 0)[tf.newaxis]

    def run(self):
        timer.start("run")

        timer.start("tfstep")
        observations = self.observations
        states = tf.concat([eng.lstm_states for eng in self.engines], 1)
        shoot_mask = tf.concat([eng.get_action_mask() for eng in self.engines], 0)
        actions, values, neglogps, new_states = model.sample(
            self.observations, states, shoot_mask
        )
        timer.stop("tfstep")

        _actions = actions.reshape(1000, 2, 4)
        _new_states = new_states.reshape(2, 1000, 2, 128)

        dones = []
        rewards = np.zeros((1000,2))
        self.observations = np.zeros((1000,2,22))

        timer.start("step")
        for i, eng in enumerate(self.engines):
            timer.start("eng")
            reward, obs, done = eng.step(_actions[i])
            timer.stop("eng")
            eng.lstm_states = _new_states[:, i]
            rewards[i] = reward
            self.observations[i] = obs
            dones.append(done)
        timer.stop("step")

        dones = np.array(dones)
        rewards = np.array(rewards).reshape(1, 2000)
        self.observations = self.observations.reshape(1, 2000,22)

        memory = (
            rewards,
            actions,
            values,
            neglogps,
            observations,
            states,
            shoot_mask,
            dones,
        )

        for eng in self.engines:
            if eng.is_finished():
                eng.init()

        timer.stop("run")
        return memory

    def train(self):
        n_robots = self.nenvs * 2
        m_actions = np.zeros((self.steps, n_robots, len(ACTION_DIMS)), dtype=np.uint8)
        m_rewards = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_values = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_dones = np.zeros((self.steps, n_robots), dtype=np.bool)
        m_neglogps = np.zeros((self.steps, n_robots), dtype=np.float32)
        m_observations = np.zeros((self.steps, n_robots, 22), dtype=np.float32)
        m_states = np.zeros((self.steps, 2, n_robots, 128), dtype=np.float32)
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
            m_dones[i] = np.repeat(dones,2)
            if self.on_step is not None:
                self.on_step()

        observations = tf.concat([eng.get_obs() for eng in self.engines], 0)[tf.newaxis]
        states = tf.concat([eng.lstm_states for eng in self.engines], 1)
        shoot_mask = tf.concat(
            [eng.get_action_mask() for eng in self.engines], 0
        )
        _, last_values, _ = model.run(observations, states, shoot_mask)

        # Insert into last slot the last values
        m_values[-1] = last_values[:, :, 0]
        m_rewards[-1] = last_values[:, :, 0]

        if self.train_iteration == 0:
            _ = old_model.run(observations, states, shoot_mask)

        # p, l, v = model.prob(observations, states, shoot_mask)
        # print([p[0:4] for p in p], [l[0:4] for l in l], v[0:4])

        # disc_reward = discounted(m_rewards, m_dones, self.gamma)
        advs, rets = get_advantage(
            m_rewards, m_values, ~m_dones, self.gamma, self.lmbda
        )

        # Assign current policy to old policy before update
        trainer.copy_to_oldmodel()

        epochs = True
        if epochs:
            num_batches = 4
            batch_size = (n_robots // num_batches) + 1
            for _ in range(4):
                order = np.arange(n_robots)
                np.random.shuffle(order)
                for i in range(num_batches):
                    slc = slice(i * batch_size, (i + 1) * batch_size)
                    # Pass data to trainer, managing the model.
                    losses = trainer.train(
                        m_observations[:, order][:, slc],
                        tf.unstack(m_states[0][:, order][:, slc]),
                        advs[:, order][:, slc],
                        rets[:, order][:, slc],
                        m_actions[:, order][:, slc],
                        m_neglogps[:, order][:, slc],
                        m_shoot_masks[:, order][:, slc],
                        m_dones[:, order][:, slc],
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
            log_dict = {
                "rewards": wandb.Histogram(m_rewards, num_bins=128),
                "mean_reward": m_rewards.mean(),
                "returns": wandb.Histogram(rets, num_bins=128),
                "mean_return": rets.mean(),
                "loss": losses.loss,
                "actor": losses.actor,
                "critic": losses.critic,
                "ratio": wandb.Histogram(losses.ratio, num_bins=128),
                "ratio_clipped": wandb.Histogram(losses.ratio_clipped, num_bins=128),
                "entropy": losses.entropy,
                "shoot_entropy": losses.entropies[0],
                "turn_entropy": losses.entropies[1],
                "move_entropy": losses.entropies[2],
                "turret_entropy": losses.entropies[3],
                "advantage": losses.advantage,
                "values": losses.value,
            }
            wandb.log(log_dict)
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
        runner.train()
        timer.stop()
        print(iteration, timer.log_str())


if __name__ == "__main__":
    args = parse_args()
    main(steps=args.steps, envs=args.envs, wandboff=args.wandboff, render=args.render)
