import argparse
from collections import defaultdict, deque
import random
import time

import numpy as np
import tensorflow as tf
from robots.app import App
from robots.engine_c.engine import Engine
from robots.robot.utils import *

import wandb
from model import Model, ModelManager, Trainer
from utils import Timer, gae
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
        default=25,
        help="Number of steps to use for training.",
    )
    return parser.parse_args()


ACTION_DIMS = (1, 3 * 3 * 3)


class EnvEngine(Engine):
    def __init__(self, i=None):
        robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
        super().__init__(robots, (300, 300))

    def init_robot(self, robot):
        robot.battle_size = self.size
        robot.opponents = [r for r in self.robots if r is not robot]
        return {
            "position": np.random.uniform(np.array(self.size)),
            "base_rotation": random.random() * 360,
            "turret_rotation": random.random() * 360,
            "energy": random.randint(30, 100),  # Randomize starting hp,
        }

    def get_obs(self):
        return np.stack([robot.get_obs() for robot in self.robots])

    def get_lstmstate(self):
        return np.stack([robot.lstmstate for robot in self.robots], 1)

    def set_lstmstate(self, states):
        for i, robot in enumerate(self.robots):
            robot.lstmstate[:] = states[:, i]

    def get_action_mask(self):
        return np.stack([r.heat > 0 for r in self.robots])

    def step(self, actions):
        timer.start("assign_actions")
        for robot, action in zip(self.robots, actions):
            robot.step_reward = 0
            # assign action contains rewards
            robot.assign_actions(action)
            robot.prev_action = action
        timer.stop("assign_actions")

        timer.start("super_step")
        super().step()
        timer.stop("super_step")

        timer.start("rewards")
        # Add extra rewards to step_reward
        for robot in self.robots:
            curr = robot.previous_energy / 100
            after = robot.energy / 100

            curr = (curr + 1 - (1 - curr) ** 4) / 2
            after = (after + 1 - (1 - after) ** 4) / 2
            robot.step_reward += (after - curr) * 2

            if self.is_finished():
                if robot.energy > 0:
                    robot.step_reward += 5
                else:
                    robot.step_reward -= 5

        rewards = np.array([r.step_reward for r in self.robots])

        for i, robot in enumerate(self.robots):
            robot.total_reward += rewards[i]
        timer.stop("rewards")

        # Sub for more than 1 robot
        rewards -= (rewards @ (1 - np.eye(len(rewards)))) / (len(rewards) - 1)
        rewards -= 0.1

        timer.start("observations")
        obs = self.get_obs()
        timer.stop("observations")
        return rewards, obs, self.is_finished()


class Runner(object):
    def __init__(self, nenvs, steps) -> None:
        self.on_step = None

        self.train_iteration = 0
        self.nenvs = nenvs
        self.steps = steps
        self.gamma = 0.99
        self.lmbda = 0.97
        model = Model(ACTION_DIMS)
        self.trainer = Trainer(model)

        self.models = {"main": model}
        self.models.update({k: Model(ACTION_DIMS) for k in range(1, 4)})
        self.load_old_models()
        self.allocation = defaultdict(list)

        self.engines = [EnvEngine(i) for i in range(nenvs)]

        self.all_robots = np.array([r for env in self.engines for r in env.robots])
        self.num_robots = len(self.all_robots)
        self.allocate_agents()

        self.lenbuffer = deque(maxlen=1000)
        self.rewbuffer = deque(maxlen=1000)
        self.bhitbuffer = deque(maxlen=1000)
        self.bbybuffer = deque(maxlen=1000)

        # Get first observations from each Engine
        self.observations = np.concatenate([eng.get_obs() for eng in self.engines], 0)[
            tf.newaxis
        ]

    def allocate_agents(self):
        idx = 0
        for i, env in enumerate(self.engines):
            env.init()
            # Allocated robots to model groups
            self.allocation["main"].append(idx)
            idx += 1
            if i < self.nenvs * 0.4:
                self.allocation["main"].append(idx)
            elif i < self.nenvs * 0.6:
                self.allocation[1].append(idx)
            elif i < self.nenvs * 0.8:
                self.allocation[2].append(idx)
            else:
                self.allocation[3].append(idx)
            idx += 1
        for k, v in self.allocation.items():
            self.allocation[k] = np.array(v)

    def load_old_models(self):
        print("Loading older models")
        for k, model in self.models.items():
            if k == "main":
                continue
            print(f"\tLoading {model} {k}")
            manager = ModelManager(model)
            manager.restore(offset=-k * 4)

    def do_sample(self, observations, states, shoot_mask):
        actions = np.zeros((1, self.num_robots, len(ACTION_DIMS)), np.uint8)
        values = np.zeros((1, self.num_robots, 1))
        new_states = np.zeros((2, self.num_robots, 128), np.float32)
        for name, indicies in self.allocation.items():
            a, v, s = self.models[name].sample(
                observations[:, indicies], states[:, indicies], shoot_mask[indicies]
            )
            actions[:, indicies] = a
            values[:, indicies] = v
            new_states[:, indicies] = s
        return actions, values, states

    def run(self):
        timer.start("run")

        timer.start("tfstep")
        observations = self.observations
        states = np.concatenate([eng.get_lstmstate() for eng in self.engines], 1)
        shoot_mask = np.concatenate([eng.get_action_mask() for eng in self.engines], 0)

        actions, values, new_states = self.do_sample(
            self.observations.astype(np.float32), states, shoot_mask
        )
        timer.stop("tfstep")

        _actions = actions.reshape(self.nenvs, 2, len(ACTION_DIMS))
        new_states = new_states.reshape(2, self.nenvs, 2, 128)

        dones = []
        rewards = np.zeros((self.nenvs, 2))
        self.observations = np.zeros((self.nenvs, 2, 23))

        timer.start("step")
        for i, eng in enumerate(self.engines):
            timer.start("eng")
            reward, obs, done = eng.step(_actions[i])
            timer.stop("eng")

            eng.set_lstmstate(new_states[:, i])
            rewards[i] = reward
            self.observations[i] = obs
            dones.append(done)
        timer.stop("step")

        dones = np.array(dones).repeat(2)
        rewards = np.array(rewards).reshape(1, self.nenvs * 2)

        trainable_indicies = self.allocation["main"]
        memory = (
            rewards[:, trainable_indicies],
            actions[:, trainable_indicies],
            values[:, trainable_indicies],
            observations[:, trainable_indicies],
            states[:, trainable_indicies],
            shoot_mask[trainable_indicies],
            dones[trainable_indicies],
        )

        for i, eng in enumerate(self.engines):
            if eng.is_finished():
                self.rewbuffer.append(np.mean([r.total_reward for r in eng.robots]))
                self.lenbuffer.append(eng.steps)
                self.bhitbuffer.append(np.mean([r.bullets_hit for r in eng.robots]))
                self.bbybuffer.append(np.mean([r.hit_by_bullets for r in eng.robots]))
                eng.init()
                # Get the new initial observations
                self.observations[i] = eng.get_obs()

        # Record observations for next iteration
        self.observations = self.observations.reshape(1, self.nenvs * 2, 23)

        timer.stop("run")
        return memory

    def train(self):
        # Can only train on actions sampled from the main policy
        indicies = self.allocation["main"]
        n_robots = len(indicies)
        m_actions = np.zeros((self.steps, n_robots, len(ACTION_DIMS)), dtype=np.uint8)
        m_rewards = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_values = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_dones = np.zeros((self.steps, n_robots), dtype=np.bool)
        m_observations = np.zeros((self.steps, n_robots, 23), dtype=np.float32)
        m_states = np.zeros((self.steps, 2, n_robots, 128), dtype=np.float32)
        m_shoot_masks = np.zeros((self.steps, n_robots), dtype=np.bool)

        for i in range(self.steps):
            (
                rewards,
                actions,
                values,
                observations,
                states,
                shoot_masks,
                dones,
            ) = self.run()
            m_actions[i] = actions
            m_values[i] = values[:, :, 0]
            m_rewards[i] = rewards
            m_observations[i] = observations
            m_states[i] = states
            m_shoot_masks[i] = shoot_masks
            m_dones[i] = dones
            if self.on_step is not None:
                self.on_step()

        observations = self.observations
        states = np.concatenate([eng.get_lstmstate() for eng in self.engines], 1)
        shoot_mask = np.concatenate([eng.get_action_mask() for eng in self.engines], 0)
        _, last_values, _ = self.models["main"].run(observations, states, shoot_mask)

        last_values = last_values.numpy()[:, indicies]

        # Insert into last slot the last values
        m_values[-1] = last_values[:, :, 0]
        m_rewards[-1] = last_values[:, :, 0]

        # disc_reward = discounted(m_rewards, m_dones, self.gamma)
        advs, rets = gae(m_rewards, m_values, ~m_dones, self.gamma, self.lmbda)

        epochs = True
        if epochs:
            num_batches = 4
            batch_size = (n_robots // num_batches) + 1
            for _ in range(3):
                order = np.arange(n_robots)
                np.random.shuffle(order)
                for i in range(num_batches):
                    slc = slice(i * batch_size, (i + 1) * batch_size)
                    # Pass data to trainer, managing the model.
                    losses = self.trainer.train(
                        m_observations[:, order][:, slc],
                        tf.unstack(m_states[0][:, order][:, slc]),
                        advs[:, order][:, slc],
                        rets[:, order][:, slc],
                        m_actions[:, order][:, slc],
                        m_shoot_masks[:, order][:, slc],
                        m_dones[:, order][:, slc],
                    )
        else:
            # Pass data to trainer, managing the model.
            losses = self.trainer.train(
                m_observations,
                tf.unstack(m_states[0]),
                advs,
                rets,
                m_actions,
                m_shoot_masks,
                m_dones,
            )

        # Checkpoint manager will save every x steps
        saved = self.trainer.model_manager.checkpoint()
        if saved:
            self.load_old_models()

        if not WANDBOFF:
            log_dict = {
                "rewards": wandb.Histogram(m_rewards, num_bins=128),
                "returns": wandb.Histogram(rets, num_bins=128),
                "mean_reward": m_rewards.mean(),
                "mean_return": rets.mean(),
                "mean_done_reward": np.mean(self.rewbuffer),
                "mean_done_length": np.mean(self.lenbuffer),
                "mean_bullets_hit": np.mean(self.bhitbuffer),
                "mean_hit_by_bullets": np.mean(self.bbybuffer),
                "loss": losses.loss,
                "actor": losses.actor,
                "grads_actor": wandb.Histogram(losses.d_grads_actor),
                "critic": losses.critic,
                "grads_critic": wandb.Histogram(losses.d_grads_critic),
                "ratio": wandb.Histogram(losses.ratio, num_bins=128),
                "ratio_clipped": wandb.Histogram(losses.ratio_clipped, num_bins=128),
                "entropy_reg": losses.entropy,
                "entropy": np.mean(losses.entropies),
                "shoot_entropy": losses.entropies[0],
                "advantage": losses.advantage,
                "values": losses.value,
                "regularisation": losses.reg_loss,
            }
            wandb.log(log_dict)
        if np.isnan(losses[0].numpy()):
            raise RuntimeError
        self.train_iteration += 1


def main(steps, envs, render=False, wandboff=False):
    global WANDBOFF
    WANDBOFF = wandboff

    runner = Runner(envs, steps)

    # Initate WandB before running
    if not WANDBOFF:
        wandb.init(project="robots_rl", entity="jchacks", resume=True)
        config = wandb.config
        config.critic_scale = runner.trainer.critic_scale
        config.entropy_scale = runner.trainer.entropy_scale
        config.learning_rate = runner.trainer.learning_rate
        config.epsilon = runner.trainer.epsilon
        config.max_steps = steps
        config.envs = envs

    if render:
        # Todo clean up this interaction with Engine and Battle
        from robots.app import App
        from robots.ui.utils import Colors
        from wrapper import AITrainingBattle

        app = App(size=(300, 300), fps_target=60)
        eng = runner.engines[-1]
        battle = AITrainingBattle(eng.robots, (300, 300), eng=eng)
        battle.bw.overlay.add_bar("step_reward", Colors.B, Colors.R, -2, 2)

        app.child = battle
        runner.on_step = app.step

    for iteration in range(1000000):
        for _ in range(5):
            timer.start()
            runner.train()
            timer.stop()
            print(iteration, timer.log_str())
        runner.trainer.copy_to_current_model()


if __name__ == "__main__":
    args = parse_args()
    main(steps=args.steps, envs=args.envs, wandboff=args.wandboff, render=args.render)
