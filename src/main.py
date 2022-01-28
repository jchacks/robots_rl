import argparse
from collections import defaultdict, deque

import numpy as np
import tensorflow as tf
from RoboArena import Engine
from robots.robot.utils import *

import wandb
from model import Model, ModelManager, Trainer
from utils import TIMER, gae
from wrapper import Dummy
import os

DEBUG = os.environ.get("DEBUG")
WANDBOFF = True


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
        default=10000,
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
        super().__init__([Dummy((255, 0, 0)), Dummy((0, 255, 0))], (300, 300))

    def init_robot(self, robot):
        robot.battle_size = self.size
        robot.opponents = [r for r in self.robots if r is not robot]
        return {
            "position": tuple(np.random.uniform(np.array(self.size))),
            "energy": np.random.uniform(30, 100),  # Randomize starting hp,
        }

    def step(self):
        for robot in self.robots:
            robot.step_reward = 0

        TIMER.start("super_step")
        super().step()
        TIMER.stop("super_step")

        TIMER.start("rewards")
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
        TIMER.stop("rewards")

        # Sub for more than 1 robot
        rewards -= (rewards @ (1 - np.eye(len(rewards)))) / (len(rewards) - 1)
        rewards -= 0.1

        return rewards, self.is_finished()


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
        self.lstm_states = np.zeros((2, self.num_robots, 128), np.float32)
        self.observations = np.zeros((1, self.num_robots, 21), np.float32)
        self.shoot_masks = np.zeros((1, self.num_robots, 1), np.bool)
        self.fill_buffers()

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

    @TIMER.wrap("fill_buffers")
    def fill_buffers(self):
        for i, robot in enumerate(self.all_robots):
            self.observations[:, i] = robot.get_observations()
            self.shoot_masks[:, i] = robot.heat > 0

    @TIMER.wrap("do_sample")
    def do_sample(self, observations, states, shoot_mask):
        actions = np.zeros((1, self.num_robots, len(ACTION_DIMS)), np.uint8)
        values = np.zeros((1, self.num_robots, 1))
        new_states = np.zeros((2, self.num_robots, 128), np.float32)
        for name, indicies in self.allocation.items():
            a, v, s = self.models[name].sample(
                observations[:, indicies],
                states[:, indicies],
                shoot_mask[:, indicies],
            )
            actions[:, indicies] = a
            values[:, indicies] = v
            new_states[:, indicies] = s
        return actions, values, states

    def run(self):
        TIMER.start("run")

        TIMER.start("tfstep")
        actions, values, new_states = self.do_sample(
            self.observations,
            self.lstm_states,
            self.shoot_masks,
        )
        TIMER.stop("tfstep")

        TIMER.start("step")
        TIMER.start("assign_actions")
        for i, robot in enumerate(self.all_robots):
            robot.assign_actions(actions[0, i])
        TIMER.stop("assign_actions")

        dones = np.zeros((self.nenvs,), np.float32)
        rewards = np.zeros((self.nenvs, 2), np.float32)
        for i, eng in enumerate(self.engines):
            reward, done = eng.step()
            rewards[i] = reward
            dones[i] = done
        TIMER.stop("step")

        dones = np.array(dones).repeat(2)
        rewards = np.array(rewards).reshape(1, self.num_robots)

        trainable_indicies = self.allocation["main"]
        memory = (
            rewards[:, trainable_indicies],
            actions[:, trainable_indicies],
            values[:, trainable_indicies],
            self.observations[:, trainable_indicies],
            self.lstm_states[:, trainable_indicies],
            self.shoot_masks[:, trainable_indicies],
            dones[trainable_indicies],
        )

        for i, eng in enumerate(self.engines):
            if eng.is_finished():
                self.rewbuffer.append(np.mean([r.total_reward for r in eng.robots]))
                self.lenbuffer.append(eng.steps)
                self.bhitbuffer.append(np.mean([r.bullets_hit for r in eng.robots]))
                self.bbybuffer.append(np.mean([r.hit_by_bullets for r in eng.robots]))
                eng.init()

        # Fill the obs, states and shoot mask buffers for the next iteration
        # Has to be here after the eng.init()
        self.fill_buffers()
        self.lstm_states[:] = new_states

        TIMER.stop("run")
        return memory

    @TIMER.wrap("train")
    def train(self):
        TIMER.start("setup")
        # Can only train on actions sampled from the main policy
        indicies = self.allocation["main"]
        n_robots = len(indicies)
        m_actions = np.zeros((self.steps, n_robots, len(ACTION_DIMS)), dtype=np.uint8)
        m_rewards = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_values = np.zeros((self.steps + 1, n_robots), dtype=np.float32)
        m_dones = np.zeros((self.steps, n_robots), dtype=np.bool)
        m_observations = np.zeros((self.steps, n_robots, 21), dtype=np.float32)
        m_states = np.zeros((self.steps, 2, n_robots, 128), dtype=np.float32)
        m_shoot_masks = np.zeros((self.steps, n_robots, 1), dtype=np.bool)
        TIMER.stop("setup")

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
            TIMER.start("memory")
            m_actions[i] = actions
            m_values[i] = values[:, :, 0]
            m_rewards[i] = rewards
            m_observations[i] = observations
            m_states[i] = states
            m_shoot_masks[i] = shoot_masks
            m_dones[i] = dones
            TIMER.stop("memory")
            if self.on_step is not None:
                self.on_step()

        # observations = self.observations
        _, last_values, _ = self.models["main"].run(
            self.observations, self.lstm_states, self.shoot_masks
        )
        last_values = last_values.numpy()[:, indicies]

        # Insert into last slot the last values
        m_values[-1] = last_values[:, :, 0]
        m_rewards[-1] = last_values[:, :, 0]

        # disc_reward = discounted(m_rewards, m_dones, self.gamma)
        advs, rets = gae(m_rewards, m_values, ~m_dones, self.gamma, self.lmbda)

        epochs = True
        if epochs:
            num_batches = 10
            batch_size = (n_robots // num_batches) + 1
            for _ in range(2):
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
        
        # Dont want to overwrite models when debugging
        if not DEBUG:
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
                "pg_loss1": wandb.Histogram(losses.pg_loss1),
                "pg_loss2": wandb.Histogram(losses.pg_loss2),
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

    num_train = 5
    runner = Runner(envs, steps)

    # Initate WandB before running
    if not WANDBOFF:
        wandb.init(project="robots_rl", entity="jchacks")
        config = wandb.config
        config.engine = "cengine"
        config.push_iterations = num_train
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
        battle = AITrainingBattle((300, 300), eng=eng)
        battle.bw.overlay.add_bar("step_reward", Colors.B, Colors.R, -2, 2)

        app.child = battle
        runner.on_step = app.step

    for iteration in range(1000000):
        for _ in range(num_train):
            TIMER.start()
            runner.train()
            TIMER.stop()
            print(iteration, TIMER.log_str())
        runner.trainer.copy_to_current_model()


if __name__ == "__main__":
    args = parse_args()
    main(steps=args.steps, envs=args.envs, wandboff=args.wandboff, render=args.render)
