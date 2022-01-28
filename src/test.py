import argparse
import time
import math
import random

import numpy as np
from robots.app import App
from RoboArena import Engine
from robots.robot.utils import *
from robots.ui.utils import Colors

from model import ModelManager
from utils import TIMER
from wrapper import AITrainingBattle, Dummy, Random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debugging information."
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="Run determinstically taking argmax.",
    )
    return parser.parse_args()


ACTION_DIMS = (1, 3, 3, 3)
model_manager = ModelManager()
model = model_manager.model


class EnvEngine(Engine):
    def __init__(self):
        robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
        super().__init__(robots, (300, 300))

    def init_robot(self, robot):
        robot.battle_size = self.size
        robot.opponents = [r for r in self.robots if r is not robot]
        return {}

    def step(self):
        for robot in self.robots:
            robot.step_reward_ema -= (1 - 0.9) * (
                robot.step_reward_ema - robot.step_reward
            )
            robot.step_reward = 0

        TIMER.start("super_step")
        super().step()
        TIMER.stop("super_step")


app = App(size=(300, 300), fps_target=60)
eng = EnvEngine()
battle = AITrainingBattle((300, 300), eng=eng)
battle.bw.overlay.add_bar("value", Colors.Y, Colors.K, -8, 0)
battle.bw.overlay.add_bar("step_reward_ema", Colors.O, Colors.W, -1, 1)
battle.bw.overlay.add_bar("norm_value", Colors.W, Colors.K)
app.child = battle

app.console.add_command("sim", battle.set_tick_rate, help="Sets the Simulation rate.")


def main(deterministic):
    battle.set_tick_rate(60)

    while True:
        eng.init()
        for robot in eng.robots:
            print(robot.fire_power)
        model_manager.restore()

        obs = np.zeros((1, len(eng.robots), 21), np.float32)
        mask = np.zeros((1, len(eng.robots), 1), np.bool)
        for i, robot in enumerate(eng.robots):
            obs[:, i] = robot.get_obs()
            mask[:, i] = robot.heat > 0
        states = np.zeros((2, len(eng.robots), 128), np.float32)

        print("Running test")
        while not eng.is_finished():
            # Calculate time to sleep
            time.sleep(max(0, battle.next_sim - time.time()))
            battle.next_sim = time.time() + battle.interval

            app.step()
            if deterministic:
                actions, value, new_states = model.run(obs, states, mask)
            else:
                actions, value, new_states = model.sample(obs, states, mask)
            actions = actions.numpy()
            value = value.numpy()
            new_states = new_states.numpy()

            for i, robot in enumerate(eng.robots):
                robot.assign_actions(actions[0, i])

            eng.step()

            for i, robot in enumerate(eng.robots):
                obs[:, i] = robot.get_obs()
                mask[:, i] = robot.heat > 0
            states[:] = new_states

            for i, robot in enumerate(eng.robots):
                robot.value = (value[0, i, 0] + 1) / 2
                robot.norm_value = (value[0, i, 0] - value.min()) / (
                    value.max() - value.min()
                )
        print(TIMER.log_str())


if __name__ == "__main__":
    args = parse_args()
    main(args.deterministic)
