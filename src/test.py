import argparse
import time
import math
import random

import numpy as np
from robots.app import App
from robots.engine_c.engine import Engine
from robots.robot.utils import *
from robots.ui.utils import Colors

from model import ModelManager
from utils import Timer
from wrapper import AITrainingBattle, Dummy, Random


timer = Timer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debugging information."
    )
    return parser.parse_args()


ACTION_DIMS = (1, 3, 3, 3)
model_manager = ModelManager()
model = model_manager.model


class EnvEngine(Engine):
    def __init__(self, i=None):
        robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
        super().__init__(robots, (300, 300))

    def init_robot(self, robot):
        robot.battle_size = self.size
        robot.opponents = [r for r in self.robots if r is not robot]
        return {}

    def get_obs(self):
        return np.stack([robot.get_obs() for robot in self.robots])

    def get_lstmstate(self):
        return np.stack([robot.lstmstate for robot in self.robots], 1)

    def set_lstmstate(self, states):
        for i, robot in enumerate(self.robots):
            robot.lstmstate = states[:, i]

    def get_action_mask(self):
        return np.stack([r.heat > 0 for r in self.robots])

    def step(self, actions):
        for robot, action in zip(self.robots, actions):
            robot.step_reward = 0
            robot.assign_actions(action)
            robot.prev_action = action

        timer.start("super_step")
        super().step()
        timer.stop("super_step")
        return self.get_obs()


app = App(size=(300, 300), fps_target=60)
eng = EnvEngine()
battle = AITrainingBattle(eng.robots, (300, 300), eng=eng)
battle.bw.overlay.add_bar("value", Colors.Y, Colors.K)
battle.bw.overlay.add_bar("step_reward", Colors.O, Colors.W, -0.5, 0.5)
battle.bw.overlay.add_bar("norm_value", Colors.W, Colors.K)
app.child = battle

app.console.add_command("sim", battle.set_tick_rate, help="Sets the Simulation rate.")


def main():
    battle.set_tick_rate(60)

    while True:
        eng.init()
        for robot in eng.robots:
            print(robot.fire_power)
        model_manager.restore()
        obs = eng.get_obs()

        print("Running test")
        while not eng.is_finished():
            # Calculate time to sleep
            time.sleep(max(0, battle.next_sim - time.time()))
            battle.next_sim = time.time() + battle.interval

            app.step()

            actions, value, new_states = model.sample(
                obs[np.newaxis].astype(np.float32),
                eng.get_lstmstate(),
                eng.get_action_mask()[np.newaxis],
            )

            actions = actions.numpy()
            value = value.numpy()
            eng.set_lstmstate(new_states.numpy())
            obs = eng.step(actions[0])

            for i, robot in enumerate(eng.robots):
                robot.value = (value[0, i, 0] + 1) / 2
                robot.norm_value = (value[0, i, 0] - value.min()) / (
                    value.max() - value.min()
                )



if __name__ == "__main__":
    args = parse_args()
    main()
