import argparse
import time
import math
import random

import numpy as np
import tensorflow as tf
from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *
from robots.ui.utils import Colors

from model import Model, ModelManager, Trainer
from utils import cast, Timer
from wrapper import AITrainingBattle, Dummy


timer = Timer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debugging information."
    )
    parser.add_argument(
        "-s",
        "--sample",
        action="store_true",
        help="Using the probability sampling instead of argmax.",
    )
    return parser.parse_args()


ACTION_DIMS = (1, 3, 3, 3)
model_manager = ModelManager()
model = model_manager.model

class EnvEngine(Engine):
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
            (2, 2, 512), dtype=tf.float32
        )  # 2[c,h], 2 robots, 128 hidden

    def init_robotdata(self, robot):
        robot.position = np.random.uniform(np.array(self.size))
        robot.base_rotation = random.random() * 2 * math.pi
        robot.turret_rotation = random.random() * 2 * math.pi
        robot.radar_rotation = robot.turret_rotation
        robot.energy = 100

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



app = App(size=(600, 600), fps_target=60)
eng = EnvEngine()
battle = AITrainingBattle(eng.robots, (600, 600), eng=eng)
battle.bw.overlay.bars.append(("value", Colors.B, Colors.R))
battle.bw.overlay.bars.append(("norm_value", Colors.W, Colors.K))
app.child = battle

app.console.add_command("sim", eng.set_rate, help="Sets the Simulation rate.")


def main(debug=False, sample=False):
    eng.set_rate(60)
    while True:
        eng.init()
        model_manager.restore()
        obs = eng.get_obs()

        print("Running test")
        while not eng.is_finished():
            # Calculate time to sleep
            time.sleep(max(0, eng.next_sim - time.time()))
            app.step()
            
            actions, value, new_states = model.sample(
                obs[np.newaxis], 
                eng.get_lstmstate(), 
                eng.get_action_mask()[np.newaxis]
            )

            actions = actions.numpy()
            value = value.numpy()
            new_states = new_states.numpy()

            _, obs, _ = eng.step(actions[0])
            eng.lstm_states = new_states

            for i, robot in enumerate(eng.robots):
                robot.value = (value[0, i, 0] + 8) / 16
                robot.norm_value = (value[0, i, 0] - value.min()) / (
                    value.max() - value.min()
                )

            if debug:
                for i, r in enumerate(robots):
                    print(
                        r.base_color,
                        r.position,
                        r.moving,
                        r.base_turning,
                        r.turret_turning,
                        r.should_fire,
                        actions[i],
                        value[i],
                    )


if __name__ == "__main__":
    args = parse_args()
    main(debug=args.verbose, sample=args.sample)
