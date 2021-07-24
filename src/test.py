import argparse
import time
import random

import numpy as np
import tensorflow as tf
from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *
from robots.ui.utils import Colors

from model import Model, Trainer
from utils import MOVING, TURNING, cast
from wrapper import AITrainingBattle, Dummy


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
model = Model(ACTION_DIMS)
robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
size = (600, 600)
app = App(size=size)


class TestingEngine(Engine):
    def init_robotdata(self, robot):
        robot.position = np.random.uniform(np.array(self.size))
        robot.base_rotation = random.random() * 360
        robot.turret_rotation = random.random() * 360
        robot.radar_rotation = robot.turret_rotation
        robot.energy = 100


eng = TestingEngine(robots, size)

# Simplify battles
eng.ENERGY_DECAY_ENABLED = False
eng.GUN_HEAT_ENABLED = True
eng.BULLET_COLLISIONS_ENABLED = False

battle = AITrainingBattle(eng.robots, (600, 600), eng=eng)
battle.bw.overlay.bars.append(("value", Colors.B, Colors.R))
app.child = battle
# Use the eng create by battle


robot_map = {}
inv_robot_map = {}
for robot in eng.robots:
    idx = len(robot_map)
    robot_map[idx] = robot
    inv_robot_map[robot] = idx


app.console.add_command("sim", eng.set_rate, help="Sets the Simulation rate.")

# Simplify battles
eng.ENERGY_DECAY_ENABLED = False
eng.GUN_HEAT_ENABLED = True
eng.BULLET_COLLISIONS_ENABLED = False


@cast(tf.float32)
def get_obs():
    return tf.stack([robot_map[i].get_obs() for i in range(len(robot_map))])


@cast(tf.float32)
def get_states():
    """Retrieves states with correct dims that were previously saved as an
    attribute on the engine instances."""
    return tf.stack([robot_map[i].lstmstate for i in range(len(robot_map))], axis=1)


@cast(tf.bool)
def get_shoot_mask():
    return tf.stack([r.turret_heat > 0 for r in eng.robots])


def main(debug=False, sample=False):
    eng.set_rate(60)
    while True:
        trainer = Trainer(model, None)
        trainer.restore(partial=True)
        eng.init(robot_kwargs={"all_robots": eng.robots})
        robot_map = {}
        inv_robot_map = {}
        for i, robot in enumerate(eng.robots):
            robot_map[i] = robot
            inv_robot_map[robot] = i
            robot.memory = []

        _states = model.initial_state(len(robot_map))
        print("Running test")
        while not eng.is_finished():
            # Calculate time to sleep
            time.sleep(max(0, eng.next_sim - time.time()))
            app.step()
            obs = get_obs()
            states = tf.unstack(tf.cast(_states, tf.float32))
            if sample:
                actions, value, _, new_states = model.sample(
                    obs, states, get_shoot_mask()
                )
            else:
                actions, value, new_states = model.run(obs, states, get_shoot_mask())

            for i, robot in robot_map.items():
                robot.assign_actions(actions[i])
                robot.value = (value[i, 0] + 1) / 2

            _states = new_states

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
            eng.step()


if __name__ == "__main__":
    args = parse_args()
    main(debug=args.verbose, sample=args.sample)
