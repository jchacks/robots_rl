import argparse
import time

import numpy as np
import tensorflow as tf
from robots.app import App
from robots.robot.utils import *
from robots.ui.utils import Colors

from model import Model, Trainer
from utils import MOVING, TURNING, cast
from wrapper import AITrainingBattle, Dummy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--verbose", action='store_true', help="Print debugging information.")
    parser.add_argument('-s', "--sample", action='store_true', help="Using the probability sampling instead of argmax.")
    return parser.parse_args()


ACTION_DIMS = (2, 3, 3, 3)
model = Model(ACTION_DIMS)
robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]

size = (600, 600)


app = App(size=size)

battle = AITrainingBattle(robots, size)
battle.bw.overlay.bars.append(('value', Colors.B, Colors.R))
app.child = battle
# Use the eng create by battle
eng = battle.eng
robot_map = {}
inv_robot_map = {}
for robot in eng.robots:
    idx = len(robot_map)
    robot_map[idx] = robot
    inv_robot_map[robot] = idx


app.console.add_command("sim", eng.set_rate, help="Sets the Simulation rate.")

# Simplify battles
eng.ENERGY_DECAY_ENABLED = True
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


def main(debug=False, sample=False):
    eng.set_rate(60)
    while True:
        trainer = Trainer(model)
        trainer.restore(partial=True)
        eng.init()
        for robot in eng.robots:
            robot.lstmstate = tf.stack(model.lstm.get_initial_state(batch_size=1, dtype=tf.float32))[:, 0]
        print("Running test")
        while not eng.is_finished():
            # Calculate time to sleep
            time.sleep(max(0, eng.next_sim - time.time()))
            app.step()
            obs = get_obs()
            states = get_states()
            if sample:
                actions, value, _, new_states = model.sample(obs, states)
            else:
                actions, value, new_states = model.run(obs, states)
            new_states = np.stack(new_states)
            for i, r in enumerate(eng.robots):
                r.assign_actions(actions[i])
                r.lstmstate = new_states[:, i]
                r.value = (value[i, 0] + 1) / 2

            if debug:
                for i, r in enumerate(robots):
                    print(r.base_color, r.position, r.moving, r.base_turning,
                          r.turret_turning, r.should_fire, actions[i], value[i])
            eng.step()


if __name__ == "__main__":
    args = parse_args()
    main(debug=args.verbose, sample=args.sample)
