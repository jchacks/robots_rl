from robots.app import App
from robots.robot.utils import *
from model import Model, Trainer
import numpy as np
import tensorflow as tf
from wrapper import Dummy, AITrainingBattle
from utils import TURNING, MOVING
import time

ACTION_DIMS = (2, 3, 3, 3)
model = Model(ACTION_DIMS)
robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
size = (600, 600)


app = App(size=size)

battle = AITrainingBattle(robots, size)
app.child = battle
# Use the eng create by battle
eng = battle.eng
app.console.add_command("sim", eng.set_rate, help="Sets the Simulation rate.")

# Simplify battles
eng.ENERGY_DECAY_ENABLED = True
eng.GUN_HEAT_ENABLED = True
eng.BULLET_COLLISIONS_ENABLED = False


def get_obs(r):
    s = np.array(size)
    center = s//2
    direction = np.sin(r.bearing * np.pi / 180), np.cos(r.bearing * np.pi / 180)
    turret = np.sin(r.turret_bearing * np.pi / 180), np.cos(r.turret_bearing * np.pi / 180)
    return tf.cast(tf.concat([
        [(r.energy/50) - 1, r.turret_heat/30],
        direction,
        turret,
        (r.position/center) - 1,
        (r.position/size),
    ], axis=0), tf.float32)


def get_position(r):
    return tf.cast(r.position/size, tf.float32)


def assign_actions(action):
    for i, robot in enumerate(robots):
        # Apply actions
        shoot, turn, move, turret = action[i]
        if robot.turret_heat > 0:
            shoot = 0
        try:
            robot.moving = MOVING[move]
            robot.base_turning = TURNING[turn]
            robot.turret_turning = TURNING[turret]
            robot.should_fire = shoot > 0
            robot.previous_energy = robot.energy
        except Exception:
            print("Failed assigning actions", i, turn, shoot)
            raise
    return action


DEBUG = False
eng.set_rate(60)
while True:
    trainer = Trainer(model)
    trainer.restore(partial=True)
    eng.init()
    print("Running test")
    while not eng.is_finished():
        # Calculate time to sleep
        time.sleep(max(0, eng.next_sim - time.time()))
        app.step()
        obs = [get_obs(r) for r in robots]
        obs = [tf.concat([obs[0], obs[1]], axis=0), tf.concat([obs[1], obs[0]], axis=0)]
        obs_batch = tf.stack(obs)
        action, value = model.run(obs_batch)
        action = assign_actions(action)
        if DEBUG:
            for r in robots:
                print(r.base_color, r.position, r.moving, r.base_turning, r.turret_turning)

        eng.step()
