from robots.config import BattleSettings
from robots.app import App, Battle
from robots.robot import Robot
from robots.robot.utils import *
from robots.engine import Engine
import random
import model
import numpy as np
import tensorflow as tf
import tqdm


turning_opts = [Turn.LEFT, Turn.NONE, Turn.RIGHT]
moving_opts = [Move.FORWARD, Move.NONE, Move.BACK]

class RLRobot(Robot):
    def run(self):

        pass
        # if random.random() > 0.5:
        #     self.moving = Move.FORWARD
        # else:
        #     self.moving = Move.BACK

        # if random.random() > 0.5:
        #     self.base_turning = Turn.LEFT
        # else:
        #     self.base_turning = Turn.RIGHT

        # if random.random() > 0.5:
        #     self.gun_turning = Turn.LEFT
        # else:
        #     self.gun_turning = Turn.RIGHT

        # if random.random() > 0.5:
        #     self.turret_turning = Turn.LEFT
        # else:
        #     self.turret_turning = Turn.RIGHT

        # if random.randint(0, 1):
        #     self.fire(random.randint(1, 3))


battle_settings = BattleSettings([
    RLRobot((255, 0, 0)),
    RLRobot((0, 255, 0))])
eng = Engine(battle_settings)


def get_obs(eng):
    return {
        'energy': [r.energy for r in eng.robots],
        'positions': [r.position for r in eng.robots],
        'done': eng.is_finished()
    }


def get_rewards(obs):
    return [np.all(pos < 50) for pos in obs['positions']]


eng.init()
for i in range(100):
    rewards = []
    a_moving = []
    a_turning = []
    values = []
    observations = []
    for i in tqdm.tqdm(range(1000)):
        obs = get_obs(eng)
        rewards.append(get_rewards(obs))
        observations.append(obs)
        moving, turning = model.actor.sample(tf.stack(obs['positions'])/(600, 400))
        value = model.critic(tf.stack(obs['positions'])).numpy()
        values.append(value)
        a_moving.append(moving)
        a_turning.append(turning)

        for robot, move, turn in zip(
            battle_settings.robots, 
            moving, 
            turning):
            robot.moving = moving_opts[move]
            robot.turning = turning_opts[turn]
        eng.step()
        if eng.is_finished():
            eng.init()

    rewards = tf.concat(rewards, axis=0)[:,tf.newaxis]

    a_moving = tf.concat(a_moving, axis=0)
    a_turning = tf.concat(a_turning, axis=0)

    values = tf.concat(values, axis=0)
    observations = tf.concat([o['positions'] for o in observations], axis=0)/(600, 400)
    losses = model.train(observations, rewards, (a_moving, a_turning), values)
    print(f"Total: {losses[0]}, Actor: {losses[1]}, Critic: {losses[2]}")