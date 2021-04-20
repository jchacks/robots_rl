from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *
import model
from model import actor, critic
import numpy as np
import tensorflow as tf
import tqdm
from wrapper import Dummy, AITrainingBattle
from utils import Memory, discounted
import time

ACTION_DIMS = (2, 3, 3)
model.actor = model.Actor(2*3*3)

TURNING = [Turn.LEFT, Turn.NONE, Turn.RIGHT]
MOVING = [Move.FORWARD, Move.NONE, Move.BACK]

robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
size = (300, 300)


render = True
if render:
    app = App(size=size)
    battle = AITrainingBattle(robots, size)
    app.child = battle
    # Use the eng create by battle
    eng = battle.eng
else:
    eng = Engine(robots, size)

# Simplify battles
eng.bullet_collisions_enabled = False
eng.gun_heat_enabled = True


def get_obs(r):
    s = np.array(size)
    center = s//2
    position = (r.position//center) - 1
    direction = np.sin(r.bearing * np.pi / 180), np.cos(r.bearing * np.pi / 180)
    turret = np.sin(r.turret_bearing * np.pi / 180), np.cos(r.turret_bearing * np.pi / 180)
    return tf.cast(tf.concat([
        [(r.energy//50) - 1, r.turret_heat/30],
        direction,
        turret,
        position
    ], axis=0), tf.float32)


def get_position(r):
    return tf.cast(r.position/size, tf.float32)


def assign_actions(action):
    for i, robot in enumerate(robots):
        # Apply actions
        # Dimension Size: Shoot = 2, Turning = 3, Move = 3, Turret = 3
        shoot, turn, move = np.unravel_index(action[i], ACTION_DIMS)
        if robot.turret_heat > 0:
            shoot = 0
        try:
            robot.moving = MOVING[move]
            robot.base_turning = TURNING[turn]
            robot.should_fire = shoot > 0
            robot.previous_energy = robot.energy
        except Exception:
            print("Failed assigning actions", i, turn, shoot)
            raise
        # action[i] = np.ravel_multi_index((shoot, turn, move), ACTION_DIMS)
    return action


def test(max_steps=200):
    eng.init()
    eng.set_rate(60)
    step = 0
    print("Running test")
    while not eng.is_finished() and step < max_steps:
        # Calculate time to sleep
        time.sleep(max(0, eng.next_sim - time.time()))
        if render:
            app.step()

        obs = [get_obs(r) for r in robots]
        obs = [tf.concat([obs[0], obs[1]], axis=0), tf.concat([obs[1], obs[0]], axis=0)]
        obs_batch = tf.stack(obs)
        action = model.run(obs_batch).numpy()
        value = critic(obs_batch).numpy()
        action = assign_actions(action)
        eng.step()
        step += 1
    eng.set_rate(-1)


def train(memory):
    # Get obs for the last state
    obs = [get_obs(r) for r in robots]
    obs = [tf.concat([obs[0], obs[1]], axis=0), tf.concat([obs[1], obs[0]], axis=0)]
    obs_batch = tf.stack(obs)
    last_values = model.critic(obs_batch).numpy()

    b_rewards = []
    b_action = []
    b_values = []
    b_obs = []

    for robot, last_value in zip(robots, last_values):
        mem = memory[robot]
        b_rewards.append(discounted(np.array(mem['rewards']), np.array(mem['dones']), + last_value, 0.99))
        b_action.append(mem['action'])
        b_values.append(mem['values'])
        b_obs.append(mem['obs'])

    b_rewards = tf.concat(b_rewards, axis=0)[:, tf.newaxis]
    b_action = tf.concat(b_action, axis=0)
    b_values = tf.concat(b_values, axis=0)
    b_obs = tf.concat(b_obs, axis=0)

    losses = model.train(b_obs, b_rewards, b_action, b_values)
    if np.isnan(losses[0].numpy()):
        raise RuntimeError
    return losses


DO_NOTHING_INDEX = np.ravel_multi_index((0, 1, 1), ACTION_DIMS)
# TODO Add random sizes
eng.init()
total_reward = {r: 0 for r in robots}
tests = 1
max_steps = 200
for iteration in range(1000000):
    # Create a memory per player
    memory = {r: Memory('rewards,action,values,obs,dones') for r in robots}
    steps = 0
    while steps <= max_steps:
        if render:
            app.step()

        obs = [get_obs(r) for r in robots]
        obs = [tf.concat([obs[0], obs[1]], axis=0), tf.concat([obs[1], obs[0]], axis=0)]
        obs_batch = tf.stack(obs)
        action = model.sample(obs_batch).numpy()
        value = critic(obs_batch).numpy()
        action = assign_actions(action)

        eng.step()
        steps += 1

        # Add to each robots memory
        for i, robot in enumerate(robots):
            # penalty = 1 if action[i] == DO_NOTHING_INDEX else 0
            reward = (robot.energy-robot.previous_energy) #- penalty
            total_reward[robot] += reward
            memory[robot].append(
                rewards=reward,
                action=action[i],
                values=value[i],
                obs=obs[i],
                dones=eng.is_finished()
            )

        if eng.is_finished():
            print(f"Reward: {list(total_reward.values())}")
            total_reward = {r: 0 for r in robots}
            # Episode is over so test every 10th
            if iteration//10 > tests:
                tests += 1
                test()
            eng.init()

    losses = train(memory)
    print(f"{iteration}: {losses[0]}, "
          f"Actor: {losses[1]}, Critic: {losses[2]}, "
          f"Entropy: {losses[3]}, "
          f"Attention: {losses[4]}")
