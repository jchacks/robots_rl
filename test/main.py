from robots.app import App
from robots.engine import Engine
from robots.robot.utils import *
import model
import numpy as np
import tensorflow as tf
import tqdm
from wrapper import Dummy, AITrainingBattle
from utils import Memory

turning_opts = [Turn.LEFT, Turn.NONE, Turn.RIGHT]
moving_opts = [Move.FORWARD, Move.NONE, Move.BACK]

robots = [Dummy((255, 0, 0)), Dummy((0, 255, 0))]
size = (600, 400)

render = True
if render:
    app = App()
    battle = AITrainingBattle(robots, size)
    app.child = battle
    # Use the eng create by battle
    eng = battle.eng
else:
    eng = Engine(robots, size)


def get_obs(r):
    direction = np.sin(r.bearing * np.pi / 180), np.cos(r.bearing * np.pi / 180)
    return tf.cast(tf.concat([[r.energy/100, r.turret_heat/30], direction, r.position/(600, 400)], axis=0), tf.float32)


def discounted(rewards, dones, gamma=0.9):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.0 - done)
        discounted.append(r)
    return discounted[::-1]


max_steps = 1000
for i in range(100):
    # Create a memory per player
    eng.init()
    memory = {r: Memory('rewards,a_moving,a_turning,a_shoot,values,obs,dones') for r in robots}
    steps = 0
    total_reward = {r:0 for r in robots}
    while not eng.is_finished():
        if render:
            app.step()

        obs = [get_obs(r) for r in robots]
        obs = [tf.concat([obs[0], obs[1]], axis=0), tf.concat([obs[1], obs[0]], axis=0)]
        obs_batch = tf.stack(obs)
        moving, turning, shoot = model.actor.sample(obs_batch)
        shoot = shoot.numpy()
        value = model.critic(obs_batch).numpy()

        for i, robot in enumerate(robots):
            # Apply actions
            if robot.turret_heat > 0:
                shoot[i] = 0
            robot.moving = moving_opts[moving[i]]
            robot.base_turning = turning_opts[turning[i]]
            robot.should_fire = shoot[i]
            robot.previous_energy = robot.energy

        eng.step()
        steps += 1

        # Add to each robots memory
        for i, robot in enumerate(robots):
            reward = robot.energy-robot.previous_energy
            total_reward[robot] += reward
            memory[robot].append(
                rewards=reward,
                a_moving=moving[i],
                a_turning=turning[i],
                a_shoot=shoot[i],
                values=value[i],
                obs=obs[i],
                dones=eng.is_finished()
            )

        if eng.is_finished() or (steps % max_steps == 0):
            steps = 0
            obs = [get_obs(r) for r in robots]
            obs = [tf.concat([obs[0], obs[1]], axis=0), tf.concat([obs[1], obs[0]], axis=0)]
            obs_batch = tf.stack(obs)
            last_values = model.critic(obs_batch).numpy()

            b_rewards = []
            b_moving = []
            b_turning = []
            b_shoot = []
            b_values = []
            b_obs = []

            for robot, last_value in zip(robots, last_values):
                mem = memory[robot]
                b_rewards.append(discounted(np.array(mem['rewards'] + last_value), np.array(mem['dones'] + [0])))
                b_moving.append(mem['a_moving'])
                b_turning.append(mem['a_turning'])
                b_shoot.append(mem['a_shoot'])
                b_values.append(mem['values'])
                b_obs.append(mem['obs'])

            b_rewards = tf.concat(b_rewards, axis=0)[:, tf.newaxis]

            b_moving = tf.concat(b_moving, axis=0)
            b_turning = tf.concat(b_turning, axis=0)
            b_shoot = tf.concat(b_shoot, axis=0)

            b_values = tf.concat(b_values, axis=0)
            b_obs = tf.concat(b_obs, axis=0)

            losses = model.train(b_obs, b_rewards, (b_moving, b_turning, b_shoot), b_values)
            print(f"Total: {losses[0]}, Actor: {losses[1]}, Critic: {losses[2]}, Reward: {total_reward.values()}")
