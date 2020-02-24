import os
import struct
from collections import namedtuple

import numpy as np
import tensorflow as tf
from proto import round_pb2

os.chdir('../robocode_robot')

buffer_dir = './buffer/'
np.set_printoptions(precision=3, suppress=True, linewidth=250)

Data = namedtuple('Data', 'state action reward')


def read_proto(file):
    with open(file, 'rb') as f:
        i = 0
        while True:
            try:
                s = struct.unpack('H', f.read(2))
            except struct.error as e:
                print(e)
                break
            tick = round_pb2.Tick()
            tick.ParseFromString(f.read(s[0]))
            yield tick
            i += 1
        print("Read %s messages" % i)


def proto_to_numpy(data):
    tmp = []
    num = -1
    for d in data:
        if d.num < num:
            a = np.stack(tmp)
            a[:, -1] = np.roll(a[:, -1], -1)
            yield a[:-1]
            tmp = []
        name = d.robot
        num = d.num
        tmp.append(
            np.array([
                d.state.position.x,
                d.state.position.y,
                d.state.bearing,
                np.sin(d.state.bearing * np.pi / 180),
                np.cos(d.state.bearing * np.pi / 180),
                d.state.enemy_scanned,
                d.state.enemy_distance,
                d.state.enemy_bearing,
                np.sin(d.state.enemy_bearing * np.pi / 180),
                np.cos(d.state.enemy_bearing * np.pi / 180),
                d.action.move,
                d.action.fire,
                d.action.turn,
                d.action.gun,
                d.action.radar,
                d.state.energy
            ], 'float32')
        )


def numpy_to_windowed(data, window=10, step=1):
    for d in data:
        actions = d[:, 10:-1]
        state = d[:,:10]
        rewards = d[:,-1]
        indexer = (np.arange(window)[None, :] + step * np.arange(len(d) - window + 1)[:, None])[:-1]

        rewards = rewards[indexer[:,-1:]]
        diff = rewards[:-1] - rewards[1:]
        yield state[indexer], actions[indexer[:,-1]], np.cumsum(diff[::-1])[::-1]


def windows_to_batch(data):
    states = []
    actions = []
    rewards = []
    while True:
        if sum(len(a) for a in actions) > 10000:
            yield np.concatenate(states), np.concatenate(actions), np.concatenate(rewards)
            states = []
            actions = []
            rewards = []
        state, action, reward = next(data)
        states.append(state)
        actions.append(action)
        rewards.append(reward)


def flatten(gen_gen):
    for gen in gen_gen:
        for item in gen:
            yield item



from a3c_model import Model

m = Model()
m.loss()

maw = tf.reduce_mean([tf.reduce_mean(tf.abs(v)) for v in tf.get_collection('full_weights', scope='actor')])
mcw = tf.reduce_mean([tf.reduce_mean(tf.abs(v)) for v in tf.get_collection('full_weights', scope='critic')])
tf.summary.scalar('mean_actor_weights', maw)
tf.summary.scalar('mean_critic_weights', mcw)

summ = m.summary()
summ_writer = tf.summary.FileWriter('./train/', m._sess.graph)

m.init()

idx = 0
while True:
    try:
        protos = flatten(read_proto(buffer_dir + file) for file in sorted(os.listdir(buffer_dir),reverse=True))
        nps = proto_to_numpy(protos)
        win = numpy_to_windowed(nps)
        data = windows_to_batch(win)

        for idx in range(idx, idx + 100000):
            print(idx,'preparing...',end='')
            state, action = next(data)
            print('running...',end='')
            res = m.train()

            print(res['actor_loss'], res['critic_loss'])
            if idx % 50 == 0:
                summ_writer.add_summary(res['summary'], idx)
            if idx % 500 == 0:
                print('saving')
                saver.save(sess, './checkpoint/graph', global_step=idx)
    except RuntimeError:
        pass
