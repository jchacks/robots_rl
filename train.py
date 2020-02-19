import os
import struct
from collections import namedtuple

import numpy as np
import tensorflow as tf
from robots.proto import round_pb2

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
        actions = d[:, -7:-2]
        scores = d[:,-1:]
        state = np.concatenate((d[:,:-7], scores),axis=-1)
        indexer = np.arange(window)[None, :] + step * np.arange(len(d) - window + 1)[:, None]
        yield state[indexer], actions[indexer[:,-1]]

def windows_to_batch(data):
    states = []
    actions = []
    while True:
        if sum(len(a) for a in actions) > 10000:
            yield np.concatenate(states), np.concatenate(actions)
            states = []
            actions = []
        state, action = next(data)
        states.append(state)
        actions.append(action)


file = os.listdir(buffer_dir)[0]
protos = read_proto(buffer_dir + file)
nps = proto_to_numpy(protos)
win = numpy_to_windowed(nps)
data = windows_to_batch(win)



import model as m

model = m.model(10, 5)
train = m.train(model.fetch['score_prediction'])

get_summ = tf.summary.merge_all()

saver = tf.train.Saver(tf.trainable_variables(), save_relative_paths=True)
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

summ_writer = tf.summary.FileWriter('./train/', sess.graph)
sess.run(tf.global_variables_initializer())

train_fetches = {
    'summary': get_summ,
    'train': train.fetch['actor_minimizer'],
    'actor_loss': train.fetch['actor_loss'],
    'critic_loss': train.fetch['critic_loss'],
}

for idx in range(100000):
    state, action = next(data)
    res = sess.run(train_fetches, {
        model.feed['state']: (state, np.ones((len(state),), dtype='int32') * 10),
        model.feed['action']: action,
        train.feed['score_target']: state[:, -1, -1][:, None]
    })

    summ_writer.add_summary(res['summary'], idx)
    print(res['actor_loss'], res['critic_loss'])
    saver.save(sess, './checkpoint/graph', global_step=idx//100)

