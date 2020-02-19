import struct
from collections import deque, namedtuple
import os
import numpy as np
from robots.proto import round_pb2

buffer_dir = 'C:/Users/jchack/Documents/projects/robocode_robot/buffer/'
np.set_printoptions(precision=3, suppress=True, linewidth=250)

Data = namedtuple('Data', 'state action reward')


def read_proto(file):
    with open(file, 'rb') as f:
        data = deque()
        i = 0
        while True:
            try:
                s = struct.unpack('H', f.read(2))
                print(s)
            except struct.error as e:
                print(e)
                break
            tick = round_pb2.Tick()
            tick.ParseFromString(f.read(s[0]))
            data.append(tick)
            i += 1
        print("Read %s messages" % i)
    return data

for file in os.listdir(buffer_dir):
    data = read_proto(buffer_dir + file)


def proto_to_numpy():
    all = []
    tmp = None
    while True:
        d = data.popleft()
        name = d.robot
        print(d.num)
        if d.num <= 1.0:

            if tmp: all.append(np.stack(tmp))
            tmp = []

        tmp.append(np.array([
            d.num,
            d.state.position.x,
            d.state.position.y,
            d.state.bearing,
            d.state.energy,
            d.action.move,
            d.action.fire,
            d.action.turn,
            d.action.gun,
            d.action.radar,
        ], 'float32'))
