import atexit
import os
import struct
import time
from collections import deque

import numpy as np
from robots import SignalRobot
from robots.proto import round_pb2
from robots.robot.utils import Move, Turn

os.chdir('../robocode_robot')

from a3c_model import Model



class AiRobot(SignalRobot):
    def __init__(self, *args, name=None, **kwargs):
        super(AiRobot, self).__init__(*args, **kwargs)
        assert name is not None, "Name should be set."
        self.name = name
        self.buffer = None
        self.memory = deque(maxlen=10)
        self.model = Model()

    def on_init(self):
        super(AiRobot, self).on_init()
        self.model.restore()
        if not self.buffer:
            self.buffer_file = './buffer/%s_%s.pb' % (int(time.time()), self.name)
            self.buffer = open(self.buffer_file, 'wb+')

            def close():
                print("Closing buffer")
                self.buffer.close()

            atexit.register(close)

        self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(5))
        self.scan = None

    def call_model(self):
        next(self.run_model)
        ph = np.zeros((1, 10, 11))
        state = np.stack(self.memory)
        ph[0, :len(state)] = state
        return self.model.run((ph, np.array([len(state)])))

    def do(self, tick):
        tick_pb = round_pb2.Tick()
        tick_pb.robot = str(self.__class__.__name__)
        tick_pb.num = int(tick)
        tick_pb.state.position.x, tick_pb.state.position.y = self.position
        tick_pb.state.bearing = self.bearing
        tick_pb.state.energy = self.energy

        scan = self.scan[0] if self.scan is not None else None
        if scan is not None:
            tick_pb.state.enemy_scanned = 1.0
            tick_pb.state.enemy_distance = scan.distance
            tick_pb.state.enemy_bearing = scan.bearing
            self.scan = None

        #TODO normalise the position or give max as a state
        state = np.array([
            *self.position,
            self.bearing,
            np.sin(self.bearing * np.pi / 180),
            np.cos(self.bearing * np.pi / 180),
            1.0 if scan is not None else 0.0,
            scan.distance if scan is not None else 0.0,
            scan.bearing if scan is not None else 0.0,
            np.sin(scan.bearing if scan is not None else 0.0 * np.pi / 180),
            np.cos(scan.bearing if scan is not None else 0.0 * np.pi / 180),
            self.energy
        ])

        self.memory.append(state)
        out = self.call_model()[0] + self.noise()
        out = np.clip(out, 0.0, 1.0)
        tick_pb.action.move = out[0]
        tick_pb.action.fire = out[1]
        tick_pb.action.turn = out[2]
        tick_pb.action.gun = out[3]
        tick_pb.action.radar = out[4]

        self.buffer.write(struct.pack('H', tick_pb.ByteSize()))
        self.buffer.write(tick_pb.SerializeToString())
        # Move Robot
        if out[0] > 0.1:
            self.moving = Move.FORWARD
        elif out[0] < -0.1:
            self.moving = Move.BACK
        else:
            self.moving = Move.NONE
        # Turn Robot
        if out[1] > 0.1:
            self.turning = Turn.RIGHT
        elif out[1] < -0.1:
            self.turning = Turn.LEFT
        else:
            self.turning = Turn.NONE
        # Turn Robot
        if out[2] > 0.1:
            self.gun.turning = Turn.RIGHT
        elif out[2] < -0.1:
            self.gun.turning = Turn.LEFT
        else:
            self.gun.turning = Turn.NONE
        # Turn Robot
        if out[3] > 0.1:
            self.radar.turning = Turn.RIGHT
        elif out[3] < -0.1:
            self.radar.turning = Turn.LEFT
        else:
            self.radar.turning = Turn.NONE

        if out[4] > 0:
            self.set_fire(3 * out[4])

    def on_scanned_robot(self, event):
        self.scan = event
