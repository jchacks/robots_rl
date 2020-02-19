import atexit
import struct

import numpy as np
from robots import SignalRobot
from robots.proto import round_pb2
from robots.robot.utils import Move, Turn
import time



class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class AiRobot(SignalRobot):
    def __init__(self, *args, **kwargs):
        super(AiRobot, self).__init__(*args, **kwargs)
        self.buffer = None

    def on_init(self):
        super(AiRobot, self).on_init()
        if not self.buffer:
            self.buffer_file = './buffer/%s_%s.pb' % (int(time.time()),id(self))
            self.buffer = open(self.buffer_file, 'wb+')
            def close():
                print("Closing buffer")
                self.buffer.close()
            atexit.register(close)

        self.call_model = OrnsteinUhlenbeckActionNoise(np.zeros(5))
        self.scan = None

    def do(self, tick):
        tick_pb = round_pb2.Tick()
        tick_pb.robot = str(self.__class__.__name__)
        tick_pb.num = int(tick)
        tick_pb.state.position.x, tick_pb.state.position.y = self.position
        tick_pb.state.bearing = self.bearing
        tick_pb.state.energy = self.energy

        scan = self.scan
        if scan is not None:
            tick_pb.state.enemy_scanned = 1.0
            tick_pb.state.enemy_distance = scan.distance
            tick_pb.state.enemy_bearing = scan.bearing
            self.scan = None

        data = np.array([
            scan.bearing if scan else -1.0,
            scan.distance if scan else -1.0,
            scan.heading if scan else -1.0,
            scan.velocity if scan else -1.0,
            *self.position,
            self.bearing,
            self.energy
        ])

        out = self.call_model()
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

        if out[4] > 0.1:
            self.set_fire(out[4])

    def on_scanned_robot(self, event):
        self.scan = event
