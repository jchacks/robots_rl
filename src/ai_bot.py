import numba as nb
import numpy as np
from robots import SignalRobot
from robots.robot.utils import Turn

np.set_printoptions(precision=3, suppress=True, linewidth=250)


def indexer(length, window, step=1):
    return np.arange(window)[None, :] + step * np.arange(length - window + 1)[:, None]


@nb.njit
def state_lstm(lstm_history: int, state):
    ph = np.zeros((1, lstm_history, 13))
    ph[0, : len(state)] = state
    return ph, np.array([len(state)])


class AiRobot(SignalRobot):
    def __init__(self, *args, **kwargs):
        super(AiRobot, self).__init__(*args, **kwargs)
        self.buffer = None
        self.records = []
        self.scan = None
        self.previous_energy = None

    def reset(self):
        super(SignalRobot, self).reset()
        self.previous_energy = 100

    def on_init(self):
        super(AiRobot, self).on_init()
        self.scan = None

    def get_obs(self):
        scan = self.scan[0] if self.scan is not None else None
        x, y = self.position
        x, y = x / 600, y / 600

        b_d = self.direction
        g_d = self.gun.direction
        r_d = self.radar.direction

        state = np.array(
            [
                x,
                y,
                # *b_d,
                *g_d,
                # *r_d,
                self.energy / 100,
                scan.energy / 100 if scan else 1,
                scan.distance / 600 if scan else -1,
                scan.direction[0] if scan else -1,
                scan.direction[1] if scan else -1,
            ]
        )
        self.scan = None
        return state

    def get_done(self):
        return self.battle.is_finished

    def do(self, tick, action):
        # # Move Robot
        # if action[0] > 0.01:
        #     self.moving = Move.FORWARD
        # elif action[0] < -0.01:
        #     self.moving = Move.BACK
        # else:
        #     self.moving = Move.NONE
        # # Turn Robot
        # if action[1] > 0.01:
        #     self.turning = Turn.RIGHT
        # elif action[1] < -0.01:
        #     self.turning = Turn.LEFT
        # else:
        #     self.turning = Turn.NONE
        # Turn Gun

        # # Would be interesting to try
        # opt = [Turn.Right, Turn.LEFT, Turn.NONE]
        # idx = action[0].argmax()
        #
        # self.gun.turning = opt[idx]
        #
        if action[0] > 0.001:
            self.gun.turning = Turn.RIGHT
        elif action[0] < -0.001:
            self.gun.turning = Turn.LEFT
        else:
            self.gun.turning = Turn.NONE
        # Turn Radar
        # if action[3] > 0.01:
        #     self.radar.turning = Turn.RIGHT
        # elif action[3] < -0.01:
        #     self.radar.turning = Turn.LEFT
        # else:
        #     self.radar.turning = Turn.NONE

        if action[1] > 0:
            self.set_fire(3 * action[1])

    def on_scanned_robot(self, event):
        self.scan = event

    def on_battle_ended(self, battle_ended_event):
        pass
