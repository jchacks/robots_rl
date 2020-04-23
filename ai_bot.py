import os
from collections import deque

import numpy as np
from robots import SignalRobot
from robots.robot.utils import Move, Turn

os.chdir('../robocode_robot')

from a2c_model import Model


def indexer(length, window, step=1):
    return (np.arange(window)[None, :] + step * np.arange(length - window + 1)[:, None])


models = {
    '1': Model()
}


class AiRobot(SignalRobot):
    def __init__(self, *args, name='1', **kwargs):
        super(AiRobot, self).__init__(*args, **kwargs)
        self.name = name
        self.buffer = None
        self.lstm_history = 25
        self.memory = deque(maxlen=self.lstm_history)
        self.model = models['1']
        self.records = []

    def on_init(self):
        super(AiRobot, self).on_init()
        # self.model.restore()
        self.scan = None

    def state_lstm(self):
        ph = np.zeros((1, self.lstm_history, 13))
        state = np.stack(self.memory)
        ph[0, :len(state)] = state
        return ph, np.array([len(state)])

    def do(self, tick):
        scan = self.scan[0] if self.scan is not None else None
        state = np.array([
            *self.position,
            np.sin(self.bearing * np.pi / 180),
            np.sin(self.gun.bearing * np.pi / 180),
            np.sin(self.radar.bearing * np.pi / 180),
            np.cos(self.bearing * np.pi / 180),
            np.cos(self.gun.bearing * np.pi / 180),
            np.cos(self.radar.bearing * np.pi / 180),
            1.0 if scan is not None else 0.0,
            scan.distance if scan is not None else 0.0,
            np.sin(scan.bearing * np.pi / 180 if scan is not None else 0.0),
            np.cos(scan.bearing * np.pi / 180 if scan is not None else 0.0),
            self.energy
        ])
        self.scan = None

        self.memory.append(state)
        state_lstm = self.state_lstm()
        action, value = self.model.run(state_lstm)
        out = action[0]
        # Record so it can be displayed
        self.last_out = out

        self.records.append((int(tick), state_lstm, value, action, self.energy))
        # Move Robot
        if out[0] > 0.25:
            self.moving = Move.FORWARD
        elif out[0] < -0.25:
            self.moving = Move.BACK
        else:
            self.moving = Move.NONE
        # Turn Robot
        if out[1] > 0.25:
            self.turning = Turn.RIGHT
        elif out[1] < -0.25:
            self.turning = Turn.LEFT
        else:
            self.turning = Turn.NONE
        # Turn Gun
        if out[2] > 0.25:
            self.gun.turning = Turn.RIGHT
        elif out[2] < -0.25:
            self.gun.turning = Turn.LEFT
        else:
            self.gun.turning = Turn.NONE
        # Turn Radar
        if out[3] > 0.25:
            self.radar.turning = Turn.RIGHT
        elif out[3] < -0.25:
            self.radar.turning = Turn.LEFT
        else:
            self.radar.turning = Turn.NONE

        if out[4] > 0:
            self.set_fire(3 * out[4])


    def on_scanned_robot(self, event):
        self.scan = event

    def on_battle_ended(self, battle_ended_event):
        tick, state_lstm, value, action, energy = zip(*self.records)
        state, seq = zip(*state_lstm)
        state = np.concatenate(state)
        seq = np.concatenate(seq)
        value = np.squeeze(value)
        action = np.concatenate(action)
        energy = np.array(energy) - 100
        rewards = energy[1:] - energy[:-1]  # Reward per state is difference in energy
        mc_value = np.cumsum(rewards[::-1])[::-1]  # Monte Carlo
        advantage = rewards + mc_value - value[:-1]
        print('advantage:', advantage.mean(), 'rewards:', rewards.mean(), 'value:', value.mean())
        res = self.model.train((state[:-1], seq[:-1]), advantage, mc_value, action[:-1])
        battle_ended_event.console.buffer += '\n' + ' '.join(str(r) for r in res)
