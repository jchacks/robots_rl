from robots.app import Battle
from robots.robot import Robot
import numpy as np
from utils import TURNING, MOVING, Memory


class AITrainingBattle(Battle):
    def step(self):
        """Null out step op so we can control it from training side."""
        pass


class Dummy(Robot):
    def init(self, *args, size=None, opponents=None, **kwargs):
        self.battle_size = size
        self.opponents = opponents

    def run(self):
        pass

    def get_state(self,):
        s = np.array(self.battle_size)
        center = s//2
        direction = np.sin(self.bearing * np.pi / 180), np.cos(self.bearing * np.pi / 180)
        turret = np.sin(self.turret_bearing * np.pi / 180), np.cos(self.turret_bearing * np.pi / 180)
        return np.concatenate([
            [(self.energy/50) - 1, self.turret_heat/30, self.velocity/8],
            direction,
            turret,
            (self.position/center) - 1,
            (self.position/self.battle_size),
        ], axis=0)

    def get_obs(self):
        return np.concatenate([self.get_state()] + [state() for state in self.opponents])

    def assign_actions(self, action):
        # Apply actions
        shoot, turn, move, turret = action
        if self.turret_heat > 0:
            shoot = 0
        try:
            self.moving = MOVING[move]
            self.base_turning = TURNING[turn]
            self.turret_turning = TURNING[turret]
            self.should_fire = shoot > 0
            self.previous_energy = self.energy
        except Exception:
            print("Failed assigning actions", self, turn, shoot)
            raise
        return action
