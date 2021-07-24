from robots.app import Battle
from robots.robot import Robot
import numpy as np
from utils import TURNING, MOVING


class AITrainingBattle(Battle):
    def step(self):
        """Null out step op so we can control it from training side."""
        pass


class Dummy(Robot):
    def init(self, size=None, **kwargs):
        self.battle_size = size
        self.value = 0.0  # Used for displaying predicted value
        self.opponents = [r for r in kwargs["all_robots"] if r != self]

    def run(self):
        pass

    def get_state(
        self,
    ):
        s = np.array(self.battle_size)
        center = s // 2
        direction = np.sin(self.bearing * np.pi / 180), np.cos(
            self.bearing * np.pi / 180
        )
        turret = np.sin(self.turret_bearing * np.pi / 180), np.cos(
            self.turret_bearing * np.pi / 180
        )
        return np.concatenate(
            [
                [(self.energy / 50) - 1, self.turret_heat / 30, self.velocity / 8],
                direction,
                turret,
                (self.position / center) - 1,
                (self.position / s),
            ],
            axis=0,
        )

    def get_obs(self):
        oppo_data = []
        size = np.array(self.battle_size)
        for r in self.opponents:
            attrs = r.get_visible_attrs()
            v = self.position - attrs["position"]
            d = np.sqrt(np.sum(v ** 2))
            oppo_data.append(
                np.array(
                    [
                        (attrs["energy"] / 50) - 1,
                        d / 100,
                        *(v / d),
                        attrs["velocity"] / 8,
                    ]
                )
            )

        return np.concatenate([self.get_state()] + oppo_data)

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
        return shoot, turn, move, turret
