from robots.app import Battle
from robots.robot import Robot
import numpy as np
from robots.robot.events import *
from utils import TURNING, MOVING
import random

ACTION_DIMS = (1, 3 * 3 * 3)
OHA_ACTIONS = [np.eye(n + 1) for n in (1, 3, 3, 3)]


class AITrainingBattle(Battle):
    def step(self):
        """Null out step op so we can control it from training side."""
        pass


def get_rot_mat(rads):
    c, s = np.cos(rads), np.sin(rads)
    return np.array(((c, -s), (s, c)))


def get_action(argmax, dims):
    "max logit locations -> actions"
    return np.unravel_index(argmax, dims)


def get_argmax(actions, dims):
    "actions -> max logit locations"
    return np.ravel_multi_index(actions, dims)


class Dummy(Robot):
    def init(self, size=None, **kwargs):
        self.battle_size = size
        self.value = 0.0  # Used for displaying predicted value
        self.norm_value = 0.0  # Used for displaying predicted value
        self.opponents = [r for r in kwargs["all_robots"] if r != self]
        self.prev_action = np.zeros(len(ACTION_DIMS), dtype=np.uint8)
        self.lstmstate = np.zeros((2, 128), dtype=np.float32)

        self.step_reward = 0
        self.total_reward = 0

        self.bullets_hit = 0
        self.hit_by_bullets = 0

        self.fire_power = np.random.uniform(0.1, 3.0)

    def run(self):
        pass

    def on_hit_by_bullet(self, event: HitByBulletEvent):
        self.step_reward -= 0.5
        self.hit_by_bullets += 1

    def on_bullet_hit(self, event: BulletHitEvent):
        self.step_reward += 2
        self.bullets_hit += 1

    def on_hit_wall(self, event: HitWallEvent):
        self.step_reward -= 0.5

    def get_obs(self):
        s = np.array(self.battle_size)
        center = s // 2
        center = center - self.position
        center = center / np.sqrt(np.sum(center ** 2))
        R = get_rot_mat(-self.base_rotation)

        turret = self.turret_rotation
        turret = np.array([np.cos(turret), np.sin(turret)])

        shoot, other = self.prev_action
        other = get_action(other, (3, 3, 3))
        p_action = np.concatenate([[shoot], other])

        oha = np.concatenate([l[act][:-1] for act, l in zip(p_action, OHA_ACTIONS)])
        obs = np.concatenate(
            [
                [(self.energy / 50) - 1, self.turret_heat / 30, self.velocity / 8],
                R @ turret,
                R @ center,
                oha,
            ],
            axis=0,
        )

        oppo_data = []
        for r in self.opponents:
            attrs = r.get_visible_attrs()
            direction = attrs["position"] - self.position
            distance = np.sqrt(np.sum(direction ** 2))
            direction = direction / distance
            oppo_data.append(
                np.array(
                    [
                        (attrs["energy"] / 50) - 1,
                        np.log(distance) / np.log(100),
                        # distance/300,
                        *(R @ direction),
                        attrs["velocity"] / 8,
                        np.dot(turret, direction),
                    ]
                )
            )

        return np.concatenate([obs] + oppo_data)

    def assign_actions(self, action):
        # Apply actions
        # shoot, turn, move, turret = action
        shoot, other = action
        turn, move, turret = get_action(other, (3, 3, 3))

        # Stop full rotations from giving rewards
        if turn > 0:
            self.step_reward -= 0.014
        if turret > 0:
            self.step_reward -= 0.014

        try:
            self.moving = MOVING[move]
            self.base_turning = TURNING[turn]
            self.turret_turning = TURNING[turret]
            self.should_fire = shoot > 0
            self.previous_energy = self.energy
        except Exception:
            print("Failed assigning actions", self, turn, shoot)
            raise
        # return shoot, get_argmax((turn, move, turret), (3, 3, 3))
        # return shoot, turn, move, turret


class Random(Dummy):
    def run(self):
        self.moving = MOVING[random.randint(0, 2)]
        self.base_turning = TURNING[random.randint(0, 2)]
        self.turret_turning = TURNING[random.randint(0, 2)]
        self.should_fire = random.randint(0, 1)

    def assign_actions(self, action):
        pass
