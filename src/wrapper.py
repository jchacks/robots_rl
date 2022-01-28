from robots.app import Battle
from RoboArena import PyRobot
import numpy as np
from robots.robot.events import *
from utils import TURNING, MOVING, TIMER
import random


ACTION_DIMS = (1, 3 * 3 * 3)
OHA_ACTIONS = [np.eye(n + 1) for n in (1, 3, 3, 3)]


def get_action_vector(action):
    shoot, other = action
    other = np.unravel_index(other, (3, 3, 3))
    return np.concatenate(
        [[shoot]] + [l[act][:-1] for act, l in zip(other, OHA_ACTIONS[1:])]
    )


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


class Dummy(PyRobot):
    def init(self):
        self.value = 0.0  # Used for displaying predicted value
        self.norm_value = 0.0  # Used for displaying predicted value
        self.prev_action = np.zeros(len(ACTION_DIMS), dtype=np.uint8)
        self.lstmstate = np.zeros((2, 128), dtype=np.float32)

        self.step_reward = 0
        self.step_reward_ema = 0
        self.total_reward = 0

        self.bullets_hit = 0
        self.hit_by_bullets = 0
        self.fire_power = np.random.uniform(0.1, 3.0)

    def run(self):
        pass

    def on_hit_by_bullet(self):
        self.step_reward -= 0.5
        self.hit_by_bullets += 1

    def on_bullet_hit(self, event: BulletHitEvent):
        self.step_reward += 2
        self.bullets_hit += 1

    def on_hit_wall(self):
        self.step_reward -= 0.5

    def get_obs(self):
        TIMER.start("self")
        R = get_rot_mat(-self.base_rotation)
        turret = self.turret_direction
        TIMER.start("prev_action")
        oha = get_action_vector(self.prev_action)
        TIMER.stop("prev_action")

        obs = [self.energy / 50 - 1, self.heat / 15 - 1, self.speed / 8, *(R @ turret)]
        TIMER.stop("self")
        TIMER.start("oppo")
        oppo_data = []
        for r in self.opponents:
            direction = r.position - self.position
            distance = np.sqrt(np.sum(direction ** 2))
            direction = direction / (distance + 1e-8)
            oppo_data.append(
                [
                    r.energy / 50 - 1,
                    distance / 500,
                    *(R @ direction),
                    r.speed / 8,
                    np.dot(turret, direction),
                ]
            )
        TIMER.stop("oppo")
        return np.concatenate([obs] + [oha] + oppo_data)

    def assign_actions(self, action):
        self.prev_action = action
        # Apply actions
        # shoot, turn, move, turret = action
        shoot, other = action
        turn, move, turret = get_action(other, (3, 3, 3))

        # Stop full rotations from giving rewards
        if turn > 0:
            self.step_reward -= 0.03
        if turret > 0:
            self.step_reward -= 0.03

        try:
            self.moving = MOVING[move].value
            self.base_turning = TURNING[turn].value
            self.turret_turning = TURNING[turret].value
            self.should_fire = shoot > 0
            self.previous_energy = self.energy
        except Exception:
            print("Failed assigning actions", self, turn, shoot)
            raise
        # return shoot, get_argmax((turn, move, turret), (3, 3, 3))
        # return shoot, turn, move, turret


class Random(Dummy):
    def run(self):
        self.moving = random.randint(-1, 1)
        self.base_turning = random.randint(-1, 1)
        self.turret_turning = random.randint(-1, 1)
        self.should_fire = random.randint(0, 1)

    def assign_actions(self, action):
        pass
