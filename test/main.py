from robots.config import BattleSettings
from robots.app import App
from robots.robot import Robot
from robots.robot.utils import *
import random


class RandomRobot(Robot):
    def run(self):
        if random.random() > 0.5:
            self.moving = Move.FORWARD
        else:
            self.moving = Move.BACK

        if random.random() > 0.5:
            self.base_turning = Turn.LEFT
        else:
            self.base_turning = Turn.RIGHT

        if random.random() > 0.5:
            self.gun_turning = Turn.LEFT
        else:
            self.gun_turning = Turn.RIGHT

        if random.random() > 0.5:
            self.turret_turning = Turn.LEFT
        else:
            self.turret_turning = Turn.RIGHT

        if random.randint(0, 1):
            self.fire(random.randint(1, 3))


battle_settings = BattleSettings([RandomRobot((255, 0, 0)), RandomRobot((0, 255, 0))])
app = App(battle_settings)
app.run()