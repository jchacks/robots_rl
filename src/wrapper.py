from robots.app import Battle
from robots.robot import Robot


class AITrainingBattle(Battle):
    def step(self):
        """Null out step op so we can control it from training side."""
        pass


class Dummy(Robot):
    def run(self):
        pass
