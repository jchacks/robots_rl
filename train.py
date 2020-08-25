import numpy as np
from robots.app import App
from robots.battle import MultiBattle

from a2c_model import Runner
from ai_bot import AiRobot

runner = Runner()


class AIBattle(MultiBattle):
    def __init__(self, *args, **kwargs):
        super(AIBattle, self).__init__(*args, **kwargs)
        self.runner = runner

    def delta(self):
        states = {}
        for battle in self.battles:
            if not battle.is_finished:
                states.update({robot: (robot.get_state(), robot.energy, False) for robot in battle.alive_robots})
            elif not battle.done:
                states.update({robot: (robot.get_state(), robot.energy, True) for robot in battle.robots})
                battle.done = True

        finished_pct = np.mean([True if battle.is_finished else False for battle in self.battles])
        if finished_pct > 0.6:
            for battle in self.battles:
                if not battle.is_finished:
                    battle.is_finished = True
                    battle.on_round_end()
                    battle.on_clean_up()

        if self.tick % 10 == 0:
            self.runner.train()

        if len(states.keys()) > 0:
            actions = self.runner.run(self.tick, states)
            for r, action in actions.items():
                r.delta(self.tick, action)

    def on_battles_ended(self):
        # self.runner.train()
        pass


app = App((1350, 720))
app.battle = AIBattle(app, (200, 200), [AiRobot, AiRobot], num_battles=80)
app.on_execute()
