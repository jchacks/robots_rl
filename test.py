from robots.app import App
from robots.battle import Battle

from a2c_model import Runner
from ai_bot import AiRobot

runner = Runner()

class AIBattle(Battle):
    def __init__(self, *args, **kwargs):
        super(AIBattle, self).__init__(*args, **kwargs)
        self.runner = runner

    def delta(self):
        if not self.is_finished:
            # Incentive to do something
            states = {}
            states.update({robot: (robot.get_state(), robot.energy) for robot in self.alive_robots})
            if len(states.keys()) > 0:
                actions = self.runner.test(0, states)
                print(actions)
                for r, action in actions.items():
                    r.delta(self.tick, action)


app = App((600, 400))
app.battle = AIBattle(app, (600, 400), [AiRobot, AiRobot])
app.on_execute()
