import threading as th

from robots.app import MultiBattleApp

from model import Model
from ai_bot import AiRobot
from runner import Env, Runner

env = Env((600, 600), [AiRobot, AiRobot], num_battles=100)
runner = Runner(env, Model(9, 2), train_steps=20)

app = MultiBattleApp(
    dimensions=(1350, 720),
    battle=env,
    simulate=False,
    rows=4,
    fps_target=120,
    # use_dirty=True
)
th = th.Thread(target=app.run, daemon=True)
th.start()
runner.train()
