import threading as th

from robots.app import MultiBattleApp

from model import Model
from ai_bot import AiRobot
from runner import Env, Runner


num_battles = 1
num_steps = 200
env = Env((600, 600), [AiRobot, AiRobot], num_battles=num_battles)
model = Model(num_steps, num_battles * 2, 9, 2)
runner = Runner(env, model, train_steps=num_steps)

app = MultiBattleApp(dimensions=(1350, 720), battle=env, simulate=False, rows=2, fps_target=60)
th = th.Thread(target=app.run, daemon=True)
th.start()
runner.test()
