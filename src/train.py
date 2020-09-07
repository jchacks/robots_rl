import threading as th

from robots.app import MultiBattleApp

from model import Model
from ai_bot import AiRobot
from runner import Env, Runner

num_battles = 20
num_steps = 200
env = Env((200, 200), [AiRobot, AiRobot], num_battles=num_battles)
model = Model(num_steps, num_battles * 2, 9, 2)
runner = Runner(env, model, train_steps=num_steps)

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
