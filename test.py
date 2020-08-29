import threading as th

from robots.app import MultiBattleApp

from a2c_model import Model
from ai_bot import AiRobot
from runner import Env, Runner

env = Env((200, 200), [AiRobot, AiRobot], num_battles=8)
runner = Runner(env, Model(9, 2))

app = MultiBattleApp(dimensions=(1350, 720),
                     battle=env,
                     simulate=False,
                     rows=2,
                     fps_target=60)
th = th.Thread(target=app.run, daemon=True)
th.start()
runner.test()
