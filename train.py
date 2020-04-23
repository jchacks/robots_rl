from robots.app import App, Battle
from ai_bot import AiRobot

size = (600, 400)
app = App(size)
app.battle = Battle(app, size, [AiRobot, AiRobot])
app.on_execute()
