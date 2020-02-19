from robots.app import App, Battle
from ai_bot import AiRobot

app = App((128, 72))
app.battle = Battle(app, (1280, 720), [AiRobot, AiRobot])
app.on_execute()