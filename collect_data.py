from robots.app import App, Battle
from ai_bot import AiRobot

app = App((1280, 720))
app.battle = Battle(app, (1280, 720), [AiRobot(name='0'), AiRobot(name='1')])
app.on_execute()
