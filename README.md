# Reinforcement Learning Robot

This repo is being developed in conjucntion with the [robots](https://github.com/jchacks/robots) AI playground repo.
It provides an environment with delayed rewards requiring, additions to classical RL algos to learn the 'value'/'critic' function. 

#### How to use:

Data collection script `collect_data.py` writes protobuff events to `./buffer/`.

Training script collects rounds from `./buffer/` and feeds them to the model.

### Todo
