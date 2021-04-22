# Reinforcement Learning Robot

This repo is being developed in conjucntion with the [robots](https://github.com/jchacks/robots) AI playground repo.
It provides an environment with delayed rewards requiring, additions to classical RL algos to learn the 'value'/'critic' function. 

_Robots package is going though an extensive rewrite and therefore makes alot of this redundant._

Checkout the run metrics on [wandb](https://wandb.ai/jchacks/robots_rl).

### Todo

* Try to model the value network with a positive network side and a negative network side
  Rational is that if a completely negative batch occurs then the network will be more robust.
  Positive side wont train.
