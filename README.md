# Reinforcement Learning Robot

This repo is being developed in conjucntion with the [robots](https://github.com/jchacks/robots) AI playground repo.
It provides an environment with delayed rewards requiring, additions to classical RL algorithms to learn the 'value'/'critic' function.

The `main.py` script with in the repo successfully trains an agent against itself that can aim and fire with a lead in order to hit its opponent.  

## Actions

The action space of the agent is (2,3,3,3) independent discrete actions.  
The actions available are:
* Shooting (none shoot)
* Moving (forward none backward)
* Base Turning (left none right)
* Turret Turning (left none right)

These are modelled as multiple categorical probability distributions.
_In the future it may be beneficial to give the agent more control and provide continuous independent actions._

## Model Structure

A very simple Multi-Layered Perceptron with relu activations is used to paramertarise the Actor and critic.  

Currently very basic observations of the state are made; the agents postition, direction vector, energy, gun heat, speed _(-8,8)_, turret direction vector are given for both itself and its opponent.  No LSTM was needed in order for convergance to be observed. 

## Notes

If changes are being made to the environment (such as reduced actionspace) then careful balancing of the critic loss is required, the agent converges too soon to a sub optimal solution.  Additional changes can be added to make the agent more robust during training. 

Checkout the run metrics on [wandb](https://wandb.ai/jchacks/robots_rl).


