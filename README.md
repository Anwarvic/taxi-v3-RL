# Taxi-v3 Environment
My attempt to solve the taxi-v3 OpenAI gym environment using various algorithms
such as:

- Q-Learning
- SARSA
- SARSAmax
- Expected-SARSA

<br>
<div align="Center">
    <img src="./assets/demo.gif" width=500>
</div>

## Install

To install dependencies for this repo, run the following command:

```
pip install -r requirements.txt
```

## Description

The workspace contains the following files:

- `agent.py`: A factory for creating RL agents (follows Factory design pattern).
- `conf.yaml`: A YAML file containing the hyper-parameters for agents.
- `DQNN_Experimentation.ipynb`: An Ipython notebook containing the experiments for DQNN algorithm.
- `main.py`: The start point for this repo. In this script, you can create agents, run them and compare between their performance.
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `q_learning.py`: An agent implemented using Q-Learning algorithm.
- `sarsa.py`: Three agents implemented using SARSA and Expected-SARSA respectively.
- `utils.py`: Helpful functions.

## Environment Benchmark

OpenAI Gym defines ["solving"](https://gym.openai.com/envs/Taxi-v3/) this task as getting average return of 9.7 over 100 consecutive trials.

The following figure shows the performance of the different algorithms in this repo:

<div align="Center">
    <img src="./assets/Figure_1.png" width=500>
</div>