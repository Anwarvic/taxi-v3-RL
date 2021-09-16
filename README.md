# Taxi-v3 Environment
My attempt to solve the taxi-v3 OpenAI gym environment using various agents.

# Description

The workspace contains three files:

- `q_agent.py`: A reinforcement learning agent that uses q-learning.
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of your agent.

# `main.py` script

When you run `main.py`, the agent that you specify in `agent.py` interacts with the environment for 20,000 episodes. The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.

- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive. So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`. This is the final score that you should use when determining how well your agent performed in the task.

## Evaluate your Performance

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 9.7 over 100 consecutive trials.

