import yaml
from agent import AgentFactory
from monitor import interact
import gym
import numpy as np


def load_conf(conf_path="conf.yaml"):
    with open(conf_path, 'r') as stream:
        return yaml.safe_load(stream)

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    conf = load_conf("conf.yaml")
    AgentModule = AgentFactory.create_agent("SARSAmax")
    agent = AgentModule(env.observation_space.n, env.action_space.n, conf)
    avg_rewards, best_avg_reward = interact(env, agent)