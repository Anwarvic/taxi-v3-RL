import time
import os
# import sys
import yaml
import matplotlib.pyplot as plt

from monitor import train
from agent import AgentFactory



def load_conf(conf_path="conf.yaml"):
    with open(conf_path, 'r') as stream:
        return yaml.safe_load(stream)


def compare(env, algorithms):
    """
    Compare between different algorithms passed
    """
    conf = load_conf("conf.yaml")
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel(f'Average reward over {conf["window"]} episodes')
    for algo in algorithms:
        print(algo)
        AgentModule = AgentFactory.create_agent(algo)
        agent = AgentModule(env.observation_space.n, env.action_space.n, conf)
        avg_rewards, _ = train(env, agent, conf)
        plt.plot(avg_rewards)
    if len(algorithms) > 1: plt.legend(labels=algorithms, loc="lower right")
    plt.show()


def interact(env, agent):
    while(True):
        os.system("clear")
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)

            # Put each rendered frame into dict for animation
            grid = env.render(mode='ansi')
            print(grid, end="")
            # sys.stdout.flush()
            time.sleep(1)
            os.system("clear")
        if input("Start a new episode (y/n): ").lower() == 'n':
            break