import gym

from utils import *




if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    conf = load_conf("conf.yaml")
    AgentModule = AgentFactory.create_agent("Expected-SARSA")
    agent = AgentModule(env.observation_space.n, env.action_space.n, conf)
    avg_rewards, best_avg_reward = train(env, agent, conf)
    interact(env, agent)

    #compare
    # compare(env, ["Q-learning", "SARSA", "Expected-SARSA"])
    #NOTE: available algorithms are: ["Q-learning", "SARSA", "Expected-SARSA"]