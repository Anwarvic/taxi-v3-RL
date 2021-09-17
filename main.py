import gym

from utils import *




if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    # conf = load_conf("conf.yaml")
    # AgentModule = AgentFactory.create_agent("SARSAmax")
    # agent = AgentModule(env.observation_space.n, env.action_space.n, conf)
    # avg_rewards, best_avg_reward = interact(env, agent, conf)
    # render(env, agent)

    
    #compare
    compare(env, ["Q-learning", "SARSA", "SARSAmax", "Expected-SARSA"])
    #NOTE: available algorithms are: [Q-learning, SARSA, SARSAmax, Expected-SARSA]