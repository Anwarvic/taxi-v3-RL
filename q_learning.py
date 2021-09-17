import numpy as np

class QLearning:

    def __init__(self, n_states, n_actions, conf):
        """ Initialize agent.

        Params
            - n_states: number of states available to the agent
            - n_actions: number of actions available to the agent
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.episode = 0
        self.epsilon = conf["epsilon"]
        self.min_epsilon = conf["min_epsilon"]
        self.alpha = conf["alpha"]
        self.gamma = conf["gamma"]

    def _epsilon_greedy(self, state):
        """
        Gets the policy using epsilon-greedy algorithm
        Args:
            - q_state: the state of the agent
        Returns:
            - pi: the policy of the agent in that state using epsilon-greedy algorithm
        """
        pi = np.full((self.n_actions), 1.0*self.epsilon/(self.n_actions)) #equiprobable-random values
        # get maximum action given state
        a_max = np.argmax(self.Q[state, :])
        # update the value of that maximum action
        pi[a_max] += 1.0 - self.epsilon
        return pi


    def select_action(self, state):
        """
        Given the state, select an action using epsilon-greedy policy
        Params
            - state: the current state of the environment
        Returns
            - action: an integer, compatible with the task's action space
        """
        policy = self._epsilon_greedy(state)
        action = np.random.choice(np.arange(self.n_actions), p=policy)
        return action
    

    def _get_next_reward(self, state, action, reward, next_state):
        next_reward = reward + self.gamma*np.max(self.Q[next_state, :])
        return next_reward


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
            - state: the previous state of the environment
            - action: the agent's previous choice of action
            - reward: last reward received
            - next_state: the current state of the environment
            - done: whether the episode is complete (True or False)
        """
        # update state-action function
        next_reward = self._get_next_reward(state, action, reward, next_state)
        self.Q[state, action] = ( ((1-self.alpha)*self.Q[state, action])
                                + (self.alpha*next_reward) )
        if done:
            #update epsilon
            self.episode += 1
            self.epsilon = max(1./self.episode, self.min_epsilon) # can't get below 0.005