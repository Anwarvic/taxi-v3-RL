from q_learning import QLearning


class SARSA(QLearning):
    def _get_next_reward(self, state, action, reward, next_state):
        next_action = self.select_action(state)
        next_reward = (reward + self.gamma*self.Q[next_state, next_action])
        return next_reward


class ExpectedSARSA(QLearning):
    def _get_next_reward(self, state, action, reward, next_state):
        policy = self._epsilon_greedy(next_state)
        next_reward = reward + self.gamma*sum(policy * self.Q[next_state, :])
        return next_reward
    