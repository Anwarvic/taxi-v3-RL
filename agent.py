class AgentFactory:
    @staticmethod
    def create_agent(name):
        name = name.lower()
        if name == "q-learning":
            from q_learning import QLearning
            return QLearning
        elif name == "sarsa":
            from sarsa import SARSA
            return SARSA
        elif name == "sarsamax":
            from sarsa import SARSAmax
            return SARSAmax
        elif name == "expected-sarsa":
            from sarsa import ExpectedSARSA
            return ExpectedSARSA
        else:
            raise NameError("Agent name is not available!!")
