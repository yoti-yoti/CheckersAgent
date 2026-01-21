from base_agent import BaseAgent
from abc import abstractmethod
from checkers_moves import get_legal_moves_mask

class CheckersAgent(BaseAgent):
    def __init__(self, policy_network=None):
        # Initialize your agent's parameters here
        self.policy_network = policy_network

    def act(self, obs):
        mask = get_legal_moves_mask(obs, player=1)

        pi_moves = self.policy_network.get_moves(obs, mask) # Assuming policy_network has a method get_moves TODO Implement this
        return pi_moves # OR return max action from pi_moves or return sampled action

    def update(self, transition):
        # Implement learning update logic here
        pass

    def save(self, path):
        # Implement model saving logic here
        pass

    def load(self, path):
        # Implement model loading logic here
        pass