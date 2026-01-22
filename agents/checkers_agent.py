from base_agent import BaseAgent
from abc import abstractmethod
from checkers_moves import get_legal_moves_mask
import torch
import os
from networks.registry import get_network_class
# from networks.base_network import PolicyValueNet

class CheckersAgent(BaseAgent):
    def __init__(self, network_name="checkers_network1", device="cpu", checkpoint_id=0):
        # Initialize your agent's parameters here
        self.policy_network = None
        if checkpoint_id==0:
            self.initialize_network(network_name, device)
        else:
            self.load(base_dir="checkpoints", network_name=network_name, checkpoint_id=checkpoint_id, device=device)
        self.update_data = []
        self.LAMBDA = 0.95
        self.GAMMA = 0.99


    def act(self, obs): #TODO make modular for which method to choose the move
        mask = get_legal_moves_mask(obs, player=1)

        pi_moves, value = self.policy_network(obs, mask) # Assuming policy_network is an extension on nn.Module TODO Implement this
        move = torch.argmax(pi_moves).item()  # Choose the action with highest probability OR sample from pi_moves
        return move, pi_moves[move], value # OR return max action from pi_moves or return sampled action

    def update(self, transition):
        # Implement learning update logic here
        self.update_data.append(transition)
        pass

    def finish_rollout(self):
        # Implement logic to finalize the rollout
        rollout = self.update_data
        rewards = [roll[3] for roll in rollout]
        values = [roll[6] for roll in rollout]
        dones = [roll[5] for roll in rollout]
        advantages = self.calculate_advantages(rewards, values, dones)
        for i in range(len(rollout)):
            rollout[i] = rollout[i] + (advantages[i],)            
        self.update_data = []  # Clear for next rollout
        return rollout
    
    def calculate_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]  # Append a zero for the last value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.GAMMA * self.LAMBDA * (1 - dones[step]) * gae
            advantages.append(gae)
        advantages.reverse()
        return advantages
    
    def learn_from_rollout(self, rollout):
        # Implement learning from the collected rollout
        # TODO
        pass

    def _next_checkpoint_id(self, save_dir: str) -> int:
        existing = [
            f for f in os.listdir(save_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        if not existing:
            return 0

        ids = [
            int(f.replace("checkpoint_", "").replace(".pt", ""))
            for f in existing
        ]
        return max(ids) + 1

    def save(self, base_dir: str, network_name: str):
        """
        Saves a new checkpoint without overwriting previous ones.
        """
        save_dir = os.path.join(base_dir, network_name)
        os.makedirs(save_dir, exist_ok=True)

        ckpt_id = self._next_checkpoint_id(save_dir)
        save_path = os.path.join(
            save_dir, f"checkpoint_{ckpt_id:03d}.pt"
        )

        torch.save(
            {
                "network_name": network_name,
                "state_dict": self.policy_network.state_dict(),
            },
            save_path,
        )

        print(f"Saved checkpoint {ckpt_id} to {save_path}")   


    def load(
        self,
        base_dir: str,
        network_name: str,
        # obs_dim: int,
        # action_dim: int,
        checkpoint_id: int | None = None,
        device="cpu",
    ):
        """
        Loads a policy network by name.
        """
        load_dir = os.path.join(base_dir, network_name)

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"No directory {load_dir}")

        if checkpoint_id is None:
            # load latest
            checkpoints = sorted(
                f for f in os.listdir(load_dir)
                if f.startswith("checkpoint_")
            )
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            ckpt_file = checkpoints[-1]
        else:
            ckpt_file = f"checkpoint_{checkpoint_id:03d}.pt"

        load_path = os.path.join(load_dir, ckpt_file)

        checkpoint = torch.load(load_path, map_location=device)

        net_cls = get_network_class(network_name)
        network = net_cls()  # obs_dim, action_dim)

        network.load_state_dict(checkpoint["state_dict"])
        network.to(device)
        network.eval()

        self.policy_network = network
        

        print(f"Loaded {network_name} from {load_path}")

    def initialize_network(self, network_name: str, device="cpu"):
        """
        Initializes a new policy network.
        
        :param network_name: type of network to initialize
        :type network_name: str
        :param device: device to load the network onto
        """
        # TODO choose how to initialize
        net_cls = get_network_class(network_name)
        network = net_cls()  # obs_dim, action_dim)

        network.to(device)
        network.eval()

        self.policy_network = network
