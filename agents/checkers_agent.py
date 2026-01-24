from base_agent import BaseAgent
from abc import abstractmethod
from checkers_moves import get_legal_moves_mask, board_after_move
import torch
import os
from networks.registry import get_network_class
import torch.nn.functional as F
# from networks.base_network import PolicyValueNet

class CheckersAgent(BaseAgent):
    def __init__(self, network_name="checkers_network1", device=torch.device("cpu"), checkpoint_id=None):
        # Initialize your agent's parameters here
        # self.policy_network = None
        if checkpoint_id is None:
            self.initialize_network(network_name, device)
        else:
            self.load(base_dir="checkpoints", network_name=network_name, checkpoint_id=checkpoint_id, device=device)
        self.update_data = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "values": [],
            "advantages": [],
            # "returns": []
        }
        self.LAMBDA = 0.95
        self.GAMMA = 0.99
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.0003)


    def act(self, obs): #TODO make modular for which method to choose the move
        #check if forced jumps, checkers env doesn't change to opponent player if there is a forced jump, and the current state will be the same as from the previous
        prev_obs = self.update_data["obs"][-1] if self.update_data["obs"] else None
        last_action = self.update_data["actions"][-1] if self.update_data["actions"] else None
        mask = get_legal_moves_mask(obs, player=1, prev_board=prev_obs, last_action=last_action) # TODO add flag for forced jumps?

        prob_of_moves, value = self.policy_network(obs) # Assuming policy_network is an extension on nn.Module
        # apply mask to prob_of_moves
        prob_of_moves = torch.softmax(prob_of_moves, dim=-1)
        prob_of_moves = prob_of_moves * torch.tensor(mask, dtype=torch.float32)
        prob_of_moves = prob_of_moves / prob_of_moves.sum()  # Re-normalize
        move = torch.argmax(prob_of_moves).item()  # Choose the action with highest probability OR sample from prob_of_moves

        return move, torch.log(prob_of_moves[int(move)]), value # OR return max action from prob_of_moves or return sampled action

    def update(self, transition):
        # Implement learning update logic here
        for data, key in zip(transition, self.update_data.keys()):
            self.update_data[key].append(data)
        pass

    def finish_rollout(self):
        # Implement logic to finalize the rollout #TODO add option for rollout per episode or multiple episodes - must count number of moves in episode?
        rollout = self.update_data
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        advantages = self._calculate_advantages(rewards, values, dones)
        rollout["advantages"].append(advantages)

        self.update_data = { # Clear for next rollout
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "values": [],
            "advantages": [],
            # "returns": []
        }  
        return rollout

    def _calculate_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]  # Append a zero for the last value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.GAMMA * self.LAMBDA * (1 - dones[step]) * gae
            advantages.append(gae)
        advantages.reverse()
        return advantages
    
    def learn_from_rollout(self, rollout, clip_eps=0.2):
        # Convert to tensors if not already
        # TODO TODO TODO TODO
        # TODO rollout is more than one game? what do i do?
        rollout = self.finish_rollout()
        states = torch.tensor(rollout["obs"], dtype=torch.int8) # TODO CHECK TYPE each obs is a board state -> a box of 8x8 TODO What type is box?
        actions = torch.tensor(rollout["actions"])
        log_probs_old = torch.tensor(rollout["log_probs"], dtype=torch.float32)
        returns = torch.tensor(rollout["returns"], dtype=torch.float32)
        advantages = torch.tensor(rollout["advantages"], dtype=torch.float32)
        
        # # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Why +1e-8? because std can be 0
        
        # # Get current policy outputs
        dist, values = self.policy_network(states)  # dist: action distribution, values: critic
        log_probs = dist.log_prob(actions)    
        entropy = dist.entropy().mean()
        
        # # PPO policy loss
        ratio = torch.exp(log_probs - log_probs_old)
        policy_loss = -torch.mean(torch.min(ratio * advantages,
                                            torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages))
        
        # # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()


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
        device=torch.device("cpu"),
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

    def initialize_network(self, network_name: str, device=torch.device("cpu")):
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
