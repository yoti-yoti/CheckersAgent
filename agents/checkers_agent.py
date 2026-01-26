from agents.base_agent import BaseAgent
from checkers_moves import get_legal_moves_mask
import torch
import os
from networks.registry import get_network_class
import torch.nn.functional as F
import numpy as np

class CheckersAgent(BaseAgent):
    def __init__(self, network_name="checkers_network1", device=torch.device("cpu"), checkpoint_id=None, player="agent", eps=0.0):
        self.device = device
        self.eps = eps

        if player not in ["agent", "opponent"]:
            raise ValueError("player must be 'agent' or 'opponent'")
        self.player = 1 if player == "agent" else -1

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
            "masks": [],
        }

        self.LAMBDA = 0.95
        self.GAMMA = 0.99
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=3e-4)

    def act(self, obs, mask=None):
        if mask is None:
            prev_obs = self.update_data["obs"][-1] if self.update_data["obs"] else None
            last_action = self.update_data["actions"][-1] if self.update_data["actions"] else None
            mask = get_legal_moves_mask(obs, player=self.player, prev_board=prev_obs, last_action=last_action)

        mask_np = np.asarray(mask, dtype=np.int8)
        legal_idx = np.flatnonzero(mask_np == 1)
        if legal_idx.size == 0:
            move = int(np.random.randint(0, 256))
            return move, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        tensor_obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        if tensor_obs.ndim == 2:
            tensor_obs = tensor_obs.unsqueeze(0)
        if tensor_obs.ndim == 3:
            tensor_obs = tensor_obs.unsqueeze(1)

        logits, value = self.policy_network(tensor_obs)
        logits = logits.squeeze(0)

        mask_t = torch.from_numpy(mask_np).to(self.device).float()
        masked_logits = logits.masked_fill(mask_t == 0, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        if torch.isnan(probs).any() or probs.sum().item() <= 0:
            move = int(np.random.choice(legal_idx))
            log_prob = torch.tensor(0.0, device=self.device)
            v = value.squeeze()
            return move, log_prob, v

        if np.random.random() < self.eps:
            move = int(np.random.choice(legal_idx))
        else:
            move = int(torch.argmax(probs, dim=-1).item())

        log_prob = torch.log(probs[move].clamp_min(1e-12))
        v = value.squeeze()
        return move, log_prob, v

    def update(self, transition: dict):
        for k in self.update_data.keys():
            if k in transition:
                self.update_data[k].append(transition[k])

    def finish_rollout(self):
        rollout = {k: list(v) for k, v in self.update_data.items()}

        rewards = [float(r) for r in rollout["rewards"]]
        dones = [float(d) for d in rollout["dones"]]
        values = [float(v) if not torch.is_tensor(v) else float(v.detach().cpu().item()) for v in rollout["values"]]

        advantages, returns = self._calculate_advantages_and_returns(rewards, values, dones)
        rollout["advantages"] = advantages
        rollout["returns"] = returns

        self.update_data = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "values": [],
            "masks": [],
        }

        return rollout

    def _calculate_advantages_and_returns(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        values_ext = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.GAMMA * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
            gae = delta + self.GAMMA * self.LAMBDA * (1.0 - dones[t]) * gae
            advantages.append(gae)
        advantages.reverse()
        returns = [advantages[i] + values[i] for i in range(len(values))]
        return advantages, returns

    def learn_from_rollout(self, rollout, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        obs_np = np.asarray(rollout["obs"], dtype=np.float32)
        states = torch.from_numpy(obs_np).to(self.device)
        if states.ndim == 3:
            states = states.unsqueeze(1)

        actions = torch.tensor(rollout["actions"], dtype=torch.int64, device=self.device)

        log_probs_old = torch.stack([
            lp if torch.is_tensor(lp) else torch.tensor(lp, device=self.device)
            for lp in rollout["log_probs"]
        ]).to(self.device).detach()

        returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout["advantages"], dtype=torch.float32, device=self.device)

        masks_np = np.asarray(rollout["masks"], dtype=np.int8)
        masks = torch.from_numpy(masks_np).to(self.device).float()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.policy_network.train()

        logits, values = self.policy_network(states)
        masked_logits = logits.masked_fill(masks == 0, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - log_probs_old)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_grad_norm)
        self.optimizer.step()

        return float(loss.item())

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
        save_dir = os.path.join(base_dir, network_name)
        os.makedirs(save_dir, exist_ok=True)

        ckpt_id = self._next_checkpoint_id(save_dir)
        save_path = os.path.join(save_dir, f"checkpoint_{ckpt_id:03d}.pt")

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
        checkpoint_id: int | None = None,
        device=torch.device("cpu"),
    ):
        load_dir = os.path.join(base_dir, network_name)

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"No directory {load_dir}")

        if checkpoint_id is None:
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
        network = net_cls()

        network.load_state_dict(checkpoint["state_dict"])
        network.to(device)
        network.train()

        self.policy_network = network
        print(f"Loaded {network_name} from {load_path}")

    def initialize_network(self, network_name: str, device=torch.device("cpu")):
        net_cls = get_network_class(network_name)
        network = net_cls()
        network.to(device)
        network.train()
        self.policy_network = network
