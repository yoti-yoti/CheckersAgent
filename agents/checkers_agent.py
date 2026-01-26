import os
import numpy as np
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from checkers_moves import get_legal_moves_mask
from networks.registry import get_network_class


def _board_to_class_idx(board_np: np.ndarray) -> np.ndarray:
    m = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    v = np.vectorize(lambda x: m[int(x)])
    return v(board_np).astype(np.int64)


class CheckersAgent(BaseAgent):
    def __init__(
        self,
        network_name="checkers_network1",
        device=torch.device("cpu"),
        checkpoint_id=None,
        player="agent",
        eps=0.0,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        use_self_supervised=False,
        aux_coef=0.1,
        temperature=1.0,
    ):
        self.device = device
        self.eps = float(eps)
        self.GAMMA = float(gamma)
        self.LAMBDA = float(lam)
        self.use_self_supervised = bool(use_self_supervised)
        self.aux_coef = float(aux_coef)
        self.temperature = float(temperature)

        if player not in ["agent", "opponent"]:
            raise ValueError("player must be 'agent' or 'opponent'")
        self.player = 1 if player == "agent" else -1

        if checkpoint_id is None:
            self.initialize_network(network_name, device)
        else:
            self.load(
                base_dir="checkpoints",
                network_name=network_name,
                checkpoint_id=checkpoint_id,
                device=device,
            )

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

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

        obs_t = torch.from_numpy(np.asarray(obs)).to(self.device)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(0)
        obs_cls = torch.from_numpy(_board_to_class_idx(obs_t.squeeze(0).detach().cpu().numpy())).to(self.device)
        obs_oh = F.one_hot(obs_cls, num_classes=5).float().permute(2, 0, 1).unsqueeze(0)

        out = self.policy_network(obs_oh)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, value = out[0], out[1]
        else:
            raise ValueError("policy_network must return (logits, value) or (logits, value, aux)")

        logits = logits.squeeze(0)
        value = value.squeeze()

        mask_t = torch.from_numpy(mask_np).to(self.device).float()
        masked_logits = logits.masked_fill(mask_t == 0, -1e9)
        if self.temperature != 1.0:
            masked_logits = masked_logits / max(self.temperature, 1e-6)

        probs = torch.softmax(masked_logits, dim=-1)

        if torch.isnan(probs).any() or probs.sum().item() <= 0:
            move = int(np.random.choice(legal_idx))
            log_prob = torch.tensor(0.0, device=self.device)
            return move, log_prob, value

        if np.random.random() < self.eps:
            move = int(np.random.choice(legal_idx))
            log_prob = torch.log(probs[move].clamp_min(1e-12))
            return move, log_prob, value

        dist = torch.distributions.Categorical(probs)
        action_t = dist.sample()
        move = int(action_t.item())
        log_prob = dist.log_prob(action_t)
        return move, log_prob, value

    def update(self, transition: dict):
        for k in self.update_data.keys():
            if k in transition:
                self.update_data[k].append(transition[k])

    def finish_rollout(self):
        rollout = {k: list(v) for k, v in self.update_data.items()}

        rewards = [float(r) for r in rollout["rewards"]]
        dones = [float(d) for d in rollout["dones"]]
        values = [
            float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
            for v in rollout["values"]
        ]

        advantages, returns = self._calc_adv_and_returns(rewards, values, dones)
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

    def _calc_adv_and_returns(self, rewards, values, dones):
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

    def learn_from_rollout(
        self,
        rollout,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        torch.autograd.set_detect_anomaly(True)
        obs_np = np.asarray(rollout["obs"])
        T = obs_np.shape[0]

        obs_cls = np.stack([_board_to_class_idx(obs_np[i]) for i in range(T)], axis=0)
        states = torch.from_numpy(obs_cls).to(self.device)
        states_oh = F.one_hot(states, num_classes=5).float().permute(0, 3, 1, 2).contiguous()

        actions = torch.tensor(rollout["actions"], dtype=torch.int64, device=self.device)

        log_probs_old = torch.stack([
            lp.detach().to(self.device) if torch.is_tensor(lp) else torch.tensor(lp, device=self.device)
            for lp in rollout["log_probs"]
        ]).detach()

        returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout["advantages"], dtype=torch.float32, device=self.device)

        masks_np = np.asarray(rollout["masks"], dtype=np.int8)
        masks = torch.from_numpy(masks_np).to(self.device).float()
        states = states.contiguous()
        masks = masks.contiguous()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.policy_network.train()

        out = self.policy_network(states_oh, actions=actions if self.use_self_supervised else None)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            
            logits, values = out[0], out[1]
            aux_logits = out[2] if (self.use_self_supervised and len(out) >= 3) else None
        else:
            raise ValueError("policy_network must return (logits, value) or (logits, value, aux_logits)")

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

        rl_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        loss = rl_loss

        aux_loss_val = 0.0
        if self.use_self_supervised:
            next_obs_np = np.asarray(rollout["next_obs"])
            next_cls = np.stack([_board_to_class_idx(next_obs_np[i]) for i in range(T)], axis=0)
            target_next = torch.from_numpy(next_cls).to(self.device)

            if aux_logits is None:
                raise ValueError("use_self_supervised=True but network did not return aux_logits")

            aux_loss = F.cross_entropy(
                aux_logits.reshape(T * 64, 5),
                target_next.reshape(T * 64),
            )
            loss = loss + self.aux_coef * aux_loss
            aux_loss_val = float(aux_loss.detach().cpu().item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "rl_loss": float(rl_loss.detach().cpu().item()),
            "aux_loss": aux_loss_val,
        }

    def _next_checkpoint_id(self, save_dir: str) -> int:
        existing = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not existing:
            return 0
        ids = [int(f.replace("checkpoint_", "").replace(".pt", "")) for f in existing]
        return max(ids) + 1

    def save(self, base_dir: str, network_name: str):
        save_dir = os.path.join(base_dir, network_name)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_id = self._next_checkpoint_id(save_dir)
        save_path = os.path.join(save_dir, f"checkpoint_{ckpt_id:03d}.pt")
        torch.save({"network_name": network_name, "state_dict": self.policy_network.state_dict()}, save_path)
        print(f"Saved checkpoint {ckpt_id} to {save_path}")

    def load(self, base_dir: str, network_name: str, checkpoint_id: int | None = None, device=torch.device("cpu")):
        load_dir = os.path.join(base_dir, network_name)
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"No directory {load_dir}")

        if checkpoint_id is None:
            checkpoints = sorted(f for f in os.listdir(load_dir) if f.startswith("checkpoint_"))
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
