import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.registry import register_network

@register_network("checkers_network1")
class CheckersPolicyValueAuxNet(nn.Module):
    def __init__(self, action_dim=256, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)

        self.action_emb = nn.Embedding(action_dim, hidden)
        self.aux = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64 * 5),
        )

    def forward(self, x, actions=None):
        z = self.enc(x)
        h = self.fc(z)
        logits = self.policy(h)
        value = self.value(h)

        if actions is None:
            return logits, value

        a = self.action_emb(actions)
        ha = torch.cat([h, a], dim=-1)
        aux_logits = self.aux(ha).reshape(-1, 64, 5)
        return logits, value, aux_logits
