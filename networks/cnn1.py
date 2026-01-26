import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# # import torchviz 
# from sklearn.model_selection import train_test_split 

# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline


from networks.registry import register_network

@register_network("cnn1")
class CheckersNetwork1(nn.Module):
    def __init__(self, hidden_size=128, output_size=256):
        super().__init__()

        # --- CNN backbone ---
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # 8x8 → 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 8x8 → 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # --- Shared FC trunk ---
        self.fc = nn.Linear(64 * 8 * 8, hidden_size)

        # --- Heads ---
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        """
        x: (batch, 1, 8, 8)
        returns:
          logits: (batch, 256)
          value:  (batch,)
        """
        x = self.backbone(x)                  # (B, 64, 8, 8)
        x = x.flatten(start_dim=1)            # (B, 4096)
        x = F.relu(self.fc(x))                # (B, hidden_size)

        logits = self.policy_head(x)           # (B, 256)
        value = self.value_head(x).squeeze(-1) # (B,)

        return logits, value
