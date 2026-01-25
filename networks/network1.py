import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# # import torchviz 
# from sklearn.model_selection import train_test_split 

# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline


from networks.registry import register_network

@register_network("checkers_network1")
class CheckersNetwork1(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.backbone(x)              # ← THIS WAS MISSING
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value
