import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# import torchviz 
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline


from networks.registry import register_network

@register_network("checkers_network1")
class CheckersNetwork1(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=256): # 256 moves + 1 value
        super(CheckersNetwork1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        ## must be same for all:
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # must be same for all:
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value