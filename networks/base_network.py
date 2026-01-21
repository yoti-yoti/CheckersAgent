import numpy as np
import pandas as pd
from sklearn.naive_bayes import abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# import torchviz
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline


class BaseNetwork(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=256):
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    @abstractmethod
    def get_moves(self, x, mask):
        pass
        # TODO (placeholder): Implement the logic to get moves based on the mask
        # x = self.forward(x)
        # x = x.masked_fill(torch.tensor(mask) == 0, float('-inf'))
        # probs = torch.softmax(x, dim=-1)
        # return probs
    
    # def forward(self, x):
    #     x = x.view(x.size(0), -1)  # Flatten the input
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x