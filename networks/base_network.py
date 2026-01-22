# import numpy as np
# import pandas as pd
# from sklearn.naive_bayes import abstractmethod
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# # import torchviz
# from sklearn.model_selection import train_test_split 

# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# # %matplotlib notebook
# # %matplotlib inline



# class PolicyValueNet(nn.Module):
#     def __init__(self, backbone: nn.Module, hidden_dim: int, action_dim: int):
#         super().__init__()
#         self.backbone = backbone
        

#     def forward(self, x):
#         return self.backbone(x)

# # class BaseNetwork(nn.Module):
# #     def __init__(self, input_size=64, hidden_size=128, output_size=256): # 256 moves + 1 value
# #         super(BaseNetwork, self).__init__()
# #         self.backbone = nn.Sequential(
# #             nn.Linear(input_size, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 256),
# #             nn.ReLU(),
# #         )
# #         self.policy_head = nn.Linear(256, output_size)
# #         self.value_head = nn.Linear(256, 1)

# #         # self.fc1 = nn.Linear(input_size, hidden_size)
# #         # self.fc2 = nn.Linear(hidden_size, output_size)
# #         # self.relu = nn.ReLU()
        
# #     # @abstractmethod
# #     # def get_moves_value(self, x, mask):
# #     #     pass
# #     #     # TODO (placeholder): Implement the logic to get moves based on the mask and value of state


# #     #     # x = self.forward(x)
# #     #     # x = x.masked_fill(torch.tensor(mask) == 0, float('-inf'))
# #     #     # probs = torch.softmax(x, dim=-1)
# #     #     # return probs
    
# #     def forward(self, x):
# #         h = self.backbone(x)
# #         logits = self.policy_head(h)
# #         value = self.value_head(h).squeeze(-1)
# #         return logits, value