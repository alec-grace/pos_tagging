# File: main.py
# Author: Alec Grace
# Created: 27 Feb 2024
# Purpose:
#   Create neural network to generate PoS tags for words

# import torch
import torch
import torch.nn as nn
# import torch.nn.functional as F


class FeedNet(nn.Module):
    """Simple Feedforward Network"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        :param input_dim (int): dimension of input
        :param hidden_dim (int): dimension of hidden layer (only one hidden layer for now)
        :param output_dim (int): dimension of output
        """
        super(FeedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.soft = nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out
