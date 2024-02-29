# File: my_dataset.py
# Author: Alec Grace
# Created: 28 Feb 2024
# Purpose:
#   Create dataset able to be loaded by Pytorch's Dataloader class

import os
import torch
from torch.utils.data import Dataset
import embeddings


class MyDataset(Dataset):
    """Words and PoS dataset"""

    def __init__(self, tuples: list):
        """
        Argument:
            tuples (list): list of tuples in the form (embedded word, pos tag)
        """
        self.data = tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
