import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ModelTrainer:
    @staticmethod
    def train_TR(
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            **kwargs
        ):
        """
        """

    @staticmethod
    def evaluate_TR(
            model:nn.Module,
            data_loader:DataLoader,
            **kwargs
        ):
        """
        """