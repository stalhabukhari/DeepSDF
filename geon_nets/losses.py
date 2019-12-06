import torch
import torch.nn as nn


class DecompositionLoss(nn.Module):
    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return (torch.relu(pred.sum(dim=0)) - self.threshold).pow(2).mean()
