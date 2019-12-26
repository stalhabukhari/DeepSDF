import torch
import torch.nn as nn


class DecompositionLoss(nn.Module):
    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return (torch.relu(pred.sum(dim=0) - self.threshold)).pow(2).mean()


class GuidanceLoss(nn.Module):
    def __init__(self, top_k_points: int):
        super().__init__()
        self.top_k_points = top_k_points

    def forward(
        self,
        transformation_matrices: torch.tensor,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.squeeze()
        true = true.squeeze()

        import ipdb

        ipdb.set_trace()

        sorted_args = torch.argsort(pred, dim=1, descending=True)
        sorted_args = sorted_args[:, : self.top_k_points]

        pred = pred.index_select(dim=-1, index=sorted_args)
        true = true.index_select(dim=-1, index=sorted_args)

        return (pred - true).pow(2).mean(dim=1).mean()


class PenalizingL1(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super().__init__()
        self.reduction = reduction
        self.l_1 = nn.L1Loss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        where_below_one = (pred <= 1).float()
        l1 = self.l_1(pred * where_below_one, true * where_below_one)
        mse = self.mse(
            pred * (1 - where_below_one), true * (1 - where_below_one)
        )

        loss_terms = l_1 + mse
        if self.reduction == "sum":
            return loss_terms.sum()
        return loss_terms.mean()
