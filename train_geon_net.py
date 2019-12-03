import typing as t
from pathlib import Path

import torch
import torch.nn as nn


class Trainer:
    def __init__(self):
        self.model: t.Optional[nn.Module] = None

    def save_model(self, directory: str, epoch: int) -> None:
        path = Path(directory)
        torch.save(
            self.model.state_dict(), (path / f"model_{epoch}.pkl").as_posix()
        )
