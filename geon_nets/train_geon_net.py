import argparse
import json
import logging
import multiprocessing as mp
import os
import typing as t
from collections import OrderedDict, defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset

from common import cfg, get_logger
from geon_nets.data import ImageToSDFDataset
from geon_nets.losses import DecompositionLoss
from networks.geon_net import Decoder, Encoder


class Trainer:
    def __init__(self, use_cuda: bool = torch.cuda.is_available()):
        self.encoder: t.Optional[Encoder] = None
        self.decoder: t.Optional[Decoder] = None

        self.l1_loss: t.Optional[nn.L1Loss] = None
        self.decomposition_loss: t.Optional[DecompositionLoss] = None

        self.optimizer: t.Optional[optim.Optimizer] = None
        self.history = defaultdict(list)

        self.cuda = use_cuda and torch.cuda.is_available()
        self.logger = get_logger("Trainer")

    def initialize_model(
        self,
        image_size: int,
        encoder_hidden_sizes: t.List[int],
        decoder_hidden_sizes: t.List[int],
        latent_size: int,
        number_of_geons: int,
        subnet_config: t.List[int],
        optimizer: t.Callable[[t.Iterable[nn.Parameter]], optim.Optimizer],
        dropout: float = 0.0,
    ):
        self.encoder = Encoder(
            image_size, encoder_hidden_sizes, latent_size, dropout
        )
        self.decoder = Decoder(
            latent_size,
            decoder_hidden_sizes,
            number_of_geons,
            subnet_config,
            dropout,
        )

        self.optimizer = optimizer(
            chain(self.encoder.parameters(), self.decoder.parameters())
        )

        self.l1_loss = torch.nn.L1Loss(reduction="sum")
        self.decomposition_loss = DecompositionLoss()

        if self.cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.l1_loss.cuda()
            self.decomposition_loss.cuda()

    def _save_file_content(
        self, file_template: str, content: t.Any, epoch: t.Optional[int] = None
    ) -> None:
        if epoch is not None:
            torch.save(content, file_template + f"_{epoch}.pkl")
        else:
            torch.save(content, file_template + ".pkl")

    def save_model(
        self, directory: str, epoch: t.Optional[int] = None
    ) -> None:
        self._ensure_dir(directory)
        self._save_file_content(
            os.path.join(directory, cfg.CONST.ENCODER_FILE_BASE_NAME),
            self.encoder.state_dict(),
            epoch,
        )
        self._ensure_dir(directory)
        self._save_file_content(
            os.path.join(directory, cfg.CONST.DECODER_FILE_BASE_NAME),
            self.decoder.state_dict(),
            epoch,
        )

    def save_optimizer(
        self, directory: str, epoch: t.Optional[int] = None
    ) -> None:
        self._ensure_dir(directory)
        self._save_file_content(
            os.path.join(directory, cfg.CONST.OPTIMIZER_FILE_BASE_NAME),
            self.optimizer.state_dict(),
            epoch,
        )

    def save_history(
        self, directory: str, epoch: t.Optional[int] = None
    ) -> None:
        self._ensure_dir(directory)
        self._save_file_content(
            os.path.join(directory, cfg.CONST.HISTORY_FILE_BASE_NAME),
            self.history,
            epoch,
        )

    def _ensure_dir(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_model(self, encoder_file: str, decoder_file: str) -> None:
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))

    def load_optimizer(self, a_file: str) -> None:
        self.optimizer.load_state_dict(torch.load(a_file))

    def load_history(self, a_file: str) -> None:
        self.history = torch.load(a_file)

    def fit(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        batch_size: int,
        epochs: int,
        save_dir: str,
        grad_clip: bool = True,
        starting_epoch: int = -1,
    ) -> "Trainer":
        self.logger.info("Saving models to %s", save_dir)

        save_dir = Path(save_dir)

        if starting_epoch < 0:
            save_dir.mkdir(parents=True, exist_ok=False)
        else:
            assert save_dir.exists()

        train_loader = DataLoader(
            train_dataset,
            num_workers=mp.cpu_count(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.history = defaultdict(list)

        if starting_epoch > 0:
            self.load_model(
                (
                    save_dir
                    / f"{cfg.CONST.ENCODER_FILE_BASE_NAME}_{starting_epoch}.pkl"
                ).as_posix(),
                (
                    save_dir
                    / f"{cfg.CONST.DECODER_FILE_BASE_NAME}_{starting_epoch}.pkl"
                ).as_posix(),
            )
            self.load_optimizer(
                (
                    save_dir
                    / f"{cfg.CONST.OPTIMIZER_FILE_BASE_NAME}_{starting_epoch}.pkl"
                ).as_posix()
            )
            self.load_history(
                (
                    save_dir
                    / f"{cfg.CONST.HISTORY_FILE_BASE_NAME}_{starting_epoch}.pkl"
                ).as_posix()
            )
            current_epoch = starting_epoch + 1
        else:
            current_epoch = 0

        for _ in range(epochs):
            self.logger.info("Epoch %d / %d", current_epoch + 1, epochs)

            batch_history = defaultdict(list)
            pbar = tqdm.tqdm(total=len(train_loader))
            self.train()
            for images, points_coordinates, distances, _ in train_loader:
                self.optimizer.zero_grad()

                images, points_coordinates, distances = self.to_cuda(
                    images, points_coordinates, distances
                )

                latent_means, latent_stds = self.encoder(images)
                pred_sdf = self.decoder(
                    latent_means, latent_stds, points_coordinates
                )

                l1_value = self.l1_loss(pred_sdf.max(dim=0), distances)
                decomp_value = self.decomposition_loss(pred_sdf)

                total_loss = l1_value + decomp_value
                total_loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_norm_(
                        chain(
                            self.encoder.parameters(),
                            self.decoder.parameters(),
                        ),
                        int(grad_clip),
                    )

                self.optimizer.step()

                batch_history["l1"].append(l1_value.item())
                batch_history["decomposition"].append(decomp_value.item())
                batch_history["total"].append(total_loss.item())

                pbar.set_postfix(
                    OrderedDict(
                        {
                            key: "%.4f" % np.mean(val)
                            for key, val in batch_history.items()
                        }
                    )
                )
                pbar.update(1)
            pbar.close()

            for key, val in batch_history.items():
                self.history[key].append(np.mean(val))

            val_results = self.score(valid_dataset, batch_size)
            self.logger.info(val_results)
            for key, val in val_results.items():
                self.history[key].append(val)
            self.history["epoch"].append(current_epoch)

            self.save_model(save_dir, current_epoch)
            self.save_optimizer(save_dir, current_epoch)
            self.save_history(save_dir, current_epoch)

            current_epoch += 1

    def score(self, dataset: Dataset, batch_size: int) -> t.Dict[str, float]:
        loader = DataLoader(
            dataset,
            num_workers=mp.cpu_count(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        history = defaultdict(list)
        self.eval()

        for images, points_coordinates, distances, _ in loader:
            images, points_coordinates, distances = self.to_cuda(
                images, points_coordinates, distances
            )

            latent_means, latent_stds = self.encoder(images)
            pred_sdf = self.decoder(
                latent_means, latent_stds, points_coordinates
            )

            l1_value = self.l1_loss(pred_sdf.max(dim=0), distances)
            decomp_value = self.decomposition_loss(pred_sdf)

            total_loss = l1_value + decomp_value
            total_loss.backward()

            history["val_l1"].append(l1_value.item())
            history["val_decomposition"].append(decomp_value.item())
            history["val_total"].append(total_loss.item())
        return {key: np.mean(val) for key, val in history}

    def to_cuda(
        self, *tensors: t.Sequence[torch.Tensor]
    ) -> t.Tuple[torch.Tensor, ...]:
        if not self.cuda:
            return tensors
        return [tensor.cuda() for tensor in tensors]

    def train(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()

    def eval(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()


def train(config_spec_path: str) -> None:
    spec = json.loads(Path(config_spec_path).read_text())
    trainer = Trainer(torch.cuda.is_available())

    if spec["optimizer"] == "adam":
        optimizer = lambda params: optim.Adam(params, lr=0.0005)
    else:
        optimizer = lambda params: optim.SGD(
            params, lr=0.05, momentum=0.9, nesterov=True
        )

    trainer.initialize_model(
        cfg.CONST.IMG_H,
        spec["encoder_hidden_sizes"],
        spec["decoder_hidden_sizes"],
        spec["latent_size"],
        spec["number_of_geons"],
        spec["subnet_sizes"],
        optimizer,
    )

    train_dataset = ImageToSDFDataset(
        spec["train_split_file"],
        cfg.CLASS_TO_INDEX_MAPPING,
        spec["subsample"],
        verbose=True,
        is_test=False,
    )

    valid_dataset = ImageToSDFDataset(
        spec["valid_split_file"],
        cfg.CLASS_TO_INDEX_MAPPING,
        spec["subsample"],
        verbose=True,
        is_test=True,
    )

    trainer.fit(
        train_dataset,
        valid_dataset,
        spec["batch_size"],
        spec["epochs"],
        spec["save_dir"],
        spec["grad_clip"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_spec", type=str, help="Path to the experiment specification"
    )

    args = parser.parse_args()

    train(args.config_spec)


if __name__ == "__main__":
    main()
