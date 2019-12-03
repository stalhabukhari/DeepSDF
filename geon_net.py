import abc
import typing as t

import numpy as np
import torch
import torch.nn as nn


class DenseNetwork(nn.Module, abc.ABC):
    def __init__(
        self,
        input_features: int,
        hidden_layers_sizes: t.List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_layers_sizes = hidden_layers_sizes
        self.layers = nn.ModuleList()
        self.dropout = dropout

        current_features = input_features
        for size in hidden_layers_sizes:
            self.layers.append(
                nn.utils.weight_norm(
                    nn.Linear(current_features, size, bias=True)
                )
            )
            current_features = size

    def forward(self, x: torch.Tensor):
        pass


class SubNetwork(DenseNetwork):
    def __init__(
        self,
        input_features: int,
        hidden_layers_sizes: t.List[int],
        dropout: float = 0.0,
    ):
        super().__init__(input_features, hidden_layers_sizes, dropout)
        self.layers.append(nn.Linear(hidden_layers_sizes[-1], 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.dropout >= 0.0:
                x = torch.dropout(x, p=self.dropout, train=self.training)
            x = torch.relu(x)
        x = torch.tanh(self.layers[-1](x))
        return x

    def get_required_parameter_count(self) -> int:
        count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                count += np.prod(module.weight.data.shape)
        return count

    def load_params_from_vectors(
        self, weights: torch.Tensor, biases: torch.Tensor
    ):
        current_features = self.input_features
        weights_current_offset = 0
        biases_current_offset = 0
        for i, size in enumerate(self.hidden_layers_sizes):
            weights_slice_start = weights_current_offset
            weights_slice_end = (
                weights_current_offset + current_features * size
            )

            biases_slice_start = biases_current_offset
            biases_slice_end = biases_slice_start + size

            weights_slice = weights[
                weights_slice_start:weights_slice_end
            ].reshape(
                (self.layers[i].out_features, self.layers[i].in_features)
            )
            biases_slice = biases[
                biases_slice_start:biases_slice_end
            ].squeeze()

            self.layers[i].weight.data = weights_slice
            self.layers[i].bias.data = biases_slice

            weights_current_offset = weights_slice_end
            biases_current_offset = biases_slice_end
            current_features = size

    def freeze_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.detach_()
                module.bias.detach_()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    @staticmethod
    def get_required_parameter_count_from_config(
        input_features: int, config: t.List[int]
    ) -> int:
        current_features = input_features
        count = 0
        for size in config:
            count += current_features * size
            current_features = size
        return count

    @staticmethod
    def get_required_biases_count_from_config(config: t.List[int]) -> int:
        return sum(config)

    @staticmethod
    def from_params(
        input_features: int,
        config: t.List[int],
        weights: torch.Tensor,
        biases: torch.Tensor,
        dropout: float = 0.0,
        requires_grad: bool = False,
    ) -> "SubNetwork":
        net = SubNetwork(input_features, config, dropout=dropout)
        net.load_params_from_vectors(weights, biases)
        if not requires_grad:
            net.freeze_layers()
        return net


class Encoder(DenseNetwork):
    def __init__(
        self,
        input_features: int,
        hidden_layers_sizes: t.List[int],
        latent_size: int,
        dropout: float = 0.0,
    ):
        super().__init__(input_features, hidden_layers_sizes, dropout)
        self.latent_size = latent_size

        self.mu_encoder = nn.Linear(
            self.hidden_layers_sizes[-1], self.latent_size
        )
        self.sigma_encoder = nn.Linear(
            self.hidden_layers_sizes[-1], self.latent_size
        )

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
            if self.dropout >= 0.0:
                x = torch.dropout(x, p=self.dropout, train=self.training)
            x = torch.relu(x)

        return self.mu_encoder(x), self.sigma_encoder(x)


class Decoder(DenseNetwork):
    def __init__(
        self,
        input_features: int,
        hidden_layers_sizes: t.List[int],
        number_of_geons: int,
        subnetwork_config: t.List[int],
        dropout: float = 0.0,
    ):
        super().__init__(input_features, hidden_layers_sizes, dropout)
        self.number_of_geons = number_of_geons
        self.subnetwork_config = subnetwork_config
        self.weights_to_predict = SubNetwork.get_required_parameter_count_from_config(
            self.hidden_layers_sizes[-1], subnetwork_config
        )
        self.biases_to_predict = SubNetwork.get_required_biases_count_from_config(
            subnetwork_config
        )

        self.param_prediction_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_layers_sizes[-1],
                    self.weights_to_predict + self.biases_to_predict,
                )
                for _ in range(number_of_geons)
            ]
        )

        self.geon_nets = nn.ModuleList(
            [
                SubNetwork(
                    input_features + 3, subnetwork_config, dropout=dropout,
                )
                for _ in range(number_of_geons)
            ]
        )
        for geon in self.geon_nets:
            geon.freeze_layers()

    @classmethod
    def reparametrize(
        cls, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        std = (sigma * 0.5).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_samples(
        self, latent_codes: torch.Tensor, point_coordinates: torch.Tensor
    ) -> torch.Tensor:
        assert len(point_coordinates.shape) == 2

        x = latent_codes
        for layer in self.layers:
            x = layer(x)
            if self.dropout >= 0.0:
                x = torch.dropout(x, p=self.dropout, train=self.training)
            x = torch.relu(x)

        latent_codes = latent_codes.repeat((len(point_coordinates), 1))
        points_features = torch.cat((point_coordinates, latent_codes), axis=-1)

        geons_outputs = []
        for i, param_prediction_layer in enumerate(
            self.param_prediction_layers
        ):
            params = param_prediction_layer(x)
            weights, biases = (
                params[0, : self.weights_to_predict],
                params[0, self.weights_to_predict :],
            )
            geon: SubNetwork = self.geon_nets[i]
            geon.load_params_from_vectors(weights, biases)
            geons_outputs.append(geon(points_features))
        geons = torch.cat(geons_outputs, dim=0)
        return geons

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        point_coordinates: torch.Tensor,
    ) -> t.List[torch.Tensor]:
        assert mu.shape[0] == 1
        x = self.reparametrize(mu, sigma)
        return self.generate_samples(x, point_coordinates)


def test():
    decoder = Decoder(256, [512, 1024], 5, [64, 32, 1])
    mu = torch.normal(0, 0.001, size=(1, 256)).float()
    sigma = torch.full((1, 256), fill_value=1).float()
    points_coordinates = torch.FloatTensor(300, 3)
    points_coordinates.uniform_(-1, 1)
    results = decoder(mu, sigma, points_coordinates)

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    test()
