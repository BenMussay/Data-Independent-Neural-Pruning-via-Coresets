from typing import Callable, Tuple, Union
import sys

import torch
import numpy as np


class Coreset:
    def __init__(self, points, weights, activation_function: Callable, upper_bound: int = 1):
        assert points.shape[0] == weights.shape[0]

        self.__points = points.cpu()
        self.__weights = weights.cpu()
        self.__activation = activation_function
        self.__beta = upper_bound
        self.__sensitivity = None
        self.indices = None

    @property
    def sensitivity(self):
        if self.__sensitivity is None:
            points_norm = self.__points.norm(dim=1)
            assert points_norm.shape[0] == self.__points.shape[0]
            weights = torch.abs(self.__weights).max(dim=1)[0]  # max returns (values, indices)
            assert weights.shape[0] == self.__points.shape[0]

            self.__sensitivity = weights * torch.abs(self.__activation(self.__beta * points_norm))
            self.__sensitivity /= self.__sensitivity.sum()

        return self.__sensitivity

    def compute_coreset(self, coreset_size):
        assert coreset_size <= self.__points.shape[0]
        prob = self.sensitivity.cpu().numpy()

        indices = set()
        idxs = []

        cnt = 0
        while len(indices) < coreset_size:
            i = np.random.choice(a=self.__points.shape[0], size=1, p=prob).tolist()[0]
            idxs.append(i)
            indices.add(i)
            cnt += 1

        hist = np.histogram(idxs, bins=range(self.__points.shape[0] + 1))[0].flatten()
        idxs = np.nonzero(hist)[0]
        self.indices = idxs
        coreset = self.__points[idxs, :]

        weights = (self.__weights[idxs].t() * torch.tensor(hist[idxs]).float()).t()
        weights = (weights.t() / (torch.tensor(prob[idxs]) * cnt)).t()

        return coreset, weights


def compress_fc_layer(layer1: Tuple[torch.Tensor, torch.Tensor],
                      layer2: Tuple[torch.Tensor, torch.Tensor],
                      compressed_size,
                      activation: Callable,
                      upper_bound,
                      device,
                      compression_type):
    num_neurons = layer1[1].shape[0]
    if compression_type == "Coreset":
        points = np.concatenate(
            (layer1[0].cpu().numpy(), layer1[1].view(num_neurons, 1).cpu().numpy()),
            axis=1)
        points = torch.tensor(points)
        weights = layer2[0].t()
        coreset = Coreset(points=points, weights=weights, activation_function=activation, upper_bound=upper_bound)
        points, weights = coreset.compute_coreset(compressed_size)
        indices = coreset.indices
        layer1 = (points[:, :-1].to(device), points[:, 1].to(device))
        weights = weights.t()
        layer2 = (weights.to(device), layer2[1].to(device))
    elif compression_type == "Uniform":
        indices = np.random.choice(num_neurons, size=compressed_size, replace=False)
        layer1 = (layer1[0][indices, :, :, :], layer1[1][indices])
        layer2 = (layer2[0][:, indices, :, :], layer2[1])
    elif compression_type == "Top-K":
        indices = torch.topk(torch.norm(layer1[0], dim=1), k=compressed_size)[1]
        layer1 = (layer1[0][indices, :], layer1[1][indices])
        layer2 = (layer2[0][:, indices], layer2[1])
    else:
        sys.exit("There is not a compression type: {}".format(compression_type))

    return layer1, layer2, indices

