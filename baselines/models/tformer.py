# Code, general network architecture and network modules are from https://github.com/tobifinn/ensemble_transformer and are used with respect to the following MIT License:

#The MIT License (MIT)
#Copyright (c) 2021, Tobias Finn

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# System modules
import logging
import abc
from typing import Tuple, Union, Dict, Any
from abc import abstractmethod

# External modules
import pytorch_lightning as pl
import torch
import torch.nn
from torch import nn as nn

import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from datetime import datetime

import numpy as np
import wandb


logger = logging.getLogger(__name__)


class transformer(nn.Module):
    # Multilayer perceptron
    def __init__(self, nr_channels, nr_heads, nr_variables, in_channels, args=None):
        super(transformer, self).__init__()

        # pass hyperparameters to model constructor
        self.nr_channels = nr_channels
        self.nr_heads = nr_heads
        self.nr_variables = nr_variables
        self.in_channels = in_channels
    
        self.embedding = Embedding(self.in_channels, self.nr_channels, 5)

        self.transformers = TransformerNet(self.nr_channels[-1], self.nr_heads)
        
        self.output_layer = EnsConv2d(in_channels=nr_channels[-1], out_channels=1, kernel_size=1)

    def forward(self, inp):
        embedded_tensor = self.embedding(inp)
        transformed_tensor = self.transformers(embedded_tensor)
        output_tensor = self.output_layer(transformed_tensor).squeeze(dim=-3)
        return output_tensor

def Tformer_prepare(args):
    if args.target_var in ['t2m']:
        return transformer([8,8,8], 16, 11, 11)
    return transformer([8,8,8], 16, 7, 7)

# Embedding module according to paper
class Embedding(nn.Module):
    def __init__(self, in_channels, nr_channels, kernel_size):
        super(Embedding, self).__init__()
        modules = []
        old_channels = in_channels
        for curr_channel in nr_channels:
            modules.append(EarthPadding((kernel_size-1)//2))
            modules.append(EnsConv2d(old_channels, curr_channel, kernel_size=kernel_size))
            modules.append(nn.SELU(inplace=True))
            old_channels = curr_channel
        self.nr_channels = nr_channels
        self.kernel_size = kernel_size
        self.net = torch.nn.Sequential(
            *modules
        )

    def forward(self, x):
        x = self.net(x)
        return x

class EarthPadding(torch.nn.Module):
    """
    Padding for ESMs with periodic and zero-padding in longitudinal and
    latitudinal direction, respectively.
    """
    def __init__(self, pad_size: int = 1):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lon_left = x[..., -self.pad_size:]
        lon_right = x[..., :self.pad_size]
        lon_padded = torch.cat([lon_left, x, lon_right], dim=-1)
        lat_zeros = torch.zeros_like(lon_padded[..., -self.pad_size:, :])
        lat_padded = torch.cat([lat_zeros, lon_padded, lat_zeros], dim=-2)
        return lat_padded

# transformer module, consitent of possibly multiple attention-modules
class TransformerNet(nn.Module):
    def __init__(self, nr_channels, nr_heads):
        super(TransformerNet, self).__init__()

        value_layer = EnsConv2d(in_channels=nr_channels, out_channels=nr_heads, kernel_size=1, bias=False)
        

        key_layer = branch_layer(nr_channels=nr_channels,nr_heads=nr_heads)
        query_layer = branch_layer(nr_channels=nr_channels,nr_heads=nr_heads)

        layer_norm = torch.nn.LayerNorm([nr_channels, 361, 720])
        value_layer = torch.nn.Sequential(layer_norm, value_layer)
        key_layer = torch.nn.Sequential(layer_norm, key_layer)
        query_layer = torch.nn.Sequential(layer_norm, query_layer)

        out_layer = EnsConv2d(in_channels=nr_heads, out_channels=nr_channels, kernel_size=1, padding=0)

        
        

        transformer_list = []
        for idx in range(1):
            curr_transformer = SelfAttentionModule(value_projector=value_layer, key_projector=key_layer, query_projector=query_layer, output_projector=out_layer, activation=nn.ReLU(inplace=True), weight_estimator=SoftmaxWeights(), reweighter=StateReweighter())
            transformer_list.append(curr_transformer)
        transformers = torch.nn.Sequential(*transformer_list)
        self.transformers = transformers

    def forward(self, x):
        x = self.transformers(x)
        return x

def branch_layer(nr_channels, nr_heads):
    return nn.Sequential(
        EnsConv2d(in_channels=nr_channels, out_channels=nr_heads, kernel_size=1, bias=False),
        nn.ReLU(inplace=True)
    )

# single attention module
class SelfAttentionModule(torch.nn.Module):
    def __init__(
            self,
            value_projector: torch.nn.Module,
            key_projector: torch.nn.Module,
            query_projector: torch.nn.Module,
            output_projector: torch.nn.Module,
            activation: torch.nn.Module,
            weight_estimator,
            reweighter
     ):
        super().__init__()
        self.value_projector = value_projector
        self.key_projector = key_projector
        self.query_projector = query_projector
        self.output_projector = output_projector
        self.activation = activation
        self.weight_estimator = weight_estimator
        self.reweighter = reweighter

    def project_input(
            self,
            in_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_projector(in_tensor)
        key = self.key_projector(in_tensor)
        query = self.query_projector(in_tensor)
        return value, key, query

    def forward(self, in_tensor: torch.Tensor):
        value, key, query = self.project_input(in_tensor)
        weights = self.weight_estimator(key, query)
        transformed_values = self.reweighter(value, weights)
        output_tensor = self.output_projector(transformed_values)
        output_tensor += in_tensor
        activated_tensor = self.activation(output_tensor)
        return activated_tensor


class Reweighter(torch.nn.Module):
    @staticmethod
    def _apply_weights(
            apply_weights_to: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        weighted_tensor = torch.einsum(
            'bcij, bic...->bjc...', weight_tensor, apply_weights_to
        )
        return weighted_tensor

    @abc.abstractmethod
    def forward(
            self,
            state_tensor: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        pass


class StateReweighter(Reweighter):
    def forward(
            self,
            state_tensor: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        _, perts_tensor = split_mean_perts(state_tensor, dim=1)
        weighted_perts = self._apply_weights(perts_tensor, weight_tensor)
        output_tensor = state_tensor + weighted_perts
        return output_tensor

class WeightEstimator(torch.nn.Module):
    @staticmethod
    def _dot_product(
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('bic..., bjc...->bcij', x, y)

    @staticmethod
    def _estimate_norm_factor(estimate_from: torch.Tensor) -> float:
        projection_size = np.prod(estimate_from.shape[3:])
        norm_factor = 1 / np.sqrt(projection_size)
        return norm_factor

    @abc.abstractmethod
    def forward(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        pass


class SoftmaxWeights(WeightEstimator):
    def forward(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        gram_mat = self._dot_product(key, query)
        norm_factor = self._estimate_norm_factor(key)
        gram_mat = gram_mat * norm_factor
        weights = torch.softmax(gram_mat, dim=-2)
        return weights

class EnsConv2d(torch.nn.Module):
    """
    Added viewing for ensemble-based 2d convolutions.
    """
    def __init__(
            self, *args, **kwargs
    ):
        super().__init__()
        self.conv2d = EnsembleWrapper(
            torch.nn.Conv2d(*args, **kwargs)
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        convolved_tensor = self.conv2d(in_tensor)
        return convolved_tensor

class EnsembleWrapper(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_tensor_batched = ens_to_batch(in_tensor)
        modified_tensor = self.base_layer(in_tensor_batched)
        out_tensor = split_batch_ens(modified_tensor, in_tensor)
        return out_tensor

def ens_to_batch(in_tensor: torch.Tensor) -> torch.Tensor:
    try:
        out_tensor = in_tensor.view(-1, *in_tensor.shape[2:])
    except RuntimeError:
        out_tensor = in_tensor.reshape(-1, *in_tensor.shape[2:]).contiguous()
    return out_tensor


def split_batch_ens(
        in_tensor: torch.Tensor,
        like_tensor: torch.Tensor
) -> torch.Tensor:
    try:
        out_tensor = in_tensor.view(
            *like_tensor.shape[:2], *in_tensor.shape[1:]
        )
    except RuntimeError:
        out_tensor = in_tensor.reshape(
            *like_tensor.shape[:2], *in_tensor.shape[1:]
        ).contiguous()
    return out_tensor


def split_mean_perts(
        in_tensor: torch.Tensor,
        dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_tensor = in_tensor.mean(dim=dim, keepdims=True)
    perts_tensor = in_tensor - mean_tensor
    return mean_tensor, perts_tensor

# for debugging
if __name__ == '__main__':
    #loss = crps_loss
    input = torch.randn(1, 5, 105, 361, 720, requires_grad=True)
    mean = torch.randn(1, 1, 361, 720, requires_grad=True)
    stddev = torch.randn(1, 1, 361, 720, requires_grad=True)
    target = torch.randn(1, 1, 361, 720)
    #loss = loss(mean, stddev, target)
    #print(loss)
    model = transformer([8,8,8], 16, 11, 105)
    output = model(input)
    print(output.shape)