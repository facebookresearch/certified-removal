# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_grad_conv import FastGradConv2d


class FastGradExtractor(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, pool_size, normalize=False):
        super(FastGradExtractor, self).__init__()
        self.normalize = normalize
        self.pool_size = pool_size
        conv_layers = []
        assert(len(num_channels) >= 2)
        self.conv_layers = nn.ModuleList([FastGradConv2d(num_channels[i], num_channels[i+1],
            kernel_size, stride) for i in range(len(num_channels)-1)])

    def forward(self, x):
        for _, conv in enumerate(self.conv_layers):
            x = F.max_pool2d(F.relu(conv(x)), self.pool_size, self.pool_size)
        out = x.view(x.size(0), -1)
        if self.normalize:
            out = F.normalize(out)
        return out


# Code for FastGradMLP is adapted from the following repository:
# https://github.com/fKunstner/fast-individual-gradients-with-autodiff/tree/master/pytorch
class FastGradMLP(nn.Module):
    """
    "Standard" MLP with support with goodfellow's backprop trick
    """
    def __init__(self, hidden_sizes):
        super(type(self), self).__init__()

        assert(len(hidden_sizes) >= 2)
        self.input_size = hidden_sizes[0]
        self.act = F.relu

        if len(hidden_sizes) == 2:
            self.hidden_layers = []
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 2)])
        self.output_layer = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])

    def forward(self, x):
        """
        Forward pass that returns also returns
        * the activations (H) and
        * the linear combinations (Z)
        of each layer, to be able to use the trick from [1].

        Args:
        - x : The inputs of the network
        Returns:
        - logits
        - activations at each layer (including the inputs)
        - linear combinations at each layer

        > [1] EFFICIENT PER-EXAMPLE GRADIENT COMPUTATIONS
        > by Ian Goodfellow
        > https://arxiv.org/pdf/1510.01799.pdf
        """
        x = x.view(-1, self.input_size)
        out = x

        # Save the model inputs, which are considered the activations of the 0'th layer.
        activations = [out]
        linearCombs = []

        for layer in self.hidden_layers:
            linearComb = layer(out)
            out = self.act(linearComb)

            # Save the activations and linear combinations from this layer.
            activations.append(out)
            linearComb.requires_grad_(True)
            linearComb.retain_grad()
            linearCombs.append(linearComb)

        logits = self.output_layer(out)

        logits.requires_grad_(True)
        logits.retain_grad()
        linearCombs.append(logits)

        return (logits, activations, linearCombs)
