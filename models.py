# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, pool_size, bias=True, normalize=False):
        super(Extractor, self).__init__()
        self.normalize = normalize
        self.pool_size = pool_size
        conv_layers = []
        assert(len(num_channels) >= 2)
        self.conv_layers = nn.ModuleList([nn.Conv2d(num_channels[i], num_channels[i+1],
            kernel_size, stride, bias=bias) for i in range(len(num_channels)-1)])

    def forward(self, x):
        for _, conv in enumerate(self.conv_layers):
            out = conv(x)
            x = F.max_pool2d(F.relu(out), self.pool_size, self.pool_size)
        out = x.view(x.size(0), -1)
        if self.normalize:
            out = F.normalize(out)
        return out


class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super(MLP, self).__init__()
        assert(len(hidden_sizes) >= 2)
        self.input_size = hidden_sizes[0]
        self.act = F.relu
        if len(hidden_sizes) == 2:
            self.hidden_layers = []
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 2)])
        self.output_layer = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        return self.output_layer(x)
