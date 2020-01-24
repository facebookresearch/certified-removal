# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGradConv2dFunction(torch.autograd.Function):
    def __init__(self, batch_size):
        super(FastGradConv2dFunction, self).__init__()
        self.batch_size = batch_size
        self.weight = None
        self.bias = None

    def forward(self, weight_param, bias_param=None):
        self.weight = weight_param.repeat(self.batch_size, 1, 1, 1)
        if bias_param is not None:
            self.bias = bias_param.repeat(self.batch_size)
        return self.weight, self.bias

    def backward(self, weight_grad, bias_grad):
        self.weight.grad = weight_grad
        self.bias.grad = bias_grad
        weight_grad = weight_grad.view(self.batch_size, -1, weight_grad.size(1),
                weight_grad.size(2), weight_grad.size(3)).sum(0)
        if bias_grad is not None:
            bias_grad = bias_grad.view(self.batch_size, -1).sum(0)
        return weight_grad, bias_grad

class FastGradConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, bias=True):
        super(FastGradConv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias)
        self.expand_func = None

    def forward(self, x):
        # delete expanded weight and bias tensors to avoid memory leak
        if self.expand_func is not None:
            del self.expand_func.weight
            del self.expand_func.bias
        batch_size = x.size(0)
        if self.train:
            self.expand_func = FastGradConv2dFunction(batch_size)
            weight, bias = self.expand_func(self.weight, self.bias)
            output = F.conv2d(x.view(1, -1, x.size(2), x.size(3)), weight, bias=bias,
                    stride=self.stride, padding=self.padding, dilation=self.dilation,
                    groups=batch_size)
            return output.view(x.size(0), -1, output.size(2), output.size(3))
        else:
            return F.conv2d(x, self.weight, self.bias, stride=self.stride,
                    padding=self.padding, dilation=self.dilation)
