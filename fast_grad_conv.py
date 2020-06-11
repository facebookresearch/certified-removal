# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGradConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_param, bias_param, batch_size=1):
        ctx.save_for_backward(weight_param)
        weight = weight_param.repeat(batch_size, 1, 1, 1)
        if bias_param is not None:
            bias = bias_param.repeat(batch_size)
        return weight, bias

    @staticmethod
    def backward(ctx, weight_grad, bias_grad):
        weight_param, = ctx.saved_tensors
        batch_size = int(weight_grad.size(0) / weight_param.size(0))
        weight_grad = weight_grad.view(batch_size, -1, weight_grad.size(1),
                weight_grad.size(2), weight_grad.size(3)).sum(0)
        if bias_grad is not None:
            bias_grad = bias_grad.view(batch_size, -1).sum(0) 
        return weight_grad, bias_grad, None

class FastGradConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, bias=True):
        super(FastGradConv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias)
        self.expanded_weight = None
        self.expanded_bias = None
    
    def forward(self, x):
        if self.train:
            del self.expanded_weight
            del self.expanded_bias
            batch_size = x.size(0)
            self.expanded_weight, self.expanded_bias = FastGradConv2dFunction.apply(self.weight, self.bias, batch_size)
            self.expanded_weight.requires_grad_(True)
            self.expanded_weight.retain_grad()
            if self.expanded_bias is not None:
                self.expanded_bias.requires_grad_(True)
                self.expanded_bias.retain_grad()
            output = F.conv2d(x.view(1, -1, x.size(2), x.size(3)), self.expanded_weight, bias=self.expanded_bias,
                    stride=self.stride, padding=self.padding, dilation=self.dilation,
                    groups=batch_size)
            return output.view(x.size(0), -1, output.size(2), output.size(3))
        else:
            return F.conv2d(x, self.weight, self.bias, stride=self.stride,
                    padding=self.padding, dilation=self.dilation)
