# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
import math
from utils import per_example_gradient, clip_and_sum_gradients, add_noisy_gradient, batch_grads_to_vec, params_to_vec, vec_to_params, compute_full_grad, loss_with_reg


# trains a regular model for a single epoch
def train(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, verbose=True):
    if extr is not None:
        extr.train()
    clf.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if extr is not None:
            output = clf(extr(data))
            if len(output) == 3:
                output = output[0]
        else:
            output = clf(data)
        loss = loss_fn(output, target)
        if args.lam > 0:
            if extr is not None:
                loss += args.lam * params_to_vec(extr.parameters()).pow(2).sum() / 2
            loss += args.lam * params_to_vec(clf.parameters()).pow(2).sum() / 2
        loss.backward()
        optimizer.step()
        if verbose and (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))

# trains a private model for a single epoch using private SGD
# clf must be a FastGradMLP
def train_private(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, C, std, include_linear=False, verbose=True):
    extr.train()
    clf.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # compute per-example gradients
        num_batches = int(math.ceil(float(data.size(0)) / args.process_batch_size))
        loss = 0
        grad_vec = None
        for i in range(num_batches):
            start = i * args.process_batch_size
            end = min((i+1) * args.process_batch_size, data.size(0))
            data_batch = data[start:end]
            target_batch = target[start:end]
            loss_batch, gradients_batch = per_example_gradient(extr, clf, data_batch, target_batch, loss_fn, include_linear=include_linear)
            loss += data_batch.size(0) * loss_batch.item()
            if i == 0:
                grad_vec = clip_and_sum_gradients(gradients_batch, C)
            else:
                grad_vec += clip_and_sum_gradients(gradients_batch, C)
        loss /= data.size(0)
        grad_vec /= data.size(0)
        noise = add_noisy_gradient(extr, clf, device, grad_vec, C, std / data.size(0), include_linear=include_linear)
        optimizer.step()
        if verbose and (batch_idx + 1) % args.log_interval == 0:
            print('Epoch %d [%d/%d]: loss = %.4f, grad_norm = %.4f, noise_norm = %.4f' % (
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), loss,
                grad_vec.norm(), noise.norm()))
