# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import math
import os
import sys
from fast_grad.goodfellow_backprop import goodfellow_backprop
from torchvision import datasets, transforms


# extracts features into a tensor
def extract_features(extr, device, data_loader):
    extr.eval()
    features = None
    labels = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = extr(data).data.cpu()
            if features is None:
                features = output.squeeze()
                labels = target
            else:
                features = torch.cat([features, output.squeeze()], dim=0)
                labels = torch.cat([labels, target], dim=0)
    return features, labels

# constructs one-hot representations of labels
def onehot(y):
    y_onehot = -torch.ones(y.size(0), y.max() + 1).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot

# loads features from a saved checkpoint or directly as raw features
def load_features(args):
    
    ckpt_file = '%s/%s_%s_extracted.pth' % (args.data_dir, args.extractor, args.dataset)
    if os.path.exists(ckpt_file):
        checkpoint = torch.load(ckpt_file)
        X_train = checkpoint['X_train'].cpu()
        y_train = checkpoint['y_train'].cpu()
        X_test = checkpoint['X_test'].cpu()
        y_test = checkpoint['y_test'].cpu()
    else:
        print('Extracted features not found, loading raw features.')
        if args.dataset == 'MNIST':
            trainset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())
            testset = datasets.MNIST(args.data_dir, train=False, transform=transforms.ToTensor())
            X_train = torch.zeros(len(trainset), 784)
            y_train = torch.zeros(len(trainset))
            X_test = torch.zeros(len(testset), 784)
            y_test = torch.zeros(len(testset))
            for i in range(len(trainset)):
                x, y = trainset[i]
                X_train[i] = x.view(784) - 0.5
                y_train[i] = y
            for i in range(len(testset)):
                x, y = testset[i]
                X_test[i] = x.view(784) - 0.5
                y_test[i] = y
            # load classes 3 and 8
            train_indices = (y_train.eq(3) + y_train.eq(8)).gt(0)
            test_indices = (y_test.eq(3) + y_test.eq(8)).gt(0)
            X_train = X_train[train_indices]
            y_train = y_train[train_indices].eq(3).float()
            X_test = X_test[test_indices]
            y_test = y_test[test_indices].eq(3).float()
        else:
            print("Error: Unknown dataset %s. Aborting." % args.dataset) 
            sys.exit(1)
        
    # L2 normalize features
    X_train /= X_train.norm(2, 1).unsqueeze(1)
    X_test /= X_test.norm(2, 1).unsqueeze(1)
    # convert labels to +/-1 or one-hot vectors
    if args.train_mode == 'binary':
        y_train_onehot = y_train
        y_train = (2 * y_train - 1)
    else:
        y_train_onehot = onehot(y_train)
    if len(y_train_onehot.size()) == 1:
        y_train_onehot = y_train_onehot.unsqueeze(1)
        
    return X_train, X_test, y_train, y_train_onehot, y_test

# computes per-example gradient of the extractor and classifier models
# clf must be a FastGradMLP
def per_example_gradient(extr, clf, x, y, loss_fn, include_linear=False):
    logits, activations, linearCombs = clf(extr(x))
    loss = loss_fn(logits, y)
    loss.backward(retain_graph=True)
    gradients = []
    for module in list(next(extr.children()).children()):
        grad = module.expanded_weight.grad * x.size(0)
        gradients.append(grad.view(x.size(0), -1, grad.size(1), grad.size(2), grad.size(3)))
        if module.expanded_bias is not None:
            gradients.append(module.expanded_bias.grad.view(x.size(0), -1) * x.size(0))
    if include_linear:
        linearGrads = torch.autograd.grad(loss, linearCombs)
        linearGrads = goodfellow_backprop(activations, linearGrads)
        gradients = gradients + linearGrads
    return loss, gradients

# clips each gradient to norm C and sum
def clip_and_sum_gradients(gradients, C):
    grad_vec = batch_grads_to_vec(gradients)
    grad_norm = grad_vec.norm(2, 1)
    multiplier = grad_norm.new(grad_norm.size()).fill_(1)
    multiplier[grad_norm.gt(C)] = C / grad_norm[grad_norm.gt(C)]
    grad_vec *= multiplier.unsqueeze(1)
    return grad_vec.sum(0)

# adds noise to computed gradients
# grad_vec should be average of gradients
def add_noisy_gradient(extr, clf, device, grad_vec, C, std, include_linear=False):
    noise = torch.randn(grad_vec.size()).to(device) * C * std
    grad_perturbed = grad_vec + noise
    extr.zero_grad()
    for param in extr.parameters():
        size = param.data.view(1, -1).size(1)
        param.grad = grad_perturbed[:size].view_as(param.data).clone()
        grad_perturbed = grad_perturbed[size:]
    if include_linear:
        clf.zero_grad()
        for param in clf.parameters():
            size = param.data.view(1, -1).size(1)
            param.grad = grad_perturbed[:size].view_as(param.data).clone()
            grad_perturbed = grad_perturbed[size:]
    return noise

# computes L2 regularized loss
def loss_with_reg(model, data, target, loss_fn, lam):
    model.zero_grad()
    loss = loss_fn(model(data), target)
    if lam > 0:
        for param in model.parameters():
            loss += lam * param.pow(2).sum() / 2
    loss.backward()
    return loss

# computes average gradient of the full dataset
def compute_full_grad(model, device, data_loader, loss_fn, lam=0):
    full_grad = None
    model.zero_grad()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        loss_with_reg(model, data, target, loss_fn, lam)
        grad = params_to_vec(model.parameters(), grad=True)
        if full_grad is None:
            full_grad = grad * data.size(0) / len(data_loader.dataset)
        else:
            full_grad += grad * data.size(0) / len(data_loader.dataset)
        model.zero_grad()
    param_vec = params_to_vec(model.parameters())
    return full_grad, param_vec

def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()

def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param

def batch_grads_to_vec(parameters):
    N = parameters[0].shape[0]
    vec = []
    for param in parameters:
        vec.append(param.view(N,-1))
    return torch.cat(vec, dim=1)

def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(param.size(0), -1).size(1)
        grads.append(vec[:, :size].view_as(param))
        vec = vec[:, size:]
    return grads
