# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression
from utils import load_features

parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
parser.add_argument('--data-dir', type=str, required=True, help='data directory')
parser.add_argument('--result-dir', type=str, default='result', help='directory for saving results')
parser.add_argument('--extractor', type=str, default='resnet50', help='extractor type')
parser.add_argument('--dataset', type=str, default='SVHN', help='dataset')
parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
parser.add_argument('--std', type=float, default=10.0, help='standard deviation for objective perturbation')
parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
parser.add_argument('--train-splits', type=int, default=1, help='number of training data splits')
parser.add_argument('--subsample-ratio', type=float, default=1.0, help='negative example subsample ratio')
parser.add_argument('--num-steps', type=int, default=100, help='number of optimization steps')
parser.add_argument('--train-mode', type=str, default='ovr', help='train mode [ovr/binary]')
parser.add_argument('--train-sep', action='store_true', default=False, help='train binary classifiers separately')
parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')
args = parser.parse_args()

device = torch.device("cuda")

def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2

def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()

def lr_grad(w, X, y, lam):
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w

def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()

def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

def ovr_lr_loss(w, X, y, lam, weight=None):
    z = batch_multiply(X, w).mul_(y)
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2

def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-10)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)

def spectral_norm(A, num_iters=20):
    x = torch.randn(A.size(0)).float().to(device)
    norm = 1
    for i in range(num_iters):
        x = A.mv(x)
        norm = x.norm()
        x /= norm
    return math.sqrt(norm)

# loads extracted features
X_train, X_test, y_train, y_train_onehot, y_test = load_features(args)
X_test = X_test.float().to(device)
y_test = y_test.to(device)

save_path = '%s/%s_%s_splits_%d_ratio_%.2f_std_%.1f_lam_%.0e.pth' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam)
if os.path.exists(save_path):
    # load trained models
    checkpoint = torch.load(save_path)
    w = checkpoint['w']
    b = checkpoint['b']
    weight = checkpoint['weight']
else:
    # train removal-enabled linear model
    start = time.time()
    if args.subsample_ratio < 1.0:
        # subsample negative examples
        subsample_indices = torch.rand(y_train_onehot.size()).lt(args.subsample_ratio).float()
        weight = (subsample_indices + y_train_onehot.gt(0).float()).gt(0).float()
        weight = weight / weight.sum(0).unsqueeze(0)
        weight = weight.to(device)
    else:
        weight = None
    # sample objective perturbation vector
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    y_train_onehot = y_train_onehot.float().to(device)
    if args.train_mode == 'ovr':
        b = args.std * torch.randn(X_train.size(1), y_train_onehot.size(1)).float().to(device)
        if args.train_sep:
            # train K binary LR models separately
            w = torch.zeros(b.size()).float().to(device)
            for k in range(y_train_onehot.size(1)):
                if weight is None:
                    w[:, k] = lr_optimize(X_train, y_train_onehot[:, k], args.lam, b=b[:, k], num_steps=args.num_steps, verbose=args.verbose)
                else:
                    w[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train_onehot[:, k][weight[:, k].gt(0)], args.lam, b=b[:, k], num_steps=args.num_steps, verbose=args.verbose)
        else:
            # train K binary LR models jointly
            w = ovr_lr_optimize(X_train, y_train_onehot, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose)
    else:
        b = args.std * torch.randn(X_train.size(1)).float().to(device)
        w = lr_optimize(X_train, y_train, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose)
    print('Time elapsed: %.2fs' % (time.time() - start))
    torch.save({'w': w, 'b': b, 'weight': weight}, save_path)

if args.train_mode == 'ovr':
    pred = X_test.mm(w).max(1)[1]
    print('Test accuracy = %.4f' % pred.eq(y_test).float().mean())
else:
    pred = X_test.mv(w)
    print('Test accuracy = %.4f' % pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean())

grad_norm_approx = torch.zeros(args.num_removes).float()
times = torch.zeros(args.num_removes)
if args.train_mode == 'ovr':
    y_train = y_train_onehot
w_approx = w.clone()
perm = torch.randperm(X_train.size(0)).to(y_train.device)
X_train = X_train.index_select(0, perm)
X_train = X_train.float().to(device)
y_train = y_train[perm].float().to(device)

# initialize K = X^T * X for fast computation of spectral norm
print('Preparing for removal')
if weight is None:
    K = X_train.t().mm(X_train)
else:
    weight = weight.index_select(0, perm.to(device))
    Ks = []
    for i in range(y_train_onehot.size(1)):
        X_sub = X_train.cpu()[weight[:, i].gt(0).cpu()]
        Ks.append(X_sub.t().mm(X_sub).to(device))

print('Testing removal')
for i in range(args.num_removes):
    start = time.time()
    if args.train_mode == 'ovr':
        # removal from all one-vs-rest models
        for k in range(y_train_onehot.size(1)):
            if weight is None or weight[i, k] > 0:
                X_rem = X_train[(i+1):]
                y_rem = y_train[(i+1):, k]
                if weight is not None:
                    X_rem = X_rem[weight[(i+1):, k].gt(0)]
                    y_rem = y_rem[weight[(i+1):, k].gt(0)]
                H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam)
                grad_i = lr_grad(w_approx[:, k], X_train[i].unsqueeze(0), y_train[i, k].unsqueeze(0), args.lam)
                # apply rank-1 down-date to K
                if weight is None:
                    K -= torch.ger(X_train[i], X_train[i])
                    spec_norm = spectral_norm(K)
                else:
                    Ks[k] -= torch.ger(X_train[i], X_train[i])
                    spec_norm = spectral_norm(Ks[k])
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                w_approx[:, k] += Delta
                grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm / 4).cpu()
    else:
        # removal from a single binary logistic regression model
        X_rem = X_train[(i+1):]
        y_rem = y_train[(i+1):]
        H_inv = lr_hessian_inv(w_approx[:], X_rem, y_rem, args.lam)
        grad_i = lr_grad(w_approx, X_train[i].unsqueeze(0), y_train[i].unsqueeze(0), args.lam)
        K -= torch.ger(X_train[i], X_train[i])
        spec_norm = spectral_norm(K)
        Delta = H_inv.mv(grad_i)
        Delta_p = X_rem.mv(Delta)
        w_approx += Delta
        grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm / 4).cpu()
            
    times[i] = time.time() - start
    print('Iteration %d: Grad norm bound = %.6f, time = %.2fs' % (i+1, grad_norm_approx[i], times[i]))

if args.train_mode == 'ovr':
    pred = X_test.mm(w_approx).max(1)[1]
    print('Test accuracy = %.4f' % pred.eq(y_test).float().mean())
else:
    pred = X_test.mv(w)
    print('Test accuracy = %.4f' % pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean())

save_path = '%s/%s_%s_splits_%d_ratio_%.2f_std_%.1f_lam_%.0e_removal.pth' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam)
torch.save({'grad_norm_approx': grad_norm_approx, 'times': times}, save_path)
