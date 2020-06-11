# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import argparse
from models import Extractor, MLP
from fast_grad_models import FastGradExtractor, FastGradMLP
from train_func import train, train_private
from test_func import test, test_linear
import utils
import time
import os
from torchdp.privacy_analysis import compute_rdp, get_privacy_spent


def main():
    parser = argparse.ArgumentParser(description='Training an SVHN model')
    parser.add_argument('--data-dir', type=str, required=True, help='directory for SVHN data')
    parser.add_argument('--save-dir', type=str, default='save', help='directory for saving trained model')
    parser.add_argument('--batch-size', type=int, default=500, help='batch size for training')
    parser.add_argument('--process-batch-size', type=int, default=500, help='batch size for processing')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lam', type=float, default=0, help='L2 regularization')
    parser.add_argument('--std', type=float, default=6.0, help='noise multiplier for DP training')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for DP training')
    parser.add_argument('--num-filters', type=int, default=64, help='number of conv filters')
    parser.add_argument('--seed', type=int, default=1, help='manual random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='logging interval')
    parser.add_argument('--train-mode', type=str, default='default', help='train mode [default/private/full_private]')
    parser.add_argument('--test-mode', type=str, default='default', help='test mode [default/linear/extract]')
    parser.add_argument('--save-suffix', type=str, default='', help='suffix for model name')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='normalize extracted features')
    parser.add_argument('--single-layer', action='store_true', default=False,
                        help='single convolutional layer')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the trained model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ])
    trainset = torchvision.datasets.SVHN(root=args.data_dir, split='train', download=True, transform=transform)
    extraset = torchvision.datasets.SVHN(root=args.data_dir, split='extra', download=True, transform=transform)
    trainset = torch.utils.data.ConcatDataset([trainset, extraset])
    testset = torchvision.datasets.SVHN(root=args.data_dir, split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.single_layer:
        extr = FastGradExtractor([3, args.num_filters], 9, 1, 2, normalize=args.normalize).to(device)
        clf = FastGradMLP([12*12*args.num_filters, 10]).to(device)
    else:
        extr = FastGradExtractor([3, args.num_filters, args.num_filters], 5, 1, 2, normalize=args.normalize).to(device)
        clf = FastGradMLP([5*5*args.num_filters, 10]).to(device)
    loss_fn = lambda x, y: F.nll_loss(F.log_softmax(x, dim=1), y)
    save_path = "%s/svhn_cnn_delta_%.2e_std_%.2f%s.pth" % (args.save_dir, args.delta, args.std, args.save_suffix)
    if not os.path.exists(save_path):
        optimizer = optim.Adam(list(extr.parameters()) + list(clf.parameters()), lr=args.lr, weight_decay=args.lam)
        C = 4
        n = len(train_loader.dataset)
        q = float(args.batch_size) / float(n)
        T = args.epochs * len(train_loader)
        # compute privacy loss using RDP analysis
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                    list(range(5, 64)) + [128, 256, 512, 1024, 2048, 4096])
        epsilon, _ = get_privacy_spent(orders, compute_rdp(q, args.std, T, orders), args.delta)
        print('RDP computed privacy loss: epsilon = %.2f at delta = %.2e' % (epsilon, args.delta))
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            if args.train_mode == 'private' or args.train_mode == 'full_private':
                include_linear = (args.train_mode == 'full_private')
                train_private(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, C, args.std, include_linear=include_linear)
            else:
                train(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch)
            test(args, extr, clf, loss_fn, device, test_loader)
        print(time.time() - start)
        if args.save_model:
            torch.save({'extr': extr.state_dict(), 'clf': clf.state_dict()}, save_path)
    else:
        checkpoint = torch.load(save_path)
        extr.load_state_dict(checkpoint['extr'])
        clf.load_state_dict(checkpoint['clf'])
        if args.test_mode == 'linear':
            test_linear(args, extr, device, train_loader, test_loader)
        elif args.test_mode == 'extract':
            # this option can be used to extract features for training the removal-enabled linear model
            X_train, y_train = utils.extract_features(extr, device, train_loader)
            X_test, y_test = utils.extract_features(extr, device, test_loader)
            torch.save({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test},
                       '%s/dp_delta_%.2e_std_%.2f_SVHN_extracted.pth' % (args.data_dir, args.delta, args.std))
        else:
            test(args, extr, clf, loss_fn, device, test_loader)

if __name__ == '__main__':
    main()
