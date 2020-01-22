# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from train_func import train
from utils import extract_features
from sklearn.linear_model import LogisticRegression


class Linear(nn.Module):
    def __init__(self, input_size):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_size, 10)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    
# computes test accuracy
def test(args, extr, clf, loss_fn, device, test_loader, verbose=True):
    if extr is not None:
        extr.eval()
    clf.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if extr is not None:
                output = clf(extr(data))
                if len(output) == 3:
                    output = output[0]
            else:
                output = clf(data)
            test_loss += output.size(0) * loss_fn(output, target).item()
            if output.size(1) > 1:
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            else:
                pred = output.gt(0).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = float(correct) / len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100 * test_acc))
    return test_loss, test_acc

# computes test accuracy of a logistic regression model using features extracted from extr
def test_linear(args, extr, device, train_loader, test_loader, verbose=True):
    X_train, y_train = extract_features(extr, device, train_loader)
    X_test, y_test = extract_features(extr, device, test_loader)
    clf = LogisticRegression(C=1/(X_train.size(0)*args.lam), solver='saga', multi_class='multinomial', verbose=int(verbose))
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    acc = clf.score(X_test.cpu().numpy(), y_test.cpu().numpy())
    print('Test accuracy = %.4f' % acc)
    return acc
