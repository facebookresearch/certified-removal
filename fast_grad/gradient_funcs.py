# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
import pdb #debugging

from goodfellow_backprop import goodfellow_backprop

def full(model, X, y):
	"""
	Computes the gradient of the complete objective function
	"""

	logits, _, _ = model.forward(X)
	loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)
	grad = torch.autograd.grad(loss, model.parameters())

	return grad

def naive(model, X, y):
	"""
	Computes the predictions in a full-batch fasion,
	then call backward on the individual losses
	"""
	grad_list = []
	logits, _, _ = model.forward(X)
	N = X.shape[0]
	for n in range(N):
		model.zero_grad()
		loss = F.binary_cross_entropy_with_logits(logits[n], y[n].view(-1,))
		loss.backward(retain_graph=True)

		grad_list.append(list([p.grad.clone() for p in model.parameters()]))

	grads = []
	for p_id in range(len(list(model.parameters()))):
		grads.append(torch.cat([grad_list[n][p_id].unsqueeze(0) for n in range(N)]))

	return grads

def goodfellow(model, X, y):
	"""
	Use Goodfellow's trick to compute individual gradients.
	Ref: Efficient per-example gradient computations
	at: https://arxiv.org/abs/1510.01799
	"""
	model.zero_grad()

	logits, activations, linearCombs = model.forward(X)
	loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)

	linearGrads = torch.autograd.grad(loss, linearCombs)
	gradients = goodfellow_backprop(activations, linearGrads)

	return gradients
