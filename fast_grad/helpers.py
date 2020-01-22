# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from time import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import cProfile, pstats
import pdb #debugging

def batch_grads_to_vec(parameters):
	r"""Convert parameters to one vector
	Arguments:
		parameters (Iterable[Tensor]): an iterator of Tensors that are of shape [N x the
			parameters of a model].
	Returns:
		The parameters represented by a single Tensor of shape [N x number of parameters in the model]
	"""
	N = parameters[0].shape[0]
	vec = []
	for param in parameters:
		vec.append(param.view(N,-1))
	return torch.cat(vec, dim=1)


def check_correctness(full, names, approximations, model, X, y):
	print()
	print("  Checking correctness")
	print("  ---")

	true_value = parameters_to_vector(full(model, X, y))
	approx_values = list()
	for i in range(len(approximations)):
		approx_value = batch_grads_to_vec(approximations[i](model, X, y))
		approx_values.append(approx_value)
		#pdb.set_trace()
		print("  - Diff. to full batch for (%5s):        %f" % (names[i], torch.norm(true_value - torch.mean(approx_value, dim=0))))
	for i in range(len(approximations)):
		for j in range(i):
			if i != j:
				print("  - Difference between (%5s) and (%5s): %f" % (names[i], names[j], torch.norm(approx_values[i] - approx_values[j])))


def simpleTiming(full, names, approximations, model, X, y, REPEATS=10):
	print()
	print("  Simple timing")
	print("  ---")

	def timeRun(method):
		start = time()
		for r in range(REPEATS):
			method(model, X, y)
		end = time()
		return (end - start)/REPEATS

	print("  - Full    : %.3fs" % timeRun(full))
	for i in range(len(approximations)):
		print("  - (%5s) : %.3fs" % (names[i], timeRun(approximations[i])))

def profiling(full, names, approximations, model, X, y, REPEATS=1, Prec=20):
	print("Profiling")

	def profile(method):
		pr = cProfile.Profile()
		pr.enable()
		for r in range(REPEATS):
			method(model, X, y)
		pr.disable()
		pr.create_stats()
		ps = pstats.Stats(pr).sort_stats("cumulative")
		ps.print_stats(Prec)

	print("Full:")
	profile(full)
	for i in range(len(approximations)):
		print(names[i])
		profile(approximations[i])

def make_data_and_model(N, D, L, seed=1):
	"""
	# N: Number of samples
	# D: Dimension of input and of each Layer
	# L: Number of hidden layers
	"""
	torch.manual_seed(seed)

	hidden_sizes = list(D for l in range(L))

	X = torch.Tensor(torch.randn(N, D))
	y = torch.Tensor(torch.round(torch.rand(N))).view(-1,)

	model = MLP(input_size = D, hidden_sizes = hidden_sizes)
	model.train(True)

	return X, y, model

