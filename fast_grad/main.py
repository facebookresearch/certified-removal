# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pdb
import helpers
from gradient_funcs import full, goodfellow, naive

def runWith(N, D, L):
	X, y, model = helpers.make_data_and_model(N, D, L)

	names = ["Goodf", "Naive"]
	methods = [goodfellow, naive]

	helpers.check_correctness(full, names, methods, model, X, y)
	helpers.simpleTiming(full, names, methods, model, X, y, REPEATS=1)
	#helpers.profiling(full, names, methods, model, X, y)

setups = [
	[2,3,1],
	[10,100,10],
	[100,100,10],
	[100,300,3],
	[32,300,50],
	[1000,100,10]
]

print("README:")
print()
print("Parameters:")
print("- N: Number of samples")
print("- D: Dimensionality of the inputs and hidden layers - width of the network")
print("- L: Number of hidden layers - depth of the network")
print()
print("Functions:")
print("- Full : Computes the averaged gradient")
print("- Naive: Compute each individual gradient by repeatedly calling backward")
print("- Goodf: Compute the individual gradients using Goodfellow's Trick,")
print("  which is equivalent to redefining the backward pass to _not_ aggregate individual gradients")
print()
print("Checking correctness is done with torch.norm()")
print("- For the diff. to the Full gradient, we first average over the sample")
print("- For the difference between individual gradient methods,")
print("  we take the L2 norm between [N x ...] matrices")

for setup in setups:
	print()
	print("Setup [N, D, L] =", setup)
	print("---")
	runWith(*setup)
