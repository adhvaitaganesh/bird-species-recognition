#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np

class SPP(nn.Module):
	# pyramid_configuration takes a list of n dimensions (w_i, h_i). These correspond to the output sizes of the pooling layers.
	# i.e. [(1, 1), (2, 2), (4, 4)] does a max-pooling on the entire image ((1, 1) output size), a max-pooling on each quadrant ((2, 2) pooling) and finally on each 16th section of the image.
	# Output will be in a single vector of length Σ(w_i * h_i) for all pooling layers. Note that this will be multiplied by the amount of feature maps given as input.
	# i.e. (1 * 1) + (2 * 2) + (4 * 4) = 1 + 4 + 16 = 21.
	def __init__(self, pyramid_configuration):
		super().__init__()
		self.pc = pyramid_configuration
		self.layers = []
		output_size = 0
		for layer_shape in self.pc:
			self.layers.append(nn.AdaptiveMaxPool2d(layer_shape))
			output_size += layer_shape[0] * layer_shape[1]
		self.output_size = output_size

	def forward(self, input):
		# Do the first layer separately to avoid if statement in the loop.
		output = torch.reshape(self.layers[0](input), (input.shape[0], -1))
		for layer in self.layers[1:]:
			result = torch.reshape(layer(input), (input.shape[0], -1)) # First reshape the pooling result to 2D Tensor. (1D tensor as in the paper for each sample)
			output = torch.cat((output, result), dim=1) # Append this tensor to the final result
		return output

	def debug(self, input):
		output = []
		print(input)
		for layer in self.layers:
			output.append(layer(input))
		return output