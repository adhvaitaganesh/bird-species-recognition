#!/usr/bin/env python
import torch
import torch.nn as nn
from networks import SPP
import numpy as np

from networks import birdClassifier
from networks import birdSpotter
from transforms import crop

class fullNetwork(nn.Module):
	def __init__(self, use_SPP):
		super(fullNetwork, self).__init__()
		# TODO: Combine birdSpotter and birdClassifier here.
		self.birdspotter = birdSpotter.birdSpotter()
		self.birdclassifier = birdClassifier.birdClassifier(use_SPP)

	def forward(self, x):
		top_left, width, height = self.birdspotter(x)
		x = crop.Crop(x, top_left, width, height)
		return self.birdclassifier(x)