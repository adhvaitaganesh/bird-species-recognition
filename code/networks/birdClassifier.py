#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision
import numpy as np
from networks import SPP

# Set finetune to True if you want to also train the base network.
class birdClassifier(nn.Module):
	def __init__(self, use_SPP=True, finetune=False):
		super(birdClassifier, self).__init__()
		self.layers = torchvision.models.resnet152(pretrained=True)
		if not finetune:
			for param in self.layers.parameters():
				param.requires_grad = False

		if use_SPP:
			config = [(1, 1), (2, 2), (4, 4)]
			single_feature_output = 0
			for entry in config:
				single_feature_output += entry[0] * entry[1]
			full_output = single_feature_output * 2048 # Because the final convolutional layer of resnet has 2048 features.


			self.layers.avgpool = SPP.SPP(config)
			self.layers.fc = nn.Linear(full_output, 555)
		else:
			self.layers.fc = nn.Linear(2048, 555)

	def forward(self, x):
		return self.layers(x)