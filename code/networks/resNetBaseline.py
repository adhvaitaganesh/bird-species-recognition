#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision
import numpy as np

# Set finetune to True if you want to also train the base network.
class resNetBaseline(nn.Module):
    def __init__(self, finetune=False):
        super(resNetBaseline, self).__init__()
        self.layers = torchvision.models.resnet152(pretrained=True)
        if not finetune:
            for param in self.layers.parameters():
                param.requires_grad = False
        self.layers.fc = nn.Linear(2048, 555)

    def forward(self, x):
        return self.layers(x)