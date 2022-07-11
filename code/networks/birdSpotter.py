#!/usr/bin/env python
import torch
import torchvision
import torch.nn as nn
#from networks import SPP
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

class birdSpotter(nn.Module):
	def __init__(self):
		super(birdSpotter, self).__init__()
		self.backbone = torchvision.models.resnet152(pretrained=True)
		for param in self.layers.parameters():
			param.requires_grad = False

		self.backbone = nn.Sequential(*list(self.layers.children())[:-2])

		self.backbone.out_channels = 2048

		num_classes = 2  # 1 class (bird) + background
		
		#anchor generator for the FasterRCNN (didn't understand fully but needed )
		anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
		
		#to do roi allign
		roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

		self.model =	FasterRCNN(self.backbone,
						num_classes = num_classes,
						rpn_anchor_generator = anchor_generator,
						box_roi_pool = roi_pooler )
		# get number of input features for the classifier
		in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    		# replace the pre-trained head with a new one
		self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	def forward(self, x) :
		out = self.model(x)
		return out


#instance Segmentation model, just for checking it out tomorrow
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
