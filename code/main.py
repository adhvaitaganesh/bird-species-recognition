#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

# Temporary testing imports
import random
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset import nabirds
import nabirdsDataset
from transforms import rescale
from transforms import crop
from networks import SPP
import matplotlib.pyplot as plt

def testSPP():
	input = torch.randn(2, 3, 4, 4)
	print(input)
	layers = [(1, 1), (2, 2), (4, 4)]
	spp = SPP.SPP(layers)
	result = spp(input)
	print(result)
	print("Actual size: ", result.size(), f"VS Expected {input.shape[0]} x", spp.output_size * input.shape[1])

def testDatasetBase():
	labels = nabirds.load_image_labels('../dataset/')
	label_count = nabirds.label_count
	print("Amount of labels in the dataset:", label_count)

def testDatasetTorch():
	# Initialize the data wrapper and show a random image.
	dl = nabirdsDataset.nabirdsWrapper('../dataset', transform=rescale.Rescale((200, 200)))
	image_id = random.choice(dl.test_images)
	dl.showImage(image_id, dl.className(image_id), show_bbox=True)
	dl.showImage('3b69ce35-b940-4f3e-b321-100c93dd2b43')
	# Create the pytorch compatible dataset objects for the training set and the test set.
	training_data = dl.generateTorchDataset(dl.train_images)
	test_data = dl.generateTorchDataset(dl.test_images)
	print("Training set has length:", len(training_data))
	print("Test set has length:", len(test_data))

	for i in range(3):
		image, label, idx = training_data[i]
		dl.showImage(idx, dl.className(img_class=label), show_bbox=True)
		print(image.shape, dl.className(img_class=label))

	# make a dataloader:
	training_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
	# TODO: Batches don't work with non-uniform image sizes. So I want to make a transform we can apply to make images uniform.
	for i, (images, labels, idx) in enumerate(training_dataloader):
		# Print 3rd batch and stop.
		if i == 3:
			for j in range(images.shape[0]):
				# Show the original image and the decomposition into color channels.
				img = images[j].numpy()
				print(f"Image shape: {img[0].shape}")
				zero = np.zeros((200, 200))
				red = np.stack((img[0], zero, zero))
				green = np.stack((zero, img[1], zero))
				blue = np.stack((zero, zero, img[2]))
				dl.showImage(idx[j])
				dl.showImage(idx[j], show_bbox=False, apply_transform=True)
				plt.figure()
				plt.subplot(2, 2, 1)
				plt.imshow(images[j].T)
				plt.subplot(2, 2, 2)
				plt.imshow(red.T)
				plt.subplot(2, 2, 3)
				plt.imshow(green.T)
				plt.subplot(2, 2, 4)
				plt.imshow(blue.T)
				plt.show()

			print(images.shape)
			print(labels)
			break

def testCropTransform():
	dl = nabirdsDataset.nabirdsWrapper('../dataset')
	image_id = random.choice(dl.test_images)
	dl.showImage(image_id, dl.className(image_id), show_bbox=False)
	img = dl.loadImage(image_id)
	top_left, width, height = dl.bbox(image_id)
	cropped_img = crop.Crop(img, top_left, width, height)
	plt.imshow(cropped_img.T)
	plt.show()

def generateResults():
	dl = nabirdsDataset.nabirdsWrapper('../dataset')
	image_id = random.choice(dl.test_images)

	plt.figure()
	plt.subplot(2, 2, 1)
	img = dl.loadImage(image_id)
	print(img.shape)
	top_left, width, height = dl.bbox(image_id)
	cropped_img = img[:, top_left[0]:top_left[0] + width, top_left[1]:top_left[1] + height]
	print(cropped_img.shape)
	plt.imshow(cropped_img.T)
	
	img = torch.from_numpy(cropped_img).float()

	layers = [(1, 1), (4, 4), (6, 6)]
	spp = SPP.SPP(layers)
	result = spp.debug(img)

	for i in range(3):
		plt.subplot(2, 2, i + 2)
		weird_img = result[i].numpy().astype(np.int32)
		print(weird_img)
		plt.imshow(weird_img.T)
	plt.show()

# testSPP()
# testDatasetBase()
testDatasetTorch()
testCropTransform()
generateResults()