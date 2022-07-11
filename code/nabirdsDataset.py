#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

UI = False
try:
	import matplotlib.pyplot as plt
	import matplotlib.patches as patches
	UI = True
except ModuleNotFoundError:
	pass

import numpy as np
from itertools import tee

from transforms import crop

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset import nabirds

class nabirdsWrapper:
	def __init__(self, path_to_dataset, transform=None):
		self.ptd = path_to_dataset
		self.transform = transform
		self.train_images, self.test_images = nabirds.load_train_test_split(self.ptd)

		# Make subsets
		#self.train_images = self.train_images[:100]
		#self.test_images = self.test_images[:10]

		self.img_paths = nabirds.load_image_paths(self.ptd)
		self.img_base = 'images'
		self.img_ids_to_class_ids = nabirds.load_image_labels(self.ptd)
		
		# First create a set of all unique bird label id's.
		class_id_set = set(self.img_ids_to_class_ids.values())
		# Then map each id to an index in 0, 1, ..., 554.
		self.class_ids_to_class_index = {label:i for i, label in enumerate(class_id_set)}
		
		self.class_ids_to_class_names = nabirds.load_class_names(self.ptd)
		self.img_sizes = nabirds.load_image_sizes(self.ptd)
		self.class_count = nabirds.label_count
		self.bboxes = nabirds.load_bounding_box_annotations(self.ptd)
		return

	def setTransform(self, transform):
		self.transform = transform

	# Returns the original class ID from the class index. Index must be a single value in the range 0-554.
	def classIndexToClassId(self, class_index):
		return list(self.class_ids_to_class_index.keys())[list(self.class_ids_to_class_index.values()).index(class_index)]

	# Generate a pytorch compatible dataset that containst the parsed images. 
	# Image_ids should be a list of image id's.
	def generateTorchDataset(self, image_ids):
		return self.nabirdsDataset(self, image_ids)

	# Returns (x, y), w, h. Here (x, y) are the coordinates of the top left (lowest x, y coordinates) of the bbox, w is the width and h is the height.
	# Optionally disable the transform that is also applied to the image
	def bbox(self, image_id, apply_transform=True):
		self.bboxes[image_id], bbox_iter = tee(self.bboxes[image_id]) # We have to do this weird trick to save a copy of the map. Otherwise we can only access the bounding box once.
		bbox_raw = list(bbox_iter)

		if apply_transform and self.transform:
			img_dims = self.loadImage(image_id).shape[1:3] # Only takes the width and height.
			return self.transform.applyBBox((bbox_raw[0], bbox_raw[1]), bbox_raw[2], bbox_raw[3], img_dims)
		
		return (bbox_raw[0], bbox_raw[1]), bbox_raw[2], bbox_raw[3]

	# Load the image as a numpy array. Has shape (3, height, width). Note that this is incompatible with imshow. For that use image.T.
	def loadImage(self, image_id):
		img_file = os.path.join(self.ptd, self.img_base, self.img_paths[image_id])
		return np.array(Image.open(img_file)).T

	# Return the class name of the image. You can pass the class or the image id.
	# img_class is either the class index as integer, or class label as tensor.
	def className(self, img_id=None, img_class=None):
		if img_id:
			return self.class_ids_to_class_names[self.img_ids_to_class_ids[img_id]]
		elif img_class is not None:
			class_id = 0
			if torch.is_tensor(img_class):
				class_id = self.classIndexToClassId(torch.argmax(img_class).item())
			else:
				class_id = self.classIndexToClassId(img_class)
			return self.class_ids_to_class_names[class_id]

	# Show the image. Optionally with title in the top left and with a bounding box around the bird.
	# Optionally disable the transform that is applied to the image.
	def showImage(self, image_id, title='', show_bbox=False, apply_transform=False):
		if not UI:
			return
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)

		img = self.loadImage(image_id)
		if apply_transform and self.transform:
			img = self.transform(img)

		plt.imshow(img.T)
		plt.text(0, -5, title)
		if show_bbox:
			top_left, width, height = self.bbox(image_id, apply_transform=apply_transform)
			bbox = patches.Rectangle(top_left, width, height, linewidth=1, edgecolor='r', facecolor='none')
			ax.add_patch(bbox)
		plt.show()
		return

	class nabirdsDataset(Dataset):
		def __init__(self, wrapper, images):
			self.images = images
			self.wrapper = wrapper

		# Needs to be defined to properly extend Dataset. Returns length of the dataset
		def __len__(self):
			return len(self.images)

		# Returns an (image, class_index, img_id) sample from the dataset. Here the image will be a numpy array.
		# The class index is a value between 0 and 554 that corresponds to the class of the bird.
		# idx just also returns the id of the image for debugging purposes.
		def __getitem__(self, idx):
			if torch.is_tensor(idx):
				idx = idx.tolist()
				print('listed')
			
			img_id = self.images[idx]
			img = self.wrapper.loadImage(img_id)

			# There's one image in the training set that somehow messes up, hence this bit of code.
			if img.shape[0] > 3:
				img = img[0:3]

			if self.wrapper.transform:
				img = self.wrapper.transform(img)

			img_class_index = self.wrapper.class_ids_to_class_index[self.wrapper.img_ids_to_class_ids[img_id]]
			return img, img_class_index, img_id
