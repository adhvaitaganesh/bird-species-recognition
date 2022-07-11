#!/usr/bin/env python
import sys
import nabirdsDataset
import networks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

UI = False
try:
    import matplotlib.pyplot as plt
    UI = True
except:
    pass

from networks import birdSpotter
from networks import birdClassifier
from networks import fullNetwork
from networks import resNetBaseline

from transforms import rescale
from transforms import crop

class Trainer:
	# Set UI to true if you want nice output. Great for local training and debugging.
	def __init__(self, net_to_train, nr_epochs=20):
		self.net_to_train = net_to_train
		self.nr_epochs = nr_epochs
		# Preload the dataset so we can configure it for the selected network.
		self.dl = nabirdsDataset.nabirdsWrapper('../dataset/')

		# Check if cuda is available. Might not be usable with varying image sizes though.
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print('Using device: %s'%self.device)
		self.prepareDataset()
		return

	def prepareDataset(self):
		training_data = None
		testing_data = None

		# REMINDER: We can only make a batched dataloader from the data if all images are the same size.
		if net_to_train == 1:
			# For resnet we'll rescale every image to 224 by 224.
			self.dl.setTransform(rescale.Rescale((224, 224)))
			training_data = self.dl.generateTorchDataset(self.dl.train_images)
			testing_data = self.dl.generateTorchDataset(self.dl.test_images)
			print("Prepared dataset for ResNet.")
		elif net_to_train == 2:
			training_data = self.dl.generateTorchDataset(self.dl.train_images)
			testing_data = self.dl.generateTorchDataset(self.dl.test_images)
			print("Prepared dataset for birdClassifier without cropping to BBox.")
		elif net_to_train == 3:
			training_data = self.dl.generateTorchDataset(self.dl.train_images)
			testing_data = self.dl.generateTorchDataset(self.dl.test_images)
			print("Prepared dataset for birdClassifier with cropping to BBox.")
		elif net_to_train == 4:
			self.trainBirdSpotter()
		elif net_to_train == 5:
			self.TrainFullNet(SPP=False)
		elif net_to_train == 6:
			self.TrainFullNet(SPP=True)
		else:
			print("Not a valid network.")

		if training_data and testing_data:
			self.training_data = training_data
			self.testing_data = testing_data
			print("Set training and testing data successfully.")
		else:
			print("Data not loaded properly.")

	def train(self):
		if net_to_train == 1:
			self.trainResNet()
		elif net_to_train == 2:
			self.trainBirdClassifier(cropping=False)
		elif net_to_train == 3:
			self.trainBirdClassifier(cropping=True)
		elif net_to_train == 4:
			self.trainBirdSpotter()
		elif net_to_train == 5:
			self.TrainFullNet(SPP=False)
		elif net_to_train == 6:
			self.TrainFullNet(SPP=True)
		else:
			print("Not a valid network.")
	
	def update_lr(self, optimizer, lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
	
	def trainResNet(self):
		# TODO: implement the training for the baseline ResNet.
		print("ResNet")
		# By default only the final classifier layers should be trained.
		network = resNetBaseline.resNetBaseline()
		learning_rate = 1e-3
		lr = learning_rate
		learning_rate_decay = 0.99
		optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)

		network.to(self.device)
		
		criterion = nn.CrossEntropyLoss()
		training_dataloader = DataLoader(self.training_data, batch_size=256, shuffle=True)
		validation_dataloader = DataLoader(self.testing_data, batch_size=256, shuffle=True)
		total_step = len(training_dataloader)
		train_loss = []
		validation_loss = []

		torch.save(network.state_dict(), 'resnet_classifier.chkp')
		for epoch in range(self.nr_epochs):
			print(f"Starting epoch {epoch + 1}/{self.nr_epochs}.")
			for i, (images, labels, img_id) in enumerate(training_dataloader):
				# Move tensors to the configured device
				images = images.to(device=self.device, dtype=torch.float)
				labels = labels.to(device=self.device, dtype=torch.long)

				# Forward pass
				outputs = network(images)
				loss = criterion(outputs, labels)

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i + 1) % 1 == 0:
					print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.nr_epochs, i + 1, total_step, loss.item()))

			# Code to update the lr
			lr *= learning_rate_decay
			self.update_lr(optimizer, lr)

			train_loss.append(loss.item())
			torch.save(network.state_dict(), 'resnet_classifier.chkp')
			with torch.no_grad():
				correct = 0
				total = 0
				for (images, labels, img_id) in validation_dataloader:
					# Move the cropped image to the configured device
					images = images.to(device=self.device, dtype=torch.float)
					labels = labels.to(device=self.device, dtype=torch.long)

					outputs = network(images)
					_, predicted = torch.max(outputs.data, 1)
					loss1 = criterion(outputs, labels)

					# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

				print('Validataion accuracy is: {} %'.format(100 * correct / total))
			validation_loss.append(loss1.item())

		if UI:
			print(train_loss)
			plt.plot(range(len(train_loss)), train_loss, label="training loss")
			plt.plot(range(len(validation_loss)), validation_loss, label="validation_loss")
			plt.legend(["Training loss", "Validation loss"])
			plt.show()

		return

	def trainBirdClassifier(self, cropping):
		network = birdClassifier.birdClassifier()
		learning_rate = 0.99e-3
		true_batch_size = 200
		learning_rate = learning_rate / true_batch_size
		learning_rate_decay = 0.99
		optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)

		network.to(self.device)
		criterion = nn.CrossEntropyLoss()
		training_dataloader = DataLoader(self.training_data, batch_size=1, shuffle=True)
		validation_dataloader = DataLoader(self.testing_data, batch_size=1, shuffle=True)

		total_step = len(training_dataloader)
		train_loss = []
		validation_loss = []
		avg_loss = 0
		
		if cropping:
			# TODO: train classifier with SPP layer and perfect cropping.
			print("BirdClassifier + SPP + Perfect Cropping")
		else:
			# TODO: train classifier with SPP layer and no cropping.
			print("BirdClassifier + SPP + No Cropping")
			
		optimizer.zero_grad()
		torch.save(network.state_dict(), f'{"crop" if cropping else "no_crop"}_classifier.chkp')
		for epoch in range(self.nr_epochs):
			print(f"Starting epoch {epoch + 1}/{self.nr_epochs}.")
			for i, (images, labels, img_id) in enumerate(training_dataloader):
				if cropping:
					# First crop the image with the bbox.
					top_left, width, height = self.dl.bbox(img_id[0])
					images = torch.from_numpy(crop.Crop(images, top_left, width, height))

				# Move the cropped image to the configured device
				images = images.to(device=self.device, dtype=torch.float)
				labels = labels.to(device=self.device, dtype=torch.long)

				# Forward pass
				outputs = network(images)
				loss = criterion(outputs, labels)
				avg_loss += loss.item() / true_batch_size
				loss.backward()

				# Backward and optimize
				if (i + 1) % true_batch_size == 0:
					print(f"Applying optimization step {(i + 1) // true_batch_size}.")
					optimizer.step()
					optimizer.zero_grad()

				if (i + 1) % true_batch_size == 0:
					print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.nr_epochs, i + 1, total_step, avg_loss))
					avg_loss = 0

			# Code to update the lr
			#lr *= learning_rate_decay
			#self.update_lr(optimizer, lr)
			train_loss.append(loss.item())
			torch.save(network.state_dict(), f'{"crop" if cropping else "no_crop"}_classifier.chkp')
			with torch.no_grad():
				correct = 0
				total = 0
				for (images, labels, img_id) in validation_dataloader:
					if cropping:
						# First crop the image with the bbox.
						top_left, width, height = self.dl.bbox(img_id[0])
						images = torch.from_numpy(crop.Crop(images, top_left, width, height))

					# Move the cropped image to the configured device
					images = images.to(device=self.device, dtype=torch.float)
					labels = labels.to(device=self.device, dtype=torch.long)

					outputs = network(images)
					_, predicted = torch.max(outputs.data, 1)
					loss1 = criterion(outputs, labels)

					# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

				print('Validataion accuracy is: {} %'.format(100 * correct / total))
			validation_loss.append(loss1.item())

		if UI:
			print(train_loss)
			plt.plot(range(len(train_loss)), train_loss, label="training loss")
			plt.plot(range(len(validation_loss)), validation_loss, label="validation_loss")
			plt.legend(["Training loss", "Validation loss"])
			plt.show()

	def trainBirdSpotter(self):
		# TODO: train the bird spotter network.
		print("BirdSpotter + SPP")
		birdSpotter.HelloWorld()

	def TrainFullNet(self, use_SPP):
		if use_SPP:
			# TODO: train the full network with SPP layer.
			print("BirdSpotter + BirdClassifier + SPP")
		else:
			# TODO: train the full network without SPP layer.
			print("BirdSpotter + BirdClassifier + No SPP")

		network = fullNetwork.fullNetwork(use_SPP)
		learning_rate = 0.99e-3
		true_batch_size = 200
		learning_rate = learning_rate / true_batch_size
		learning_rate_decay = 0.99
		optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)

		network.to(self.device)
		criterion = nn.CrossEntropyLoss()
		training_dataloader = DataLoader(self.training_data, batch_size=1, shuffle=True)
		validation_dataloader = DataLoader(self.testing_data, batch_size=1, shuffle=True)

		total_step = len(training_dataloader)
		train_loss = []
		validation_loss = []
		avg_loss = 0
			
		optimizer.zero_grad()
		torch.save(network.state_dict(), f'full_net_{"SSP" if use_SSP else "no_SSP"}_classifier.chkp')
		for epoch in range(self.nr_epochs):
			print(f"Starting epoch {epoch + 1}/{self.nr_epochs}.")
			for i, (images, labels, img_id) in enumerate(training_dataloader):
				if cropping:
					# First crop the image with the bbox.
					top_left, width, height = self.dl.bbox(img_id[0])
					images = torch.from_numpy(crop.Crop(images, top_left, width, height))

				# Move the cropped image to the configured device
				images = images.to(device=self.device, dtype=torch.float)
				labels = labels.to(device=self.device, dtype=torch.long)

				# Forward pass
				outputs = network(images)
				loss = criterion(outputs, labels)
				avg_loss += loss.item() / true_batch_size
				loss.backward()

				# Backward and optimize
				if (i + 1) % true_batch_size == 0:
					print(f"Applying optimization step {(i + 1) // true_batch_size}.")
					optimizer.step()
					optimizer.zero_grad()

				if (i + 1) % true_batch_size == 0:
					print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.nr_epochs, i + 1, total_step, avg_loss))
					avg_loss = 0

			# Code to update the lr
			#lr *= learning_rate_decay
			#self.update_lr(optimizer, lr)
			train_loss.append(loss.item())
			torch.save(network.state_dict(), f'full_net_{"SSP" if use_SSP else "no_SSP"}_classifier.chkp')
			with torch.no_grad():
				correct = 0
				total = 0
				for (images, labels, img_id) in validation_dataloader:
					if cropping:
						# First crop the image with the bbox.
						top_left, width, height = self.dl.bbox(img_id[0])
						images = torch.from_numpy(crop.Crop(images, top_left, width, height))

					# Move the cropped image to the configured device
					images = images.to(device=self.device, dtype=torch.float)
					labels = labels.to(device=self.device, dtype=torch.long)

					outputs = network(images)
					_, predicted = torch.max(outputs.data, 1)
					loss1 = criterion(outputs, labels)

					# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

				print('Validataion accuracy is: {} %'.format(100 * correct / total))
			validation_loss.append(loss1.item())

		if UI:
			print(train_loss)
			plt.plot(range(len(train_loss)), train_loss, label="training loss")
			plt.plot(range(len(validation_loss)), validation_loss, label="validation_loss")
			plt.legend(["Training loss", "Validation loss"])
			plt.show()

net_to_train = int(sys.argv[1]) # Takes the parsed command line argument.
nr_epochs = int(sys.argv[2])
trainer = Trainer(net_to_train, nr_epochs=nr_epochs)
trainer.train()
