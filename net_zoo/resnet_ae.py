import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .layers import (StemBlockRGB, StemBlockHSI, ResBlock, 
					 UpsampleResBlock, CLFBlock, ProjectionHead,
					 LinearHead, Bottleneck)


class ResNet(nn.Module):

	def __init__(self, in_chns=1, outputs=1):
		
		super().__init__()

		torch.manual_seed(0)

		# stem_block_hsi = StemBlockHSI(25,3, hidden_filters=128)
		# stem_block_hsi = StemBlockHSI(25,3, hidden_filters=128)
		stem_gray = StemBlockRGB(in_chns, 64)

		stage_1 = nn.Sequential(ResBlock(64,128, downsample=True, projection=True),
								ResBlock(128,256, projection=True),
								ResBlock(256,256),
								ResBlock(256,256))
		
		stage_2 = nn.Sequential(ResBlock(256,512, downsample=True, projection=True),
								ResBlock(512,512),
								ResBlock(512,512),
								ResBlock(512,512),
								ResBlock(512,512),
								ResBlock(512,512))

		stage_3 = nn.Sequential(ResBlock(512,1024, downsample=True, projection=True),
								ResBlock(1024,1024),
								ResBlock(1024,1024),
								ResBlock(1024,1024))
		
		upconv_3 = UpsampleResBlock(512,1024,256)
		upconv_2 = UpsampleResBlock(256,256,256)
		upconv_1 = UpsampleResBlock(64,256,256)
		upconv_0 = UpsampleResBlock(1,256,256)
		dec_proj_head = ProjectionHead(256, 128, outputs=outputs)
		
		clf = CLFBlock(256, 1024)
		output_head = nn.Conv2d(1024, outputs, 1, stride=1, padding=0, bias=True)

		
		# self.params = nn.ModuleDict({
		# 	'stem_gray': stem_gray,
		# 	'encoder': nn.ModuleDict({"stage_1": stage_1,
		# 							  "stage_2": stage_2,
		# 							  "stage_3": stage_3,
		# 							}),
			
		# 	'dec_proj_head': dec_proj_head,
		# 	# 'linear_head': lin_head,

		# 	'decoder': nn.ModuleDict({"upconv_0": upconv_0,
		# 							  "upconv_1": upconv_1,
		# 							  "upconv_2": upconv_2,
		# 							  "upconv_3": upconv_3,
		# 							  "clf": clf,
		# 							  "output_head": output_head
		# 							})
		# }) 
			
		self.stem_gray = stem_gray
		self.encoder = nn.ModuleDict({"stage_1": stage_1,
								  "stage_2": stage_2,
								  "stage_3": stage_3,
								})
		
		self.dec_proj_head = dec_proj_head
		
		self.decoder = nn.ModuleDict({"upconv_0": upconv_0,
								  	  "upconv_1": upconv_1,
								  	  "upconv_2": upconv_2,
								  	  "upconv_3": upconv_3,
								  	  "clf": clf,
								  	  "output_head": output_head
									})
									
		self.initialize_weights()
		

	def initialize_weights(self):

		for m in self.modules():

			if isinstance(m, nn.Conv2d):
				# Initializing Conv2d layers with Kaiming Normal initialization
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				# Initializing BatchNorm2d layers with a small positive constant
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				# Initializing Linear layers with Kaiming Normal initialization
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)


	def check_gradients(self):

		means = .0
		steps = 0
		for param in self.params.encoder.parameters():

			if param.grad is not None:
				means += param.grad.data.mean()
				steps += 1
		
		print(means/steps)


	def forward(self, X, modality="segmentation"):
		
		x = self.stem_gray(X)
		c_1 = self.encoder.stage_1(x)
		c_2 = self.encoder.stage_2(c_1)
		c_3 = self.encoder.stage_3(c_2)
		
		if modality=="segmentation":
			p_3 = self.decoder.upconv_3(c_2, c_3)
			p_2 = self.decoder.upconv_2(c_1, p_3)
			p_1 = self.decoder.upconv_1(x, p_2)
			p_0 = self.decoder.upconv_0(X, p_1)
			p_0 = self.decoder.clf(p_0)
			x_out = self.decoder.output_head(p_0)

		elif modality=="output_dec-embed":
			p_3 = self.decoder.upconv_3(c_2, c_3)
			p_2 = self.decoder.upconv_2(c_1, p_3)
			p_1 = self.decoder.upconv_1(x, p_2)
			p_0 = self.decoder.upconv_0(X, p_1)
			x_out = self.dec_proj_head(p_0)

		else:
			raise ValueError(f"Modality: {modality} not defined.")

		return x_out
		



# class ResNetSE(nn.Module):

# 	def __init__(self):
		
# 		super().__init__()

# 		torch.manual_seed(0)

# 		stem_block_hsi = StemBlockHSI(25,3, hidden_filters=128)
# 		stem_block_rgb = StemBlockRGB(3,64)

# 		stage_1 = nn.Sequential(Bottleneck(64,128, downsample=True, projection=True, expansion=2),
# 									 Bottleneck(256,64, projection=True),
# 									 Bottleneck(256,64),
# 									 Bottleneck(256,64))
		
# 		stage_2 = nn.Sequential(Bottleneck(256,128, downsample=True, projection=True),
# 									 Bottleneck(512,128),
# 									 Bottleneck(512,128),
# 									 Bottleneck(512,128),
# 									 Bottleneck(512,128),
# 									 Bottleneck(512,128))

# 		stage_3 = nn.Sequential(Bottleneck(512, 256, downsample=True, projection=True),
# 									 Bottleneck(1024,256),
# 									 Bottleneck(1024,256),
# 									 Bottleneck(1024,256))
# 		proj_head = ProjectionHead(1024, 256)
# 		lin_head = LinearHead(1024, 256, 1)

# 		upconv_3 = UpsampleResBlock(512,1024,256)
# 		upconv_2 = UpsampleResBlock(256,256,256)
# 		upconv_1 = UpsampleResBlock(64,256,256)
# 		upconv_0 = UpsampleResBlock(3,256,256, scale=2.09)
# 		clf = CLFBlock(256, 1024)
# 		output_head = nn.Conv2d(1024, 2, 1, stride=1, padding=0, bias=True)

# 		self.params = nn.ModuleDict({
# 			'stem_rgb': stem_block_rgb,
# 			'stem_hsi': stem_block_hsi,
# 			'encoder': nn.ModuleDict({"stage_1": stage_1,
# 									  "stage_2": stage_2,
# 									  "stage_3": stage_3,
# 									}),
			
# 			'proj_head': proj_head,
# 			'linear_head': lin_head,

# 			'decoder': nn.ModuleDict({"upconv_0": upconv_0,
# 									  "upconv_1": upconv_1,
# 									  "upconv_2": upconv_2,
# 									  "upconv_3": upconv_3,
# 									  "clf": clf,
# 									  "output_head": output_head
# 									})
# 		}) 
									
# 		self.initialize_weights()
		

# 	def initialize_weights(self):

# 		for m in self.modules():

# 			if isinstance(m, nn.Conv2d):
# 				# Initializing Conv2d layers with Kaiming Normal initialization
# 				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0)
# 			elif isinstance(m, nn.BatchNorm2d):
# 				# Initializing BatchNorm2d layers with a small positive constant
# 				nn.init.constant_(m.weight, 1)
# 				nn.init.constant_(m.bias, 0)
# 			elif isinstance(m, nn.Linear):
# 				# Initializing Linear layers with Kaiming Normal initialization
# 				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0)


# 	def freeze_layers(self, layers):

# 		for layer in layers:
# 			for name, param in self.params[layer].named_parameters():
# 				# self.params[layer].requires_grad_(False)
# 				param.requires_grad = False
# 				param.grad = None
# 				self.params[layer].training = False
# 				self.params[layer].eval()


# 	def unfreeze_layers(self, layers):

# 		for layer in layers:
# 			for name, param in self.params[layer].named_parameters():
# 				# self.params[layer].requires_grad_(False)
# 				param.requires_grad = True
# 				self.params[layer].training = True
# 				self.params[layer].train()


# 	def check_status(self):

# 		print(self.params.decoder.clf.batch_norm_0.running_mean)
# 		print("**********")

# 		return
# 		for layer in self.params.keys():
# 			for name, param in self.params[layer].named_parameters():
				
# 				# if "stem_rgb.conv_block_0.0.weight" in f"{layer}.{name}":
# 				# 	print(param)
				
# 				print(f"{layer}.{name}, REQ GRAD:{param.requires_grad}, GRAD NONE:{param.grad is None}")
		

# 	def check_gradients(self):

# 		means = .0
# 		steps = 0
# 		for param in self.params.encoder.parameters():

# 			if param.grad is not None:
# 				means += param.grad.data.mean()
# 				steps += 1
		
# 		print(means/steps)


# 	def forward(self, X, source="rgb", modality="segmentation"):
		
# 		if source=="hsi":
# 			X = self.params.stem_hsi(X)
		
# 		x = self.params.stem_rgb(X)	
# 		c_1 = self.params.encoder.stage_1(x)
# 		c_2 = self.params.encoder.stage_2(c_1)
# 		c_3 = self.params.encoder.stage_3(c_2)
		
# 		if modality=="pre-train_encoder":
# 			x_out = self.params.proj_head(c_3), self.params.linear_head(c_3)
		
# 		elif modality=="segmentation":
# 			p_3 = self.params.decoder.upconv_3(c_2, c_3)
# 			p_2 = self.params.decoder.upconv_2(c_1, p_3)
# 			p_1 = self.params.decoder.upconv_1(x, p_2)
# 			p_0 = self.params.decoder.upconv_0(X, p_1)
# 			p_0 = self.params.decoder.clf(p_0)
# 			x_out = self.params.decoder.output_head(p_0)

# 		else:
# 			raise ValueError(f"Modality: {modality} not defined.")

# 		return x_out