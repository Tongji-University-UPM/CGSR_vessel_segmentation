import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class SqueezeExcitation(torch.nn.Module):
	def __init__(self, input_channels, squeeze_channels):
	
		super().__init__()
		self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
		self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
		self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
		
		self.act = nn.ReLU(inplace=True)
		self.scale_activation = torch.nn.Sigmoid()

	def _scale(self, input):
		scale = self.avgpool(input)
		scale = self.fc1(scale)
		scale = self.act(scale)
		scale = self.fc2(scale)
		return self.scale_activation(scale)
	
	def forward(self, input):
			scale = self._scale(input)
			return scale * input
	

class Bottleneck(nn.Module):

	def __init__(self, in_filters, base_filters, expansion=4,
			  	 stride=1, downsample=False, projection=True,
				 se_module=True):
		
		super().__init__()

		self.downsample = downsample
		self.projection = projection
		self.se_module = se_module

		if downsample:
			stride=2

		self.conv_I_0 = nn.Conv2d(in_filters, base_filters, 1, stride=1, padding=0, bias=False)
		self.batch_norm_0 = nn.BatchNorm2d(base_filters, affine=True)

		self.conv_1 = nn.Conv2d(base_filters, base_filters, 3, stride=stride, padding=1, bias=False)
		self.batch_norm_1 = nn.BatchNorm2d(base_filters, affine=True)

		self.conv_I_1 = nn.Conv2d(base_filters, base_filters*expansion, 1, stride=1, padding=0, bias=False)
		self.batch_norm_2 = nn.BatchNorm2d(base_filters*expansion, affine=True)

		self.avg_pool_res = nn.AvgPool2d(2, stride=2)
		self.conv_I_res = nn.Conv2d(in_filters, base_filters*expansion, 1, stride=1, padding=0, bias=False)
		self.batch_norm_res = nn.BatchNorm2d(base_filters*expansion, affine=True)
		
		self.squeeze_excite = SqueezeExcitation(base_filters*expansion, base_filters)

		self.act = nn.LeakyReLU(0.1, inplace=True)


	def forward(self, X):

		x_i = X.clone()

		x = self.act(self.batch_norm_0(self.conv_I_0(X)))
		x = self.act(self.batch_norm_1(self.conv_1(x)))
		x = self.batch_norm_2(self.conv_I_1(x))

		if self.se_module:
			x = self.squeeze_excite(x)

		if self.downsample:
			x_i = self.avg_pool_res(x_i)
		
		if self.projection:
			x_i = self.batch_norm_res(self.conv_I_res(x_i))

		x += x_i

		return self.act(x)



class ResBlock(nn.Module):

	def __init__(self, in_filters, base_filters,
			  	 stride=1, downsample=False, projection=False):
		
		super().__init__()

		self.downsample = downsample
		self.projection = projection

		if downsample:
			stride=2

		self.conv_1 = nn.Conv2d(in_filters, base_filters, 3, stride=stride, padding=1, bias=False)
		self.batch_norm_1 = nn.BatchNorm2d(base_filters, affine=True)

		self.conv_2 = nn.Conv2d(base_filters, base_filters, 3, stride=1, padding=1, bias=False)
		self.batch_norm_2 = nn.BatchNorm2d(base_filters, affine=True)

		self.avg_pool_res = nn.AvgPool2d(2, stride=2)
		self.conv_I_res = nn.Conv2d(in_filters, base_filters, 1, stride=1, padding=0, bias=False)
		self.batch_norm_res = nn.BatchNorm2d(base_filters, affine=True)
		
		self.act = nn.LeakyReLU(0.1, inplace=True)


	def forward(self, X):

		x_i = X.clone()

		x = self.act(self.batch_norm_1(self.conv_1(X)))
		x = self.batch_norm_2(self.conv_2(x))

		if self.downsample:
			x_i = self.avg_pool_res(x_i)
		
		if self.projection:
			x_i = self.batch_norm_res(self.conv_I_res(x_i))

		x += x_i

		return self.act(x)


class UpsampleResBlock(nn.Module):

	def __init__(self, in_filters_enc, in_filters_dec, base_filters, scale=2):

		super().__init__()

		self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')

		self.conv_I_res = nn.Conv2d(in_filters_enc, in_filters_dec, 1, stride=1, padding=0, bias=False)
		self.batch_norm_res = nn.BatchNorm2d(in_filters_dec, affine=True)

		self.conv_1 = nn.Conv2d(in_filters_dec, base_filters, 3, stride=1, padding=1, bias=False)
		self.batch_norm_1 = nn.BatchNorm2d(base_filters, affine=True)

		self.conv_2 = nn.Conv2d(base_filters, base_filters, 3, stride=1, padding=1, bias=False)
		self.batch_norm_2 = nn.BatchNorm2d(base_filters, affine=True)
		self.act = nn.LeakyReLU(0.1, inplace=True)

		
	def forward(self, X_enc, X):

		x = self.upsample(X)
		x += self.batch_norm_res(self.conv_I_res(X_enc))

		x = self.act(x)

		x = self.act(self.batch_norm_1(self.conv_1(x)))
		return self.act(self.batch_norm_2(self.conv_2(x)))


class StemBlockRGB(nn.Module):

	def __init__(self, in_filters, out_filters):
		
		super().__init__()

		self.conv_block_0 = nn.Sequential(nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1, bias=False),
					  					  nn.BatchNorm2d(out_filters, affine=True),
					  					  nn.LeakyReLU(0.1, inplace=True))

		self.conv_block_1 = nn.Sequential(nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1, bias=False),
					  					  nn.BatchNorm2d(out_filters, affine=True),
					  					  nn.LeakyReLU(0.1, inplace=True))

		self.conv_block_2 = nn.Sequential(nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1, bias=False),
					  					  nn.BatchNorm2d(out_filters, affine=True),
					  					  nn.LeakyReLU(0.1, inplace=True))
		
	def forward(self, X):
		x = self.conv_block_0(X)
		x = self.conv_block_1(x)
		return self.conv_block_2(x)
	

class StemBlockHSI(nn.Module):

	def __init__(self, in_filters, out_filters, hidden_filters=256):
		
		super().__init__()

		self.conv_block_0 = nn.Sequential(nn.Conv2d(in_filters, hidden_filters, 3, stride=1, padding=1, bias=False),
					  					  nn.BatchNorm2d(hidden_filters, affine=True),
					  					  nn.LeakyReLU(0.1, inplace=True))

		self.conv_block_1 = nn.Sequential(nn.Conv2d(hidden_filters, hidden_filters, 3, stride=1, padding=1, bias=False),
					  					  nn.BatchNorm2d(hidden_filters, affine=True),
					  					  nn.LeakyReLU(0.1, inplace=True))

		self.conv_block_2 = nn.Sequential(nn.Conv2d(hidden_filters, out_filters, 3, stride=1, padding=1, bias=False),
					  					  nn.BatchNorm2d(out_filters, affine=True),
					  					  nn.LeakyReLU(0.1, inplace=True))
		
	def forward(self, X):
		x = self.conv_block_0(X)
		x = self.conv_block_1(x)
		return self.conv_block_2(x)


# class StemBlockHSI(nn.Module):

# 	def __init__(self, in_filters, out_filters, hidden_filters=256):
		
# 		super().__init__()

# 		self.res_block_0 = ResBlock(in_filters, hidden_filters, projection=True)
# 		self.res_block_1 = ResBlock(hidden_filters, hidden_filters, projection=True)
# 		self.res_block_2 = ResBlock(hidden_filters, out_filters, projection=True)

# 		# self.conv_block_0 = nn.Sequential(nn.Conv2d(hidden_filters, hidden_filters, 1, stride=1, padding=0, bias=False),
# 		# 			  					  nn.BatchNorm2d(hidden_filters, affine=True),
# 		# 			  					  nn.LeakyReLU(0.1, inplace=True))
		
# 		# self.conv_block_1 = nn.Sequential(nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0, bias=False),
# 		# 			  					  nn.BatchNorm2d(out_filters, affine=True),
# 		# 			  					  nn.LeakyReLU(0.1, inplace=True))
		
# 	def forward(self, X):
# 		x = self.res_block_0(X)
# 		x = self.res_block_1(x)
# 		# x = self.conv_block_0(x)
# 		return self.res_block_2(x)


class CLFBlock(nn.Module):

	def __init__(self, in_filters, base_filters):
		
		super().__init__()

		self.conv_I_0 = nn.Conv2d(in_filters, base_filters, 1, stride=1, padding=0, bias=False)
		self.batch_norm_0 = nn.BatchNorm2d(base_filters, affine=True)
		self.act_0 = nn.LeakyReLU(0.1, inplace=True)

		self.conv_I_1 = nn.Conv2d(base_filters, base_filters, 1, stride=1, padding=0, bias=False)
		self.batch_norm_1 = nn.BatchNorm2d(base_filters, affine=True)
		self.act_1 = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, X):

		x = self.act_0(self.batch_norm_0(self.conv_I_0(X)))
		return self.act_1(self.batch_norm_1(self.conv_I_1(x)))
	

class NormLayer(nn.Module):
	def __init__(self, dim=1, eps=1e-12):
		super(NormLayer, self).__init__()
		self.dim = dim
		self.eps = eps

	def forward(self, x):
		return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class SqueezeLayer(nn.Module):
	def __init__(self, *args):
		super(SqueezeLayer, self).__init__()
		self.args = args

	def forward(self, x):
		return x.squeeze((self.args))


class _ProjectionHead(nn.Module):

	def __init__(self, in_filters, base_filters):
		
		super().__init__()
		
		self.projection_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
								  		NormLayer(dim=1),
					  					nn.Conv2d(in_filters, base_filters, 1),
										nn.LeakyReLU(0.1),
					  					nn.Conv2d(base_filters, base_filters, 1),
										NormLayer(dim=1),
										SqueezeLayer(2,3))

	def forward(self, X):
		return self.projection_head(X)


class LinearHead(nn.Module):

	def __init__(self, in_filters, base_filters, outputs):
		
		super().__init__()
	
		self.linear_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
								   		 nn.Conv2d(in_filters, base_filters, 1),
										 nn.LeakyReLU(0.1),
										 nn.Conv2d(base_filters, outputs, 1),
										 SqueezeLayer(2,3))

	def forward(self, X):
		return self.linear_head(X)
	

class ProjectionHead(nn.Module):

	def __init__(self, in_filters, base_filters, outputs=1):
		
		super().__init__()
		
		self.projection_head = nn.Sequential(NormLayer(dim=1),
					  						 nn.Conv2d(in_filters, base_filters, 1),
											 nn.LeakyReLU(0.1),
					  						 nn.Conv2d(base_filters, base_filters, 1),
											 NormLayer(dim=1))
		
		self.clf_head = nn.Sequential(nn.Conv2d(in_filters, base_filters, 1, bias=False),
									  nn.BatchNorm2d(base_filters, affine=True),
									  nn.LeakyReLU(0.1),
									  nn.Conv2d(base_filters, base_filters, 1, bias=False),
									  nn.BatchNorm2d(base_filters, affine=True),
									  nn.LeakyReLU(0.1),
									  nn.Conv2d(base_filters, outputs, 1))

	def forward(self, X):

		proj_out = self.projection_head(X)
		clf_out = self.clf_head(X)
		
		return proj_out.squeeze(2,3), clf_out.squeeze(2,3)