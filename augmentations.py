import cv2
import random
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
# import kornia.filters as kf

from random import choice

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


def random_patch(data_dict, w_size, n_patches=4):
	
	img = data_dict["img"]
	plabel = data_dict["plabel"]

	
	pos_ixs = torch.nonzero(plabel[:-w_size, :-w_size], as_tuple=False)

	crop_img_list = []
	crop_plab_list = []
	for _ in range(n_patches):
	
		if len(pos_ixs) == 0:
			raise ValueError("No annotated points found in the mask.")

		y, x = pos_ixs[random.randint(0, len(pos_ixs) - 1)]

		crop_img_list.append(TF.crop(img, y, x, w_size, w_size))
		crop_plab_list.append(TF.crop(plabel, y, x, w_size, w_size))
		
	data_dict["img"] = torch.stack(crop_img_list, 0)
	data_dict["plabel"] = torch.stack(crop_plab_list, 0)

	return data_dict


def distort_color(img, s=.5):
	color_jitter = transforms.ColorJitter(brightness=0.1, contrast=.3, saturation=0.1, hue=0.3)
	return color_jitter(img)

def apply_blur(img, kernel_max=7, sigma_min=0.1, sigma_max=2.0):
	kernel_size = 2*(np.random.randint(1,kernel_max)//2)+1
	sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()
	return TF.gaussian_blur(img, kernel_size, [sigma, sigma])
	

# def apply_blur(img, kernel_max=11, sigma_min=0.1, sigma_max=2.0):
# 	kernel_size = 2*(np.random.randint(3,kernel_max)//2)+1
# 	sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()
# 	return kf.box_blur(img.unsqueeze(0), [kernel_size,kernel_size]).squeeze(0)
	

def apply_noise(img, PSNR_rng=[18,25]):
	PSNR = torch.randint(*PSNR_rng, (1,))
	N = torch.randn(img.size())
	As = torch.max(img)
	MSE = (As**2)/10**(PSNR/10)
	noise_fact = torch.sqrt((MSE/(torch.square(N).mean())))
	return torch.clamp(img + noise_fact*N, min=0, max=1)


""" FLIPS """

class RandomFlips_gen(object):

	def __init__(self):
		self.trf_name = ""

	def __call__(self):
		
		if random.random()<.5:
			trf_fn = TF.hflip
			self.trf_name = "Img_hflip"
			
		else:
			trf_fn = TF.vflip
			self.trf_name = "Img_vflip"
			
		return trf_fn

	def __str__(self):
		return self.trf_name

""" ROTATIONS """

class Rotation(object):

	def __init__(self, angle):
		self.angle = angle

	def __call__(self, input):
		return TF.rotate(input, self.angle)
	
	def __str__(self):
		return f"Rotation_{self.angle}deg"

class RandRotation_gen(object):

	def __init__(self):
		self.angles = [90, 180, 270]
		self.fn_angle = 0

	def __call__(self):
		self.fn_angle = choice(self.angles)
		return Rotation(self.fn_angle)



class TransformationSet(object):
	
	def __init__(self):
		
		self.rand_rotation = RandRotation_gen()
		self.rand_flips = RandomFlips_gen()

	def __call__(self):

		rand_rot_fn = self.rand_rotation()
		rand_flip_fn = self.rand_flips()

		# transforms = [rand_flip_fn, rand_rot_fn, rand_resize_crop_fn]
		transform_list = [rand_flip_fn, rand_rot_fn]

		return choice(transform_list)



class RandomResizedCrop(object):

	def __init__(self, size=512, scale=(0.7, 1.0), ratio=(0.56, 1.77), p=.5):
		
		self.size = size
		self.scale = scale
		self.ratio = ratio
		self.p = p

	def __call__(self, data_dict):

		img = data_dict["img"]
		plabel = data_dict["plabel"]

		if np.random.rand()<= self.p:
			size = torch.squeeze(img).shape

			params = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)

			img = TF.resized_crop(img, *params, size, antialias=True,
								interpolation=transforms.InterpolationMode.BILINEAR)
			
			plabel = TF.resized_crop(plabel, *params, size, antialias=False,
									interpolation=transforms.InterpolationMode.NEAREST)
									
		return {"img": img, "plabel": plabel}


class RandomZoomOut(object):

	def __init__(self, size=512, scale=(1, 1.4), p=.5):
		
		self.scale = scale
		self.p = p

	def __call__(self, data_dict):

		img = data_dict["img"]
		plabel = data_dict["plabel"]

		size = torch.squeeze(img).shape

		zoom_out_trf = v2.RandomZoomOut(0, self.scale, p=self.p)

		params = zoom_out_trf.make_params(img)

		if np.random.rand()<= self.p:
			""" FOR NEWER PYTORCH VERSION"""
			img = zoom_out_trf.transform(img, params)
			plabel = zoom_out_trf.transform(plabel, params)
			
			""" FOR OLDER PYTORCH VERSION"""
			# img = zoom_out_trf._transform(img, zoom_out_trf._get_params(img))
			# plabel = zoom_out_trf._transform(plabel, zoom_out_trf._get_params(plabel))
			
			img = transforms.functional.resize(img, size,
								transforms.InterpolationMode.BILINEAR, None, True)
		
			plabel = transforms.functional.resize(plabel, size,
											transforms.InterpolationMode.NEAREST, None, True)
		
		return {"img": img, "plabel": plabel}

class RandomImageDistort(object):

	def __init__(self, size=512, p=.5):
		
		self.p = p

	def __call__(self, data_dict):

		img = data_dict["img"]
		plabel = data_dict["plabel"]

		color_jitter = transforms.ColorJitter(brightness=0.5, contrast=[.3,.9],
											  saturation=0.1, hue=0.3)
		
		if np.random.rand()<= self.p:
			img = color_jitter(img)

		if np.random.rand()<= self.p:
			img = apply_noise(img)
		
		if np.random.rand()<= self.p:
			img = apply_blur(img)
		
		return {"img": img, "plabel": plabel}


class RandomImageResize(object):
	def __init__(self, p=.5, size=384):
		self.p = p

		self.rand_res_crop = RandomResizedCrop(size, p=0.7, scale=(0.5, 1.0))
		self.rand_zoomout = RandomZoomOut(p=0.7)


	def __call__(self, data_dict):

		if np.random.rand()<= self.p:
			data_dict_trf = self.rand_res_crop(data_dict)
		else:
			data_dict_trf = self.rand_zoomout(data_dict)
		
		return data_dict_trf


class Bypass(object):
	def __init__(self, p=None, size=None):
		pass
	def __call__(self, data_dict):
		return data_dict



aug_dict = {"img_distort": RandomImageDistort,
			"rand_crop": RandomResizedCrop,
			"rand_zoomout": RandomZoomOut,
			"rand_resize": RandomImageResize,
			"bypass": Bypass}


distort_transform = transforms.Compose([transforms.RandomApply([apply_blur], p=.5),
										transforms.RandomApply([apply_noise], p=.5),
										transforms.RandomApply([distort_color], p=.5),
									])



	

# patch_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
#                                     transforms.RandomVerticalFlip(p=0.5),
#                                     transforms.RandomApply([distort_color], p=0.5),
#                                     transforms.RandomApply([apply_blur], p=0.5),
#                                     transforms.RandomApply([apply_noise], p=.5)
#                                     ])


# distort_transform = transforms.Compose([transforms.RandomApply([distort_color], p=0.5),
#                                         transforms.RandomApply([apply_blur], p=0.5),
#                                         transforms.RandomApply([apply_noise], p=.5)
#                                     ])

distort_transform = transforms.Compose([transforms.RandomApply([apply_blur], p=.5),
										transforms.RandomApply([apply_noise], p=.5),
										transforms.RandomApply([distort_color], p=.5),
									])


patch_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
									  transforms.RandomVerticalFlip(p=0.5)])

if __name__ == "__main__":

	from PIL import Image
	from os import listdir
	import matplotlib
	matplotlib.use('TKAgg')
	import matplotlib.pyplot as plt

	resize = 384

	idp = "122"
	img_path = "/data/local/guillermo.vazquez/Data/Tongji/134XCA/test/images/"

	img = np.array(Image.open(f"{img_path}{idp}.png"), dtype=np.uint8)

	img_rs_xca = transforms.functional.resize(torch.from_numpy(img[None,...]), [resize, resize],
									transforms.InterpolationMode.BILINEAR, None, True)[0,...]
	


	idp = "003_PPA_-29_PSA_29_2"
	img_path = "/data/local/guillermo.vazquez/Data/Tongji/XCAD/training/trainB/"

	img = np.array(Image.open(f"{img_path}{idp}.png"), dtype=np.uint8)
	plabel = np.load(f"/data/local/guillermo.vazquez/Data/Tongji/XCAD/training/plabel_maps/003_PPA_-29_PSA_29_2_S:63.npy")

	img_rs = transforms.functional.resize(torch.from_numpy(img[None,...]), [resize, resize],
									transforms.InterpolationMode.BILINEAR, None, True)[0,...]

	
	
	
	clahe = cv2.createCLAHE(clipLimit=2)
	img = clahe.apply(img)
	img = torch.from_numpy(img).float()[None,...]/255
	plabel = torch.from_numpy(plabel).long()[None,...]

	img_rs = transforms.functional.resize(img[None,...], [resize, resize],
									transforms.InterpolationMode.BILINEAR, None, True)[0,...]
	
	plabel_rs = transforms.functional.resize(plabel[None,...], [resize, resize],
									transforms.InterpolationMode.NEAREST, None, True)[0,...]
	
	data_dict = {"img": img_rs, "plabel":plabel_rs}

	composed_transform = transforms.Compose([RandomImageResize(),
											 RandomImageDistort()])
											
	
	img_trf = RandomResizedCrop(p=100)

	
	for _ in range(10):

		color_jitter = transforms.ColorJitter(brightness=0.5, contrast=[.3,.9],
											  saturation=0.1, hue=0.3)

		img_trf = color_jitter(img_rs)

		plt.figure()
		plt.imshow(img_trf.numpy().squeeze(), vmin=0, vmax=1)
		plt.show()
		continue

		data_dict_trf = composed_transform(data_dict)

		# data_dict_trf = img_trf(data_dict)


		# plt.figure()
		# plt.imshow(img.numpy().squeeze(), vmin=0, vmax=1)


		plt.figure()
		plt.imshow(img_rs_xca.squeeze().numpy(), vmin=0, vmax=255)

		plt.figure()
		plt.imshow(data_dict_trf["img"].numpy().squeeze(), vmin=0, vmax=1)

		plt.figure()
		plt.imshow(data_dict_trf["plabel"].numpy().squeeze())

	
		plt.figure()
		plt.imshow(data_dict["img"].numpy().squeeze(), vmin=0, vmax=1)

		plt.figure()
		plt.imshow(data_dict["plabel"].numpy().squeeze())

		plt.show()

