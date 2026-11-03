import re
import os
import cv2
import sys
import copy
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

from random import choice

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from augmentations import (TransformationSet, aug_dict, Bypass)


def get_data_names(data_path, img_folder,
				   plabel_folder, ds_name,
				   pctg=1., min_score=0):
	
	img_path = f"{data_path}{ds_name}/{img_folder}" 
	plabel_path = f"{data_path}{ds_name}/{plabel_folder}"

	fn_list = []

	files_set = list(os.listdir(img_path))
	plabels_set = list(os.listdir(plabel_path))
	subset_len = int(round(len(files_set)*pctg))
	files_subset = files_set[:subset_len]

	for fn in files_subset:

		for pl_elem in plabels_set:
			if fn.split(".")[0] in pl_elem:
				pfn = pl_elem

		if min_score>0:
			score = int(re.findall(r'\d+', pfn)[-1])/100

			if (score<min_score or score>1):
				continue
			
		img_fn = img_path+fn.split(".")[0]
		pl_fn = plabel_path+pfn.split(".")[0]

		fn_list.append((img_fn, pl_fn))

	return fn_list



def custom_collate(batch):

	elem = batch[0]
	clone = copy.copy(elem)

	for key in elem:

		stacked_elems = []
		for d in batch:

			stacked_elems.append(d[key])

		if torch.is_tensor(d[key]):
			stacked_elems = torch.stack(stacked_elems, 0)

		clone.update({key: stacked_elems})

	return clone



class ImageDataset(data.Dataset):

	def __init__(self, ds_dict,
			  	 central_window=7, bit_res=8,
				 resize=-1,
				 equivariant_trf=False,
				 fill_pl_blanks=False,
				 eq_img_hist = True,
				 augmentations = [],
				 plabel_ext="npy"):

		super(ImageDataset, self).__init__()

		self.ds_names = list(ds_dict.keys())

		self.ds_lens = []
		for name in self.ds_names:
			self.ds_lens.append(len(ds_dict[name]))
		
		self.plabel_ext = plabel_ext
		self.resize = resize
		
		self.trf_set = TransformationSet()
		self.equivariant_trf = equivariant_trf

		self.central_window = central_window
		self.bit_res = bit_res
		self.fill_pl_blanks = fill_pl_blanks
		self.eq_img_hist = eq_img_hist
		self.augmentations = augmentations

		self.clahe = cv2.createCLAHE(clipLimit=2)

		self.img_list = []
		self.plabel_list = []
		self.idp_list = []
		self.ds_list = []
		for name in self.ds_names:
			sel_dict = ds_dict[name]

			self.img_list.extend([idp[0] for idp in sel_dict])
			self.plabel_list.extend([idp[1] for idp in sel_dict])
			self.idp_list.extend([idp[0].split("/")[-1] for idp in sel_dict])
			self.ds_list.extend([name]*len(sel_dict))

	def __getitem__(self, index):


		img_path = self.img_list[index]
		plabel_path = self.plabel_list[index]
		idp = self.idp_list[index]
		ds_name = self.ds_list[index]
		
		img = np.array(Image.open(f"{img_path}.png"), dtype=np.uint8)
		
		if self.plabel_ext == "npy":
			plabel = np.load(f"{plabel_path}.npy")

		elif self.plabel_ext == "png":
			plabel = np.array(Image.open(f"{plabel_path}.png"), dtype=np.uint8)


		if self.eq_img_hist:
			img = self.clahe.apply(img)

		img = img/((2**self.bit_res)-1)
		img = torch.from_numpy(img).float()
		plabel = torch.from_numpy(plabel).float()

		img_size = img.shape
		
		if self.resize>0:
			
			img = transforms.functional.resize(img[None,...], [self.resize, self.resize],
									transforms.InterpolationMode.BILINEAR, None, True)[0,...]
			
			plabel = transforms.functional.resize(plabel[None,...], [self.resize, self.resize],
									transforms.InterpolationMode.NEAREST, None, True)[0,...]
			
			img_size = self.resize


		# composed_transform = transforms.Compose([RandomResizedCrop(img_size),
		# 										 RandomZoomOut(),
		# 										 RandomImageDistort()])

		if not self.augmentations:
			composed_transform = Bypass()
		else:
			compose_list = []
			for mod in self.augmentations:
				compose_list.append(aug_dict[mod](size=img_size))
			
			composed_transform = transforms.Compose(compose_list)
		
		# composed_transform = transforms.Compose([RandomImageDistort()])
				

		data_dict = {"img":img[None,...], "plabel":plabel,
			   		 "ds_name":ds_name, "idp":idp}

		if self.equivariant_trf:
			
			transform_fn = self.trf_set()

			data_dict["plabel"] = data_dict["plabel"][None,...]
			data_dict = composed_transform(data_dict)

			data_dict["img_trf"] = transform_fn(data_dict["img"])
			data_dict["plabel_trf"] = transform_fn(data_dict["plabel"]).squeeze(0)
			data_dict["plabel"] = data_dict["plabel"].squeeze(0)
			data_dict["transform_fn"] = transform_fn

		if self.fill_pl_blanks:
			data_dict["plabel"][data_dict["plabel"]==0] = 2
			
			if self.equivariant_trf:
				data_dict["plabel_trf"][data_dict["plabel_trf"]==0] = 2
			

		return data_dict

	def __len__(self):

		return len(self.img_list)
	
	