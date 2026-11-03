"""
AUXILIAR IMAGE HOMOGENIZATION SCRIPT

Takes the desired input dataset and copies its content into the
required folder structure and ".png" file extension.

"""

import os
import re
import cv2
import h5py
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from utils import check_dirs

def main():
	
	ds_name = "DSA"
	ds_path = "/path/to/datasets_folder/"

	img_path = f"{ds_path}{ds_name}/training/images_jpg/"
	
	mask_path = f"{ds_path}{ds_name}/training/1st_manual/"
	
	out_img_path = f"{ds_path}{ds_name}/training/images/"
	out_mask_path = f"{ds_path}{ds_name}/training/masks/"
	out_label_path = f"{ds_path}{ds_name}/training/label_maps/"

	check_dirs(out_img_path, out_mask_path, out_label_path)

	if ds_name== "STARE":

		for fn in list(os.listdir(img_path)):
			
			print(f"---> {fn}")

			id_num = re.findall(r'\d+', fn)[0]

			fid = fn.split(".")[0]
			fext = fn.split(".")[-1]

			img = np.array(Image.open(f"{img_path}{fn}"), dtype=np.uint8)

			if len (img.shape)>2:
				img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

			if mask_path is not None:	
				for mfn in list(os.listdir(mask_path)):
					mid_num = re.findall(r'\d+', mfn)[0]
					
					if(id_num==mid_num):
						mask_img = np.array(Image.open(f"{mask_path}{mfn}"), dtype=np.uint8)			
				
				Image.fromarray(mask_img).save(f"{out_mask_path}{fid}.png")

			Image.fromarray(img).save(f"{out_img_path}{fid}.png")


	elif ds_name == "134XCA":

		for fn in list(os.listdir(img_path)):
			
			print(f"---> {fn}")

			id_num = re.findall(r'\d+', fn)[0]

			fid = fn.split(".")[0]
			fext = fn.split(".")[-1]

			img = np.array(Image.open(f"{img_path}{fn}"), dtype=np.uint8)

			if "gt" in fn:
				Image.fromarray(gt).save(f"{out_mask_path}{id_num}.png")

			else:
				Image.fromarray(img).save(f"{out_img_path}{id_num}.png")

	elif ds_name == "30XCA":
		
		f = h5py.File(img_path+"test1_ori.hdf5", "r")
		img_ds = f.get("image")[:]

		f = h5py.File(img_path+"test1_gt.hdf5", "r")
		gt_ds = f.get("image")[:]

		for i in range(img_ds.shape[0]):
			img = np.moveaxis(img_ds[i,0,...], 0,-1)
			gt = gt_ds[i,0,...]

			img = img[...,2]

			Image.fromarray(gt.astype('uint8')).save(f"{out_mask_path}{i}.png")
			Image.fromarray(img.astype('uint8')).save(f"{out_img_path}{i}.png")

	elif ds_name == "DSA":

		for fn in list(os.listdir(img_path)):
			print(fn)
			idp = fn.split(".")[0]
			img = np.array(Image.open(f"{img_path}{idp}.jpg"), dtype=np.uint8)
			mask = np.array(Image.open(f"{mask_path}{idp}.png"), dtype=np.uint8)

			mask = mask[...,0]
			img = img[...,0]

			label = mask/mask.max()
			label += 1
			
			Image.fromarray(mask.astype('uint8')).save(f"{out_mask_path}{idp}.png")
			Image.fromarray(img.astype('uint8')).save(f"{out_img_path}{idp}.png")
			np.save(f"{out_label_path}{idp}.npy", label)


	elif ds_name == "FP":

		for fn in list(os.listdir(img_path)):
			print(fn)
			idp = fn.split(".")[0]
			img = np.array(Image.open(f"{img_path}{idp}.png"), dtype=np.uint8)

			if len (img.shape)>2:
				img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

			print(img.shape)

			Image.fromarray(img.astype('uint8')).save(f"{out_img_path}{idp}.png")

	elif ds_name == "UWF":

		for fn in list(os.listdir(img_path)):
			print(fn)
			idp = fn.split(".")[0]
			img = np.array(Image.open(f"{img_path}{idp}.png"), dtype=np.uint8)

			if len (img.shape)>2:
				img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

			print(img.shape)

			# img = img[...,0]

			Image.fromarray(img.astype('uint8')).save(f"{out_img_path}{idp}.png")



if __name__ == "__main__":
	main()