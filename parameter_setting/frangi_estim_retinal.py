import os
import re
import sys
import cv2
import json
import random
import numpy as np
from PIL import Image


import cupy as cp

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from parser import parse_in_args
from utils import check_dirs, dilate, erode
from segmentation import get_largest_contour, apply_frangi, filter_range
from filtering import apply_prewitt
from img_processing import get_clusters


def get_temp_masks(data_path, id_list,
				   gamma, f_ext="png"):
	
	for idp in id_list:

		clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

		pos_img = np.array(Image.open(f"{data_path}{idp}"), dtype=np.uint8)
		# pos_img_eq = cv2.equalizeHist(pos_img)
		pos_img_eq = cv2.equalizeHist(cv2.medianBlur(pos_img, 5))
		pos_img_clahe_filt = clahe.apply(cv2.medianBlur(pos_img, 5))
		
		gradients = apply_prewitt(pos_img_clahe_filt)
		mask_img = 1*(filter_range(gradients, f=1)>0)
		mask_img = dilate(erode(mask_img.astype('uint8')))

		frangi_map = apply_frangi(pos_img_eq, gamma=gamma)
		frangi_map_ = apply_frangi(pos_img_clahe_filt, gamma=gamma)
		comb_mask = np.maximum(frangi_map, mask_img).astype('uint8')
		comb_mask_ = np.maximum(frangi_map_, mask_img).astype('uint8')


		plt.figure()
		plt.imshow(comb_mask)

		plt.figure()
		plt.imshow(comb_mask_)
		plt.show()



def main(args):

	ds_name = args.ds_name
	params_fn = f"./params_files/{ds_name}_params.json"
	storage_path = f"/data/local/guillermo.vazquez/Data/Tongji/{ds_name}/"
	
	pos_train_img_path =  f"{storage_path}training/images/"
	temp_masks_path = f"{storage_path}training/temp_masks/"

	f_ext = "png"

	# try:
	# 	params_file_r = open(params_fn, "r")
	# 	params_dict = json.load(params_file_r)
	# except FileNotFoundError:
	# 	print("Params file not found. Run noise estimator first.")
	# 	exit()
	# except json.decoder.JSONDecodeError:
	# 	print("Error parsing params file. Check file content.")
	# 	exit()

	# try:
	# 	k_blr = params_dict["k_blr"]
	# 	f_blr = params_dict["f_blr"]
	# except KeyError:
	# 	print("'k_blr' and 'f_blr' not found in params file. Run noise estimator first.")
	# 	exit()

	check_dirs(temp_masks_path)

	ds_pctg = args.ds_pctg

	file_list = os.listdir(pos_train_img_path)
	crop_len = int(ds_pctg*len(file_list))
	
	random.seed(0)
	random.shuffle(file_list)

	# get_temp_masks(pos_train_img_path, file_list, 9.0)

	top_gammas_0 = []
	top_gammas_1 = []

	img_list, mask_list, ret_list = [], [], []

	params_dict = {}
	check_dirs("./params_files/")
	params_file = open(params_fn, "w")


	for fn in file_list[:crop_len]:
		
		idp = fn.split(".")[0]
		print(f"\n---> {idp}")

		try:
			# pos_img = np.array(Image.open(f"{pos_train_img_path}{idp}.{f_ext}"), dtype=np.uint8)
			pos_img = np.array(Image.open(f"{pos_train_img_path}{fn}"), dtype=np.uint8)
			# pos_img = cv2.cvtColor(pos_img, cv2.COLOR_RGB2GRAY)
			# neg_img = np.array(Image.open(f"{neg_train_img_path}{idp}.png"), dtype=np.uint8)
		
		except FileNotFoundError:
			print(f"ID {idp} not found.")
			exit()

		# if pos_img.shape[0] > max_size:
		# 	continue


		clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

		pos_img_eq = cv2.equalizeHist(pos_img)
		
		# pos_img_clahe_filt = cv2.bilateralFilter(pos_img_clahe, k_blr, k_blr, k_blr*f_blr)
		pos_img_clahe_filt = clahe.apply(cv2.medianBlur(pos_img, 5))
		
		_, ret_mask = get_clusters(pos_img_clahe_filt, k=4)
		ret_mask = get_largest_contour(ret_mask)


		pos_img_clahe_filt[ret_mask==1] = 255

		# scores, klist = test_clusters_score(cv2.bilateralFilter(pos_img_clahe_filt, k_blr, k_blr * 2, k_blr / 2), cl=24, ch=256, step=8)
		
		# plt.plot(range(24, 256, 8), scores)
		# plt.show()

		# _, bin_mask = get_clusters(pos_img_clahe_filt, k=8)

		gradients = apply_prewitt(pos_img_clahe_filt)

		bin_mask_0 = 1*(filter_range(gradients, f=1)>0)
		bin_mask_1 = 1*(filter_range(gradients, f=0)>0)

		bin_mask_0 = dilate(erode(bin_mask_0.astype('uint8')))
		bin_mask_1 = dilate(erode(bin_mask_1.astype('uint8')))

		dsc_0_list = []
		dsc_1_list = []

		img_list.append(pos_img_eq)
		mask_list.append(bin_mask_0)
		ret_list.append(ret_mask)

		for g_test in range(3,19,3):
						
			frangi_map = apply_frangi(pos_img_eq, gamma=g_test)
			dsc_0 = 2*(bin_mask_0*frangi_map).sum()/(bin_mask_0.sum() + frangi_map.sum())
			dsc_1 = 2*(bin_mask_1*frangi_map).sum()/(bin_mask_1.sum() + frangi_map.sum())

			# conn = mask_connectivity(frangi_map)
			
			dsc_0_list.append(dsc_0)
			dsc_1_list.append(dsc_1)

		top_gamma_0 = list(range(3,19,3))[np.argmax(dsc_0_list)]
		top_gamma_1 = list(range(3,19,3))[np.argmax(dsc_1_list)]
		top_gammas_0.append(top_gamma_0)
		top_gammas_1.append(top_gamma_1)


	""" TEMP MASK SAVING"""
	gamma = float(np.mean(top_gammas_0))
	params_dict["gamma"] = gamma

	for idp, img, mask_img, ret_img in zip(file_list, img_list, mask_list, ret_list):

		frangi_map = apply_frangi(img, gamma=gamma)

		ret_img = dilate(ret_img.astype('uint8'), k=5)
		# mask_img = filter_mask(mask_img)
		comb_mask = np.maximum(frangi_map, mask_img)
		comb_mask = (comb_mask*(1-ret_img)).astype('uint8')

		Image.fromarray(comb_mask).save(f"{temp_masks_path}{idp}")
		print(f"Saved {idp}")

	
	json.dump(params_dict, params_file, indent=4)
	params_file.close()


if __name__ == "__main__":
	args = parse_in_args()
	main(args)