import os
import re
import sys
import cv2
import json
import mahotas
import random
import numpy as np
from PIL import Image, ImageFilter


import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from parser import parse_in_args
from utils import check_dirs, dilate, erode
from segmentation import get_largest_contour, filter_range
from filtering import apply_prewitt
from img_processing import get_clusters

import time


def main(args):

	ds_name = args.ds_name
	params_fn = f"./params_files/{ds_name}_params.json"
	storage_path = f"/data/local/guillermo.vazquez/Data/Tongji/{ds_name}/"
	# storage_path = "/data/local/guillermo.vazquez/Data/Tongji/STARE/"
	
	pos_train_img_path = f"{storage_path}training/images/"
	f_ext = "png"

	params_dict = {}
	check_dirs("./params_files/")
	params_file = open(params_fn, "w")

	file_list = os.listdir(pos_train_img_path)	
	# random.shuffle(file_list)

	k_blr = 6

	top_x = []
	top_y = []

	for fn in file_list:
		
		idp = fn.split(".")[0]
		print(f"\n---> {idp}")

		try:
			pos_img = np.array(Image.open(f"{pos_train_img_path}{idp}.{f_ext}"), dtype=np.uint8)
			# pos_img = cv2.cvtColor(pos_img, cv2.COLOR_RGB2GRAY)
			# neg_img = np.array(Image.open(f"{neg_train_img_path}{idp}.png"), dtype=np.uint8)
		
		except FileNotFoundError:
			print(f"Skipping ID {idp}: Not pos-neg pairing.")
			continue

		# if pos_img.shape[0] > max_size:
		# 	continue


		clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
		pos_img_clahe = clahe.apply(pos_img)

		pos_img_eq = cv2.equalizeHist(pos_img)
		
		pos_img_clahe_filt = cv2.bilateralFilter(pos_img_clahe, k_blr, k_blr * 2, k_blr / 2)
		
		_, ret_mask = get_clusters(pos_img_clahe_filt, k=4)
		ret_mask = get_largest_contour(ret_mask)


		pos_img_clahe_filt[ret_mask==1] = 255

		x_list = []
		y_list = []
		conn_list = []
		
		for x in range(3,19,2):
			for y in range(3,19,2):
			
				pos_img_clahe_filt = cv2.bilateralFilter(pos_img_clahe, x, x, y*x)

				# pos_img_clahe_filt = cv2.medianBlur(pos_img_clahe, x)

				gradients = apply_prewitt(pos_img_clahe_filt)
				# log_grads = np.log(gradients+1)
				# bin_mask_0 = 1*(log_grads>(log_grads.mean()+log_grads.std()))
				bin_mask_0 = 1*(filter_range(gradients)>0)

				bin_mask_0 = dilate(erode(bin_mask_0.astype('uint8')))
				label_map = mahotas.label(dilate(bin_mask_0.astype('uint8')))[0]
				connectivity = np.sum(bin_mask_0)/np.sqrt(len(np.unique(label_map)[1:]))

				conn_list.append(connectivity)
				x_list.append(x)
				y_list.append(y)


		max_ix = np.argmax(conn_list)
		top_x.append(x_list[max_ix])
		top_y.append(y_list[max_ix])
	


	params_dict["k_blr"] = round(np.mean(top_x))
	params_dict["f_blr"] = np.mean(top_y)

	json.dump(params_dict, params_file)
	params_file.close()

if __name__ == "__main__":
	args = parse_in_args()
	main(args)