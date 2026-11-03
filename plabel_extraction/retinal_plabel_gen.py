import os
import sys
import cv2
import json
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
from utils import check_dirs, dilate, erode, colorize_mask, dice_score
from segmentation import get_largest_contour, get_plabel_map, apply_frangi
from img_processing import get_clusters, post_process_map, separate_background, filter_mask



def main(args):

	ds_name = args.ds_name
	data_path = args.data_path
	params_fn = f"../parameter_setting/params_files/{ds_name}_params.json"
	storage_path = f"{data_path}/{ds_name}/"

	pos_train_img_path =  f"{storage_path}training/images/"
	plabel_save_path = f"{storage_path}training/plabel_maps/"
	rgb_plabel_save_path = f"{storage_path}training/rgb_plabel_maps/"
	frangi_save_path = f"{storage_path}training/frangi_maps/"
	linear_save_path = f"{storage_path}training/linear_maps/"


	try:
		params_file_r = open(params_fn, "r")
		params_dict = json.load(params_file_r)
	except FileNotFoundError:
		print("Params file not found. Run noise estimator first.")
		exit()
	except json.decoder.JSONDecodeError:
		print("Error parsing params file. Check file content.")
		exit()

	try:
		gamma = params_dict["gamma"]
		lin_params_dict = params_dict["linear_params"]
	except KeyError:
		print("'gamma' and 'linear_params' not found in params file. Run noise estimator first.")
		exit()
	
	device = args.device

	f_ext = "png"

	check_dirs(plabel_save_path, rgb_plabel_save_path, "./figs",
			   frangi_save_path, linear_save_path)

	file_list = os.listdir(pos_train_img_path)
	
	random.shuffle(file_list)

	scores = []

	for fn in file_list:
		
		idp = fn.split(".")[0]

		print(f"\n---> {idp}")


		try:
			pos_img = np.array(Image.open(f"{pos_train_img_path}{idp}.{f_ext}"), dtype=np.uint8)
		
		except FileNotFoundError:
			print(f"Skipping ID {idp}: Not pos-neg pairing.")
			continue


		clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
		pos_img_clahe = clahe.apply(pos_img)

		_, ret_mask = get_clusters(pos_img, k=4)
		ret_mask = get_largest_contour(ret_mask).astype('uint8')

		pos_img_eq = cv2.equalizeHist(pos_img)

		pos_img_eq[ret_mask==1] = np.mean(pos_img_eq[ret_mask==0])
		pos_img_clahe[ret_mask==1] = np.mean(pos_img_clahe[ret_mask==0])	
		
		ret_mask = dilate(ret_mask.astype('uint8'), k=15)


		pos_pl_map_eq = get_plabel_map(pos_img_clahe,
								 	   params_dict=lin_params_dict,
							  		   device=device, smooth=False,
									   single_cont=False, filter_masks=True)/255
		
		frangi_map = apply_frangi(pos_img_eq, gamma=gamma)
		img_clst, img_ctr_bin = get_clusters(pos_img_clahe, k=8)

		filt_vsl_map = 1*(pos_pl_map_eq>0)*np.maximum(frangi_map, img_ctr_bin)
		
		filt_vsl_map = filt_vsl_map*(1-ret_mask)
		frangi_map = frangi_map*(1-ret_mask)

		filt_vsl_map_pp = post_process_map(filt_vsl_map, hard_filt=False, f=-.5)
		background_map = np.maximum(frangi_map, pos_pl_map_eq)>0
		vsl_bg, outer_bg = separate_background(filt_vsl_map_pp, background_map)

		frangi_out = (1-get_largest_contour(dilate(frangi_map, k=5)))*frangi_map

		outer_map = np.maximum(outer_bg, frangi_out)

		outer_map = outer_map*(1-vsl_bg)

		""" SCORE CALC """		
		score = dice_score(filt_vsl_map_pp, filt_vsl_map)
		scores.append(score)

		label_map = np.zeros_like(filt_vsl_map_pp)
		label_map[(1-vsl_bg)>0] = 2
		label_map[outer_map>0] = 3
		label_map[filt_vsl_map_pp>0] = 1

		print(f"SCORE: {score}")

		np.save(f"{frangi_save_path}{idp}.npy", frangi_map)
		np.save(f"{linear_save_path}{idp}.npy", pos_pl_map_eq)

		np.save(f"{plabel_save_path}{idp}_S:{int(round(score*100))}.npy", label_map)
		Image.fromarray(colorize_mask(label_map)).save(f"{rgb_plabel_save_path}{idp}_S:{int(round(score*100))}.png")
	
	
	plt.figure()
	plt.hist(np.array(scores),bins=128)
	plt.savefig(f"./figs/{ds_name}_score_hist.png")

if __name__ == "__main__":
	args = parse_in_args()
	main(args)