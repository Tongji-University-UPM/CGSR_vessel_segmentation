import os
import sys
import cv2
import json
import random
import pathlib
import numpy as np
from PIL import Image, ImageFilter

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from parser import parse_in_args
from utils import check_dirs, dilate, erode, colorize_mask, dice_score
from segmentation import get_largest_contour, get_plabel_map, apply_frangi
from img_processing import (get_clusters, post_process_map, separate_background,
							merge_masks, filter_mask, closure, dilate)

import time


def get_plabel_fn(idp, plabel_save_path):
	
	found = False
	for fn in os.listdir(plabel_save_path):
		if idp==fn.split(":")[0][:-2]:
			found = True
			break

	if not found:
		raise FileNotFoundError
	
	return fn


def refine_plabel(fn, pos_train_img_path, pred_path, prev_plab_path,
				  plabel_save_path, rgb_plabel_save_path, f_ext="png"):
		
	idp = fn.split(".")[0]
	# print(f"\n---> {idp}")

	plab_fn = get_plabel_fn(idp, prev_plab_path)

	pos_img = np.array(Image.open(f"{pos_train_img_path}{idp}.{f_ext}"), dtype=np.uint8)

	_, ret_mask = get_clusters(pos_img, k=4)
	ret_mask = get_largest_contour(ret_mask).astype('uint8')
	ret_mask = dilate(ret_mask, k=15)

	pred_img = np.array(Image.open(f"{pred_path}{idp}.png"), dtype=np.uint8)
	pred_vsl = 1*(pred_img==0)
	pred_outer = 1*(pred_img==2)

	prev_plab = np.load(f"{prev_plab_path}{plab_fn}")
	prev_plab_vsl = 1*(prev_plab==1)
	prev_plab_uncert = 1*(prev_plab==0)
	prev_plab_outer = 1*(prev_plab==3)
	

	pred_vsl_pp = post_process_map(pred_vsl, hard_filt=False, f=-.5)
	
	""" MIN TO INTERSECT MAX TO UNION """
	comb_vsl = np.maximum(prev_plab_vsl, pred_vsl_pp)
	# comb_vsl = np.minimum(prev_plab_vsl, pred_vsl_pp)
	""""""

	uncert_map = np.maximum(prev_plab_vsl, pred_vsl)

	""" TO PRESERVE UNCERTAINTY """
	uncert_map_pot, _ = separate_background(uncert_map, prev_plab_uncert, dil_k=3)
	uncert_map = np.maximum(uncert_map, uncert_map_pot)
	uncert_map = post_process_map(uncert_map, hard_filt=False, f=-.5)
	""""""

	wrong_vsl_preds = pred_vsl*(1-uncert_map)
	prev_outer_map = np.maximum(np.minimum(pred_outer, prev_plab_outer), 
			   		 np.minimum(pred_outer, prev_plab_uncert))

	outer_map = np.maximum(wrong_vsl_preds, prev_outer_map)
	outer_map = post_process_map(outer_map, hard_filt=False, f=-.5)

	ref_plabel = 2*np.ones_like(pred_vsl)

	ref_plabel[uncert_map==1] = 0
	ref_plabel[outer_map==1] = 3
	ref_plabel[comb_vsl==1] = 1

	ref_plabel[ret_mask==1] = 2

	score = dice_score(prev_plab_vsl, post_process_map(pred_vsl, hard_filt=False, f=-.5))

	np.save(f"{plabel_save_path}{idp}_S:{int(round(score*100))}.npy", ref_plabel)
	Image.fromarray(colorize_mask(ref_plabel)).save(f"{rgb_plabel_save_path}{idp}_S:{int(round(score*100))}.png")



def main(args):

	train_mode = args.train_mode
	
	if len(train_mode) > 0:
		train_mode = "_"+train_mode

	seq_proc = args.sequential_proc

	ds_name = args.ds_name
	data_path = args.data_path
	storage_path = f"{data_path}/{ds_name}/"
	pred_path = args.pred_path+"/"
	prev_plab_path = args.prev_plab_path+"/"

	pos_train_img_path =  f"{storage_path}training/images/"
	plabel_save_path = f"{storage_path}training/ref_plabel_maps{train_mode}/"
	rgb_plabel_save_path = f"{storage_path}training/ref_rgb_plabel_maps{train_mode}/"
	
	check_dirs(plabel_save_path, rgb_plabel_save_path, "./figs")

	file_list = os.listdir(pos_train_img_path)
	
	random.shuffle(file_list)

	

	if seq_proc:
		for fn in file_list:
			refine_plabel(fn, pos_train_img_path,
						pred_path, prev_plab_path,
						plabel_save_path, rgb_plabel_save_path)

	# 	with ProcessPoolExecutor(max_workers=4) as executor:
			
	# 		for fn in file_list:
	# 			executor.submit(refine_plabel, fn, pos_train_img_path,
	# 							pred_path, prev_plab_path,
	# 							plabel_save_path, rgb_plabel_save_path)
	

	else:		
		
		try:
			with ProcessPoolExecutor(max_workers=4) as executor:	
				for fn in file_list:
					executor.submit(refine_plabel, fn, pos_train_img_path,
									pred_path, prev_plab_path,
									plabel_save_path, rgb_plabel_save_path)
					
		except KeyboardInterrupt:
				print("KeyboardInterrupt detected — stopping...")
				
				for p in executor._processes.values():
					p.terminate()  
				
				executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
	args = parse_in_args()
	main(args)