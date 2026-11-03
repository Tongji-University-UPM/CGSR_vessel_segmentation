import cv2
import json
import argparse
import random
import pickle
import argparse
import numpy as np
from PIL import Image

from json_parser import parse_json_args, parse_in_args

import dataloader as dl

from model import Model

from utils import (check_dirs, colorize_classes,
				   convert2binary, erode, dilate)

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from segmentation import filter_mask, get_largest_contour


def main(args):

	args = parse_json_args(args)

	out_plabel_path = "/data/local/guillermo.vazquez/Data/Tongji/XCAD/train/refined_plabel_map/"
	
	mask_results_path = args.test_results_path+"mask/"
	rgb_results_path = args.test_results_path+"rgb/"
	
	check_dirs(args.model_path, mask_results_path,
			   rgb_results_path, out_plabel_path)
	
	train_ids = dl.get_data_names(args.train_img_path, args.train_plabel_path,
								   	  pctg=args.dataset_pctg, min_score=args.plabel_score)
	print(train_ids)
	train_ds = dl.ImageDataset(train_ids, args.train_img_path,
							   args.train_plabel_path,
							   downsample=args.downsample,
							   patchify=args.patchify,
							   equivariant_trf=False)
	
	model = Model(device=args.device,
				  model_save_path=args.model_path+args.model_name)

	model.load_model(f"{args.model_path}{args.model_name}")


	""" PREDICT """
	seg_imgs, idps = model.test_pred(train_ds)

	# for i in range(len(idps)):
	for id_set in train_ids:

		idp, pl_idp = id_set
		ix = idps.index(idp)

		print(f"---> {pl_idp}")

		pred_classes = seg_imgs[ix,...]

		plabel = np.load(f"{args.train_plabel_path}{pl_idp}.npy")		

		# pred_img = np.moveaxis(pred_img, 0,-1)

		rgb_map = colorize_classes(np.moveaxis(pred_classes, 0,-1))
		vsl_img = np.array(Image.open(f"{args.train_img_path}{idp}.png"), dtype=np.uint8)

		pred_img = convert2binary(np.moveaxis(pred_classes, 0,-1))

		if args.downsample:
			H,W = plabel.shape
			rgb_map = cv2.resize(rgb_map, (W, H), interpolation=cv2.INTER_LINEAR)
			pred_img = cv2.resize(pred_img, (W, H), interpolation=cv2.INTER_LINEAR)

		pred_img = 1.*(pred_img>1e-3)
		comb_pred = np.maximum(1*(plabel==1), pred_img)
		comb_pred = get_largest_contour(erode(dilate(comb_pred, k=5), k=5))


		plabel[comb_pred>1e-3] = 1
		plabel[plabel==0] = 3

		np.save(f"{out_plabel_path}{pl_idp}.npy", plabel)
		continue
		
		
		plt.figure()
		plt.imshow(vsl_img)

		# plt.figure()
		# plt.imshow(rgb_map)

		plt.figure("COMB")
		plt.imshow(comb_pred)
		
		plt.figure("PRED")
		plt.imshow(pred_img)

		plt.figure("PLABEL")
		plt.imshow(plabel)
		
		plt.figure()
		plt.imshow(plabel)
		plt.show()


if __name__ == "__main__":

	args = parse_in_args()
	main(args)