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

from utils import (check_dirs, colorize_pred, get_metrics,
				   colorize_classes, convert2binary,
				   colorize_probs)

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



def main(args):

	args = parse_json_args(args)

	skip_train = args.skip_train
	
	mask_results_path = args.train_results_path
	
	check_dirs(args.model_path)
	
	ds_dict_train = {}
	ds_dict_val = {}
	ds_dict_pred = {}


	for ds_name in args.ds_list:
				
		check_dirs(f"{mask_results_path}{ds_name}/mask/",
				   f"{mask_results_path}{ds_name}/mask_3c/",
				   f"{mask_results_path}{ds_name}/rgb_probs/")

		train_val_ids_pred = dl.get_data_names(args.data_path, args.image_folder,
										  	   args.plabel_folder, ds_name,
										  	   pctg=args.dataset_pctg,
										  	   min_score=-1)
		
		ds_dict_pred[ds_name] = train_val_ids_pred

		if not args.skip_train:
			train_val_ids = dl.get_data_names(args.data_path, args.image_folder,
											args.plabel_folder, ds_name,
											pctg=args.dataset_pctg,
											min_score=args.plabel_score)
			
			random.Random(args.seed).shuffle(train_val_ids)

			val_len = int(len(train_val_ids)*args.validation_pctg)
			train_ids = train_val_ids[val_len:]
			val_ids = train_val_ids[:val_len]

			ds_dict_train[ds_name] = train_ids
			ds_dict_val[ds_name] = val_ids
			
		
	if not args.skip_train:
			
		train_ds = dl.ImageDataset(ds_dict_train,
								resize=args.resize,
								equivariant_trf=True,
								augmentations=args.augmentations,
								fill_pl_blanks=args.fill_pl_blanks)
		
		val_ds = dl.ImageDataset(ds_dict_val,
								eq_img_hist=True,
								resize=args.resize,
								fill_pl_blanks=args.fill_pl_blanks)
		
	train_ds_pred = dl.ImageDataset(ds_dict_pred,
									resize=args.resize)
	
	model = Model(device=args.device,
				  module=args.model_type,
				  model_save_path=args.model_path+args.model_name)

	""" TRAIN """
	if skip_train:
		model.load_model(f"{args.model_path}{args.model_name}")
		
	else:
		if args.pretrained_model:
			model.load_model(f"{args.model_path}{args.pretrained_model_name}")

		model.train(train_ds, val_ds, batch_size=args.batch_size, 
					lr=args.lr, epochs=args.epochs,
					checkpoint=args.save_chkpt,
					contrastive_train=args.contrastive_train,
					pretrained_model=args.pretrained_model)
		
		if not args.save_chkpt:
			model.save_model(f"{args.model_path}{args.model_name}")

	
	""" SAVE CFG FILE """
	with open(f"{mask_results_path}/used_cfg.json", 'w') as fp:
			json.dump(vars(args), fp, indent=4)

	""" TRAIN PREDICT """
	seg_imgs, idps, ds_names = model.test_pred(train_ds_pred)
	
	for i in range(len(idps)):
		
		print(idps[i])

		pred_classes = seg_imgs[i,...]

		pred_img = convert2binary(np.moveaxis(pred_classes, 0,-1))
		pred_img_3c = np.argmax(np.moveaxis(pred_classes, 0,-1), axis=-1).astype("uint8")
		probs_rgb = colorize_probs(np.moveaxis(pred_classes, 0,-1))

		img_path = f"{args.data_path}{ds_names[i]}/{args.image_folder}" 
		vsl_img = np.array(Image.open(f"{img_path}{idps[i]}.png"), dtype=np.uint8)

		if args.resize>0:
			H,W = vsl_img.shape
			# gt = cv2.resize(gt, (W//2, H//2), interpolation=cv2.INTER_AREA)
			pred_img = cv2.resize(pred_img, (W, H), interpolation=cv2.INTER_LINEAR)
			pred_img_3c = cv2.resize(pred_img_3c, (W, H), interpolation=cv2.INTER_LINEAR)
			probs_rgb = cv2.resize(probs_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

		mask_pred = (pred_img*255).astype('uint8')
		probs_rgb = (probs_rgb*255).astype('uint8')
		Image.fromarray(mask_pred).save(f"{mask_results_path}{ds_names[i]}/mask/{idps[i]}.png")
		Image.fromarray(pred_img_3c).save(f"{mask_results_path}{ds_names[i]}/mask_3c/{idps[i]}.png")
		Image.fromarray(probs_rgb).save(f"{mask_results_path}{ds_names[i]}/rgb_probs/{idps[i]}.png")

if __name__ == "__main__":

	args = parse_in_args()
	main(args)