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

	mask_results_path = args.test_results_path
	
	check_dirs(args.model_path)
	
	model = Model(device=args.device,
			   	  module=args.model_type)
	
	model.load_model(f"{args.model_path}{args.pretrained_model_name}")


	for ds_name in args.ds_list:

		print(f"\n-----> {ds_name}")
				
		check_dirs(f"{mask_results_path}{ds_name}/mask/",
				   f"{mask_results_path}{ds_name}/mask_3c/",
				   f"{mask_results_path}{ds_name}/rgb/",
				   f"{mask_results_path}{ds_name}/rgb_probs/")

		test_ids = dl.get_data_names(args.data_path, args.image_folder,
									 args.test_gt_folder, ds_name,
									 pctg=args.dataset_pctg,
									 min_score=-1)
		
		ds_dict_test = {}
		ds_dict_test[ds_name] = test_ids
		
		test_ds = dl.ImageDataset(ds_dict_test,
								  resize=args.resize,
								  plabel_ext="png")


		
		""" TEST PREDICT """
		seg_imgs, idps, ds_names = model.test_pred(test_ds)

		pred_results = {}

		prec_list, jacc_list, dice_list, acc_list, f1_list, hr_list, rec_list  = [], [], [], [], [], [], []

		for i in range(len(idps)):
			
			print(idps[i])
			
			pred_classes = seg_imgs[i,...]

			gt = np.array(Image.open(f"{args.data_path}{ds_names[i]}/{args.test_gt_folder}{idps[i]}.png"), dtype=np.uint8)
			pred_img = convert2binary(np.moveaxis(pred_classes, 0,-1))
			pred_img_3c = np.argmax(np.moveaxis(pred_classes, 0,-1), axis=-1).astype("uint8")
			probs_rgb = colorize_probs(np.moveaxis(pred_classes, 0,-1))

			if args.resize>0:
				H,W = gt.shape
				# gt = cv2.resize(gt, (W//2, H//2), interpolation=cv2.INTER_AREA)
				pred_img = cv2.resize(pred_img, (W, H), interpolation=cv2.INTER_LINEAR)
				pred_img_3c = cv2.resize(pred_img_3c, (W, H), interpolation=cv2.INTER_LINEAR)
				probs_rgb = cv2.resize(probs_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

			pred_metrics = get_metrics(gt/255, pred_img)
			
			f1_metric = pred_metrics["F1"]
			acc_metric = pred_metrics["ACC"]
			jaccard = pred_metrics["JACC"]
			precision = pred_metrics["PREC"]
			dice_ix = pred_metrics["DICE"]
			hit_rate = pred_metrics["HR"]
			recall = pred_metrics["REC"]

			rgb_pred = colorize_pred(pred_img, gt)

			rgb_pred = (rgb_pred*255).astype('uint8')
			mask_pred = (pred_img*255).astype('uint8')
			probs_rgb = (probs_rgb*255).astype('uint8')
			
			Image.fromarray(rgb_pred).save(f"{mask_results_path}{ds_names[i]}/rgb/{idps[i]}.png")
			Image.fromarray(mask_pred).save(f"{mask_results_path}{ds_names[i]}/mask/{idps[i]}.png")
			Image.fromarray(pred_img_3c).save(f"{mask_results_path}{ds_names[i]}/mask_3c/{idps[i]}.png")
			Image.fromarray(probs_rgb).save(f"{mask_results_path}{ds_names[i]}/rgb_probs/{idps[i]}.png")

			pred_results[idps[i]] = round(float(dice_ix), 4)

			dice_list.append(dice_ix)
			acc_list.append(acc_metric)
			f1_list.append(f1_metric)
			hr_list.append(hit_rate)
			prec_list.append(precision)
			jacc_list.append(jaccard)
			rec_list.append(recall)

		pred_results["MEAN DICE: "] = f"{round(float(np.mean(dice_list)),3)} ± {round(float(np.std(dice_list)),3)}"
		pred_results["MEAN JACCARD: "] = f"{round(float(np.mean(jacc_list)),3)} ± {round(float(np.std(jacc_list)),3)}"
		pred_results["MEAN PRECISION: "] = f"{round(float(np.mean(prec_list)),3)} ± {round(float(np.std(prec_list)),3)}"
		pred_results["MEAN ACC: "] = f"{round(float(np.mean(acc_list)),3)} ± {round(float(np.std(acc_list)),3)}"
		pred_results["MEAN F1: "] = f"{round(float(np.mean(f1_list)),3)} ± {round(float(np.std(f1_list)),3)}"
		pred_results["MEAN HR: "] = f"{round(float(np.mean(hr_list)),3)} ± {round(float(np.std(hr_list)),3)}"
		pred_results["MEAN RECALL: "] = f"{round(float(np.mean(rec_list)),3)} ± {round(float(np.std(rec_list)),3)}"

		with open(f"{args.test_results_path}{ds_names[i]}/results.json", 'w') as fp:
			json.dump(pred_results, fp, indent=4)

	""" SAVE CFG FILE """
	with open(f"{mask_results_path}/used_cfg.json", 'w') as fp:
			json.dump(vars(args), fp, indent=4)

if __name__ == "__main__":

	args = parse_in_args()
	main(args)