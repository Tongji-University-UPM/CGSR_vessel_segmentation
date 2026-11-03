import os
import re
import sys
import cv2
import json
import mahotas
import optuna
import optuna_distributed
import random
import numpy as np
from PIL import Image

import cupy
import cupy_backends

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from parser import parse_in_args
from segmentation import get_plabel_map



def objective(trial, img_list, mask_list, device):

	k_0 = trial.suggest_int('k0', 3, 7, step=2)
	k_1 = trial.suggest_int('k1', 9, 19, step=2)
	k_2 = trial.suggest_int('k2', 21, 29, step=2)

	d_0 = 1
	d_1 = trial.suggest_int('d1', 1, 3, step=1)
	d_2 = trial.suggest_int('d2', 1, 3, step=1)

	t_0 = trial.suggest_float('th0', .1, .24, step=.02)
	t_1 = trial.suggest_float('th1', .1, .24, step=.02)
	t_2 = trial.suggest_float('th2', .1, .24, step=.02)

	params_dict = {}
	params_dict["k0"] = k_0
	params_dict["k1"] = k_1
	params_dict["k2"] = k_2

	params_dict["d0"] = d_0
	params_dict["d1"] = d_1
	params_dict["d2"] = d_2

	params_dict["th0"] = t_0
	params_dict["th1"] = t_1
	params_dict["th2"] = t_2

	dsc_list = []
	for step, (img, mask) in enumerate(zip(img_list, mask_list)):
		
		seg_map = get_plabel_map(img, 
								 params_dict=params_dict,
								 device=device, smooth=False,
								 single_cont=False, filter_masks=True)/255
		
		label_map = mahotas.label(seg_map.astype('uint8'))[0]
		connectivity = np.sum(seg_map)/np.sqrt(len(np.unique(label_map)[1:]))

		dsc = 2*(mask*seg_map).sum()/(mask.sum() + seg_map.sum())
		dsc_list.append(dsc+connectivity)

		trial.report(dsc+connectivity, step=step)

		if trial.should_prune():
			raise optuna.TrialPruned()
	
	mean_dsc = np.mean(dsc_list)
	trial.set_user_attr("mean_dsc", mean_dsc)

	return mean_dsc


def main(args):

	ds_name = args.ds_name
	params_fn = f"./params_files/{ds_name}_params.json"
	storage_path = f"/data/local/guillermo.vazquez/Data/Tongji/{ds_name}/"
	
	pos_train_img_path =  f"{storage_path}training/images/"
	temp_masks_path = f"{storage_path}training/temp_masks/"

	f_ext = "png"
	
	try:
		params_file_r = open(params_fn, "r")
		params_dict = json.load(params_file_r)
	except FileNotFoundError:
		print("Params file not found. Run noise estimator first.")
		exit()
	except json.decoder.JSONDecodeError:
		print("Error parsing params file. Check file content.")
		exit()

	# try:
	# 	k_blr = params_dict["k_blr"]
	# 	f_blr = params_dict["f_blr"]
	# except KeyError:
	# 	print("'k_blr' and 'f_blr' not found in params file. Run noise estimator first.")
	# 	exit()

	f_ext = "png"
	ds_pctg = args.ds_pctg

	device = args.device

	file_list = os.listdir(pos_train_img_path)
	crop_len = int(ds_pctg*len(file_list))
	
	random.seed(0)
	random.shuffle(file_list)

	img_list = []
	mask_list = []
	for fn in file_list[:crop_len]:
		
		idp = fn.split(".")[0]
		print(f"\n---> {idp}")

		try:
			pos_img = np.array(Image.open(f"{pos_train_img_path}{idp}.{f_ext}"), dtype=np.uint8)
			temp_mask = np.array(Image.open(f"{temp_masks_path}{idp}.png"), dtype=np.uint8)
			# temp_mask = filter_mask(temp_mask)

		except FileNotFoundError:
			print(f"Skipping ID {idp}: Not pos-neg pairing.")
			continue


		clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
		# pos_img_clahe = clahe.apply(pos_img)
		# pos_img_clahe = cv2.bilateralFilter(pos_img_clahe, k_blr, k_blr, k_blr*f_blr)

		# pos_img_clahe = clahe.apply(cv2.medianBlur(pos_img, 5))
		pos_img_clahe = clahe.apply(pos_img)

		img_list.append(pos_img_clahe)
		mask_list.append(temp_mask)

	pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

	study = optuna.create_study(direction="maximize",
								pruner=pruner,
							 	sampler=optuna.samplers.TPESampler(n_startup_trials=5))
	
	study = optuna_distributed.from_study(study)
	study.optimize(lambda trial: objective(trial, img_list, mask_list, device),
					n_trials=200, n_jobs=4, show_progress_bar=True,
					catch=(cupy.cuda.memory.OutOfMemoryError, cupy_backends.cuda.api.runtime.CUDARuntimeError))
	
	lin_params_dict = study.best_params
	lin_params_dict["d0"] = 1

	params_dict["linear_params"] = lin_params_dict

	params_file_w = open(params_fn, "w")
	json.dump(params_dict, params_file_w, indent=4)
	params_file_w.close()
	params_file_r.close()

if __name__ == "__main__":
	args = parse_in_args()
	main(args)