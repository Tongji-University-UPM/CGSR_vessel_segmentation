import os
import json
import shlex
import subprocess



is_coronary = {"XCAD": True,
				"DSA": True,
				"UWF": False,
				"FP": False,
				"STARE": False}


alias_dict = {"XCAD": "XC",
			  "DSA": "DSA",
			  "UWF": "UWF",
			  "FP": "FP",
			  "STARE": "ST"}


iter_suffixes = ["_ref", "_2nd_ref", "_3d_ref", "_4th_ref", "_5th_ref"]
base_plabel_names = ["plabel_maps", "ref_plabel_maps", "ref_plabel_maps", "ref_plabel_maps"]

def get_exp_name(ds_list, extra_info=""):
	
	if extra_info:
		extra_info = extra_info+"_"

	alias_list = []
	for ds in ds_list:
		alias_list.append(alias_dict[ds])

	aliases = "-".join(alias_list)
	return f"MULTI_{extra_info}{aliases}"

def main():

	device = 0

	iters = 4
	train_scores = [60, 80, 80, 80, 80, 80]
	assert len(train_scores)>=(iters+1)

	epochs = 15
	ds_pctg = 1.0

	skip_pretrain = False
	skip_ref = False
	only_pretrain = False

	train_ds_list = ["XCAD", "STARE"]
	test_ds_list = ["DSA", "30XCA", "134XCA", "XCAD", "DRIVE", "STARE"]

	results_path = "/path/to/store/results/"
	models_path = "/path/to/save/models/"
	data_path = "/path/where/datasets/are/located/"

	project_path = "/path/to/code_project/"
	extra_info = ""

	exec_path = f"{project_path}CGSR/"
	plabel_exec_path = f"{project_path}plabel_refinement/"
	config_train_path = f"{project_path}CGSR/config_train/resnet/"
	config_test_path = f"{project_path}CGSR/config_test/"

	pretrain_json_fn = "pretrain_evc_cfg.json"
	train_json_fn = "train_evc_cfg.json"
	test_json_fn = "test_evc_cfg.json"

	experiment_name = get_exp_name(train_ds_list, extra_info=extra_info)

	pretr_json_dict = json.load(open(f"{config_train_path}{pretrain_json_fn}", "r"))
	test_json_dict = json.load(open(f"{config_test_path}{test_json_fn}", "r"))
	train_json_dict = json.load(open(f"{config_train_path}{train_json_fn}", "r"))

	
	train_iter_sfx = ""

	model_path = f"{models_path}{experiment_name}/"

	train_results_path = f"{results_path}{experiment_name}/EV_3C_{train_scores[0]}s_dist/training/"
	test_results_path = f"{results_path}{experiment_name}/EV_3C_{train_scores[0]}s_dist/test/"
	model_name_pretrain = f"MULTI_evc_model_3c_{str(train_scores[0])}s_dist"


	if not skip_pretrain:

		""" PRETRAINING """
		pretr_json_dict["plabel_score"] = float(train_scores[0])/100
		pretr_json_dict["pretrained_model_name"] = ""
		pretr_json_dict["ds_list"] = train_ds_list
		pretr_json_dict["model_path"] = model_path
		pretr_json_dict["model_name"] = model_name_pretrain
		
		pretr_json_dict["train_results_path"] = train_results_path

		pretr_json_dict["epochs"] = epochs
		pretr_json_dict["dataset_pctg"] = ds_pctg

		pretr_config_str = json.dumps(pretr_json_dict)
		subprocess.Popen(f"CUDA_VISIBLE_DEVICES={device} python3 {exec_path}nn_train.py --config {shlex.quote(pretr_config_str)} --config_type s")


		""" TESTING PRETRAIN """
		test_json_dict["ds_list"] = test_ds_list
		test_json_dict["model_path"] = model_path
		test_json_dict["pretrained_model_name"] = model_name_pretrain
		
		test_json_dict["test_results_path"] = test_results_path

		test_config_str = json.dumps(test_json_dict)
		subprocess.Popen(f"CUDA_VISIBLE_DEVICES={device} python3 {exec_path}nn_test.py --config {shlex.quote(test_config_str)} --config_type s")


	prior_model_name = model_name_pretrain


	for it in range(iters):

		train_iter_sfx = iter_suffixes[it]

		if not skip_ref:
			""" PLABEL REFINEMENT """
			print("-----> STARTING REFINEMENT")
			processes = []
			for dsname in train_ds_list:

				storage_path = f"{data_path}{dsname}/"
				plabel_save_path = f"{storage_path}training/ref_plabel_maps_{experiment_name}_{str(train_scores[it])}{train_iter_sfx}/"
				rgb_plabel_save_path = f"{storage_path}training/ref_rgb_plabel_maps_{experiment_name}_{str(train_scores[it])}{train_iter_sfx}/"

				subprocess.Popen(f"rm -rf {plabel_save_path}")
				subprocess.Popen(f"rm -rf {rgb_plabel_save_path}")

				plabel_path = f"{storage_path}/training/"
				if it <1:
					plabel_folder = f"{base_plabel_names[it]}/"
				else:
					plabel_folder = f"{base_plabel_names[it]}_{experiment_name}_{str(train_scores[it-1])}{iter_suffixes[it-1]}/"

				# print(plabel_save_path)
				# print(f"{plabel_path}{plabel_folder}")

				if is_coronary[dsname]:
					p = subprocess.Popen(["python3", f"{plabel_exec_path}coronary_plabel_ref.py",
					                      "--data_path", data_path,
										  "--ds_name", dsname,
										  "--pred_path", f"{train_results_path}{dsname}/mask_3c/",
										  "--prev_plab_path", f"{plabel_path}{plabel_folder}",
										  "--train_mode", f"{experiment_name}_{str(train_scores[it])}{train_iter_sfx}"])
					processes.append(p)

				else:
					
					p = subprocess.Popen(["python3", f"{plabel_exec_path}retinal_plabel_ref.py",
					                      "--data_path", data_path,
					                      "--ds_name", dsname,
										  "--pred_path", f"{train_results_path}{dsname}/mask_3c/",
										  "--prev_plab_path", f"{plabel_path}{plabel_folder}",
										  "--train_mode", f"{experiment_name}_{str(train_scores[it])}{train_iter_sfx}"])
					processes.append(p)
				
			for p in processes:
				p.wait()
			
			print("-----> FINISHING REFINEMENT")
			if only_pretrain:
				exit()

		skip_ref = False

		ref_plabel_folder = f"training/ref_plabel_maps_{experiment_name}_{str(train_scores[it])}{train_iter_sfx}/"

		train_results_path = f"{results_path}{experiment_name}/EV_3C_{str(train_scores[it+1])}s_distres{train_iter_sfx}/training/"
		test_results_path = f"{results_path}{experiment_name}/EV_3C_{str(train_scores[it+1])}s_distres{train_iter_sfx}/test/"
		model_name_train = f"MULTI_evc_model_3c_{str(train_scores[it+1])}s_distres{train_iter_sfx}"


		""" TRAINING """
		train_json_dict["plabel_score"] = float(train_scores[it+1])/100
		train_json_dict["pretrained_model_name"] = prior_model_name
		train_json_dict["ds_list"] = train_ds_list
		train_json_dict["model_path"] = model_path
		train_json_dict["model_name"] = model_name_train
		train_json_dict["plabel_folder"] = ref_plabel_folder
		
		train_json_dict["train_results_path"] = train_results_path

		train_json_dict["epochs"] = epochs
		train_json_dict["dataset_pctg"] = ds_pctg

		train_config_str = json.dumps(train_json_dict)
		subprocess.Popen(f"CUDA_VISIBLE_DEVICES={device} python3 {exec_path}nn_train.py --config {shlex.quote(train_config_str)} --config_type s")

		prior_model_name = model_name_train


		""" TESTING PRETRAIN """
		test_json_dict["ds_list"] = test_ds_list
		test_json_dict["model_path"] = model_path
		test_json_dict["pretrained_model_name"] = model_name_train
		test_json_dict["test_results_path"] = test_results_path

		test_config_str = json.dumps(test_json_dict)
		subprocess.Popen(f"CUDA_VISIBLE_DEVICES={device} python3 {exec_path}nn_test.py --config {shlex.quote(test_config_str)} --config_type s")


if __name__ == "__main__":
	main()