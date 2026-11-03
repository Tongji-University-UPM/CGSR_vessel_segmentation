import os
import re
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

def dice_score(gt, probs):
	upper_term = np.sum(probs*gt)
	lower_term = np.sum(probs) + np.sum(gt)
	return 2*(upper_term/lower_term)


def main():
	
	ds_name = "DSA"

	step = 10
	plabel_folder = "/data/local/guillermo.vazquez/Data/Tongji/DSA/training/IntersecShrink/ref_plabel_maps_MULTI_DSA_80_3d_ref/"
	
	label_folder = "/data/local/guillermo.vazquez/Data/Tongji/DSA/training/masks/"
	
	score_list = []
	plabel_list = []
	label_list = []

	plabels_set = list(os.listdir(plabel_folder))
	labels_set = list(os.listdir(label_folder))

	for pfn in plabels_set:

		score = int(re.findall(r'\d+', pfn)[-1])

		for l_elem in labels_set:
			if l_elem.split(".")[0] in pfn:
				lfn = l_elem

		plabel_map = np.load(f"{plabel_folder}{pfn}")
		plabel_map = 1*(plabel_map==1)
		label_map = np.asanyarray(Image.open(f"{label_folder}{lfn}"))
		label_map = label_map/label_map.max()

		score_list.append(score)
		plabel_list.append(plabel_map)
		label_list.append(label_map)

	score_list = np.array(score_list)

	mean_dices = []
	std_dices = []
	max_dices, min_dices = [], []
	num_pat = []

	mean_dices_cum = []
	std_dices_cum = []
	max_dices_cum, min_dices_cum = [], []
	num_pat_cum = []

	"""
	GET ALSO CUMULATIVE DICE
	"""

	for ix in range(0,100,step):

		indices = np.where((score_list>=ix) & (score_list<ix+step))[0]
		
		if len(indices)==0:
			mean_dices.append(0)
			std_dices.append(0)
			num_pat.append(0)
			min_dices.append(0)
			max_dices.append(0)
			continue
			
		dice_scores = []
		
		for ix in indices:
			plabel_map = plabel_list[int(ix)]
			label_map = label_list[int(ix)]
			dice_scores.append(100*dice_score(label_map, plabel_map))

		mean_dices.append(np.mean(dice_scores))
		std_dices.append(np.std(dice_scores))
		min_dices.append(np.min(dice_scores))
		max_dices.append(np.max(dice_scores))
		num_pat.append(len(indices))



	for ix in range(0,100,step):
		indices_cum = np.where(score_list>=ix)[0]
		cum_dice_scores = []
		for cix in indices_cum:
			plabel_map = plabel_list[int(cix)]
			label_map = label_list[int(cix)]
			cum_dice_scores.append(100*dice_score(label_map, plabel_map))

		if not cum_dice_scores:
			mean_dices_cum.append(0)
			std_dices_cum.append(0)
			num_pat_cum.append(0)	
			min_dices_cum.append(0)
			max_dices_cum.append(0)
		else:
			mean_dices_cum.append(np.mean(cum_dice_scores))
			std_dices_cum.append(np.std(cum_dice_scores))
			num_pat_cum.append(len(indices_cum))
			min_dices_cum.append(np.min(cum_dice_scores))
			max_dices_cum.append(np.max(cum_dice_scores))

	num_pat_cum = np.array(num_pat_cum)
	per_pat_cum = num_pat_cum/len(plabels_set)


	fig = plt.figure("MEAN DICES")
	ax = fig.gca()
	ax.set_axisbelow(True)
	ax.set_xticks(list(range(0,100,step)))
	ax.set_yticks(np.arange(0, 100, 5))
	ax.grid(linestyle="-.")

	plt.bar(list(range(0,100,step)), mean_dices)
	y_err = np.stack([np.array(mean_dices)-np.array(min_dices), np.array(max_dices)-np.array(mean_dices)])
	y_err_std = std_dices
	plt.errorbar(list(range(0,100,step)), mean_dices, yerr=y_err, fmt="o", color="r")
	plt.errorbar(list(range(0,100,step)), mean_dices, yerr=y_err_std, fmt="o", color="g")

	for i, (mu, std) in enumerate(zip(mean_dices, std_dices)):
		mu_ = round(mu,2)
		std_ = round(std,2)
		print(f"({i+1}, {mu_}) +- (0, {std_})")
	
	# for i in np.array(std_dices):
	# 	print()

	# print("___")
	# for i in np.array(mean_dices):
	# 	print(round(i,2))

	plt.figure("NUM PATIENTS")
	plt.bar(list(range(0,100,step)), num_pat)


	fig = plt.figure("CUMULATIVE MEAN DICES")
	ax = fig.gca()
	ax.set_axisbelow(True)
	ax.set_xticks(list(range(0,100,step)))
	ax.set_yticks(np.arange(0, 100, 5))
	ax.grid(linestyle="-.")
	
	plt.bar(list(range(0,100,step)), mean_dices_cum)
	y_err_cum = np.stack([np.array(mean_dices_cum)-np.array(min_dices_cum), np.array(max_dices_cum)-np.array(mean_dices_cum)])
	y_err_cum_std = std_dices_cum
	plt.errorbar(list(range(0,100,step)), mean_dices_cum, yerr=y_err_cum, fmt="o", color="r")
	plt.errorbar(list(range(0,100,step)), mean_dices_cum, yerr=y_err_cum_std, fmt="o", color="g")


	plt.figure("CUMULATIVE NUM PATIENTS")
	plt.bar(list(range(0,100,step)), num_pat_cum)

	print("___")

	for i, num in enumerate(np.array(num_pat_cum)):
		print(f"({i+1},{np.around(100*num/np.amax(num_pat_cum))})")
		
	print("___")

	for i, num in enumerate(np.array(num_pat)):
		print(f"({i+1},{np.around(100*num/np.amax(num_pat_cum))})")

	# plt.figure("CUMULATIVE NUM PATIENTS PER DICE")
	# plt.bar(list(range(0,100,step)), mean_dices_cum*per_pat_cum)

	plt.show()




if __name__ == "__main__":
	main()