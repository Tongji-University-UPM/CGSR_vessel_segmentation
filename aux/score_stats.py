"""
AUXILIAR SCRIPT TO OBTAIN PSEUDO-LABEL SCORE STATS


"""

import os
import re
import numpy as np

import matplotlib.pyplot as plt

def main():
	

	plabel_path = "/path/to/pseudo_label_folder/"

	step = 5
	scores = []
	num_pat = []
	num_pat_cum = []
	for fn in os.listdir(plabel_path):
		
		score = int(re.findall(r'\d+', fn)[-1])
		scores.append(score)
	
	scores = np.array(scores)


	for ix in range(0,100,step):

		indices = np.where((scores>=ix) & (scores<ix+step))[0]
		
		if len(indices)==0:
			num_pat.append(0)
			continue
		num_pat.append(len(indices))


	for ix in range(0,100,step):
		indices_cum = np.where(scores>=ix)[0]
		num_pat_cum.append(len(indices_cum))


	plt.figure("NUM PATIENTS")
	plt.bar(list(range(0,100,step)), num_pat)

	plt.figure("CUMULATIVE NUM PATIENTS")
	plt.bar(list(range(0,100,step)), num_pat_cum)

	plt.show()



if __name__ == "__main__":
	main()