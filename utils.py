import os
import cv2
import numpy as np

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
							 precision_score, accuracy_score,
							 jaccard_score, f1_score, recall_score)

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

def colorize_mask(mask, n_labels=4):

	color_map = np.array([[  0, 0, 0],
					   	  [  31, 165, 83],
						  [  255,   255,   255],
					   	  [200, 135,   0]])

	if len(mask.shape)<3:

		H,W = mask.shape
		one_hot_map = np.arange(n_labels).reshape(1,1,-1)
		one_hot_map = np.tile(one_hot_map, (H,W,1))==mask[...,None]
	else:
		one_hot_map = mask

	rgb_img = one_hot_map @ color_map
	return rgb_img.astype('uint8')


def colorize_probs(mask, n_labels=4):

	color_map = np.array([[  31, 165, 83],
						  [  255,   255,   255],
					   	  [200, 135,   0]])

	rgb_img = mask @ color_map
	return rgb_img.astype('uint8')

def adjust_mean_bright(img, target_bright, clip=255):
	img_adj = img.astype(float)*target_bright/np.mean(img[img>0])
	return np.clip(img_adj, 0, clip).astype('uint8')


def difference_img(cube):

	H,W,C = cube.shape
	central_spect = cube[[H//2],[W//2],:][None,...]
	return np.sqrt(np.sum(np.square(cube-central_spect),axis=-1))
	

def line_coords(orientation, H, W):

	if orientation == "LR":
		""" top left-right diagonal """
		rr = np.arange(0,H-1)
		cc = np.linspace(0,W-1,len(rr)).astype(int)
	
	elif orientation == "RL":
		""" top right-left diagonal """
		rr = np.arange(0,H-1)
		cc = np.linspace(0,W-1,len(rr)).astype(int)[::-1]

	elif orientation == "V":
		""" vertical line """
		rr = np.arange(0,H-1)
		cc = (np.ones(len(rr))*W//2).astype(int)

	elif orientation == "H":
		""" horizontal line """
		rr = (np.ones(H-1)*H//2).astype(int)
		cc = np.linspace(0,W-1,len(rr)).astype(int)[::-1]

	return rr, cc



def check_dirs(*args):
	for dir_ in args:
		if not os.path.exists(dir_):
			os.makedirs(dir_)

def convert_to_1hot(data, num_labels):
	
	H,W = data.shape

	data_flat = data.ravel().astype(int)
	one_hot_data = np.zeros((H*W, num_labels))
	one_hot_data[np.arange(data.size), data_flat] = 1

	return one_hot_data.reshape(H,W,num_labels)


# def get_metrics(probs, gt):

# 	preds = np.argmax(probs, axis=-1)+1

# 	labeled_gt = gt[gt>0]
# 	oacc = np.sum(preds[gt>0]==labeled_gt)/np.sum(gt>0)

# 	acc = {1:-1, 2:-1, 3:-1, 4:-1}
	
# 	for l in np.unique(gt)[1::]:

# 		l_acc = np.sum(preds[gt==l]==l)/np.sum(gt==l)
# 		acc[l] = l_acc
	
# 	return oacc, acc[2]
	



def save_confussion_mat(pred_img, gt, save_path):

	try:
		
		labeled_gt = gt[gt>0].astype(int)
		preds = np.argmax(pred_img[gt>0,:], axis=-1)+1

		cm = confusion_matrix(labeled_gt, preds, labels=[1,2,3,4])
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["H", "T", "V", "D"])
		disp.plot()
		plt.savefig(f"{save_path}.png")
		plt.close()
		
	except ImportError as e:
		print("Confussion matrix could not be saved due to backend import error.\n")


def get_confussion_mat(pred_img, gt):
	
	labeled_gt = gt[gt>0].astype(int)
	preds = np.argmax(pred_img[gt>0,:], axis=-1)+1
	return confusion_matrix(labeled_gt, preds, labels=[1,2,3,4])


def dilate(img, k=3, ellipse=True):
	if ellipse:
		st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
	else:
		st_elem = np.ones((k,k), np.uint8)
	return cv2.dilate(img, st_elem)

def erode(img, k=3, ellipse=True):

	if ellipse:
		st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
	else:
		st_elem = np.ones((k,k), np.uint8)
	return cv2.erode(img, st_elem)


def hd95_index(pred, gt):

	H,W = gt.shape

	coords1 = np.column_stack(np.nonzero(pred))
	coords2 = np.column_stack(np.nonzero(gt))

	dims = np.array([H,W]).reshape(1,2)

	coords1 = coords1/dims
	coords2 = coords2/dims

	distances = cdist(coords1, coords2)

	print(distances.shape)
	print(coords1.shape, coords2.shape)
	quit()

	min1 = np.min(distances, axis=1)
	min2 = np.min(distances, axis=0)
	all_min_dists = np.concatenate([min1, min2])

	return np.percentile(all_min_dists, 95)


def dice_score(gt, probs):
	upper_term = np.sum(probs*gt)
	lower_term = np.sum(probs) + np.sum(gt)
	return 2*(upper_term/(lower_term+1e-8))

def get_metrics(gt, probs, thresh=.5):

	gt = gt.astype(int)

	preds = ((probs>=thresh)*1.).astype(int)

	""" F1 """
	f1_metric = f1_score(gt.ravel(), preds.ravel(), average="binary")
	
	""" ACC """
	acc_metric = accuracy_score(gt.ravel(), preds.ravel())

	""" PRECISION """
	precision = precision_score(gt.ravel(), preds.ravel(), average="binary")

	""" JACCARD """
	jaccard = jaccard_score(gt.ravel(), preds.ravel(), average="binary")

	""" DICE """
	dice_ix = dice_score(gt, probs)

	""" HIT RATE """
	hit_rate = np.sum((gt*preds)>0)/np.sum(gt)

	""" RECALL """
	recall = recall_score(gt.ravel(), preds.ravel(), average="binary")

	return {"F1": f1_metric,
		 	"ACC": acc_metric,
			"DICE": dice_ix,
			"HR": hit_rate,
			"PREC": precision,
			"JACC": jaccard,
			"REC": recall}


def non_neg_mean(data, axis=0):

	data = np.array(data)
	pos_mask = (data>=0)*1.
	return np.sum(data*pos_mask, axis=axis)/np.sum(pos_mask, axis=axis)


def colorize_pred(pred, gt):

	gt = gt/gt.max()

	color_mx = np.array([[0, 1., 0],
					  	 [1., 0, 0],
					  	 [0.5, 0., 1.]])
	
	tp_img = pred*gt
	fn_img = (1-pred)*gt
	fp_img = pred*(1-gt)
	
	pred_img = np.stack([tp_img, fn_img, fp_img],axis=-1)

	return pred_img @ color_mx


def colorize_probs(probs):

	color_mx = np.array([[0.12156863, 0.64705882, 0.3254902 ],
						 [1.        , 1.        , 1.        ],
						 [0.78431373, 0.52941176, 0.        ]])


	return probs @ color_mx


def colorize_classes(pred):


	color_mx = np.array([[0.12156863, 0.64705882, 0.3254902 ],
						 [1.        , 1.        , 1.        ],
						 [0.78431373, 0.52941176, 0.        ]])
	
	return pred @ color_mx


def convert2binary(probs):
	max_prob = np.amax(probs[...,[1,2]], axis=-1)
	return probs[...,0]/(probs[...,0] + max_prob)
	