import os
import re
import sys
import cv2
import mahotas
import random
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from sklearn.metrics import calinski_harabasz_score

from utils import check_dirs, dilate, erode

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



def mask_connectivity(mask):

	mask = dilate(erode(mask.astype('uint8')))
	label_map = mahotas.label(dilate(mask.astype('uint8')))[0]
	return np.sum(mask)/np.sqrt(len(np.unique(label_map)[1:]))


def test_clusters_score(image, cl, ch, step):
	s_scores = []
	kmeans_list = []
	""" GLOBAL LABELS """
	for n in range(cl, ch, step):
		kmeans_obj = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(image.reshape(-1,1))
		# score = silhouette_score(trans_img.reshape((-1,3)), kmeans_hc.labels_)
		score = calinski_harabasz_score(image.reshape(-1,1), kmeans_obj.labels_)
		s_scores.append(score)
		kmeans_list.append(kmeans_obj)
		# print(f"{n}: {silhouette_avg}")
	
	return s_scores, kmeans_list[np.argmax(s_scores)]



def get_largest_contour(img):

	labels, _ = mahotas.label(img>1e-3)
	masks = []
	
	if len(np.unique(labels)[1::])>1:
		for l in np.unique(labels)[1::]:
			masks.append(labels==l)

		masks = np.array(masks)
		areas = masks.sum(axis=(1,2))
		largest_mask = masks[np.argmax(areas),...]
		return img*largest_mask
	else:
		return img



def filter_mask(mask, f=1.):

	label_map = mahotas.label(mask)[0]

	if len(np.unique(label_map)[1:]) > 5:

		diag_list = []
		label_list = []
		for l in np.unique(label_map)[1:]:

			r,c = np.where(label_map==l)
			diag = np.sqrt(np.square(np.amax(r)-np.amin(r)) + np.square(np.amax(c)-np.amin(c)))
			diag_list.append(diag)
			label_list.append(l)
		
		sel_ix = np.array(diag_list)>(np.mean(diag_list)+f*np.std(diag_list))
		sel_labels = np.array(label_list)[sel_ix]
		mask_filt = np.isin(label_map, sel_labels)*1

	else:
		mask_filt = mask

	return mask_filt



def get_clusters(img, k=3):

	# clahe = cv2.createCLAHE(clipLimit=2)
	# img = clahe.apply(img)
	# img = cv2.bilateralFilter(img, k_blr, k_blr * 2, k_blr / 2)

	H,W = img.shape

	out_img = np.zeros_like(img)
	kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(img.reshape(-1,1))

	label_img = kmeans.labels_.reshape(H,W)

	for l in np.unique(label_img):
		out_img[label_img==l] = kmeans.cluster_centers_[l]

	img = img.astype(float)
	log_img = np.log(img+1)
	log_out_img = np.log(out_img+1)
	
	bin_mask = log_out_img<(np.mean(log_img)-np.std(log_img))

	bin_mask_filt = filter_mask(bin_mask)

	return out_img, bin_mask_filt
	# out_img



def get_SP_map(img, n_pix=40, k_blr=10):

	# clahe = cv2.createCLAHE(clipLimit=2)
	# img = clahe.apply(img)
	img = cv2.bilateralFilter(img, k_blr, k_blr * 2, k_blr / 2)

	H,W = img.shape

	num_segments = int(H*W//n_pix)

	segments = slic(img, n_segments=num_segments, compactness=.05, slic_zero=False, channel_axis=None)

	bright_levs = []
	for i in np.unique(segments):
		bright_levs.append(np.mean(img[segments==i]))
		
	bright_levs = np.array(bright_levs)
	bright_levs_log = np.log(bright_levs+1)

	sel_mask = bright_levs_log<(np.mean(bright_levs_log)-np.std(bright_levs_log))
	sel_segments = (np.arange(np.amax(segments))+1)[sel_mask]
	
	sel_mask = np.isin(segments,sel_segments)
	
	return 1*sel_mask


def post_process_map(img, hard_filt=True, f=1.5):
	
	img = filter_mask(img, f=0)
	img = erode(dilate(img.astype('uint8'), k=3), k=3)
	
	if hard_filt:
		img = get_largest_contour(img).astype(float)
	else:
		img = filter_mask(img, f=f)
	# return cv2.GaussianBlur(img, (3,3), 0)
	return img


def separate_background(vessel_img, bg_conts, dil_k=3):

	outer_bg_map = np.zeros_like(vessel_img)
	vessel_bg_map = np.zeros_like(vessel_img)

	bg_conts = bg_conts.astype("uint8")

	# bg_conts = (1-bg_img)
	if dil_k>1:
		bg_conts_d = dilate(bg_conts, k=dil_k, ellipse=False)
	else:
		bg_conts_d = bg_conts

	label_map = mahotas.label(bg_conts_d)[0]

	for l in np.unique(label_map)[1:]:
		
		if ((label_map==l)*vessel_img).sum() == 0:
			outer_bg_map = np.maximum((label_map==l)*bg_conts, outer_bg_map)
		
		else:
			vessel_bg_map = np.maximum((label_map==l)*bg_conts, vessel_bg_map)

	# vessel_bg_map = vessel_bg_map-vessel_img
	
	return vessel_bg_map, outer_bg_map


def merge_masks(P,F,M):

	P_c = 1-P
	F_c = 1-F
	M_c = 1-M

	t_1 = np.maximum(np.maximum(P,F),M)
	t_2 = np.maximum(np.maximum(P,F),M_c)
	t_3 = np.maximum(np.maximum(P,F_c),M)
	t_4 = np.maximum(np.maximum(P_c,F),M)
	

	return t_1*t_2*t_3*t_4


def closure(mask, k):
	return erode(dilate(mask.astype('uint8'), k=k), k=k)

def opening(mask, k):
	return erode(dilate(mask.astype('uint8'), k=k), k=k)