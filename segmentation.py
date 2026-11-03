import os
import cv2
import torch
import struct
import mahotas
import numpy as np
from skimage.filters import frangi

import time

# os.environ["CUPY_CACHE_DIR"] = "/data/intercambio2/guillermo.vazquez/cupy_cache"


import cupy as cp
import cupy.lib.stride_tricks as st


import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from liner import get_contour
from img_processing import filter_mask, get_largest_contour
from utils import erode, dilate





def filter_range(img, f=1):
	
	feats = img[img>0]
	log_img = np.log(img+1)
	log_feats = np.log(feats+1)
	
	img[log_img<(log_feats.mean()+f*log_feats.std())] = 0
	return img


def convert2color(labels, shape):
		
	color_labels = plt.cm.viridis(labels/len(np.unique(labels)))[:,:-1]*255

	if len(labels.shape)==1:
		color_labels = color_labels.reshape((shape[0], shape[1], 3))

	return color_labels.astype('uint8')

def pred2color(preds, palette):
	pass

def colorize_gt(gt):

	gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3))

	gt_rgb[(gt>=100)*(gt<200)] = [0,255,0]

	gt_rgb[(gt>=200)*(gt<300)] = [255,0,0]
	gt_rgb[(gt>=300)*(gt<320)] = [0,0,255]
	gt_rgb[gt==320] = [218, 112, 214]

	return gt_rgb


def apply_frangi(img, gamma=7, apply_filter=True):
	
	# clahe = cv2.createCLAHE(clipLimit=2)
	# img = clahe.apply(img)
	H,W = img.shape

	img = cv2.resize(img, (0,0), fx=.5, fy=.5)

	frangi_filt = frangi(img, alpha=.5, beta=.5, gamma=gamma)
	frangi_filt = cv2.resize(frangi_filt, (W,H))
	
	frangi_filt = 1.*(frangi_filt>.15)
	if apply_filter:
		frangi_filt = filter_mask(frangi_filt)

	return erode(dilate(frangi_filt.astype('uint8')))


def init_operator(operator, eps=1e-8):

	angle = 180/operator.shape[-1]
	half_edge = operator.shape[0]//2 + 0
	center = (operator.shape[0]//2, operator.shape[0]//2)
	left_points = []
	
	for a in np.arange(0,45.+angle,angle):
		left_points.append(round(np.tan(a*np.pi/180)*half_edge))
		

	#right_points = np.array(left_points[1:])+half_edge-1
	right_points = np.array(left_points[1:])+half_edge
	left_points = np.abs(np.array(left_points)[::-1]-half_edge)
	points = np.concatenate((left_points, right_points))

	edge_points_top = np.vstack((points[:-1], np.zeros_like(points[:-1]))).T
	edge_points_bottom = np.vstack((points[::-1][:-1], np.ones_like(points[:-1])*operator.shape[0]-1)).T

	edge_points_left = np.vstack((np.zeros_like(points[1:]), points[1:])).T
	edge_points_right = np.vstack((np.ones_like(points[1:])*operator.shape[0]-1, points[::-1][1:])).T
	
	centers = np.tile(center, (len(points[:-1])*2,1))

	edge_points_0 = np.vstack((edge_points_top, edge_points_left))
	edge_points_1 = np.vstack((edge_points_bottom, edge_points_right))

	for i, (ep0, ep1, c) in enumerate(zip(edge_points_0, edge_points_1, centers)):
		
		coords_t = get_contour([ep0], [c])
		coords_b = get_contour([ep1], [c])

		operator[coords_t[1], coords_t[0],i] = 1
		operator[coords_b[1], coords_b[0],i] = 1

	return operator


def line_detector(img, k=15, lines=12, stride=1, dilation=1,
				  out_strength=True, thresh=80, eps=1e-8):

	operator = init_operator(np.zeros((k,k,lines)))

	if dilation >1:
		for l_ix in range(lines):
			operator[...,l_ix] = cv2.dilate(operator[...,l_ix], 
								   np.ones((dilation, dilation), np.uint8), 
								   iterations=1) 

	img = img.astype(np.float32)
	img_i = 255-img

	out_img = np.zeros_like(img[:,:], dtype=float)
	
	st_x = k//2
	st_y = k//2
	end_x = img.shape[1] - st_x-1
	end_y = img.shape[0] - st_y-1

	denom = (np.sum(operator>0,axis=(0,1))+eps)

	for i in range(st_y, end_y, stride):
		for j in range(st_x, end_x, stride):

			roi = img_i[i-k//2:i+k//2+1, j-k//2:j+k//2+1]
			filt = np.multiply(operator, roi[...,None])

			avg_line = np.sum(filt,axis=(0,1))/denom
			
			avg_roi = np.average(roi)

			strength = avg_line-avg_roi

			if out_strength:
				S = operator[:,:,np.argmax(strength)]*(max(0,np.amax(strength)))
				out_img[i-k//2:i+k//2+1, j-k//2:j+k//2+1] += S
			
			else:
				S = filt[:,:,np.argmax(strength)]-avg_roi
				out_patch = np.maximum(out_img[i-k//2:i+k//2+1, j-k//2:j+k//2+1], S)
				out_img[i-k//2:i+k//2+1, j-k//2:j+k//2+1] = out_patch


	out_img[out_img<=thresh] = 0

	vessel_samps = out_img[out_img>0]

	vessel_samps = vessel_samps.astype('int')

	out_img[out_img>0] = vessel_samps

	return out_img


def get_plabel_map(img,
				   k0=7, k1=11, k2=27,
				   d0=1, d1=3, d2=2,
				   th0=.12, th1=.2, th2=.2,
				   params_dict=None,
				   device=6, single_cont=True,
				   filter_masks=True, smooth=False):

	f_path = os.path.realpath(__file__)
	file_path = "/".join(f_path.split("/")[:-1])
	file_path = file_path+"/_cupy_cache"

	os.environ["CUPY_CACHE_DIR"] = file_path

	if params_dict is not None:
		k0 = params_dict["k0"]
		k1 = params_dict["k1"]
		k2 = params_dict["k2"]
		d0 = params_dict["d0"]
		d1 = params_dict["d1"]
		d2 = params_dict["d2"]
		th0 = params_dict["th0"]
		th1 = params_dict["th1"]
		th2 = params_dict["th2"]


	# img = cv2.bilateralFilter(img, k_blr, k_blr * 2, k_blr / 2)
	# img_blur = cv2.bilateralFilter(img, k_blr, k_blr * 2, k_blr / 2)
	H,W = img.shape
	
	# img = cv2.resize(img, (W//2, H//2), interpolation=cv2.INTER_AREA)
	# img_blur = cv2.resize(img_blur, (W//2, H//2), interpolation=cv2.INTER_AREA)

	if device == "cpu":
		
		vessels_0 = line_detector(img, k=k0, dilation=d0, thresh=0, lines=12)
		vessels_1 = line_detector(img, k=k1, dilation=d1, thresh=0, lines=12)
		vessels_2 = line_detector(img, k=k2, dilation=d2, thresh=0, lines=12)

	else:

		cp.cuda.runtime.setDevice(device)
		
		with cp.cuda.Device(device):
			
			mempool = cp.get_default_memory_pool()
			pinned_mempool = cp.get_default_pinned_memory_pool()

			img_cu = cp.asarray(img)
			# img_blur = cp.asarray(img_blur)

			vessels_0_cu = line_detector_cu(img_cu, k=k0, dilation=d0, thresh=0, lines=12)
			vessels_1_cu = line_detector_cu(img_cu, k=k1, dilation=d1, thresh=0, lines=12)
			vessels_2_cu = line_detector_cu(img_cu, k=k2, dilation=d2, thresh=0, lines=12)

			vessels_0 = vessels_0_cu.get()
			vessels_1 = vessels_1_cu.get()
			vessels_2 = vessels_2_cu.get()

			del vessels_0_cu
			del vessels_1_cu
			del vessels_2_cu
			del img
			
			mempool.free_all_blocks()
			pinned_mempool.free_all_blocks()

	# plt.figure()
	# plt.imshow(vessels_0)

	# plt.figure()
	# plt.imshow(vessels_1)

	# plt.figure()
	# plt.imshow(vessels_2)
	# plt.show()

	# vessels_0 = filter_range(vessels_0)
	# vessels_1 = filter_range(vessels_1)
	# vessels_2 = filter_range(vessels_2)

	vessels_0 = (vessels_0-vessels_0.min())/(vessels_0.max()-vessels_0.min())
	vessels_0[vessels_0<th0]=0
	
	vessels_1 = (vessels_1-vessels_1.min())/(vessels_1.max()-vessels_1.min())
	vessels_1[vessels_1<th1]=0

	vessels_2 = (vessels_2-vessels_2.min())/(vessels_2.max()-vessels_2.min())
	vessels_2[vessels_2<th2]=0

	vessels_0 = (vessels_0-vessels_0.min())/(vessels_0.max()-vessels_0.min())
	vessels_1 = (vessels_1-vessels_1.min())/(vessels_1.max()-vessels_1.min())
	vessels_2 = (vessels_2-vessels_2.min())/(vessels_2.max()-vessels_2.min())


	vessels_0 = (255*vessels_0).astype('uint8')
	vessels_1 = (255*vessels_1).astype('uint8')
	vessels_2 = (255*vessels_2).astype('uint8')

	
	if filter_masks:
		vessels_0 = filter_mask(vessels_0, f=0).astype('uint8')
		vessels_1 = filter_mask(vessels_1, f=0).astype('uint8')
		vessels_2 = filter_mask(vessels_2, f=0).astype('uint8')

	vessels_comb = np.maximum(vessels_0, vessels_1)
	vessels_comb = np.maximum(vessels_comb, vessels_2)

	if filter_masks:
		vessels_comb = filter_mask(vessels_comb, f=0).astype('uint8')

	vessels_comb[vessels_comb>0] = 255
	# vessels_comb = cv2.GaussianBlur(cv2.medianBlur(vessels_comb, 3), (7,7), 0)

	if single_cont:
		vessels_comb = get_largest_contour(vessels_comb).astype('uint8')

	if smooth:
		vessels_comb = cv2.GaussianBlur(vessels_comb, (3,3), 0)

	vessels_comb = cv2.resize(vessels_comb, (W, H), interpolation=cv2.INTER_AREA)

	return vessels_comb




def log_correction(img):

	img[img<30] = 0

	img[img>0] = 255*np.log(img[img>0])/np.log(np.amax(img[img>0]))

	return img.astype('uint8')


def histogram_filter(img, mask):

	k = 21
	H,W = mask.shape

	roi_img = img[H//2-k:H//2+k+1, W//2-k:W//2+k+1]
	roi_mask = mask[H//2-k:H//2+k+1, W//2-k:W//2+k+1]>0

	dilation=5

	roi_mask_d = cv2.dilate((roi_mask*255).astype('uint8'), np.ones((dilation, dilation), np.uint8), 
			   iterations=1) 
	
	roi_mask = (roi_mask_d*1.-roi_mask*1.).astype('uint8')
	
	plt.figure()
	plt.hist(roi_img[roi_mask>0].ravel(), 256)

	plt.figure()
	plt.hist(img[mask>0].ravel(), 256)
	

def get_main_contour(gt, img):
	
	H,W = img.shape
	D = np.sqrt(W**2 + H**2)
	
	labels, _ = mahotas.label(img>0, np.ones((3,3)))
	aux_img = np.zeros_like(img)

	for l in np.unique(labels)[1::]:

		label_mask = (labels==l)
		if np.sum(gt*label_mask )>0:
			aux_img = np.maximum(img*label_mask, aux_img)

	return aux_img


def filter_vessel_noise(img, thresh=45):

	H,W = img.shape
	D = np.sqrt(W**2 + H**2)
	
	labels, _ = mahotas.label(img>0, np.ones((3,3)))
	aux_img = np.zeros_like(img)

	for l in np.unique(labels)[1::]:

		c,r = np.where(labels==l)

		w = np.amax(r)-np.amin(r)
		h = np.amax(c)-np.amin(c)

		diag = np.sqrt(w**2 + h**2)

		if diag>(D/thresh):
			aux_img[labels==l] = 1

	return img*aux_img


""" CUPY-BASED DETECTION """

def line_detector_cu(img, k=15, lines=12, stride=1, dilation=1,
					 out_strength=True, thresh=80, eps=1e-8,
					 device=6):

	operator = init_operator(np.zeros((k,k,lines)))

	if dilation >1:
		for l_ix in range(lines):
			operator[...,l_ix] = cv2.dilate(operator[...,l_ix], 
								   np.ones((dilation, dilation), np.uint8), 
								   iterations=1) 

	H, W = img.shape

	operator = cp.asarray(operator.astype(np.float16))

	img_i = 255 - img 

	denom = (cp.sum(operator > 0, axis=(0, 1)) + eps)

	wview = st.sliding_window_view(img_i, (k, k))
	
	Hv, Wv, kh, kw = wview.shape

	xc, yc = cp.meshgrid(cp.arange(Wv), cp.arange(Hv))

	denom = cp.expand_dims(denom, axis=(0,1,2,3))
	operator = cp.expand_dims(operator, axis=(0, 1))
	operator_img = cp.tile(operator, (Hv, Wv, 1, 1, 1))
	wview = cp.expand_dims(wview, axis=-1)
	
	prod = cp.multiply(operator_img, wview)

	avg_line = cp.sum(prod, axis=(2, 3), keepdims=True) / denom  
	avg_roi = cp.mean(wview, axis=(2, 3), keepdims=True)
	
	strength = avg_line - avg_roi                    
	max_ix = cp.argmax(strength, axis=-1).squeeze()
	max_val = cp.amax(strength, axis=-1) 

	line_img = operator_img[yc, xc, :, :, max_ix] * max_val

	out_img = reconstruct_view(line_img, [H,W], k)

	out_img[out_img<=thresh] = 0
	return out_img.astype('int')



def reconstruct_view(patches, image_shape, k):
	H, W = image_shape

	out_H, out_W, _, _ = patches.shape  # assuming shape (H−k+1, W−k+1, k, k)

	# Create base image and weight mask
	rec_img = cp.zeros(image_shape, dtype=patches.dtype)

	# Compute index arrays
	row_idx = cp.arange(out_H)[:, None] + cp.arange(k)  # shape (out_H, k)
	col_idx = cp.arange(out_W)[:, None] + cp.arange(k)  # shape (out_W, k)

	row_idx = row_idx[:, None, :, None]  # shape (out_H, 1, k, 1)
	col_idx = col_idx[None, :, None, :]  # shape (1, out_W, 1, k)

	# Broadcast to patch shape (out_H, out_W, k, k)
	row_idx = cp.broadcast_to(row_idx, patches.shape)
	col_idx = cp.broadcast_to(col_idx, patches.shape)

	# Flatten everything to 1D
	flat_rows = row_idx.ravel()
	flat_cols = col_idx.ravel()
	flat_vals = patches.ravel()

	# Accumulate values and weights
	cp.add.at(rec_img, (flat_rows, flat_cols), flat_vals)

	return rec_img


""" TORCH-BASED DETECTION """

def line_detector_torch(img, k=15, lines=12, stride=1, dilation=1,
					 	out_strength=True, thresh=80, eps=1e-8,
					 	device="cuda:6"):

	operator = init_operator(np.zeros((k,k,lines)))

	if dilation >1:
		for l_ix in range(lines):
			operator[...,l_ix] = cv2.dilate(operator[...,l_ix], 
								   np.ones((dilation, dilation), np.uint8), 
								   iterations=1) 

	H, W = img.shape

	operator = torch.from_numpy(operator.astype(np.float32)).to(device)

	img_i = 255 - img 

	denom = (torch.sum(operator > 0, axis=(0, 1)) + eps)

	wview = img_i.unfold(0, k, 1).unfold(1, k, 1)

	# wview = st.sliding_window_view(img_i, (k, k))
	
	Hv, Wv, kh, kw = wview.shape

	xc, yc = torch.meshgrid(torch.arange(Wv), torch.arange(Hv), indexing="ij")
	
	denom = denom[None,None,None,None,...]
	operator = operator[None,None,...]
	wview = wview.unsqueeze(-1)
	
	operator_img = torch.tile(operator, (Hv, Wv, 1, 1, 1))
	
	prod = torch.multiply(operator_img, wview)

	avg_line = torch.sum(prod, axis=(2, 3), keepdims=True) / denom  
	avg_roi = torch.mean(wview, axis=(2, 3), keepdims=True)
	
	strength = avg_line - avg_roi                    
	max_ix = torch.argmax(strength, axis=-1).squeeze()
	max_val = torch.max(strength, axis=-1)[0]

	line_img = operator_img[yc.int(), xc.int(), ..., max_ix.int()] * max_val

	out_img = reconstruct_view_torch(line_img, [H,W], k, device)

	out_img[out_img<=thresh] = 0
	return out_img.int()



def reconstruct_view_torch(patches, image_shape, k, device):
	H, W = image_shape

	out_H, out_W, _, _ = patches.shape  # assuming shape (H−k+1, W−k+1, k, k)

	# Create base image and weight mask
	rec_img = torch.zeros(image_shape, dtype=patches.dtype, device=device)
	weight = torch.zeros(image_shape, dtype=torch.float32, device=device)

	# Compute index arrays
	row_idx = torch.arange(out_H)[:, None] + torch.arange(k)  # shape (out_H, k)
	col_idx = torch.arange(out_W)[:, None] + torch.arange(k)  # shape (out_W, k)

	row_idx = row_idx[:, None, :, None]  # shape (out_H, 1, k, 1)
	col_idx = col_idx[None, :, None, :]  # shape (1, out_W, 1, k)

	# Broadcast to patch shape (out_H, out_W, k, k)
	row_idx = torch.broadcast_to(row_idx, patches.shape)
	col_idx = torch.broadcast_to(col_idx, patches.shape)

	# Flatten everything to 1D
	flat_rows = row_idx.ravel()
	flat_cols = col_idx.ravel()
	flat_vals = patches.ravel()

	# Accumulate values and weights
	# torch.add.at(rec_img, (flat_rows, flat_cols), flat_vals)

	return torch.index_put(torch.zeros_like(rec_img), (flat_rows, flat_cols), flat_vals, accumulate=True)

	return rec_img



