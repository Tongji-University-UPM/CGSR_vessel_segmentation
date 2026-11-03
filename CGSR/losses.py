import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

"""
NTXent_Loss:
https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb

SupConLoss:
https://github.com/HobbitLong/SupContrast/blob/master/losses.py

"""

class DiceBCELoss(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.bce_criterion = nn.BCEWithLogitsLoss()

	
	def dice_loss_fn(self, X, Y):
		up_term = torch.sum(torch.mul(Y, X), dim=(2,3))
		low_term = torch.sum(X, dim=(2,3)) + torch.sum(Y, dim=(2,3))
		return (1-2*up_term/(low_term+torch.finfo().eps)).mean()
	
	def forward(self, X, Y):

		X_prob = F.sigmoid(X)
		
		dice_loss = self.dice_loss_fn(X_prob, Y)
		bce_loss = self.bce_criterion(X,Y)

		# return self.bce_criterion(pred, labels)+hull_loss, vsl_loss
		return dice_loss + bce_loss
		# return bce_loss


def masked_ce_loss(img, labels, reduction="mean"):
	
	active_mask = (labels>0)

	img = torch.moveaxis(img, 1,-1)
	logits = img[active_mask,:]
	labels_masked = labels[active_mask]
	
	bce_loss_batch = F.cross_entropy(logits, (labels_masked-1).long(), reduction=reduction)
	
	return bce_loss_batch



def masked_dice_loss(preds, labels):

	probs = F.softmax(preds, dim=1)
	probs = probs.permute(0, 2, 3, 1)

	one_hot_labs = F.one_hot(labels.long(), num_classes=4).float()
	active_mask = 1-one_hot_labs[..., [0]]
	one_hot_labs = one_hot_labs[..., 1:]
	
	upper_term = (one_hot_labs*probs).sum()
	lower_term = one_hot_labs.sum() + (probs*active_mask).sum()

	return 1-2*upper_term/lower_term

	

class EVLoss(nn.Module):

	def __init__(self, epochs):
			super().__init__()
			
			self.epochs = epochs
			self.dice_bce_loss_fn = DiceBCELoss()
			self.sup_con_loss = SupConLoss()


	def getCAM(self, X, eps=1e-8):
		# return F.relu(X)/(torch.amax(F.relu(X), dim=(1,2))[:,None,None]+eps)
		return F.softmax(X, dim=1)


	def forward(self, X, X_trf, X_emb, X_emb_trf,
			 	YP, YP_trf, transforms, e,
				subsamp=.5, samp_thresh=2.5e4):


		device = X.get_device()
		device = "cpu" if device<0 else f"cuda:{device}"


		dice_loss = masked_dice_loss(torch.cat([X, X_trf], dim=0).squeeze(1),
									 torch.cat([YP, YP_trf], dim=0))

		# ce_loss = masked_ce_loss(torch.cat([X, X_trf], dim=0).squeeze(1),
		# 						 torch.cat([YP, YP_trf], dim=0))

		""" EQUIVARIANT LOSS """
		c_weight = get_current_consistency_weight(e, self.epochs)

		CAM = self.getCAM(X)
		CAM_trf = self.getCAM(X_trf)

		CAM_ev, X_emb_ev, YP_ev = [], [], []
		for i in range(len(transforms)):
			trf_fn = transforms[i]
			CAM_ev.append(trf_fn(CAM[i,...]))
			X_emb_ev.append(trf_fn(X_emb[i,...]))
			YP_ev.append(trf_fn(YP[[i],...]))

		CAM_ev = torch.stack(CAM_ev, 0)
		YP_ev = torch.stack(YP_ev, 0)
		X_emb_ev = torch.stack(X_emb_ev, 0)
		l2_loss = F.mse_loss(CAM_ev, CAM_trf)*c_weight


		# ce_loss = masked_ce_loss(torch.cat([X, X_trf], dim=0),
		# 		  				 torch.cat([YP, YP_trf], dim=0))
		
		# loss_order = torch.flip(torch.argsort(ce_loss).view(-1,1), [0,1])
		# samps_len = len(loss_order)*subsamp/2


		""" CONTRASTIVE LOSS """
		vsl_mask = (YP_ev==1).squeeze(1)
		out_mask = (YP_ev==3).squeeze(1)


		X_emb_ev = X_emb_ev.permute(0, 2, 3, 1)
		X_emb_trf = X_emb_trf.permute(0, 2, 3, 1)

		B = vsl_mask.shape[0]
		batch_sc_loss = 0
		for b in range(B):
			
			vsl_samps = X_emb_ev[b,...][vsl_mask[b],:]
			vsl_samps_trf = X_emb_trf[b,...][vsl_mask[b],:]

			out_samps = X_emb_ev[b,...][out_mask[b],:]
			out_samps_trf = X_emb_trf[b,...][out_mask[b],:]

			# print(vsl_samps.shape, out_samps.shape)

			if vsl_samps.shape[0]>0 and out_samps.shape[0]>0:

				base_samps = torch.cat([vsl_samps, out_samps], dim=0).unsqueeze(1)
				trf_samps = torch.cat([vsl_samps_trf, out_samps_trf], dim=0).unsqueeze(1)

				con_labs = torch.cat([torch.zeros(vsl_samps.shape[0]),\
									torch.ones(out_samps.shape[0])])
				
				# if len(con_labs)>samp_thresh:
				# 	subsamp = subsamp/2
				# neg_len = min(vsl_mask[b].sum(), con_labs.shape[0]*subsamp)

				sel_ixs = torch.randperm(con_labs.shape[0])[:int(con_labs.shape[0]*subsamp)]

				sc_loss = self.sup_con_loss(torch.cat([base_samps[sel_ixs,...],
													trf_samps[sel_ixs,...]], dim=1),
											con_labs[sel_ixs], device=X_emb.device)

				batch_sc_loss += sc_loss
		"""  *****  """

		return l2_loss + dice_loss + batch_sc_loss/B
		# return l2_loss + dice_loss
		# return ctx_loss, vsl_loss
		


class _EVLoss(nn.Module):

	def __init__(self, epochs):
			super().__init__()
			
			self.epochs = epochs
			self.dice_bce_loss_fn = DiceBCELoss()
			self.sup_con_loss = SupConLoss()


	def getCAM(self, X, eps=1e-8):
		# return F.relu(X)/(torch.amax(F.relu(X), dim=(1,2))[:,None,None]+eps)
		return F.softmax(X, dim=1)


	def forward(self, X, X_trf, X_emb, X_emb_trf,
			 	YP, YP_trf, transforms, e,
				subsamp=.3, samp_thresh=2.5e4):


		device = X.get_device()
		device = "cpu" if device<0 else f"cuda:{device}"


		dice_loss = masked_dice_loss(torch.cat([X, X_trf], dim=0).squeeze(1),
									 torch.cat([YP, YP_trf], dim=0))

		# ce_loss = masked_ce_loss(torch.cat([X, X_trf], dim=0).squeeze(1),
		# 						 torch.cat([YP, YP_trf], dim=0))

		""" EQUIVARIANT LOSS """
		c_weight = get_current_consistency_weight(e, self.epochs)

		CAM = self.getCAM(X)
		CAM_trf = self.getCAM(X_trf)

		CAM_ev, X_emb_ev, YP_ev = [], [], []
		for i in range(len(transforms)):
			trf_fn = transforms[i]
			CAM_ev.append(trf_fn(CAM[i,...]))
			X_emb_ev.append(trf_fn(X_emb[i,...]))
			YP_ev.append(trf_fn(YP[[i],...]))

		CAM_ev = torch.stack(CAM_ev, 0)
		YP_ev = torch.stack(YP_ev, 0)
		X_emb_ev = torch.stack(X_emb_ev, 0)
		l2_loss = F.mse_loss(CAM_ev, CAM_trf)*c_weight


		""" CONTRASTIVE LOSS """
		vsl_mask = (YP_ev==1).squeeze(1)
		out_mask = (YP_ev==3).squeeze(1)

		X_emb_ev = X_emb_ev.permute(0, 2, 3, 1)
		X_emb_trf = X_emb_trf.permute(0, 2, 3, 1)

		vsl_samps = X_emb_ev[vsl_mask,:].mean(dim=0, keepdim=True)
		vsl_samps_trf = X_emb_trf[vsl_mask,:].mean(dim=0, keepdim=True)

		out_samps = X_emb_ev[out_mask,:].mean(dim=0, keepdim=True)
		out_samps_trf = X_emb_trf[out_mask,:].mean(dim=0, keepdim=True)

		base_samps = torch.cat([vsl_samps, out_samps], dim=0).unsqueeze(1)
		trf_samps = torch.cat([vsl_samps_trf, out_samps_trf], dim=0).unsqueeze(1)

		con_labs = torch.cat([torch.zeros(vsl_samps.shape[0]),\
							torch.ones(out_samps.shape[0])])
		
	
		sc_loss = self.sup_con_loss(torch.cat([base_samps, trf_samps], dim=1),
									con_labs, device=X_emb.device)

		return l2_loss + dice_loss + sc_loss


"""
https://github.com/HZS23660/DLBCL-segmentation/tree/main
"""

def sigmoid_rampup(current, rampup_length):
	"""Exponential rampup from https://arxiv.org/abs/1610.02242"""
	if rampup_length == 0:
		return 1.0
	else:
		current = np.clip(current, 0.0, rampup_length)
		phase = 1.0 - current / rampup_length
		return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, epochs):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return .5 * sigmoid_rampup(epoch, epochs)


class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=0.07, contrast_mode='all',
				 base_temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def forward(self, features, labels=None, mask=None, device=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf

		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
				has the same class as sample i. Can be asymmetric.
		Returns:
			A loss scalar.
		"""

		if device is None:
			device = (torch.device('cuda')
					if features.is_cuda
					else torch.device('cpu'))

		if len(features.shape) < 3:
			raise ValueError('`features` needs to be [bsz, n_views, ...],'
							 'at least 3 dimensions are required')
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature
			anchor_count = contrast_count
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature)
			
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)
		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
			0
		)
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		# modified to handle edge cases when there is no positive pair
		# for an anchor point. 
		# Edge case e.g.:- 
		# features of shape: [4,1,...]
		# labels:            [0,1,1,2]
		# loss before mean:  [nan, ..., ..., nan] 
		mask_pos_pairs = mask.sum(1)
		mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss
	

class EVLossCE(nn.Module):

	def __init__(self, epochs):
			super().__init__()
			
			self.epochs = epochs
			self.dice_bce_loss_fn = DiceBCELoss()


	def getCAM(self, X, eps=1e-8):
		# return F.relu(X)/(torch.amax(F.relu(X), dim=(1,2))[:,None,None]+eps)
		return F.sigmoid(X)


	def forward(self, X, X_trf, X_emb, X_emb_trf,
			 	YP, YP_trf, transforms, e,
				subsamp=.5, samp_thresh=2.5e4):


		device = X.get_device()
		device = "cpu" if device<0 else f"cuda:{device}"


		dice_loss = masked_dice_loss(torch.cat([X, X_trf], dim=0).squeeze(1),
									 torch.cat([YP, YP_trf], dim=0))
		

		# ce_loss = masked_ce_loss(torch.cat([X, X_trf], dim=0).squeeze(1),
		# 						 torch.cat([YP, YP_trf], dim=0))
		

		""" EQUIVARIANT LOSS """
		c_weight = get_current_consistency_weight(e, self.epochs)

		CAM = self.getCAM(X)
		CAM_trf = self.getCAM(X_trf)

		CAM_ev, X_emb_ev, YP_ev = [], [], []
		for i in range(len(transforms)):
			trf_fn = transforms[i]
			CAM_ev.append(trf_fn(CAM[i,...]))
			X_emb_ev.append(trf_fn(X_emb[i,...]))
			YP_ev.append(trf_fn(YP[[i],...]))

		CAM_ev = torch.stack(CAM_ev, 0)
		YP_ev = torch.stack(YP_ev, 0)
		X_emb_ev = torch.stack(X_emb_ev, 0)
		l2_loss = F.mse_loss(CAM_ev, CAM_trf)*c_weight


		# return l2_loss + dice_loss + ce_loss
		return l2_loss + dice_loss
		