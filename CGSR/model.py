import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

from sklearn.metrics import f1_score

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# from net_zoo import ResNet as Net
import net_zoo
# from nets.net_zoo import ResNetBN as Net

from dataloader import custom_collate
from losses import EVLoss, EVLossCE

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



class Model(nn.Module):
	
	def __init__(self, model_save_path=None,
			     in_chns=1, outputs=3,
			  	 module="ResNet", device="cuda:0"):
		
		super().__init__()

		self.model_save_path = model_save_path

		self.net = getattr(net_zoo, module)(in_chns=in_chns,
									  		outputs=outputs)
		self.device = device

	def train(self, train_ds, validation_ds,
		   	  lr=.000001, epochs=1500, 
	   		  batch_size=1, checkpoint=True, 
			  pretrained_model=False, modality="output_dec-embed",
			  contrastive_train=True,
			  frozen_modules=[], plot_path="./"):
		

		if contrastive_train:
			ev_loss = EVLoss(epochs)
		else:
			ev_loss = EVLossCE(epochs)
		
		# optimizer = optim.AdamW([
		# 	{"params": self.net.params.parameters(), "lr": lr}],
		# 	lr=lr, weight_decay=.001, amsgrad=True)

		optimizer = optim.AdamW([
			{"params": self.net.parameters(), "lr": lr}],
			lr=lr, weight_decay=.001, amsgrad=True)

		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs+1, eta_min=lr/10)

		train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
										 	   pin_memory=True, shuffle=True,
											   drop_last=False, num_workers=4,
											   collate_fn=custom_collate)
		
		validation_dl = torch.utils.data.DataLoader(validation_ds, batch_size=batch_size,
											  		pin_memory=True, shuffle=True,
													collate_fn=custom_collate, drop_last=False)
		
		global_loss = []
		global_val_vsl_f1 = []

		val_epochs = []
		
		running_val_f1_prev = .0

		self.net.to(self.device)
		# self.conv_net.to(self.device)
		
		for e in range(epochs+1):

			# print(e, optimizer.param_groups[0]["lr"])
			# lrs.append(optimizer.param_groups[0]["lr"])
			# scheduler.step(e)
			# continue

			running_loss = .0
			steps = 0

			self.net.train()

			for train_xy in train_dl:

				x_train_gpu = train_xy["img"].to(self.device)
				x_train_trf_gpu = train_xy["img_trf"].to(self.device)
				pl_train_gpu = train_xy["plabel"].to(self.device)
				pl_train_trf_gpu = train_xy["plabel_trf"].to(self.device)
				transform_fn = train_xy["transform_fn"]

				# plt.figure()
				# plt.imshow((x_train_gpu[0,0,...]).detach().cpu().numpy())

				# plt.figure()
				# plt.imshow((x_train_trf_gpu[0,0,...]).detach().cpu().numpy())

				# plt.figure()
				# plt.imshow((pl_train_gpu[0,...]).detach().cpu().numpy())
				
				# plt.figure()
				# plt.imshow((pl_train_trf_gpu[0,...]).detach().cpu().numpy())
				# plt.show()

				optimizer.zero_grad()

				x_comb_train = torch.cat([x_train_gpu, x_train_trf_gpu], dim=0)
				bsz = pl_train_gpu.shape[0]

				out_comb_gpu = self.net(x_comb_train, modality=modality)

				pred_comb_gpu = out_comb_gpu[1]
				embed_comb_gpu = out_comb_gpu[0]

				pred_train_gpu, pred_train_trf_gpu = torch.split(pred_comb_gpu, [bsz, bsz], dim=0)
				embed_train_gpu, embed_train_trf_gpu = torch.split(embed_comb_gpu, [bsz, bsz], dim=0)

				# loss = bce_loss(pred_train_gpu, y_train_gpu)

				loss_val = ev_loss(pred_train_gpu, pred_train_trf_gpu,
					   			   embed_train_gpu, embed_train_trf_gpu,
								   pl_train_gpu, pl_train_trf_gpu,
								   transform_fn, e)
				
				loss_val.backward()
				
				optimizer.step()

				running_loss += loss_val.detach().item()
				
				steps += 1
				# acc_train += acc_train_it

			scheduler.step()

			global_loss.append(running_loss/steps)

			if e%10==0:
				running_val_f1 = self.validation_pred(validation_dl, modality)
				global_val_vsl_f1.append(running_val_f1)
				val_epochs.append(e)
				
				print(f"EPOCH: {e}, CTX_LOSS: {np.around(global_loss[-1],3)}, ",
					  f" VAL_F1: {np.around(global_val_vsl_f1[-1],3)}")
				
				if checkpoint and (running_val_f1>running_val_f1_prev):
					self.save_model(self.model_save_path)
					running_val_f1_prev = running_val_f1
				
		try:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			plt.plot(global_loss, label="Loss")
			plt.legend()
			ax.grid(which='minor', alpha=0.2)
			ax.grid(which='major', alpha=0.5)
			plt.savefig(f"{plot_path}/loss.png")
			plt.close()
			
			
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			plt.plot(val_epochs, global_val_vsl_f1, label="Dice")
			plt.legend()
			ax.grid(which='minor', alpha=0.2)
			ax.grid(which='major', alpha=0.5)
			plt.savefig(f"{plot_path}/val_acc.png")
			plt.close()
		
		except ImportError as e:
			print("Training metrics plotting failed due to backend import error.\n")
			

	def train_metrics(self, pred_gpu, y_train_gpu):

		probs = F.softmax(pred_gpu, dim=-1)
		labels = torch.argmax(probs, dim=-1)
		acc = torch.sum(y_train_gpu==labels)/labels.shape[0]
		return acc.cpu().detach().numpy()


	def validation_pred(self, validation_dl, modality):

		# self.device = torch.self.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		val_f1 = .0
		steps = 0

		for val_xy in validation_dl:

			x_val_gpu = val_xy["img"].to(self.device)
			y_val_gpu = val_xy["plabel"].to(self.device)

			self.net.eval()
			with torch.no_grad():
				pred_val_gpu = self.net(x_val_gpu, modality=modality)[1]
				probs_gpu = F.softmax(pred_val_gpu, dim=1)			

				probs_gpu = torch.moveaxis(probs_gpu, 1,-1)

				# print(probs_gpu.shape)

				# plt.figure()
				# plt.imshow((probs_gpu[0,...,0]).detach().cpu().numpy())

				# plt.figure()
				# plt.imshow((probs_gpu[0,...,1]).detach().cpu().numpy())

				# plt.figure()
				# plt.imshow((probs_gpu[0,...,2]).detach().cpu().numpy())
				
				# plt.figure()
				# plt.imshow((y_val_gpu[0,...]==1).detach().cpu().numpy())
				# plt.show()

				active_mask = y_val_gpu>0
				
				step_f1 = multiclass_f1_score(probs_gpu[active_mask,:],
											  y_val_gpu[active_mask].long()-1,
											  average='weighted', num_classes=3)

			val_f1 += step_f1.detach().cpu().numpy()
			steps += 1

		return val_f1/steps


	def test_pred(self, test_ds, modality="output_dec-embed", batch_size=64):

		pred_vsl_imgs = []
		pred_ids = []
		pred_ds = []

		test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
											  shuffle=False, drop_last=False,
											  collate_fn=custom_collate)
		
		self.net.to(self.device)

		for test_xy in test_dl:

			x_test_gpu = test_xy["img"].to(self.device)
			idp_test = test_xy["idp"]
			ds_name_test = test_xy["ds_name"]
			
			pred_ids.extend(idp_test)
			pred_ds.extend(ds_name_test)

			self.net.eval()
			with torch.no_grad():
				
				pred_train_gpu = self.net(x_test_gpu, modality=modality)[1]
				probs_vsl_gpu = F.softmax(pred_train_gpu, dim=1)
				pred_vsl_imgs.append(probs_vsl_gpu)
				
		# pred_vsl_imgs = torch.cat(pred_vsl_imgs, dim=0)[:,0,...]
		pred_vsl_imgs = torch.cat(pred_vsl_imgs, dim=0)

		return pred_vsl_imgs.cpu().detach().numpy(), pred_ids, pred_ds



	def save_model(self, path):

		torch.save(self.net.state_dict(), path+"_state_dict.pt")
		# torch.save(self.net, path+".pt")

		print(f"\nModel {path} saved")


	def load_model(self, path, reset_modules=[], forget_modules=[]):

		pretrained_dict = torch.load(path+"_state_dict.pt", map_location=torch.device('cpu'))
		rand_dict = self.net.state_dict()
		final_dict = {}

		for full_k in pretrained_dict.keys():
			
			mod_k = full_k.split(".")[0]
			mod_dict_k = full_k.split(".")[1]

			if mod_k in forget_modules:
				continue

			if not ((mod_k in reset_modules) or (mod_dict_k in reset_modules)):
				final_dict[full_k] = pretrained_dict[full_k]

		if reset_modules:
			for full_k in rand_dict.keys():
				mod_k = full_k.split(".")[0]
				mod_dict_k = full_k.split(".")[1]

				if (mod_k in reset_modules) or (mod_dict_k in reset_modules):
					final_dict[full_k] = rand_dict[full_k]
		
		self.net.load_state_dict(final_dict, strict=False)

