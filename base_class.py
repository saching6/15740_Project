import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataformatter import *

import matplotlib.pyplot as plt
from copy import deepcopy
import os

class BaseClass:
	"""
	Basic implementation of a general Knowledge Distillation framework

	:param teacher_model (torch.nn.Module): Teacher model
	:param student_model (torch.nn.Module): Student model
	:param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
	:param optimizer_student (torch.optim.*): Optimizer used for training student
	:param loss_fn (torch.nn.Module): Loss Function used for distillation
	:param temp (float): Temperature parameter for distillation
	:param distil_weight (float): Weight paramter for distillation loss
	:param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
	:param log (bool): True if logging required
	:param logdir (str): Directory for storing logs
	"""
	def __init__(
		self,
		teacher_model,
		student_model,
		optimizer_teacher,
		optimizer_student,
		dataset,
		batch_size,
		shuffle,
		loss_fn=nn.KLDivLoss(),
		temp=20.0,
		distil_weight=0.5,
		device="cpu",
	):

# 		self.train_loader = train_loader
# 		self.val_loader = val_loader
		self.optimizer_teacher = optimizer_teacher
		self.optimizer_student = optimizer_student
		self.setwise_dataset=dataset
		self.temp = temp
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.distil_weight = distil_weight

		try:
			torch.Tensor(0).to(device)
			self.device = device
		except:
			print(
				"Either an invalid device or CUDA is not available. Defaulting to CPU."
			)
			self.device = torch.device("cpu")

		try:
			self.teacher_model = teacher_model.to(self.device)
		except:
			print("Warning!!! Teacher is NONE.")
		self.student_model = student_model.to(self.device)
		try:
			self.loss_fn = loss_fn.to(self.device)
			self.ce_fn = nn.CrossEntropyLoss().to(self.device)
		except:
			self.loss_fn = loss_fn
			self.ce_fn = nn.CrossEntropyLoss()
			print("Warning: Loss Function can't be moved to device.")



	def run_epoch_student():
		num_egs = 0
		for batch in data_iterator:
			loss, acc, bsz, m_out = model(batch)
			stats.append([loss.item(), acc.item()])
			num_egs += bsz
			if mode == 'train':
				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
				optimizer.step()
		stats = np.array(stats)
		avg_loss = np.mean(stats[:, 0])
		avg_acc = (stats[:, 1]).sum() / num_egs
		return avg_loss, avg_acc
	def _train_student(
		self,
		epochs=10
	):
		"""
		Function to train student model - for internal use only.
		:param epochs (int): Number of epochs you want to train the teacher
		:param plot_losses (bool): True if you want to plot the losses
		:param save_model (bool): True if you want to save the student model
		:param save_model_pth (str): Path where you want to save the student model
		"""
		self.teacher_model.eval()
		self.student_model.train()
		loss_arr = []
		print("Training Student...")
		stats = []
		for epoch_ in range(epochs):
			accs = []
			setwise_keys = list(self.setwise_dataset.keys())
			perm = np.random.permutation(len(setwise_keys))
			setwise_keys = np.array(setwise_keys)[perm]
			for set_id  in setwise_keys:
				dataset = self.setwise_dataset[set_id]
				self.student_model.remap_embedders(dataset, set_id)
				self.teacher_model.remap_embedders(dataset, set_id)
				data_iterator = get_batch_iterator(dataset, self.batch_size, shuffle=self.shuffle)
				for batch in data_iterator:
					_,acc_student,bsz_student,student_out,label = self.student_model(batch)
					_,acc_teacher,bsz_teacher,teacher_out,label = self.teacher_model(batch)
					distill_loss = self.calculate_kd_loss(student_out, teacher_out, label.long())
					pred = student_out.argmax(dim=1, keepdim=True)
	# 				correct += pred.eq(label.view_as(pred)).sum().item()
					self.optimizer_student.zero_grad()
					distill_loss.backward()
					self.optimizer_student.step()
			print("Epoch: {} | Distillation Loss: {}".format(epoch_,distill_loss.item()))

				
# 			acc_stats = np.min(accs), np.mean(accs), np.median(accs), np.max(accs)
# 			print('Min Acc {} | Mean Acc : {} | Median Acc {} | Max Acc {} '.format(*acc_stats))

	def train_student(
		self,
		epochs=20,
		plot_losses=True,
		save_model=True,
		save_model_pth="./models/student.pt",
	):
		"""
		Function that will be training the student

		:param epochs (int): Number of epochs you want to train the teacher
		:param plot_losses (bool): True if you want to plot the losses
		:param save_model (bool): True if you want to save the student model
		:param save_model_pth (str): Path where you want to save the student model
		"""
		self._train_student(epochs)

	def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
		"""
		Custom loss function to calculate the KD loss for various implementations

		:param y_pred_student (Tensor): Predicted outputs from the student network
		:param y_pred_teacher (Tensor): Predicted outputs from the teacher network
		:param y_true (Tensor): True labels
		"""

		raise NotImplementedError

	def _evaluate_model(self, model, verbose=True):
		"""
		Evaluate the given model's accuaracy over val set.
		For internal use only.
		:param model (nn.Module): Model to be used for evaluation
		:param verbose (bool): Display Accuracy
		"""
		model.eval()
		length_of_dataset = len(self.val_loader.dataset)
		correct = 0
		outputs = []

		with torch.no_grad():
			for data, target in self.val_loader:
				data = data.to(self.device)
				target = target.to(self.device)
				output = model(data)

				if isinstance(output, tuple):
					output = output[0]
				outputs.append(output)

				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(target.view_as(pred)).sum().item()
				accuracy = correct / length_of_dataset

		if verbose:
			print("-" * 80)
			print(f"Accuracy: {accuracy}")
		return outputs, accuracy

	def evaluate(self, teacher=False):
		"""
		Evaluate method for printing accuracies of the trained network

		:param teacher (bool): True if you want accuracy of the teacher network
		"""
		if teacher:
			model = deepcopy(self.teacher_model).to(self.device)
		else:
			model = deepcopy(self.student_model).to(self.device)
		_, accuracy = self._evaluate_model(model)

		return accuracy

	def get_parameters(self):
		"""
		Get the number of parameters for the teacher and the student network
		"""
		teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
		student_params = sum(p.numel() for p in self.student_model.parameters())

		print("-" * 80)
		print(f"Total parameters for the teacher network are: {teacher_params}")
		print(f"Total parameters for the student network are: {student_params}")

	def post_epoch_call(self, epoch):
		"""
		Any changes to be made after an epoch is completed.

		:param epoch (int) : current epoch number
		:return            : nothing (void)
		"""

		pass
