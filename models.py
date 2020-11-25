import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_, kaiming_normal_


# DEFINE MODEL SPECIFIC HYPER-PARAMS HERE
FC_CONFIG = {
    'layers': [100, 200, 2],
    'dropout': 0.2
}

TRANSFORMER_CONFIG = {
    'dropout': 0.2,
}


# Weight initialization
def weight_init(init_method):
	def initfn(layer):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if init_method == 'xavier_unif':
				xavier_uniform_(layer.weight.data)
			elif init_method == 'kaiming_unif':
				kaiming_uniform_(layer.weight.data)
			elif init_method == 'kaiming_normal':
				kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				zeros_(layer.bias.data)
		elif isinstance(layer, nn.BatchNorm2d):
			layer.weight.data.fill_(1)
			layer.bias.data.zero_()
	return initfn


def get_model(model_type, **kwargs):
	# TODO [All] - we need to implement the individual functions
	if model_type == 'FC':
		kwargs = FC_CONFIG
		assert 'layers' in kwargs, 'Need to specify layers for MLP model'
		assert 'dropout' in kwargs, 'Need to specify dropout for MLP model'
		loss_name = 'CE' if kwargs['layers'][-1] != 1 else 'BCE'
		model = MLP(kwargs['layers'], dp_ratio=kwargs['dropout'])
	elif model_type == 'TRANSFORMER':
		kwargs = TRANSFORMER_CONFIG
		pass
	return model


# Super-class encapsulating all model related functions
class Model(nn.Module):
	def __init__(
					self, loss_name='CE'
				):
		super(Model, self).__init__()
		self.loss_fn_name = loss_name
		self.loss_fn = self.get_loss_fn(loss_name)

	def get_loss_fn(self, fn_name, reduction='mean'):
		if fn_name == 'CE':
			return nn.CrossEntropyLoss(reduction=reduction)
		elif fn_name == 'BCE':
			return nn.BCELoss(reduction=reduction)
		elif fn_name == 'MSE':
			return nn.MSELoss(reduction=reduction)

	def forward(self, x, y):
		# Assumes the data has already been formatted appropriately
		m_out = self.model(x)
		if self.loss_fn_name == 'BCE':
			# We need to do a sigmoid if we're using binary labels
			m_out = torch.sigmoid(m_out)
		loss = self.loss_fn(m_out, y)
		# Get the acccuracy
		acc = self.get_accuracy(m_out, y)
		return loss, acc

	def get_accuracy(self, pred, targets):
		with torch.no_grad():
			argmax = pred.argmax(dim=-1).squeeze()
			if pred.shape[-1] == 1:  # We have only 1 output - so sigmoid.
				argmax = (pred > 0.5).squeeze().float()
			return argmax.eq(targets).sum()

	def criterion(self, outs, target):
		if self.loss_fn_name == 'BCE':
			target = target.float().unsqueeze(1)
		return self.loss_fn(outs, target)


class MLP(Model):
	def __init__(
					self, layers, init_method='xavier_unif',
					loss_name='CE', dp_ratio=0.3):
		super(MLP, self).__init__(
									loss_name=loss_name,
								)
		sequence = []
		for i in range(len(layers) - 1):
			sequence.append(nn.Dropout(dp_ratio))
			sequence.append(nn.Linear(layers[i], layers[i + 1]))
			sequence.append(nn.ReLU())
		self.model = nn.Sequential(*sequence[:-1])  # [:-1] to remove the last relu
		self.model.apply(weight_init(init_method))

