import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_, kaiming_normal_
from collections import defaultdict
import pdb

FC_CONFIG = {
    'layers': [100, 200, 400, 1],
    'dropout': 0.2,
	'feat_map': {'Program Counter' : 0, 'Set': 1, 'Cache Friendly': 2}
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
		model = MLP(
						kwargs['layers'], dp_ratio=kwargs['dropout'],
						loss_name=loss_name, feat_idx_map=kwargs['feat_map']
					)
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

	def forward(self, batch):
		x, y = self.format_batch(batch)
		# Assumes the data has already been formatted appropriately
		m_out = self.model(x)
		if self.loss_fn_name == 'BCE':
			# We need to do a sigmoid if we're using binary labels
			m_out = torch.sigmoid(m_out)
		loss = self.loss_fn(m_out, y)
		acc = self.get_accuracy(m_out, y)
		return loss, acc

	def get_accuracy(self, pred, targets):
		with torch.no_grad():
			argmax = pred.argmax(dim=-1).squeeze()
			if pred.shape[-1] == 1:  # We have only 1 output - so sigmoid.
				argmax = (pred > 0.5).squeeze().float()
				targets = targets.squeeze()
			return argmax.eq(targets).sum()

	def criterion(self, outs, target):
		if self.loss_fn_name == 'BCE':
			target = target.float().unsqueeze(1)
		return self.loss_fn(outs, target)
	
	def prep_for_data(self, dataset):
		# Expect the data to be in this form
		pass
	
	def format_batch(self, batch):
		pass


class MLP(Model):
	def __init__(
					self, layers, init_method='xavier_unif',
					loss_name='CE', dp_ratio=0.3, feat_idx_map=None):
		super(MLP, self).__init__(
									loss_name=loss_name,
								)
		assert feat_idx_map is not None, 'Need to specify a mapping from feature name to its position'
		self.feat_idx_map = feat_idx_map
		# TODO [all] - assuming we are concatenating features
		assert layers[0] % (len(feat_idx_map) - 1) == 0, 'The input dimension should be divisible by # of features'
		self.layers = layers
		self.emb_dim = int(layers[0] / (len(feat_idx_map) - 1))
		sequence = []
		self.layers = layers
		for i in range(len(layers) - 1):
			sequence.append(nn.Dropout(dp_ratio))
			sequence.append(nn.Linear(layers[i], layers[i + 1]))
			sequence.append(nn.ReLU())
		self.model = nn.Sequential(*sequence[:-1])  # [:-1] to remove the last relu
		self.model.apply(weight_init(init_method))
	
	def rm_feature(self, feat_name):
		del self.feat_idx_map[feat_name]
		self.emb_dim = int(self.layers[0] / (len(self.feat_idx_map) - 1))

	def create_embedder(self, dataset, feat_idx, temporal_order=True):
		all_entries = [b[feat_idx] for b in dataset]
		if not temporal_order:
			all_entries = set(all_entries)
		counter = 1
		this_map = defaultdict(int)
		for id_ in all_entries:
			if id_ not in this_map:
				this_map[id_] = counter
				counter += 1
		embedder = torch.nn.Embedding(max(counter, 500), self.emb_dim, 0) # Making this 2x so there there is slack.
		return this_map, embedder

	def remap_embedder(self, entries, cntr_max):
		counter = 1
		this_map = defaultdict(int)
		for id_ in entries:
			if id_ not in this_map:
				this_map[id_] = counter
				counter += 1
			if len(this_map) >= cntr_max - 1:
				# We are assuming everything that comes after is padding
				counter = 0
		return this_map

	def remap_embedders(self, dataset, set_id):
		if not hasattr(self, 'setid_to_map_map'):
			self.setid_to_map_map = defaultdict(lambda: {})
		if 'Program Counter' in self.feat_idx_map:
			if set_id not in self.setid_to_map_map['Program Counter']:
				max_emb_cnt = self.pc_embedding.weight.shape[0]
				feat_idx = self.feat_idx_map['Program Counter']
				entries = [b[feat_idx] for b in dataset]
				self.setid_to_map_map['Program Counter'][set_id] = self.remap_embedder(entries, max_emb_cnt)
			self.pc_emb_map = self.setid_to_map_map['Program Counter'][set_id]
		if 'Set' in self.feat_idx_map:
			if set_id not in self.setid_to_map_map['Set']:
				max_emb_cnt = self.set_embedding.weight.shape[0]
				feat_idx = self.feat_idx_map['Set']
				entries = [b[feat_idx] for b in dataset]
				self.setid_to_map_map['Set'][set_id] = self.remap_embedder(entries, max_emb_cnt)
			self.set_emb_map = self.setid_to_map_map['Set'][set_id]

	def get_data_columns(self):
		ncols = len(self.feat_idx_map)
		col_names = ['' for _ in range(ncols)]
		for k, v in self.feat_idx_map.items():
			if v >= ncols:
				# This is the target column
				continue
			col_names[v] = k
		return col_names

	def prep_for_data(self, dataset, temp_order=True):
		if 'Program Counter' in self.feat_idx_map:
			feat_idx = self.feat_idx_map['Program Counter']
			self.pc_emb_map, self.pc_embedding = self.create_embedder(dataset, feat_idx, temporal_order=temp_order)
		if 'Set' in self.feat_idx_map:
			feat_idx = self.feat_idx_map['Set']
			self.set_emb_map, self.set_embedding = self.create_embedder(dataset, feat_idx, temporal_order=temp_order)
		# TODO [all] - add to this as more features are introduced

	def format_batch(self, batch):
		# TODO [ldery]
		# NOTE : ASSUMING THE LAST INDEX IS THE TARGET
		assert hasattr(self, 'pc_emb_map'), 'This model should have an embedding map for pc'
		x_pc, x_set, y = [], [], []
		for b in batch:
			y.append(b[-1])
			# Convert the program counter to an embedding
			if 'Program Counter' in self.feat_idx_map:
				x_pc.append(self.pc_emb_map[b[self.feat_idx_map['Program Counter']]])
			if 'Set' in self.feat_idx_map:
				x_set.append(self.set_emb_map[b[self.feat_idx_map['Set']]])
		x = torch.tensor(x_pc)
		x = self.pc_embedding(x)
		y = torch.tensor(y).unsqueeze(-1).float()
		# TODO [ldery / all] - this is not neat - figure out a better way to do this
		if len(self.feat_idx_map) == 2:
			return x, y
		elif len(self.feat_idx_map) == 3:
			x_set = self.set_embedding(torch.tensor(x_set))
			x = torch.cat([x, x_set], dim=-1)
		return x, y
