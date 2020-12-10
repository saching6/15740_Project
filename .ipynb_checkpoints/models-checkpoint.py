import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_, kaiming_normal_
from collections import defaultdict
import pdb
import numpy as np
import math

FC_CONFIG = {
    'layers': [192, 400, 2],
    'dropout': 0.2,
	'feat_map': {'Program Counter' : 0, 'Set Occupancy': 1, 'Belady Friendly': 2} #'Set': 1,
}

TFORMER_CONFIG = {
	'd_model': 192,
	'n_head': 4,
	'num_encoder_layers': 3,
	'dim_feedforward': 256,
	'dropout': 0.2,
	'final_out_sz': 2,
	'pred_window_sz': 5,
	'feat_map': {'Program Counter' : 0, 'Set Occupancy': 1, 'Belady Friendly': 2} #'Set': 1,
}

TFORMER_CONFIG_1 = {
	'd_model': 96,
	'n_head': 2,
	'num_encoder_layers': 3,
	'dim_feedforward': 128,
	'dropout': 0.2,
	'final_out_sz': 2,
	'pred_window_sz': 8,
	'feat_map': {'Program Counter' : 0, 'Set Occupancy': 1, 'Belady Friendly': 2} #'Set': 1,
}

TFORMER_CONFIG_2 = {
	'd_model': 384,
	'n_head': 6,
	'num_encoder_layers': 3,
	'dim_feedforward': 256,
	'dropout': 0.3,
	'final_out_sz': 2,
	'pred_window_sz': 8,
	'feat_map': {'Program Counter' : 0, 'Set Occupancy': 1, 'Belady Friendly': 2} #'Set': 1,
}

def reshape(x, p_win_sz=32):
	d = x.shape[0]
	# Truncate the array to make it divisible by p_win_sz * 2
	if np.mod( d, p_win_sz*2 ) > 0:
		x = x[:-np.mod( d, p_win_sz*2 )]
	#get the largest section of the array starting at i that is divisible by p_win_sz * 2
	g = lambda x_, i: x_[ i :- ( 2 * p_win_sz - np.mod( i, 2 * p_win_sz ) ) ]
	#Reshape the array to shape (p_win_sz*2,:)
	f = lambda x_, i: g( x_, i ).reshape(
			int( g( x_, i ).shape[0] / ( 2 * p_win_sz ) ),
			2 * p_win_sz
	)
	#Get all via list comprehension
	res = [ f( x, i ) for i in range( p_win_sz * 2  ) ]
	return np.vstack(res)

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
	elif 'TRANSFORMER' in model_type:
		if '1' in model_type:
			kwargs = TFORMER_CONFIG_1
		elif '2' in model_type:
			kwargs = TFORMER_CONFIG_2
		else:
			kwargs = TFORMER_CONFIG
		kwargs['loss_name'] = 'CE' if kwargs['final_out_sz'] != 1 else 'BCE'
		model = TFormer(**kwargs)
	return model

def gen_bias_mask(max_length):
	"""
	Generates bias values (-Inf) to mask future timesteps during attention
	"""
	np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
	torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

	return torch_mask.unsqueeze(0).unsqueeze(1)

def gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
	"""
	Generates a [1, length, channels] timing signal consisting of sinusoids
	"""
	position = np.arange(length)
	num_timescales = channels // 2
	log_timescale_increment = (
					math.log(float(max_timescale) / float(min_timescale)) /
					(float(num_timescales) - 1))
	inv_timescales = min_timescale * np.exp(
					np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
	scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)


	signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
	signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
					'constant', constant_values=[0.0, 0.0])
	signal =  signal.reshape([1, length, channels])
	return torch.from_numpy(signal).type(torch.FloatTensor)


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
		return loss, acc, len(y)

	def get_accuracy(self, pred, targets):
		with torch.no_grad():
			argmax = pred.argmax(dim=-1).squeeze()
			if pred.shape[-1] == 1:  # We have only 1 output - so sigmoid.
				argmax = (pred > 0.5).squeeze().float()
				targets = targets.squeeze()
			return argmax.eq(targets).sum()
	
	def rm_feature(self, feat_name):
		assert hasattr(self, 'feat_idx_map'), 'Does not have a feature index map'
		del self.feat_idx_map[feat_name]
		self.emb_dim = int(self.layers[0] / (len(self.feat_idx_map) - 1))
	
	def criterion(self, outs, target):
		if self.loss_fn_name == 'BCE':
			target = target.float().unsqueeze(1)
		return self.loss_fn(outs, target)
	
	def prep_for_data(self, dataset):
		# Expect the data to be in this form
		pass
	
	def format_batch(self, batch):
		pass
	
	def get_data_columns(self):
		ncols = len(self.feat_idx_map)
		col_names = ['' for _ in range(ncols)]
		for k, v in self.feat_idx_map.items():
			if v >= ncols:
				assert 'We should not be here : v {} vrs {}'.format(v, self.feat_idx_map)
			col_names[v] = k
		return col_names

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
		if torch.cuda.is_available:
			embedder = embedder.cuda()
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
		# We are assuming that the set occupancy map is shared accross all sets


	def prep_for_data(self, dataset, temp_order=True):
		if 'Program Counter' in self.feat_idx_map:
			feat_idx = self.feat_idx_map['Program Counter']
			self.pc_emb_map, self.pc_embedding = self.create_embedder(dataset, feat_idx, temporal_order=temp_order)
		if 'Set' in self.feat_idx_map:
			feat_idx = self.feat_idx_map['Set']
			self.set_emb_map, self.set_embedding = self.create_embedder(dataset, feat_idx, temporal_order=temp_order)
		if 'Set Occupancy' in self.feat_idx_map:
			feat_idx = self.feat_idx_map['Set Occupancy']
			self.set_occ_emb_map, self.set_occ_embedding = self.create_embedder(dataset, feat_idx, temporal_order=temp_order)
		# TODO [all] - add to this as more features are introduced

class TFormer(Model):
	def __init__(self, **kwargs):
		super(TFormer, self).__init__(loss_name=kwargs['loss_name'])
		self.feat_idx_map = kwargs['feat_map']
		assert kwargs['d_model'] % (len(self.feat_idx_map) - 1) == 0, 'The input dimension should be divisible by # of features'
		self.emb_dim = int(kwargs['d_model'] / (len(self.feat_idx_map) - 1))
		self.pred_window_sz = kwargs['pred_window_sz']
		encoder_layer = torch.nn.TransformerEncoderLayer(kwargs['d_model'], kwargs['n_head'], kwargs['dim_feedforward'], kwargs['dropout'], 'relu')
		encoder_norm = torch.nn.LayerNorm(kwargs['d_model'])
		self.encoder = torch.nn.TransformerEncoder(encoder_layer, kwargs['num_encoder_layers'], encoder_norm)
		self.proj = nn.Linear(kwargs['d_model'], kwargs['final_out_sz'])
		# Adding for the reuse-distance
# 		self.proj_reuse = nn.Linear(kwargs['d_model'], kwargs['reuse_out_sz'])
		self.pred_window_sz = kwargs['pred_window_sz']

	def format_batch(self, batch):
		x_pc, x_set, y = [], [], []
		x_set_occ = []
		for b in batch:
			y.append(b[-1])
			# Convert the program counter to an embedding
			if 'Program Counter' in self.feat_idx_map:
				x_pc.append(self.pc_emb_map[b[self.feat_idx_map['Program Counter']]])
			if 'Set' in self.feat_idx_map:
				x_set.append(self.set_emb_map[b[self.feat_idx_map['Set']]])
			if 'Set Occupancy' in self.feat_idx_map:
				x_set_occ.append(self.set_occ_emb_map[b[self.feat_idx_map['Set Occupancy']]])
		x_pc = reshape(np.array(x_pc), p_win_sz=self.pred_window_sz)
		y = reshape(np.array(y), p_win_sz=self.pred_window_sz)
		mask = gen_bias_mask(x_pc.shape[-1]).squeeze() # After doing the reshaping
		pos_emb = gen_timing_signal(x_pc.shape[-1], self.emb_dim * (len(self.feat_idx_map) - 1))
		pos_emb = torch.transpose(pos_emb, 1, 0)
		if self.use_cuda:
			x = torch.tensor(x_pc).cuda()
		else:
			x = torch.tensor(x_pc)
		x = self.pc_embedding(x)
		y = torch.tensor(y).float()
		x = torch.transpose(x, 1, 0)
		y = torch.transpose(y, 1, 0)
		if 'Set' in self.feat_idx_map:
			x_set = reshape(np.array(x_set), p_win_sz=self.pred_window_sz)
			x_set = torch.tensor(x_set)
			if self.use_cuda:
				x_set = x_set.cuda()
			x_set = self.set_embedding(x_set)
			x_set = torch.transpose(x_set, 1, 0)
			x = torch.cat([x, x_set], dim=-1)
		if 'Set Occupancy' in self.feat_idx_map:
			x_set_occ = reshape(np.array(x_set_occ), p_win_sz=self.pred_window_sz)
			x_set_occ = torch.tensor(x_set_occ)
			if self.use_cuda:
				x_set_occ = x_set_occ.cuda()
			x_set_occ = self.set_occ_embedding(x_set_occ)
			x_set_occ = torch.transpose(x_set_occ, 1, 0)
			x = torch.cat([x, x_set_occ], dim=-1)
		if self.use_cuda:
			x, y, mask, pos_emb = x.cuda(), y.cuda(), mask.cuda(), pos_emb.cuda()
		return x, y, mask, pos_emb


	def forward(self, batch):
		x, y, mask, pos_embed = self.format_batch(batch)
		x = x + pos_embed  # Add the positional embedding
		m_out = self.encoder(x, mask)
		m_out = F.relu(m_out)
		m_out = self.proj(m_out)
		# Need to do the right amount of sub-indexing
		m_out = m_out[-self.pred_window_sz:, :, :]
		m_out = m_out.view(-1, m_out.shape[-1])
		y = y[-self.pred_window_sz:].flatten()
		bsz = len(y)
		if self.loss_fn_name == 'BCE':
			# We need to do a sigmoid if we're using binary labels
			m_out = torch.sigmoid(m_out)
		if self.loss_fn_name == 'CE':
			y = y.long()
		loss = self.loss_fn(m_out, y)
		acc = self.get_accuracy(m_out, y)
		return loss, acc, bsz

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


	def format_batch(self, batch):
		x_pc, x_set, y = [], [], []
		x_set_occ = []
		for b in batch:
			y.append(b[-1])
			# Convert the program counter to an embedding
			if 'Program Counter' in self.feat_idx_map:
				x_pc.append(self.pc_emb_map[b[self.feat_idx_map['Program Counter']]])
			if 'Set' in self.feat_idx_map:
				x_set.append(self.set_emb_map[b[self.feat_idx_map['Set']]])
			if 'Set Occupancy' in self.feat_idx_map:
				x_set_occ.append(self.set_occ_emb_map[b[self.feat_idx_map['Set Occupancy']]])
		if self.use_cuda:
			x = torch.tensor(x_pc).cuda()
		else:
			x = torch.tensor(x_pc)
		x = self.pc_embedding(x)
		y = torch.tensor(y).float()
		if 'Set' in self.feat_idx_map:
			x_set = torch.tensor(x_set)
			if self.use_cuda:
				x_set = x_set.cuda()
			x_set = self.set_embedding(x_set)
			x = torch.cat([x, x_set], dim=-1)
		if 'Set Occupancy' in self.feat_idx_map:
			x_set_occ = torch.tensor(x_set_occ)
			if self.use_cuda:
				x_set_occ = x_set_occ.cuda()
			x_set_occ = self.set_occ_embedding(x_set_occ)
			x = torch.cat([x, x_set_occ], dim=-1)
		if self.use_cuda:
			x, y = x.cuda(), y.cuda()
		return x, y