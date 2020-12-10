import numpy as np
import torch
import pandas as pd
import pdb

NWAYS = 16
def get_reuse_distances(header_names, np_df):
	addresses = np_df[:, header_names.index('Physical Address'), None]
	set_hdr_ids = [header_names.index("Cache_Set_{}".format(i)) for i in range(NWAYS)]
	cache_sets = np_df[:, set_hdr_ids]
	reuse_dists = [header_names.index("Reuse Distance {}".format(i)) for i in range(NWAYS)]
	this_ids = np.argmax(cache_sets == addresses, axis=-1)
	pdb.set_trace()
	reuse_dist_feat = reuse_dists[:, this_ids]
	pdb.set_trace()

def csv_to_data(fname, chosen_columns=['Program Counter', 'Set', 'Set Occupancy', 'Cache Friendly']):
	df = pd.read_csv(fname)
	header_names = list(df.columns)
	df_npy = df.to_numpy()
	set_occ = 'Set Occupancy' in chosen_columns
	if set_occ:
		# Do the set-occupancy calculations
		header_ids = [header_names.index("Cache_Set_{}".format(i)) for i in range(NWAYS)]
		set_infos = NWAYS - (df_npy[:, header_ids] == -1.0).sum(axis=-1)
		set_infos = np.expand_dims(set_infos, axis=-1)
		all_cols = []
		for chosen_col in chosen_columns:
			if chosen_col == 'Set Occupancy':
				all_cols.append(set_infos)
			else:        
				all_cols.append(df_npy[:, header_names.index(chosen_col), None])
		cols = np.hstack(all_cols)
	else:
		c_ids = [header_names.index(x) for x in chosen_columns]
		cols = df_npy[:, c_ids]
	return cols

def get_batch_iterator(data, batch_sz, shuffle=True, batch_info=False):
    num_pts = len(data)
    perm = np.random.permutation(num_pts)
    perm = perm if shuffle else np.arange(num_pts)
    num_batches = max(num_pts // batch_sz, 1)
    for i in range(num_batches):
        idxs = perm[(i * batch_sz) : (i + 1) * batch_sz]
        samples = [data[idx_] for idx_ in idxs]
        if not batch_info:
            yield samples
        else:
            yield samples, num_batches

def group_by_set(dataset, set_idx=1):
	all_sets = dataset[:, set_idx]
	unique_set_ids = np.unique(all_sets)
	grouped_data = {}
	for s_id in unique_set_ids:
		grouped_data[s_id] = dataset[dataset[:, set_idx] == s_id]
	return grouped_data

if __name__ == '__main__':
	print('I am here')
	fname = "hawkeye_trace_belady_graph.csv"
	df = pd.read_csv(fname)
	header_names = list(df.columns)
	df_npy = df.to_numpy()
	get_reuse_distances(header_names, df_npy)
    
