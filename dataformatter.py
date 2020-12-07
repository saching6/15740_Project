import numpy as np
import torch
import pandas as pd


def csv_to_data(fname, chosen_columns=['Program Counter', 'Set', 'Cache Friendly']):
    df = pd.read_csv(fname)
    header_names = list(df.columns)
    df_npy = df.to_numpy()
    c_ids = [header_names.index(x) for x in chosen_columns]
    return df_npy[:, c_ids]

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
    
