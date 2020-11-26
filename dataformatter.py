import numpy as np
import torch
import pandas as pd


def csv_to_data(fname, chosen_columns=['Program Counter', 'Cache Friendly']):
    df = pd.load_csv(fname)
	header_names = list(df.columns())
	df_npy = df.to_numpy()
	c_ids = [header_names.index(x) for x in chosen_columns]
	return df_npy[:, c_ids]

def get_batch_iterator(data, batch_sz, shuffle=True):
    num_pts = len(data)
    perm = np.random.permutation(num_pts)
    perm = perm if shuffle else np.arange(num_pts)
    num_batches = num_pts // batch_sz
    for i in range(num_batches):
        idxs = perm[(i * batch_sz) : (i + 1) * batch_sz]
        samples = [data[idx_] for idx_ in idxs]
        yield samples
    