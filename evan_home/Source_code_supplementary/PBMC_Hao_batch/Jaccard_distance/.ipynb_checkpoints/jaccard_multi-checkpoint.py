import os
import numpy as np
import pandas as pd
import scanpy as sc
import copy
import matplotlib.pyplot as plt
import umap
import scipy
import sys

import multiprocessing as mp
from sklearn.utils import gen_batches
from pathos.multiprocessing import ProcessingPool

def pairwise_jaccard(adata, idx_1, idx_2):
    a = adata.X[idx_1].tocsr()
    b = adata.X[idx_2].tocsr()
    
    intersection = len(np.intersect1d(a.indices, b.indices))
    union = len(np.union1d(a.indices, b.indices))
    jaccard = intersection / union if union != 0 else 0
    return jaccard

def compute_jaccard_matrix_batch(args):
    adata, batch_idx = args
    print('=====Starting batch:', batch_idx)
    n_samples = adata.shape[0]
    start_idx = batch_idx[0]
    stop_idx = batch_idx[1]
    # jaccard_matrix_batch = np.zeros((n_samples, n_samples))
    jaccard_matrix_batch = np.zeros((stop_idx - start_idx, n_samples))
    for i in range(start_idx, stop_idx):
        for j in range(i, n_samples):
            jaccard = pairwise_jaccard(adata, i, j)
            jaccard_matrix_batch[i, j] = jaccard
            # jaccard_matrix_batch[j, i] = jaccard
    print('=====Finished batch:', batch_idx)
    return jaccard_matrix_batch

def parallel_jaccard(adata, num_splits=21):
    n_samples = adata.shape[0]
    # batches = list(gen_batches(n_samples, n_samples // num_splits))
    k = 184
    batches = [[0, 16], 
               [16, 16+k], 
               [16+k, 16+2*k],
               [16+2*k, 16+3*k],
               [16+3*k, 16+4*k],
               [16+4*k, 16+5*k],
               [16+5*k, 16+6*k],
               [16+6*k, 16+7*k],
               [16+7*k, 16+8*k],
               [16+8*k, 16+9*k],
               [16+9*k, 16+10*k],
               [16+10*k, 16+11*k],
               [16+11*k, 16+12*k],
               [16+12*k, 16+13*k],
               [16+13*k, 16+14*k],
               [16+14*k, 16+15*k],
               [16+15*k, 16+16*k],
               [16+16*k, 16+17*k],
               [16+17*k, 16+18*k],
               [16+18*k, 16+19*k],
               [16+19*k, 16+20*k]]
    
    # Map the compute function to each batch of indices
    # pool = multiprocessing.Pool(processes=num_splits)
    # results = pool.starmap(compute_jaccard_matrix, [(adata, batch) for batch in batches])
    # pool.close()
    # pool.join()
    with ProcessingPool(processes=num_splits) as pool:
        results = pool.map(compute_jaccard_matrix_batch, [(adata, batch) for batch in batches])

    return results
    # Combine results
    # full_matrix = np.zeros((n_samples, n_samples))
    # for i, idx in enumerate(indices):
    #     full_matrix[np.ix_(idx, idx)] = results[i]
    result_dict = {}
    for i, idx in enumerate(batches):
        result_dict[idx] = results[i]

    return result_dict


adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378_raw.h5ad')
# adata = sc.read_h5ad(r"C:\Users\evanlee\Documents\Research_datasets\PBMC_Hao\GSE164378_Hao\Hao_PBMC_GSE164378_raw.h5ad")
# adata = sc.read_h5ad('/Users/evanli/Documents/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378_raw.h5ad')
print(adata.shape)

# early stop
sys.exit()

jaccard_matrix_para = parallel_jaccard(adata, num_splits=21)

# save to pickle
import pickle
try:
    with open('jaccard_matrix_para.pkl', 'wb') as f:
        pickle.dump(jaccard_matrix_para, f)
except:
    print('Error saving jaccard_matrix_para.pkl')

try:
    i = 0
    for mat in jaccard_matrix_para:
        sparse_mat = scipy.sparse.csr_matrix(mat)
        scipy.sparse.save_npz(f'Pancreas_jaccard_matrix_{i}.npz', sparse_mat)
except:
    print('Error saving Pancreas_jaccard_matrix')

i = 0
for mat in jaccard_matrix_para:
    np.savetxt(f'Pancreas_jaccard_matrix_{i}.csv', mat, delimiter=',')
    i += 1
