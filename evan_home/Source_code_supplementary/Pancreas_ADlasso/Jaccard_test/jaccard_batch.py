import sympy as sp
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import copy
import matplotlib.pyplot as plt
import scipy
import sys

import multiprocessing as mp
from sklearn.utils import gen_batches
from pathos.multiprocessing import ProcessingPool

## TODO: improve performance by utilizing sparse matrix and CuPy
# ChatGPT: https://chat.openai.com/share/d3ed10de-00f9-4bdd-be97-677321a64274

def split_isosceles_right_triangle(l, n):
    # Define the symbol for height
    h = sp.symbols('h')
    
    # Calculate the total area of the triangle
    total_area = 0.5 * l * l
    
    # Area of each section
    section_area = total_area / n
    
    # Function to calculate area up to height h in the triangle
    def area_up_to_h(h):
        return 0.5 * h * h
    
    # Calculate heights for each section
    heights = []
    for i in range(1, n):
        # Calculate the height for the i-th section, where i varies from 1 to n-1
        height = sp.solve(area_up_to_h(h) - i * section_area, h)[1]  # We take the positive root
        heights.append(float(height))
    
    # Append the total height of the triangle as the final height
    heights.append(l)

    # turn into intervals
    heights.insert(0, 0)
    heights = [h for h in reversed(heights)]
    intervals = []
    for i in range(len(heights)-1):
        curr_h = int(heights[i])
        next_h = int(heights[i+1])
        intervals.append([l-curr_h, l-next_h])
    
    return intervals

# Example usage
l = 3696  # side length of the triangle
n = 16     # number of equal parts
intervals = split_isosceles_right_triangle(l, n)
print(intervals)

# sys.exit()

# %%
def pairwise_jaccard(adata, idx_1, idx_2):
    a = adata.X[idx_1].tocsr()
    b = adata.X[idx_2].tocsr()
    del adata

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
            jaccard_matrix_batch[i - start_idx, j] = jaccard
            # jaccard_matrix_batch[j, i] = jaccard
    print('=====Finished batch:', batch_idx)
    return jaccard_matrix_batch

# def parallel_jaccard(adata, n_split=16):
#     batches = 

# %%
import pickle

if __name__ == '__main__':
    # adata = sc.read_h5ad('/home/jovyan/work/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378_raw.h5ad')
    # adata = sc.read_h5ad(r"C:\Users\evanlee\Documents\Research_datasets\PBMC_Hao\GSE164378_Hao\Hao_PBMC_GSE164378_raw.h5ad")
    # adata = sc.read_h5ad('/Users/evanli/Documents/Research_datasets/PBMC_Hao/GSE164378_Hao/Hao_PBMC_GSE164378_raw.h5ad')
    adata = scv.datasets.pancreas('/home/jovyan/work/Research_datasets/Pancreas/endocrinogenesis_day15.h5ad')
    print(adata.shape)

    n_samples = adata.shape[0]
    n_jobs = 16
    batches = split_isosceles_right_triangle(n_samples, n_jobs)
    print(batches)

    # sys.exit()

    with ProcessingPool(processes=n_jobs) as pool:
        para_results = pool.map(compute_jaccard_matrix_batch, [(adata, batch) for batch in batches])
    
    # save to pickle
    try:
        with open('jaccard_para_results.pkl', 'wb') as f:
            pickle.dump(para_results, f)
    except:
        print('Error saving pkl')
    
    try:
        i = 0
        for mat in para_results:
            sparse_mat = scipy.sparse.csr_matrix(mat)
            scipy.sparse.save_npz(f'Pancreas_jaccard_matrix_{i}.npz', sparse_mat)
            i += 1
    except:
        print('Error saving sparse Pancreas_jaccard_matrix')
    
    try:
        i = 0
        for mat in para_results:
            np.savetxt(f'Pancreas_jaccard_matrix_{i}.csv', mat, delimiter=',')
            i += 1
    except:
        print('Error saving dense Pancreas_jaccard_matrix')






