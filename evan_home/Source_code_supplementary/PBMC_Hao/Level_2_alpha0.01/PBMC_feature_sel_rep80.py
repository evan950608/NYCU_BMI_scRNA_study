import os
os.chdir('/home/evanlee/PBMC_Hao')
from ADlasso2 import AD2_w_utils_para as ad

import os
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import json
import time

# os.chdir('/Users/evanli/Documents/Research_datasets/PBMC_Hao')
# Read representative cells
adata = sc.read('/home/evanlee/PBMC_Hao/Hao_PBMC_rep_cells.h5ad')
print(adata.shape)  # row is cells, column is gene

label = adata.obs['celltype.l2'].tolist()
types = np.unique(label).tolist()


# %% Feature selection with optimal lambda
def pipeline_feature_selection(data, celltype, label, opt_lmbd, output_path=''):
    print('====================')
    print('Starting job for {}'.format(celltype))
    # Binary classification of a celltype
    celltype_label = [1 if x == celltype else 0 for x in label]
    # create index for a celltype
    celltype_indices = [idx for idx, label in enumerate(celltype_label) if label == 1]

    # Find marker genes with optimal lambda
    pvl = ad.get_prevalence(data.X, celltype_indices)
    print('Fitting with optimal lambda:', opt_lmbd)
    opt_res = ad.ADlasso2(lmbd=opt_lmbd, echo=True, device='cpu')
    opt_res.fit(data.X, celltype_label, pvl)
    
    # Export selection results
    os.chdir(output_path)
    opt_res.writeList(outpath=output_path+f'/{celltype}_features.txt', featureNameList=data.var_names)
    print(f'{celltype} feature list exported.')

    # Ouput description
    description = f'''Optimal lambda: {opt_lmbd}
    median of selected prevalence: {np.median([pvl[i]  for i, w in enumerate(opt_res.feature_set) if w != 0])}
    total selected feature: {np.sum(opt_res.feature_set)}\n'''
    print('---Selection result for {}'.format(celltype))
    print(description)

    with open(f'{celltype}_description.txt', 'w') as f:
        f.write(description)


# %%
# cts = types[:6]
# print(cts)

# for celltype in cts:
#     # read optimal lambda
#     os.chdir('/home/evanlee/PBMC_Hao/AD2_result')
#     with open(f'./{celltype}/{celltype}_opt_lambda.txt', 'r') as f:
#         opt_lmbd = float(f.read())
#     print(celltype, 'Optimal lambda:', opt_lmbd)

#     pipeline_feature_selection(adata, celltype, label, opt_lmbd, output_path='/home/evanlee/PBMC_Hao/AD2_result')


# %% Multi-processing
import multiprocessing as mp

cts = types[7:12]
print(cts)

# Define a function to execute pipeline_feature_selection for a single cell type
def run_pipeline_feature_selection(celltype):
    # read optimal lambda
    os.chdir('/home/evanlee/PBMC_Hao/AD2_result')
    with open(f'./{celltype}_opt_lambda.txt', 'r') as f:
        opt_lmbd = float(f.read())
    print(celltype, 'Optimal lambda:', opt_lmbd, '\n')

    pipeline_feature_selection(adata, celltype, label, opt_lmbd, output_path='/home/evanlee/PBMC_Hao/AD2_result')

# Create a pool of worker processes
with mp.Pool(processes=len(cts)) as pool:
    # Map the run_pipeline_feature_selection function to each cell type using the pool of workers
    pool.map(run_pipeline_feature_selection, cts)


